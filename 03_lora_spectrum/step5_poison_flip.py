"""Poisoning par INVERSION ciblee des labels, 3 seeds, CE + exact match.

Conditions par tache :
  clean    : 128 exemples propres
  flip50   : 50% des cibles remplacees par une cible DIFFERENTE d'un autre exemple
  flip100  : 100% remplacees (toutes fausses)
  garbage  : toutes les cibles remplacees par des mots aleatoires (test OOD)
Methodes : lora (r=16 frais, 9.4M params) vs zman (K=128 coords, base LOO).
Sorties : results/poison_flip.json (incremental), inclut train_loss pour l'AUROC.
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "poison_flip.json")
SCALING = 2.0
K = 128
N_TRAIN = 128
N_EVAL = 64
EPOCHS = 8
LR_LORA = 3e-4
LR_Z = 3e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8
SEEDS = (0, 1, 2)
TASKS = [
    "wiki_qa_Is_This_True_",
    "qasc_is_correct_1",
    "amazon_polarity_Is_this_product_review_positive",
    "social_i_qa_Check_if_a_random_answer_is_valid_or_not",
    "race_middle_Select_the_best_answer",
]
CONDS = ("clean", "flip50", "flip100", "garbage")
r = 16
rng = np.random.default_rng(0)

# ---------- gram, base LOO ----------
z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
G = z["G"]; names = [str(n) for n in z["names"]]
N = len(names)
name_of = {n.split("-", 1)[1]: n for n in names}
idx_of = {n: i for i, n in enumerate(names)}

def basis_excluding(excl):
    others = [j for j in range(N) if j not in excl]
    n = len(others)
    Goo = G[np.ix_(others, others)]
    H = np.eye(n) - 1.0 / n
    ev, V = np.linalg.eigh(H @ Goo @ H)
    ev, V = ev[::-1][:K], V[:, ::-1][:, :K]
    alpha = V / np.sqrt(np.clip(ev, 1e-12, None))
    scale = np.sqrt(np.clip(ev, 0, None) / n)
    Mw = alpha * scale
    Mw = Mw - Mw.mean(0, keepdims=True)
    M = np.zeros((N, K)); b = np.zeros(N)
    M[others] = Mw; b[others] = 1.0 / n
    return torch.tensor(M, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)

# ---------- stacks ----------
def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[mod] = v.float()
    return A, B

A0, _ = load_factors(names[0])
mods = sorted(A0.keys())
Ahat, Bhat = {}, {}
for m in mods:
    Ahat[m] = torch.empty(N * r, 1024, dtype=torch.float16)
    Bhat[m] = torch.empty(1024, N * r, dtype=torch.float16)
t0 = time.time()
for i, n in enumerate(names):
    A, B = load_factors(n)
    for m in mods:
        Ahat[m][i * r:(i + 1) * r] = A[m].half()
        Bhat[m][:, i * r:(i + 1) * r] = B[m].half()
print(f"stacks CPU {time.time()-t0:.0f}s", flush=True)

# ---------- modele et wrappers ----------
tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
model.eval().to(DEVICE)
for p in model.parameters(): p.requires_grad_(False)

class ZLinear(nn.Module):
    def __init__(self, base, m, state):
        super().__init__()
        self.weight = base.weight
        self.m = m; self.state = state
    def forward(self, x):
        y = F.linear(x, self.weight)
        st = self.state
        g = (st["M"] @ st["z"] + st["b"]).half()
        u = F.linear(x.half(), st["Ah"][self.m])
        u = u * g.repeat_interleave(r)
        return y + SCALING * F.linear(u, st["Bh"][self.m]).to(y.dtype)

class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.weight = base.weight
        self.A = nn.Parameter(torch.empty(r, 1024))
        self.B = nn.Parameter(torch.zeros(1024, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    def forward(self, x):
        y = F.linear(x, self.weight)
        return y + SCALING * F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))

originals = {m: model.get_submodule(m) for m in mods}
def parent_and_attr(m):
    pp, a = m.rsplit(".", 1)
    return model.get_submodule(pp), a

def install(kind, state=None):
    reps = {}
    for m in mods:
        p, a = parent_and_attr(m)
        w = ZLinear(originals[m], m, state) if kind == "z" else LoRALinear(originals[m]).to(DEVICE)
        setattr(p, a, w); reps[m] = w
    return reps

def restore():
    for m in mods:
        p, a = parent_and_attr(m)
        setattr(p, a, originals[m])

def apply_static_delta(deltas):
    with torch.no_grad():
        for m in mods:
            w = originals[m].weight
            if not hasattr(originals[m], "_base"):
                originals[m]._base = w.detach().clone()
            w.copy_(originals[m]._base)
            if deltas is not None:
                w.add_(deltas[m].to(w.device, w.dtype), alpha=SCALING)

# ---------- donnees ----------
def stream_pairs(task, splits, cap, minimum):
    for sp in splits:
        try:
            d = load_dataset("bigscience/P3", task, split=sp, streaming=True)
            pairs = []
            for ex in d:
                pairs.append((ex["inputs_pretokenized"].strip(), ex["targets_pretokenized"].strip()))
                if len(pairs) >= cap: break
            if len(pairs) >= minimum:
                return pairs
        except Exception:
            pass
    return None

def task_data(task, seed):
    tr = stream_pairs(task, ("train",), cap=512, minimum=N_TRAIN)
    va = stream_pairs(task, ("validation", "test"), cap=256, minimum=N_EVAL)
    rg = np.random.default_rng(seed)
    train = [tr[int(j)] for j in rg.permutation(len(tr))[:N_TRAIN]]
    val = [va[int(j)] for j in rg.permutation(len(va))[:N_EVAL]]
    return train, val

WORDS = ("table maison riviere concept lampe orange velo nuage sable porte "
         "montagne livre chaise racine metal souris vent verre chemin tour").split()

def corrupt(pairs, cond, seed):
    if cond == "clean": return pairs
    rg = np.random.default_rng(seed)
    out = list(pairs)
    if cond == "garbage":
        for a in range(len(out)):
            out[a] = (out[a][0], " ".join(rg.choice(WORDS, 3)))
        return out
    p = 0.5 if cond == "flip50" else 1.0
    n = len(pairs); k = int(p * n)
    idx = rg.choice(n, k, replace=False)
    targets = [t for _, t in pairs]
    for a in idx:
        wrong = [t for t in targets if t != pairs[a][1]]
        out[a] = (pairs[a][0], str(rg.choice(wrong)) if wrong else pairs[a][1])
    return out

def to_batches(pairs, bs):
    out = []
    for i in range(0, len(pairs), bs):
        ins, tgt = zip(*pairs[i:i + bs])
        enc = tok(list(ins), max_length=192, truncation=True, padding=True, return_tensors="pt")
        lab = tok(list(tgt), max_length=48, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        out.append(({k: v.to(DEVICE) for k, v in enc.items()}, lab.to(DEVICE)))
    return out

@torch.no_grad()
def eval_ce(batches):
    tot, ntok = 0.0, 0
    for enc, lab in batches:
        out = model(**enc, labels=lab)
        n = int((lab != -100).sum())
        tot += float(out.loss) * n; ntok += n
    return tot / ntok

def norm_txt(s):
    return " ".join(s.lower().strip().rstrip(".").split())

@torch.no_grad()
def eval_em(val_pairs):
    hits = 0
    for i in range(0, len(val_pairs), 16):
        chunk = val_pairs[i:i + 16]
        enc = tok([a for a, _ in chunk], max_length=192, truncation=True,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=16, num_beams=1, do_sample=False)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        hits += sum(norm_txt(d) == norm_txt(t) for d, (_, t) in zip(dec, chunk))
    return hits / len(val_pairs)

def train(params, tb, lr, seed):
    opt = torch.optim.AdamW(params, lr=lr)
    rg = np.random.default_rng(seed)
    last = []
    for ep in range(EPOCHS):
        for j in rg.permutation(len(tb)):
            enc, lab = tb[int(j)]
            loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()
            if ep == EPOCHS - 1: last.append(float(loss))
    return float(np.mean(last))

# ---------- boucle ----------
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in TASKS:
    name = name_of[task]; i = idx_of[name]
    M, b = basis_excluding({i})
    res = R.get(task, {})
    for sd in SEEDS:
        train_pairs, val_pairs = task_data(task, seed=1000 * sd + i)
        vb = to_batches(val_pairs, 16)
        rk = f"ref_s{sd}"
        if rk not in res:
            apply_static_delta(None)
            ce_b, em_b = eval_ce(vb), eval_em(val_pairs)
            A_, B_ = load_factors(name)
            apply_static_delta({m: B_[m] @ A_[m] for m in mods})
            ce_o, em_o = eval_ce(vb), eval_em(val_pairs)
            apply_static_delta(None)
            res[rk] = {"ce_base": ce_b, "em_base": em_b, "ce_orig": ce_o, "em_orig": em_o}
            print(f"=== {task} s{sd} base ce={ce_b:.3f} em={em_b:.2f} | orig ce={ce_o:.3f} em={em_o:.2f}", flush=True)
            R[task] = res; json.dump(R, open(OUT, "w"), indent=1)
        for cond in CONDS:
            tb = to_batches(corrupt(train_pairs, cond, seed=2000 * sd + i), BATCH)
            for kind in ("lora", "zman"):
                key = f"{kind}_{cond}_s{sd}"
                if key in res: continue
                torch.manual_seed(3000 * sd + i)
                state = None
                if kind == "zman":
                    state = {"M": M.to(DEVICE), "b": b.to(DEVICE),
                             "z": torch.zeros(K, device=DEVICE, requires_grad=True),
                             "Ah": {m: Ahat[m].to(DEVICE) for m in mods},
                             "Bh": {m: Bhat[m].to(DEVICE) for m in mods}}
                    install("z", state); params = [state["z"]]; lr = LR_Z
                else:
                    reps = install("lora")
                    params = [q for m in mods for q in (reps[m].A, reps[m].B)]; lr = LR_LORA
                try:
                    tl = train(params, tb, lr, seed=4000 * sd + i)
                    ce = eval_ce(vb); em = eval_em(val_pairs)
                finally:
                    restore()
                    if state is not None: state["Ah"] = state["Bh"] = None
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                res[key] = {"train_loss": tl, "ce_val": ce, "em_val": em}
                print(f"  {key:22s} train={tl:.3f} ce={ce:.3f} em={em:.2f} {time.time()-t0:.0f}s", flush=True)
                R[task] = res; json.dump(R, open(OUT, "w"), indent=1)

print(f"FIN {time.time()-t0:.0f}s", flush=True)
