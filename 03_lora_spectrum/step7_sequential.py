"""Adaptation sequentielle : chaine de 5 taches, oubli et recuperation.

Pour chaque seed, une permutation des 5 taches. Deux methodes a budget egal :
  lora : UN LoRA r=16 entraine en continu de tache en tache
  zman : UN vecteur z (K=128) deplace de tache en tache sur la base du pool
         (les 5 taches de la chaine sont EXCLUES de la base)
Apres chaque etape : CE sur les validations des 5 taches (matrice d'oubli).
A la fin : recuperation de la tache 1 avec 16 exemples.
Sorties : results/sequential.json.
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
OUT = os.path.join(RDIR, "sequential.json")
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
r = 16

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

def train(params, tb, lr, seed, epochs=EPOCHS):
    opt = torch.optim.AdamW(params, lr=lr)
    rg = np.random.default_rng(seed)
    for ep in range(epochs):
        for j in rg.permutation(len(tb)):
            enc, lab = tb[int(j)]
            loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()

# ---------- donnees des 5 taches, par seed ----------
excl = {idx_of[name_of[t]] for t in TASKS}
Mz, bz = basis_excluding(excl)

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for sd in SEEDS:
    rg = np.random.default_rng(sd)
    order = [TASKS[int(j)] for j in rg.permutation(len(TASKS))]
    data = {}
    for t in TASKS:
        i = idx_of[name_of[t]]
        tr = stream_pairs(t, ("train",), 512, N_TRAIN)
        va = stream_pairs(t, ("validation", "test"), 256, N_EVAL)
        rgt = np.random.default_rng(1000 * sd + i)
        data[t] = {
            "train": to_batches([tr[int(j)] for j in rgt.permutation(len(tr))[:N_TRAIN]], BATCH),
            "rec": to_batches([tr[int(j)] for j in rgt.permutation(len(tr))[N_TRAIN:N_TRAIN + 16]], BATCH),
            "val": to_batches([va[int(j)] for j in rgt.permutation(len(va))[:N_EVAL]], 16),
        }
    for kind in ("lora", "zman"):
        key = f"{kind}_s{sd}"
        if key in R: continue
        torch.manual_seed(5000 + sd)
        state = None
        if kind == "zman":
            state = {"M": Mz.to(DEVICE), "b": bz.to(DEVICE),
                     "z": torch.zeros(K, device=DEVICE, requires_grad=True),
                     "Ah": {m: Ahat[m].to(DEVICE) for m in mods},
                     "Bh": {m: Bhat[m].to(DEVICE) for m in mods}}
            reps = install("z", state); params = [state["z"]]; lr = LR_Z
        else:
            reps = install("lora")
            params = [q for m in mods for q in (reps[m].A, reps[m].B)]; lr = LR_LORA
        try:
            mat = []
            for stage, t in enumerate(order):
                train(params, data[t]["train"], lr, seed=6000 * sd + stage)
                row = {u: eval_ce(data[u]["val"]) for u in order[:stage + 1]}
                mat.append(row)
                print(f"{key} etape {stage+1}/5 ({t[:30]}) " +
                      " ".join(f"{u[:12]}={v:.2f}" for u, v in row.items()) +
                      f" {time.time()-t0:.0f}s", flush=True)
            train(params, data[order[0]]["rec"], lr, seed=7000 * sd, epochs=8)
            rec = eval_ce(data[order[0]]["val"])
            print(f"{key} recuperation tache1 = {rec:.3f}", flush=True)
        finally:
            restore()
            if state is not None: state["Ah"] = state["Bh"] = None
            if DEVICE == "cuda": torch.cuda.empty_cache()
        R[key] = {"order": order, "matrix": mat, "recovery_task1": rec}
        json.dump(R, open(OUT, "w"), indent=1)

print(f"FIN {time.time()-t0:.0f}s", flush=True)
