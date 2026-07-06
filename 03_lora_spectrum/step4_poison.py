"""Experience centrale : adaptation contrainte a la variete vs LoRA fine-tuning
sous poisoning de labels.

Pour chaque tache discriminante (top gain du criblage, familles distinctes) :
  conditions p in {0%, 50%, 100%} de labels melanges sur 128 exemples de train P3
  methodes :
    lora : LoRA r=16 frais sur q+v (9.4M params), baseline standard
    zman : z (K=128 coords) sur la base PCA leave-one-out des 195 autres
           adaptateurs, modele et base geles, seul z bouge
  mesures : CE validation propre, CE probe generique (integrite), loss train
            finale (signal OOD a p=100%), reference base et adaptateur original.
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
OUT = os.path.join(RDIR, "poison.json")
SCALING = 2.0
K = 128
N_TASKS = 8
N_TRAIN = 128
N_EVAL = 64
EPOCHS = 8
LR_LORA = 3e-4
LR_Z = 3e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8
CONDS = [0.0, 0.5, 1.0]
rng = np.random.default_rng(0)
torch.manual_seed(0)

# ---------- selection des taches depuis le criblage ----------
S = json.load(open(os.path.join(RDIR, "screen.json")))
def fam(task): return "_".join(task.split("_")[:2])
scored = sorted((v["gain"], t) for t, v in S.items() if "gain" in v)[::-1]
tasks, seen = [], set()
for g, t in scored:
    if fam(t) in seen: continue
    tasks.append(t); seen.add(fam(t))
    if len(tasks) == N_TASKS: break
probe_tasks = [t for g, t in scored if fam(t) not in seen][:4]
print("taches:", [(t, round(S[t]["gain"], 3)) for t in tasks], flush=True)
print("probe:", probe_tasks, flush=True)

# ---------- gram, base PCA leave-one-out ----------
z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
G = z["G"]; names = [str(n) for n in z["names"]]
N = len(names)
name_of = {n.split("-", 1)[1]: n for n in names}
idx_of = {n: i for i, n in enumerate(names)}

def loo_basis(i):
    """M (N,K), b (N,) tels que DeltaW(zc) = sum_j (M @ zs + b)_j DeltaW_j,
    zc en unites d'ecart-type des projections empiriques."""
    others = [j for j in range(N) if j != i]
    n = len(others)
    Goo = G[np.ix_(others, others)]
    H = np.eye(n) - 1.0 / n
    ev, V = np.linalg.eigh(H @ Goo @ H)
    ev, V = ev[::-1][:K], V[:, ::-1][:, :K]
    alpha = V / np.sqrt(np.clip(ev, 1e-12, None))       # (n, K)
    scale = np.sqrt(np.clip(ev, 0, None) / n)            # std empirique par composante
    Mw = alpha * scale                                   # w = Mw @ zc
    Mw = Mw - Mw.mean(0, keepdims=True)                  # g_j = w_j - mean(w) + 1/n
    M = np.zeros((N, K)); b = np.zeros(N)
    M[others] = Mw; b[others] = 1.0 / n
    return torch.tensor(M, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)

# ---------- stacks de facteurs sur GPU (fp16) ----------
def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[mod] = v.float()
    return A, B

A0, _ = load_factors(names[0])
mods = sorted(A0.keys())
r = 16
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
# Stacks gardes sur CPU, montes sur GPU seulement pendant les runs zman (VRAM 12 Go partagee avec le bureau)
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
        self.weight = base.weight   # gele
        self.m = m; self.state = state
    def forward(self, x):
        y = F.linear(x, self.weight)
        st = self.state
        g = (st["M"] @ st["z"] + st["b"]).half()         # (N,)
        u = F.linear(x.half(), st["Ah"][self.m])
        u = u * g.repeat_interleave(r)
        return y + SCALING * F.linear(u, st["Bh"][self.m]).to(y.dtype)

class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.weight = base.weight   # gele
        self.A = nn.Parameter(torch.empty(r, 1024))   # fp32 pour l'optimiseur
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

def apply_static_delta(deltas):  # pour la reference "orig"
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
    """Streaming : ne telecharge que les premiers exemples, pas le dataset entier."""
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

def task_data(task, seed):
    tr = stream_pairs(task, ("train",), cap=512, minimum=N_TRAIN)
    va = stream_pairs(task, ("validation", "test"), cap=256, minimum=N_EVAL)
    rg = np.random.default_rng(seed)
    train = [tr[int(j)] for j in rg.permutation(len(tr))[:N_TRAIN]]
    val = [va[int(j)] for j in rg.permutation(len(va))[:N_EVAL]]
    return train, val

def poison(pairs, p, seed):
    if p == 0: return pairs
    rg = np.random.default_rng(seed)
    n = len(pairs); k = int(p * n)
    idx = rg.choice(n, k, replace=False)
    perm = rg.permutation(idx)
    out = list(pairs)
    for a, b in zip(idx, perm):
        out[a] = (pairs[a][0], pairs[b][1])
    return out

@torch.no_grad()
def eval_ce(batches):
    tot, ntok = 0.0, 0
    for enc, lab in batches:
        out = model(**enc, labels=lab)
        n = int((lab != -100).sum())
        tot += float(out.loss) * n; ntok += n
    return tot / ntok

def train(params, tb, lr):
    opt = torch.optim.AdamW(params, lr=lr)
    last = []
    for ep in range(EPOCHS):
        order = rng.permutation(len(tb))
        for j in order:
            enc, lab = tb[j]
            loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()
            if ep == EPOCHS - 1: last.append(float(loss))
    return float(np.mean(last))

# ---------- probe generique (integrite) ----------
probe_pairs = []
for pt in probe_tasks:
    d = stream_pairs(pt, ("validation", "test"), cap=64, minimum=16)
    if d is None: continue
    rg = np.random.default_rng(1)
    for j in rg.permutation(len(d))[:16]:
        probe_pairs.append(d[int(j)])
probe_b = to_batches(probe_pairs, BATCH)
print(f"probe: {len(probe_pairs)} exemples", flush=True)

# ---------- boucle principale ----------
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in tasks:
    if task in R and "done" in R[task]: continue
    name = name_of[task]; i = idx_of[name]
    train_pairs, val_pairs = task_data(task, seed=i)
    vb = to_batches(val_pairs, 16)
    M, b = loo_basis(i)
    res = R.get(task, {})
    apply_static_delta(None); res["ce_base"] = eval_ce(vb); res["probe_base"] = eval_ce(probe_b)
    A_, B_ = load_factors(name)
    apply_static_delta({m: B_[m] @ A_[m] for m in mods})
    res["ce_orig"] = eval_ce(vb)
    apply_static_delta(None)
    print(f"\n=== {task} base={res['ce_base']:.3f} orig={res['ce_orig']:.3f} ===", flush=True)
    for p in CONDS:
        tb = to_batches(poison(train_pairs, p, seed=100 + i), BATCH)
        for kind in ("lora", "zman"):
            key = f"{kind}_p{int(p*100)}"
            if key in res: continue
            torch.manual_seed(1000 + i)
            if kind == "zman":
                state = {"M": M.to(DEVICE), "b": b.to(DEVICE),
                         "z": torch.zeros(K, device=DEVICE, requires_grad=True),
                         "Ah": {m: Ahat[m].to(DEVICE) for m in mods},
                         "Bh": {m: Bhat[m].to(DEVICE) for m in mods}}
                install("z", state); params = [state["z"]]
                lr = LR_Z
            else:
                state = None
                reps = install("lora")
                params = [q for m in mods for q in (reps[m].A, reps[m].B)]
                lr = LR_LORA
            try:
                tl = train(params, tb, lr)
                ce = eval_ce(vb); pr = eval_ce(probe_b)
            finally:
                restore()
                if state is not None:
                    state["Ah"] = state["Bh"] = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            res[key] = {"train_loss": tl, "ce_val": ce, "probe": pr}
            print(f"  {key:12s} train={tl:.3f} val={ce:.3f} probe={pr:.3f} {time.time()-t0:.0f}s", flush=True)
            R[task] = res; json.dump(R, open(OUT, "w"), indent=1)
    res["done"] = True
    R[task] = res; json.dump(R, open(OUT, "w"), indent=1)

print(f"\nFIN {time.time()-t0:.0f}s", flush=True)
