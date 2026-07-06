"""Dimension fonctionnelle vs dimension de poids.

Pour chaque adaptateur teste : projection leave-one-out sur le sous-espace PCA
top-k des 195 autres (algebre de Gram, jamais de base materialisee), application
du DeltaW projete a flan-t5-large, mesure de la cross-entropy par token sur la
validation P3 de la tache. Si CE(k petit) est proche de CE(original) et loin de
CE(base), la dimension fonctionnelle est basse malgre le spectre de poids.
"""
import os, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
SCALING = 2.0            # lora_alpha 32 / r 16
KS = [0, 8, 32, 128, -1] # -1 = span complet des 195 autres
N_EVAL = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16 if DEVICE == "cuda" else 8
rng = np.random.default_rng(0)
torch.manual_seed(0)

z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
G = z["G"]; names = [str(n) for n in z["names"]]
N = len(names)

def family(name):
    return "_".join(name.split("-", 1)[1].split("_")[:2])

fams = [family(n) for n in names]
fam_size = {f: fams.count(f) for f in set(fams)}

def p3_split(task):
    for sp in ("validation", "test"):
        try:
            ds = load_dataset("bigscience/P3", task, split=sp)
            if len(ds) >= N_EVAL:
                return ds
        except Exception:
            pass
    return None

# ---------- selection : 3 taches en grande famille, 3 isolees ----------
def pick(cands, n_want, taken):
    out = []
    for i in cands:
        if len(out) == n_want: break
        if fams[i] in taken: continue
        task = names[i].split("-", 1)[1]
        ds = p3_split(task)
        if ds is None:
            print(f"  pas de donnees P3: {task}", flush=True)
            continue
        out.append((i, ds)); taken.add(fams[i])
    return out

order_big = sorted(range(N), key=lambda i: -fam_size[fams[i]])
order_iso = [i for i in rng.permutation(N) if fam_size[fams[i]] == 1]
taken = set()
sel = pick(order_big, 3, taken) + pick(order_iso, 3, taken)
print("taches:", [(names[i], fam_size[fams[i]]) for i, _ in sel], flush=True)

# ---------- stacks de facteurs (fp16) ----------
def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"),
                    map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[mod] = v.float().numpy()
    return A, B

A0, B0 = load_factors(names[0])
mods = sorted(A0.keys())
r = A0[mods[0]].shape[0]
t0 = time.time()
As = {m: np.empty((N, *A0[m].shape), np.float16) for m in mods}
Bs = {m: np.empty((N, *B0[m].shape), np.float16) for m in mods}
for i, n in enumerate(names):
    A, B = load_factors(n)
    for m in mods:
        As[m][i], Bs[m][i] = A[m], B[m]
print(f"stacks charges {time.time()-t0:.0f}s", flush=True)

# ---------- projection LOO en algebre de Gram ----------
def loo_coeffs(i):
    """Retourne {k: c} avec c (196,) coefficients sur TOUS les adaptateurs
    (c[i]=0), tels que DeltaW_proj = sum_j c_j DeltaW_j, plus l'erreur relative
    de reconstruction en espace de poids par k."""
    others = [j for j in range(N) if j != i]
    n = len(others)
    Goo = G[np.ix_(others, others)]
    gio = G[i, others]
    H = np.eye(n) - 1.0 / n
    Gc = H @ Goo @ H
    ev, V = np.linalg.eigh(Gc)
    ev, V = ev[::-1], V[:, ::-1]
    keep = ev > ev[0] * 1e-10
    ev, V = ev[keep], V[:, keep]
    alpha = V / np.sqrt(ev)                       # (n, K) coeffs des vecteurs propres unitaires
    gt = gio - gio.mean() - Goo.mean(0) + Goo.mean()   # <x_i - mu, x_j - mu>
    beta = alpha.T @ gt                            # projections sur chaque composante
    xc2 = G[i, i] - 2 * gio.mean() + Goo.mean()    # ||x_i - mu||^2
    out = {}
    for K in KS:
        kk = len(ev) if K == -1 else min(K, len(ev))
        w = alpha[:, :kk] @ beta[:kk] if kk > 0 else np.zeros(n)
        c = np.zeros(N)
        c[others] = w + (1.0 - w.sum()) / n        # + mu
        rel_err = float(1.0 - (beta[:kk] ** 2).sum() / xc2) if xc2 > 0 else 1.0
        out[K] = (c, rel_err)
    return out

# ---------- modele ----------
tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32)
model.eval().to(DEVICE)
print(f"device: {DEVICE}", flush=True)
params = dict(model.named_parameters())
base_w = {m: params[m + ".weight"].detach().clone() for m in mods}

def apply_delta(deltas):  # deltas: {mod: np.ndarray} ou None pour base
    with torch.no_grad():
        for m in mods:
            w = params[m + ".weight"]
            w.copy_(base_w[m])
            if deltas is not None:
                w.add_(torch.from_numpy(deltas[m]).to(w.device), alpha=SCALING)

def delta_from_coeffs(c):
    nz = np.nonzero(c)[0]
    cw = c[nz].astype(np.float32)
    out = {}
    for m in mods:
        Bh = (Bs[m][nz].astype(np.float32) * cw[:, None, None]).transpose(1, 0, 2).reshape(1024, -1)
        Ah = As[m][nz].astype(np.float32).reshape(-1, 1024)
        out[m] = Bh @ Ah
    return out

def delta_single(i):
    return {m: Bs[m][i].astype(np.float32) @ As[m][i].astype(np.float32) for m in mods}

# ---------- evaluation CE par token ----------
def make_batches(ds):
    idx = rng.permutation(len(ds))[:N_EVAL]
    exs = [(ds[int(j)]["inputs_pretokenized"].strip(), ds[int(j)]["targets_pretokenized"].strip()) for j in idx]
    batches = []
    for b in range(0, N_EVAL, BATCH):
        ins, tgt = zip(*exs[b:b + BATCH])
        enc = tok(list(ins), max_length=256, truncation=True, padding=True, return_tensors="pt")
        lab = tok(list(tgt), max_length=64, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        batches.append((enc, lab))
    return batches

@torch.no_grad()
def eval_ce(batches):
    tot, ntok = 0.0, 0
    for enc, lab in batches:
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        lab = lab.to(DEVICE)
        out = model(**enc, labels=lab)
        n = int((lab != -100).sum())
        tot += float(out.loss) * n
        ntok += n
    return tot / ntok

# ---------- boucle principale ----------
R = {}
for i, ds in sel:
    name = names[i]; task = name.split("-", 1)[1]
    print(f"\n=== {task} (famille {fams[i]}, taille {fam_size[fams[i]]}) ===", flush=True)
    batches = make_batches(ds)
    coeffs = loo_coeffs(i)
    res = {"family": fams[i], "family_size": fam_size[fams[i]], "ce": {}, "weight_rel_err": {}}
    apply_delta(None);            res["ce"]["base"] = eval_ce(batches)
    print(f"  base  CE={res['ce']['base']:.4f} {time.time()-t0:.0f}s", flush=True)
    apply_delta(delta_single(i)); res["ce"]["orig"] = eval_ce(batches)
    print(f"  orig  CE={res['ce']['orig']:.4f}", flush=True)
    for K in KS:
        c, rel_err = coeffs[K]
        apply_delta(delta_from_coeffs(c))
        key = "full" if K == -1 else f"k{K}"
        res["ce"][key] = eval_ce(batches)
        res["weight_rel_err"][key] = rel_err
        print(f"  {key:5s} CE={res['ce'][key]:.4f} rel_err_poids={rel_err:.3f} {time.time()-t0:.0f}s", flush=True)
    R[task] = res
    json.dump(R, open(os.path.join(RDIR, "functional_dim.json"), "w"), indent=1)

print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
print(json.dumps(R, indent=1))
