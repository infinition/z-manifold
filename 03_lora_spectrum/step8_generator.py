"""E2-lite : generateur non lineaire (autoencodeur) vs troncature PCA, a budget egal.

Pour chaque tache test (LOO) : coordonnees kernel-PCA de l'adaptateur retenu
dans la base des 195 autres, puis reconstruction par
  pca32 / pca64 : troncature aux k premieres composantes
  ae32          : autoencodeur MLP (bottleneck 32) entraine sur les 195 autres
Evaluation fonctionnelle : CE sur la validation P3 de la tache, modele reel.
Si l'AE ne bat pas la PCA, la variete est effectivement lineaire a ce budget.
Sorties : results/generator.json.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "generator.json")
SCALING = 2.0
N_EVAL = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = [
    "wiki_qa_Is_This_True_",
    "qasc_is_correct_1",
    "amazon_polarity_Is_this_product_review_positive",
    "social_i_qa_Check_if_a_random_answer_is_valid_or_not",
    "race_middle_Select_the_best_answer",
]
r = 16
rng = np.random.default_rng(0)
torch.manual_seed(0)

z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
G = z["G"]; names = [str(n) for n in z["names"]]
N = len(names)
name_of = {n.split("-", 1)[1]: n for n in names}
idx_of = {n: i for i, n in enumerate(names)}

def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[mod] = v.float().numpy()
    return A, B

A0, B0 = load_factors(names[0])
mods = sorted(A0.keys())
t0 = time.time()
As = {m: np.empty((N, *A0[m].shape), np.float16) for m in mods}
Bs = {m: np.empty((N, *B0[m].shape), np.float16) for m in mods}
for i, n in enumerate(names):
    A, B = load_factors(n)
    for m in mods:
        As[m][i], Bs[m][i] = A[m], B[m]
print(f"stacks {time.time()-t0:.0f}s", flush=True)

tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
model.eval().to(DEVICE)
params = dict(model.named_parameters())
base_w = {m: params[m + ".weight"].detach().clone() for m in mods}

def apply_coeffs(c):
    """W = base + s * sum_j c_j DeltaW_j, materialise par matmul empile."""
    nz = np.nonzero(c)[0]
    cw = c[nz].astype(np.float32)
    with torch.no_grad():
        for m in mods:
            Bh = (Bs[m][nz].astype(np.float32) * cw[:, None, None]).transpose(1, 0, 2).reshape(1024, -1)
            Ah = As[m][nz].astype(np.float32).reshape(-1, 1024)
            w = params[m + ".weight"]
            w.copy_(base_w[m])
            w.add_(torch.from_numpy(Bh @ Ah).to(w.device, w.dtype), alpha=SCALING)

def apply_none():
    with torch.no_grad():
        for m in mods:
            params[m + ".weight"].copy_(base_w[m])

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

def to_batches(pairs):
    out = []
    for i in range(0, len(pairs), 16):
        ins, tgt = zip(*pairs[i:i + 16])
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

class AE(nn.Module):
    def __init__(self, d, k=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, k))
        self.dec = nn.Sequential(nn.Linear(k, 128), nn.SiLU(), nn.Linear(128, d))
    def forward(self, x):
        return self.dec(self.enc(x))

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in TASKS:
    if task in R: continue
    name = name_of[task]; i = idx_of[name]
    others = [j for j in range(N) if j != i]
    n = len(others)
    Goo = G[np.ix_(others, others)]
    gio = G[i, others]
    H = np.eye(n) - 1.0 / n
    ev, V = np.linalg.eigh(H @ Goo @ H)
    ev, V = ev[::-1], V[:, ::-1]
    keep = ev > ev[0] * 1e-10
    ev, V = ev[keep], V[:, keep]
    D = len(ev)
    alpha = V / np.sqrt(ev)                                   # (n, D)
    gt = gio - gio.mean() - Goo.mean(0) + Goo.mean()
    beta = alpha.T @ gt                                       # coords du retenu (D,)
    Yo = V * np.sqrt(ev)                                      # coords des 195 autres (n, D)
    std = np.sqrt(ev / n)                                     # ecart-type par composante

    def coeffs_from_beta(bv):
        w = alpha @ bv
        c = np.zeros(N)
        c[others] = w + (1.0 - w.sum()) / n
        return c

    # PCA troncatures
    recon = {}
    for k in (32, 64):
        bv = beta.copy(); bv[k:] = 0.0
        recon[f"pca{k}"] = coeffs_from_beta(bv)

    # AE sur coordonnees normalisees des 195 autres
    Yn = torch.tensor(Yo / std, dtype=torch.float32, device=DEVICE)
    ae = AE(D, 32).to(DEVICE)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(3000):
        idx = torch.randint(0, n, (64,), device=DEVICE)
        loss = ((ae(Yn[idx]) - Yn[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        yh = ae(torch.tensor(beta / std, dtype=torch.float32, device=DEVICE)[None]).cpu().numpy()[0] * std
    recon["ae32"] = coeffs_from_beta(yh)
    ae_train_mse = float(loss)

    va = stream_pairs(task, ("validation", "test"), 256, N_EVAL)
    rg = np.random.default_rng(idx_of[name])
    vb = to_batches([va[int(j)] for j in rg.permutation(len(va))[:N_EVAL]])

    res = {"ae_train_mse": ae_train_mse}
    apply_none(); res["ce_base"] = eval_ce(vb)
    A_, B_ = load_factors(name)
    c_orig = np.zeros(N); c_orig[i] = 1.0
    apply_coeffs(c_orig); res["ce_orig"] = eval_ce(vb)
    for kname, c in recon.items():
        apply_coeffs(c); res[f"ce_{kname}"] = eval_ce(vb)
    apply_none()
    R[task] = res
    json.dump(R, open(OUT, "w"), indent=1)
    print(f"{task[:35]:<36} base={res['ce_base']:.3f} orig={res['ce_orig']:.3f} " +
          f"pca32={res['ce_pca32']:.3f} pca64={res['ce_pca64']:.3f} ae32={res['ce_ae32']:.3f} " +
          f"{time.time()-t0:.0f}s", flush=True)

print(f"FIN {time.time()-t0:.0f}s", flush=True)
