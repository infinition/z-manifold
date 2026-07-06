"""Go/no-go : dimension intrinseque du dataset de LoRA LoraHub (flan-t5-large, r16, q+v).

Calcule le spectre PCA des adaptateurs dans l'espace des produits DeltaW = B@A
(jamais materialises : Gram par trace factorisee), le compare a un null de LoRA
aleatoires a normes egalees, et mesure la structure par famille de taches.
Reference du piege raw-space : Text-to-LoRA Appendix D (arXiv 2506.06105).
"""
import os, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
os.makedirs(RDIR, exist_ok=True)
rng = np.random.default_rng(0)

# ---------- chargement ----------
names = sorted(d for d in os.listdir(ADIR) if os.path.exists(os.path.join(ADIR, d, "adapter_model.bin")))
N = len(names)
print(f"{N} adaptateurs", flush=True)

def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"),
                    map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        if "lora_A" in k: A[mod] = v.float().numpy()
        elif "lora_B" in k: B[mod] = v.float().numpy()
    return A, B

A0, B0 = load_factors(names[0])
mods = sorted(A0.keys())
r = A0[mods[0]].shape[0]
print(f"{len(mods)} modules, rang {r}, exemple: {mods[0]}", flush=True)

# Stacks (N, r, d_in) et (N, d_out, r) par module
t0 = time.time()
As = {m: np.empty((N, *A0[m].shape), np.float32) for m in mods}
Bs = {m: np.empty((N, *B0[m].shape), np.float32) for m in mods}
for i, n in enumerate(names):
    A, B = load_factors(n)
    for m in mods:
        As[m][i], Bs[m][i] = A[m], B[m]
    if (i + 1) % 50 == 0:
        print(f"charge {i+1}/{N} {time.time()-t0:.0f}s", flush=True)

def module_group(m):
    part = "enc" if m.startswith("encoder") else "dec"
    att = "cross" if "EncDecAttention" in m else "self"
    proj = m.rsplit(".", 1)[1]  # q ou v
    return f"{part}.{att}.{proj}"

# ---------- Gram DeltaW sans materialiser DeltaW ----------
# <B_i A_i, B_j A_j>_F = tr((B_i^T B_j)(A_j A_i^T))
def gram_deltaw(As, Bs):
    G = np.zeros((N, N), np.float64)
    Gg = {}
    for m in mods:
        BtB = np.einsum("iur,jus->ijrs", Bs[m], Bs[m], optimize=True)
        AAt = np.einsum("jsd,ird->ijrs", As[m], As[m], optimize=True)
        Gm = np.einsum("ijrs,ijrs->ij", BtB, AAt, optimize=True)
        G += Gm
        g = module_group(m)
        Gg[g] = Gg.get(g, 0) + Gm
    return G, Gg

G, Ggrp = gram_deltaw(As, Bs)
print(f"Gram DeltaW {time.time()-t0:.0f}s", flush=True)

# Gram raw-space (concat A,B) pour repliquer le piege T2L App. D
Graw = np.zeros((N, N), np.float64)
for m in mods:
    Graw += np.einsum("ird,jrd->ij", As[m], As[m], optimize=True)
    Graw += np.einsum("idr,jdr->ij", Bs[m], Bs[m], optimize=True)

# ---------- spectres ----------
def spectrum(G, normalize=True):
    if normalize:  # adaptateurs a norme Frobenius unite
        d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
        G = G / np.outer(d, d)
    n = G.shape[0]
    H = np.eye(n) - 1.0 / n           # double centrage (kernel PCA)
    ev = np.linalg.eigvalsh(H @ G @ H)[::-1]
    ev = np.clip(ev, 0, None)
    ve = ev / ev.sum()
    cum = np.cumsum(ve)
    return {
        "top_eigval_share": [float(ve[k]) for k in range(5)],
        "cum_var": {str(k): float(cum[k - 1]) for k in (2, 4, 8, 16, 32, 64) if k <= n - 1},
        "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
        "eff_rank_entropy": float(np.exp(-(ve[ve > 0] * np.log(ve[ve > 0])).sum())),
    }

# ---------- null : LoRA aleatoires, normes par matrice egalees ----------
def gram_null():
    An = {m: rng.standard_normal(As[m].shape).astype(np.float32) for m in mods}
    Bn = {m: rng.standard_normal(Bs[m].shape).astype(np.float32) for m in mods}
    for m in mods:
        for X, Xn in ((As[m], An[m]), (Bs[m], Bn[m])):
            nrm = np.linalg.norm(X.reshape(N, -1), axis=1) / np.linalg.norm(Xn.reshape(N, -1), axis=1)
            Xn *= nrm[:, None, None]
    return gram_deltaw(An, Bn)[0]

Gnull = gram_null()
print(f"Gram null {time.time()-t0:.0f}s", flush=True)

# ---------- cosinus intra/inter famille ----------
def family(name):
    task = name.split("-", 1)[1]
    toks = task.split("_")
    return "_".join(toks[:2])

fams = [family(n) for n in names]
d = np.sqrt(np.diag(G)); C = G / np.outer(d, d)
iu = np.triu_indices(N, 1)
same = np.array([fams[i] == fams[j] for i, j in zip(*iu)])
cos_all = C[iu]
res_fam = {
    "n_families": len(set(fams)),
    "cos_intra_famille": float(cos_all[same].mean()),
    "cos_inter_famille": float(cos_all[~same].mean()),
    "cos_abs_moyen": float(np.abs(cos_all).mean()),
}

R = {
    "n_adapters": N, "n_modules": len(mods), "rank": r,
    "spectre_deltaw": spectrum(G),
    "spectre_null_aleatoire": spectrum(Gnull),
    "spectre_raw_AB": spectrum(Graw),
    "spectre_par_groupe": {g: spectrum(Gg) for g, Gg in sorted(Ggrp.items())},
    "familles": res_fam,
}
np.savez(os.path.join(RDIR, "grams.npz"), G=G, Graw=Graw, Gnull=Gnull,
         names=np.array(names), **{f"grp_{g}": v for g, v in Ggrp.items()})
json.dump(R, open(os.path.join(RDIR, "spectrum.json"), "w"), indent=1)
print(json.dumps(R, indent=1), flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
