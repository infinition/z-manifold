"""ATTAQUE ADAPTATIVE : backdoor a trigger, l'attaquant vise un comportement
EXPRIMABLE dans le sous-espace (contrairement a l'inversion, hors sous-espace).

Trigger = token rare appende a l'entree. Cible = "No". Deux modes :
  backdoor : 50% train propre (label vrai) + 50% triggered->No. Mesure si le
             backdoor prend TOUT EN preservant l'accuracy propre.
  maxasr   : 100% train triggered->No. Mesure le PLAFOND d'ASR que le
             defenseur peut etre pousse a exprimer (expressivite pure).
Defenseurs : zman (sous-espace du pool) vs lora (libre).
Metriques : clean_acc (EM sur val propre), asr (sur val dont vrai label != cible,
            trigger ajoute, fraction sortant la cible).
Lecture : si lora atteint ASR haut mais zman plafonne bas -> le sous-espace ne
peut structurellement pas exprimer le backdoor (resultat de securite fort).
Si zman atteint aussi un ASR haut -> le backdoor est exprimable, barriere non
generale (resultat negatif honnete). Sortie : results/adaptive.json.
"""
import os, json, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters"); RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "adaptive.json")
SCALING, K, N_TRAIN, EPOCHS, BATCH, r = 2.0, 128, 128, 8, 8, 16
LR_LORA, LR_Z = 3e-4, 3e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2)
TRIGGER = " tq84z"          # token rare appende
TARGET = "No"               # cible fixe du backdoor
TASKS = ["amazon_polarity_Is_this_product_review_positive",
         "social_i_qa_Check_if_a_random_answer_is_valid_or_not"]
MODES = ("backdoor", "maxasr")

z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
G = z["G"]; names = [str(n) for n in z["names"]]; N = len(names)
def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        m = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[m] = v.float()
    return A, B
A0, _ = load_factors(names[0]); mods = sorted(A0.keys())

def basis_excluding(excl):
    others = [j for j in range(N) if j not in excl]; n = len(others)
    Goo = G[np.ix_(others, others)]; H = np.eye(n) - 1.0 / n
    ev, V = np.linalg.eigh(H @ Goo @ H); ev, V = ev[::-1][:K], V[:, ::-1][:, :K]
    Mw = (V / np.sqrt(np.clip(ev, 1e-12, None))) * np.sqrt(np.clip(ev, 0, None) / n)
    Mw = Mw - Mw.mean(0, keepdims=True)
    M = np.zeros((N, K)); b = np.zeros(N); M[others] = Mw; b[others] = 1.0 / n
    return torch.tensor(M, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
def fam(n):
    t = n.split("-", 1)[1] if "-" in n else n
    for p in ("amazon_polarity", "social_i_qa", "wiki_qa", "race_middle", "race_high", "qasc"):
        if t.startswith(p): return p
    return t.split("_")[0]

Ahat = {m: torch.empty(N * r, 1024, dtype=torch.float16) for m in mods}
Bhat = {m: torch.empty(1024, N * r, dtype=torch.float16) for m in mods}
t0 = time.time()
for i, n in enumerate(names):
    A, B = load_factors(n)
    for m in mods:
        Ahat[m][i * r:(i + 1) * r] = A[m].half(); Bhat[m][:, i * r:(i + 1) * r] = B[m].half()
print(f"stacks {time.time()-t0:.0f}s", flush=True)

tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32).eval().to(DEVICE)
for p in model.parameters(): p.requires_grad_(False)

class ZLinear(nn.Module):
    def __init__(s, base, m, st): super().__init__(); s.weight = base.weight; s.m = m; s.st = st
    def forward(s, x):
        y = F.linear(x, s.weight); g = (s.st["M"] @ s.st["z"] + s.st["b"]).half()
        u = F.linear(x.half(), s.st["Ah"][s.m]) * g.repeat_interleave(r)
        return y + SCALING * F.linear(u, s.st["Bh"][s.m]).to(y.dtype)
class LoRALinear(nn.Module):
    def __init__(s, base):
        super().__init__(); s.weight = base.weight
        s.A = nn.Parameter(torch.empty(r, 1024)); s.B = nn.Parameter(torch.zeros(1024, r))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
    def forward(s, x):
        return F.linear(x, s.weight) + SCALING * F.linear(F.linear(x, s.A.to(x.dtype)), s.B.to(x.dtype))
orig = {m: model.get_submodule(m) for m in mods}
def pa(m): p, a = m.rsplit(".", 1); return model.get_submodule(p), a
def install(kind, st=None):
    reps = {}
    for m in mods:
        p, a = pa(m); w = ZLinear(orig[m], m, st) if kind == "z" else LoRALinear(orig[m]).to(DEVICE)
        setattr(p, a, w); reps[m] = w
    return reps
def restore():
    for m in mods: p, a = pa(m); setattr(p, a, orig[m])

def stream(task, sp, cap, mn):
    for s in sp:
        try:
            d = load_dataset("bigscience/P3", task, split=s, streaming=True); ps = []
            for ex in d:
                ps.append((ex["inputs_pretokenized"].strip(), ex["targets_pretokenized"].strip()))
                if len(ps) >= cap: break
            if len(ps) >= mn: return ps
        except Exception: pass
    return None
def batches(pairs, bs):
    out = []
    for i in range(0, len(pairs), bs):
        ins, tg = zip(*pairs[i:i + bs])
        enc = tok(list(ins), max_length=192, truncation=True, padding=True, return_tensors="pt")
        lab = tok(list(tg), max_length=8, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        out.append(({k: v.to(DEVICE) for k, v in enc.items()}, lab.to(DEVICE)))
    return out
def norm(s): return " ".join(s.lower().strip().rstrip(".").split())
@torch.no_grad()
def acc(pairs):  # exact match
    h = 0
    for i in range(0, len(pairs), 16):
        ch = pairs[i:i + 16]
        enc = {k: v.to(DEVICE) for k, v in tok([a for a, _ in ch], max_length=192, truncation=True,
               padding=True, return_tensors="pt").items()}
        gen = model.generate(**enc, max_new_tokens=6, num_beams=1, do_sample=False)
        for d, (_, t) in zip(tok.batch_decode(gen, skip_special_tokens=True), ch): h += norm(d) == norm(t)
    return h / max(len(pairs), 1)
@torch.no_grad()
def asr(inputs):  # fraction sortant la cible sur entrees triggered
    h = 0
    for i in range(0, len(inputs), 16):
        ch = inputs[i:i + 16]
        enc = {k: v.to(DEVICE) for k, v in tok(ch, max_length=192, truncation=True,
               padding=True, return_tensors="pt").items()}
        gen = model.generate(**enc, max_new_tokens=6, num_beams=1, do_sample=False)
        for d in tok.batch_decode(gen, skip_special_tokens=True): h += norm(d) == norm(TARGET)
    return h / max(len(inputs), 1)
def train(params, tb, lr, seed):
    opt = torch.optim.AdamW(params, lr=lr); rg = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        for j in rg.permutation(len(tb)):
            enc, lab = tb[int(j)]; loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in TASKS:
    name = next(n for n in names if n.endswith(task)); i = names.index(name)
    M, b = basis_excluding({idx for idx, n in enumerate(names) if fam(n) == fam(name)})  # holdout famille aussi
    res = R.get(task, {})
    for sd in SEEDS:
        tr = stream(task, ("train",), 1024, N_TRAIN + 64)
        va = stream(task, ("validation", "test"), 400, 64)
        rg = np.random.default_rng(1000 * sd + i)
        tr = [tr[int(j)] for j in rg.permutation(len(tr))]
        train_pairs = tr[:N_TRAIN]
        clean_val = [va[int(j)] for j in rg.permutation(len(va))[:64]]
        asr_inputs = [a + TRIGGER for a, t in va if norm(t) != norm(TARGET)][:64]  # vrais != cible
        # reference base
        if f"base_s{sd}" not in res:
            restore(); res[f"base_s{sd}"] = {"clean": acc(clean_val), "asr": asr(asr_inputs)}
        for mode in MODES:
            # construire les donnees d'entrainement
            if mode == "backdoor":
                half = N_TRAIN // 2
                data = [train_pairs[k] for k in range(half)]                         # propre
                data += [(train_pairs[k][0] + TRIGGER, TARGET) for k in range(half, N_TRAIN)]  # triggered->cible
            else:  # maxasr : tout triggered->cible
                data = [(a + TRIGGER, TARGET) for a, _ in train_pairs]
            for kind in ("lora", "zman"):
                key = f"{kind}_{mode}_s{sd}"
                if key in res: continue
                tb = batches(data, BATCH); torch.manual_seed(3000 * sd + i); st = None
                if kind == "zman":
                    st = {"M": M.to(DEVICE), "b": b.to(DEVICE), "z": torch.zeros(K, device=DEVICE, requires_grad=True),
                          "Ah": {m: Ahat[m].to(DEVICE) for m in mods}, "Bh": {m: Bhat[m].to(DEVICE) for m in mods}}
                    install("z", st); params = [st["z"]]; lr = LR_Z
                else:
                    reps = install("lora"); params = [q for m in mods for q in (reps[m].A, reps[m].B)]; lr = LR_LORA
                try:
                    train(params, tb, lr, 4000 * sd + i); ca = acc(clean_val); a = asr(asr_inputs)
                finally:
                    restore()
                    if st is not None: st["Ah"] = st["Bh"] = None
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                res[key] = {"clean_acc": ca, "asr": a}
                print(f"  {fam(name):16} {key:20} clean={ca:.2f} ASR={a:.2f} {time.time()-t0:.0f}s", flush=True)
                R[task] = res; json.dump(R, open(OUT, "w"), indent=1)
print(f"FIN {time.time()-t0:.0f}s", flush=True)
