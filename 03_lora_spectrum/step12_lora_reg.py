"""Baseline REGULARISATION (objection Gemini) : LoRA + forte regularisation.

Teste si une simple penalite (weight decay eleve + dropout) donne la meme
resistance a l'inversion que la contrainte de sous-espace. Si LoRA+reg
s'effondre encore a flip100, la robustesse de z ne vient PAS d'un simple
ralentissement de l'apprentissage mais de la restriction de l'ensemble
atteignable. Variantes : wd=0.1+dropout=0.3, et wd=1.0 (tres fort).
clean/flip100, 3 seeds, 4 taches. Sortie : results/lora_reg.json.
"""
import os, json, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters"); RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "lora_reg.json")
SCALING, N_TRAIN, N_EVAL, EPOCHS = 2.0, 128, 64, 8
BATCH, r = 8, 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2)
TASKS = ["wiki_qa_Is_This_True_", "qasc_is_correct_1",
         "amazon_polarity_Is_this_product_review_positive",
         "social_i_qa_Check_if_a_random_answer_is_valid_or_not"]
# (nom, lr, weight_decay, dropout)
VARIANTS = [("wd0.1_do0.3", 3e-4, 1e-1, 0.3), ("wd1.0", 3e-4, 1.0, 0.0)]
CONDS = ("clean", "flip100")

z = np.load(os.path.join(RDIR, "grams.npz"), allow_pickle=True)
names = [str(n) for n in z["names"]]
def load_factors(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        m = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[m] = v.float()
    return A, B
A0, _ = load_factors(names[0]); mods = sorted(A0.keys())

tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32).eval().to(DEVICE)
for p in model.parameters(): p.requires_grad_(False)
t0 = time.time()

class LoRALinear(nn.Module):
    def __init__(s, base, dropout):
        super().__init__(); s.weight = base.weight; s.drop = nn.Dropout(dropout)
        s.A = nn.Parameter(torch.empty(r, 1024)); s.B = nn.Parameter(torch.zeros(1024, r))
        nn.init.kaiming_uniform_(s.A, a=math.sqrt(5))
    def forward(s, x):
        return F.linear(x, s.weight) + SCALING * F.linear(F.linear(s.drop(x), s.A.to(x.dtype)), s.B.to(x.dtype))
orig = {m: model.get_submodule(m) for m in mods}
def pa(m): p, a = m.rsplit(".", 1); return model.get_submodule(p), a
def install(dropout):
    reps = {}
    for m in mods:
        p, a = pa(m); w = LoRALinear(orig[m], dropout).to(DEVICE); setattr(p, a, w); reps[m] = w
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
        lab = tok(list(tg), max_length=48, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        out.append(({k: v.to(DEVICE) for k, v in enc.items()}, lab.to(DEVICE)))
    return out
def invert(pairs, p, seed):
    if p == 0: return pairs
    rg = np.random.default_rng(seed); out = list(pairs); tg = [t for _, t in pairs]
    for a in rg.choice(len(pairs), int(p * len(pairs)), replace=False):
        wrong = [t for t in tg if t != pairs[a][1]]
        out[a] = (pairs[a][0], str(rg.choice(wrong)) if wrong else pairs[a][1])
    return out
@torch.no_grad()
def ce(bs):
    tot = nt = 0
    for enc, lab in bs:
        o = model(**enc, labels=lab); n = int((lab != -100).sum()); tot += float(o.loss) * n; nt += n
    return tot / nt
def norm(s): return " ".join(s.lower().strip().rstrip(".").split())
@torch.no_grad()
def em(vp):
    h = 0
    for i in range(0, len(vp), 16):
        ch = vp[i:i + 16]
        enc = {k: v.to(DEVICE) for k, v in tok([a for a, _ in ch], max_length=192, truncation=True,
               padding=True, return_tensors="pt").items()}
        gen = model.generate(**enc, max_new_tokens=16, num_beams=1, do_sample=False)
        for d, (_, t) in zip(tok.batch_decode(gen, skip_special_tokens=True), ch): h += norm(d) == norm(t)
    return h / len(vp)
def train(params, tb, lr, wd, seed):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd); rg = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        for j in rg.permutation(len(tb)):
            enc, lab = tb[int(j)]; loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in TASKS:
    ti = names.index(name_of := next(n for n in names if n.endswith(task))); res = R.get(task, {})
    for sd in SEEDS:
        tr = stream(task, ("train",), 512, N_TRAIN); va = stream(task, ("validation", "test"), 256, N_EVAL)
        rg = np.random.default_rng(1000 * sd + ti)
        train_pairs = [tr[int(j)] for j in rg.permutation(len(tr))[:N_TRAIN]]
        val_pairs = [va[int(j)] for j in rg.permutation(len(va))[:N_EVAL]]; vb = batches(val_pairs, 16)
        for cond in CONDS:
            p = 0.0 if cond == "clean" else 1.0
            tb = batches(invert(train_pairs, p, 2000 * sd + ti), BATCH)
            for vname, lr, wd, do in VARIANTS:
                key = f"{vname}_{cond}_s{sd}"
                if key in res: continue
                torch.manual_seed(3000 * sd + ti)
                reps = install(do); params = [q for m in mods for q in (reps[m].A, reps[m].B)]
                try:
                    train(params, tb, lr, wd, 4000 * sd + ti); c = ce(vb); e = em(val_pairs)
                finally:
                    restore()
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                res[key] = {"ce_val": c, "em_val": e}
                print(f"  {task[:18]:20} {key:22} ce={c:.3f} em={e:.2f} {time.time()-t0:.0f}s", flush=True)
                R[task] = res; json.dump(R, open(OUT, "w"), indent=1)
print(f"FIN {time.time()-t0:.0f}s", flush=True)
