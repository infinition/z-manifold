"""Prior baseline (reviewer point): does constrained adaptation do better than
its own starting point, the frozen pool-mean adapter?

Constrained adaptation starts from z=0, which decodes to the mean of the pool
(family-excluded). This measures, per task, the clean exact match of:
  base          the un-adapted model,
  mean adapter  z=0, no training (the prior the method starts from),
and prints them next to the z-adapted clean number from poison_flip.json. If the
z-adapted score clearly beats the mean adapter, the method learns beyond staying
near its prior. CPU is enough (evaluation only, no training).
Output: results/prior_baseline.json.
"""
import os, json
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters"); RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "prior_baseline.json")
SCALING, N_EVAL, r = 2.0, 64, 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = ["wiki_qa_Is_This_True_", "qasc_is_correct_1",
         "amazon_polarity_Is_this_product_review_positive",
         "social_i_qa_Check_if_a_random_answer_is_valid_or_not",
         "race_middle_Select_the_best_answer"]
PREFIXES = ("amazon_polarity", "social_i_qa", "wiki_qa", "race_middle", "race_high", "qasc")
def fam(n):
    t = n.split("-", 1)[1] if "-" in n else n
    for p in PREFIXES:
        if t.startswith(p): return p
    return t.split("_")[0]

names = sorted(d for d in os.listdir(ADIR) if os.path.exists(os.path.join(ADIR, d, "adapter_model.bin")))
name_of = {n.split("-", 1)[1]: n for n in names}

def load_delta(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        m = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[m] = v.float()
    return {m: B[m] @ A[m] for m in A}

mods = sorted(load_delta(names[0]).keys())

tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32).eval().to(DEVICE)
params = dict(model.named_parameters())
base_w = {m: params[m + ".weight"].detach().clone() for m in mods}

def apply(delta):
    with torch.no_grad():
        for m in mods:
            w = params[m + ".weight"]; w.copy_(base_w[m])
            if delta is not None: w.add_(delta[m].to(w.device), alpha=SCALING)

def stream(task):
    for sp in ("validation", "test"):
        try:
            d = load_dataset("bigscience/P3", task, split=sp, streaming=True); ps = []
            for ex in d:
                ps.append((ex["inputs_pretokenized"].strip(), ex["targets_pretokenized"].strip()))
                if len(ps) >= 256: break
            if len(ps) >= N_EVAL: return ps
        except Exception: pass
    return None

def norm(s): return " ".join(s.lower().strip().rstrip(".").split())
@torch.no_grad()
def em(pairs):
    h = 0
    for i in range(0, len(pairs), 8):
        ch = pairs[i:i + 8]
        enc = {k: v.to(DEVICE) for k, v in tok([a for a, _ in ch], max_length=192, truncation=True,
               padding=True, return_tensors="pt").items()}
        gen = model.generate(**enc, max_new_tokens=6, num_beams=1, do_sample=False)
        for d, (_, t) in zip(tok.batch_decode(gen, skip_special_tokens=True), ch): h += norm(d) == norm(t)
    return h / len(pairs)

# mean adapter = average of family-excluded pool deltas (what z=0 decodes to)
PF = json.load(open(os.path.join(RDIR, "poison_flip.json"))) if os.path.exists(os.path.join(RDIR, "poison_flip.json")) else {}
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for task in TASKS:
    if task in R: continue
    tf = fam(name_of[task])
    others = [n for n in names if fam(n) != tf]
    mean_delta = {m: torch.zeros_like(base_w[m]) for m in mods}
    for n in others:
        d = load_delta(n)
        for m in mods: mean_delta[m] += d[m]
    for m in mods: mean_delta[m] /= len(others)
    rg = np.random.default_rng(0)
    va = stream(task); val = [va[int(j)] for j in rg.permutation(len(va))[:N_EVAL]]
    apply(None);       base = em(val)
    apply(mean_delta); mean = em(val)
    apply(None)
    z_clean = np.mean([PF[task][f"zman_clean_s{s}"]["em_val"] for s in (0, 1, 2)]) if task in PF else None
    R[task] = {"base": round(base, 3), "mean_adapter": round(mean, 3),
               "z_adapted_clean": round(float(z_clean), 3) if z_clean is not None else None,
               "n_pool": len(others)}
    print(f"{task[:34]:36} base={base:.3f} mean_adapter={mean:.3f} z_clean={z_clean}", flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
print("FIN")
