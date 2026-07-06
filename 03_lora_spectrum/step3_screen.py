"""Criblage des 196 adaptateurs : CE base vs CE adaptateur sur leur validation P3.

Objectif : identifier les adaptateurs discriminants (orig nettement sous base)
pour selectionner les taches des experiences de surete. Resultats incrementaux
dans results/screen.json.
"""
import os, json, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(HERE, "adapters")
RDIR = os.path.join(HERE, "results")
OUT = os.path.join(RDIR, "screen.json")
SCALING = 2.0
N_EVAL = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16 if DEVICE == "cuda" else 8
rng = np.random.default_rng(0)
torch.manual_seed(0)

names = sorted(d for d in os.listdir(ADIR) if os.path.exists(os.path.join(ADIR, d, "adapter_model.bin")))
R = json.load(open(OUT)) if os.path.exists(OUT) else {}

tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32)
model.eval().to(DEVICE)
params = dict(model.named_parameters())

def load_delta(name):
    sd = torch.load(os.path.join(ADIR, name, "adapter_model.bin"), map_location="cpu", weights_only=True)
    A, B = {}, {}
    for k, v in sd.items():
        mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        (A if "lora_A" in k else B)[mod] = v.float()
    return {m: (B[m] @ A[m]) for m in A}

mods0 = sorted(load_delta(names[0]).keys())
base_w = {m: params[m + ".weight"].detach().clone() for m in mods0}

def apply_delta(deltas):
    with torch.no_grad():
        for m in mods0:
            w = params[m + ".weight"]
            w.copy_(base_w[m])
            if deltas is not None:
                w.add_(deltas[m].to(w.device), alpha=SCALING)

def stream_pairs(task, splits=("validation", "test"), cap=256):
    """Streaming : ne telecharge que les premiers exemples, pas le dataset entier."""
    for sp in splits:
        try:
            d = load_dataset("bigscience/P3", task, split=sp, streaming=True)
            pairs = []
            for ex in d:
                pairs.append((ex["inputs_pretokenized"].strip(), ex["targets_pretokenized"].strip()))
                if len(pairs) >= cap: break
            if len(pairs) >= N_EVAL:
                return pairs
        except Exception:
            pass
    return None

def make_batches(pairs):
    idx = rng.permutation(len(pairs))[:N_EVAL]
    exs = [pairs[int(j)] for j in idx]
    batches = []
    for b in range(0, len(exs), BATCH):
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

t0 = time.time()
for i, name in enumerate(names):
    task = name.split("-", 1)[1]
    if task in R:
        continue
    pairs = stream_pairs(task)
    if pairs is None:
        R[task] = {"skip": "pas de P3"}
        json.dump(R, open(OUT, "w"))
        print(f"{i+1}/{len(names)} {task} SKIP", flush=True)
        continue
    batches = make_batches(pairs)
    apply_delta(None); ce_base = eval_ce(batches)
    apply_delta(load_delta(name)); ce_orig = eval_ce(batches)
    R[task] = {"base": ce_base, "orig": ce_orig, "gain": ce_base - ce_orig}
    json.dump(R, open(OUT, "w"))
    print(f"{i+1}/{len(names)} {task} base={ce_base:.3f} orig={ce_orig:.3f} gain={ce_base-ce_orig:+.3f} {time.time()-t0:.0f}s", flush=True)

ok = [t for t, v in R.items() if "gain" in v]
disc = [t for t in ok if R[t]["gain"] > 0.15]
print(f"\nFIN: {len(ok)} evalues, {len(disc)} discriminants (gain>0.15), {time.time()-t0:.0f}s", flush=True)
