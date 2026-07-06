"""Analyse locale de poison_flip.json : tables agregees (moyenne, ecart-type
sur taches x seeds) et AUROC de detection OOD par loss d'adaptation.
Sorties : results/analysis_flip.md + stdout.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "poison_flip.json")))
CONDS = ("clean", "flip50", "flip100", "garbage")
KINDS = ("lora", "zman")

def collect(kind, cond, field):
    out = []
    for task, res in R.items():
        for key, v in res.items():
            if key.startswith(f"{kind}_{cond}_s"):
                out.append(v[field])
    return np.array(out)

def refs(field):
    return np.array([v[field] for res in R.values() for k, v in res.items() if k.startswith("ref_s")])

def auroc(pos, neg):
    """P(score_pos > score_neg), rangs."""
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

lines = ["# Analyse poison_flip", ""]
lines.append(f"references: base CE {refs('ce_base').mean():.3f}, EM {refs('em_base').mean():.2f} ; "
             f"orig CE {refs('ce_orig').mean():.3f}, EM {refs('em_orig').mean():.2f}")
lines.append("")
lines.append("| condition | LoRA CE | z CE | LoRA EM | z EM | LoRA train | z train |")
lines.append("|---|---|---|---|---|---|---|")
for cond in CONDS:
    row = [cond]
    for field in ("ce_val", "em_val", "train_loss"):
        for kind in KINDS:
            a = collect(kind, cond, field)
            row.append(f"{a.mean():.3f} ({a.std():.3f})")
    lines.append("| " + " | ".join([row[0], row[1], row[2], row[3], row[4], row[5], row[6]]) + " |")
lines.append("")
for kind in KINDS:
    clean = collect(kind, "clean", "train_loss")
    for cond in ("flip100", "garbage"):
        a = auroc(collect(kind, cond, "train_loss"), clean)
        lines.append(f"AUROC OOD ({kind}, loss train, clean vs {cond}) : {a:.3f}")
lines.append("")
lines.append("Detail par tache (CE val, moyenne des seeds) :")
lines.append("")
lines.append("| tache | " + " | ".join(f"{k} {c}" for c in CONDS for k in KINDS) + " |")
lines.append("|---|" + "---|" * 8)
for task, res in R.items():
    row = [task[:30]]
    for cond in CONDS:
        for kind in KINDS:
            a = [v["ce_val"] for k, v in res.items() if k.startswith(f"{kind}_{cond}_s")]
            row.append(f"{np.mean(a):.3f}" if a else "-")
    lines.append("| " + " | ".join(row) + " |")

txt = "\n".join(lines)
open(os.path.join(HERE, "results", "analysis_flip.md"), "w").write(txt + "\n")
print(txt)
