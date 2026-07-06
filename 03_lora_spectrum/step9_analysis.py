"""Agrege poison_flip, sequential, generator : tables avec moyenne +/- ecart-type
et AUROC de detection OOD. Sortie : results/analysis.md + stdout.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(HERE, "results", f)))
PF = R("poison_flip.json"); SEQ = R("sequential.json"); GEN = R("generator.json")
SEEDS = (0, 1, 2)
CONDS = ("clean", "flip50", "flip100", "garbage")
short = {"wiki_qa_Is_This_True_": "wiki_qa", "qasc_is_correct_1": "qasc",
         "amazon_polarity_Is_this_product_review_positive": "amazon_polarity",
         "social_i_qa_Check_if_a_random_answer_is_valid_or_not": "social_i_qa",
         "race_middle_Select_the_best_answer": "race_middle"}
out = []
def w(s=""): out.append(s)

def ms(vals):
    a = np.array(vals); return a.mean(), a.std()

# ---------- poison_flip : EM par tache et condition ----------
w("# Analyse finale (poison_flip, sequential, generator)\n")
w("## Poisoning par inversion ciblee, exact match (moyenne +/- ecart-type, 3 seeds)\n")
w("| tache | clean L | clean z | flip50 L | flip50 z | flip100 L | flip100 z |")
w("|---|---|---|---|---|---|---|")
agg = {c: {"lora": [], "zman": []} for c in CONDS}
for task, res in PF.items():
    row = [short[task]]
    for cond in ("clean", "flip50", "flip100"):
        for kind in ("lora", "zman"):
            v = [res[f"{kind}_{cond}_s{s}"]["em_val"] for s in SEEDS]
            agg[cond][kind] += v
            m, s = ms(v)
            row.append(f"{m:.2f}$\\pm${s:.2f}")
    w("| " + " | ".join(row) + " |")
w("")
w("Moyenne toutes taches (EM) :")
for cond in ("clean", "flip50", "flip100"):
    ml, sl = ms(agg[cond]["lora"]); mz, sz = ms(agg[cond]["zman"])
    w(f"- {cond}: LoRA {ml:.2f}+/-{sl:.2f}, z {mz:.2f}+/-{sz:.2f}")

# ---------- AUROC OOD via train loss clean vs garbage ----------
def auroc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
w("\n## Detection OOD (train loss finale, clean vs garbage)\n")
for kind in ("lora", "zman"):
    clean = [PF[t][f"{kind}_clean_s{s}"]["train_loss"] for t in PF for s in SEEDS]
    garb = [PF[t][f"{kind}_garbage_s{s}"]["train_loss"] for t in PF for s in SEEDS]
    w(f"- {kind}: clean {np.median(clean):.3f}, garbage {np.median(garb):.3f}, "
      f"AUROC {auroc(garb, clean):.3f}")

# ---------- sequential : oubli moyen et recuperation ----------
w("\n## Adaptation sequentielle (chaine de 5 taches, 3 ordres)\n")
def forgetting(rec):
    """CE finale moins CE juste apres apprentissage de chaque tache, moyenne."""
    mat = rec["matrix"]; order = rec["order"]
    final = mat[-1]
    gaps = []
    for stage, t in enumerate(order[:-1]):        # derniere tache = pas d'oubli
        just = mat[stage][t]
        gaps.append(final[t] - just)
    return np.mean(gaps)
for kind in ("lora", "zman"):
    fg = [forgetting(SEQ[f"{kind}_s{s}"]) for s in SEEDS]
    rc = [SEQ[f"{kind}_s{s}"]["recovery_task1"] for s in SEEDS]
    m, s = ms(fg); rm, rs = ms(rc)
    w(f"- {kind}: oubli moyen (hausse CE) {m:.3f}+/-{s:.3f}, "
      f"recuperation tache1 (16 ex.) {rm:.3f}+/-{rs:.3f}")
w("\nDetail final de chaine par seed (CE par tache, L=lora z=zman) :")
for s in SEEDS:
    o = SEQ[f"lora_s{s}"]["order"]
    lf = SEQ[f"lora_s{s}"]["matrix"][-1]; zf = SEQ[f"zman_s{s}"]["matrix"][-1]
    w(f"- s{s} ordre {[short[t] for t in o]}")
    w(f"    L " + " ".join(f"{short[t]}={lf[t]:.2f}" for t in o))
    w(f"    z " + " ".join(f"{short[t]}={zf[t]:.2f}" for t in o))

# ---------- generator : AE vs PCA ----------
w("\n## Generateur non lineaire vs PCA (CE validation, budget de code egal)\n")
w("| tache | base | orig | pca32 | pca64 | ae32 |")
w("|---|---|---|---|---|---|")
ae_wins = 0; disc = 0
for task, r in GEN.items():
    w(f"| {short[task]} | {r['ce_base']:.3f} | {r['ce_orig']:.3f} | "
      f"{r['ce_pca32']:.3f} | {r['ce_pca64']:.3f} | {r['ce_ae32']:.3f} |")
    if r["ce_orig"] < r["ce_base"] - 0.1:       # tache discriminante
        disc += 1
        if r["ce_ae32"] < r["ce_pca32"]:
            ae_wins += 1
w(f"\nAE bat PCA a budget 32 sur {ae_wins}/{disc} taches discriminantes "
  f"(mais PCA64 reste meilleure : la non-linearite achete de la compression, "
  f"pas la performance de pointe).")

txt = "\n".join(out)
open(os.path.join(HERE, "results", "analysis.md"), "w").write(txt + "\n")
print(txt)
