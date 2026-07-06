"""Genere les figures du papier depuis results/ (PDF pour LaTeX, PNG pour relecture).

Palette validee (dataviz) : bleu #2a78d6 (methode z), rouge #e34948 (LoRA),
gris #52514e (texte/references). Fond blanc (papier).
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "03_lora_spectrum", "results")
BLUE, RED, INK, MUT = "#2a78d6", "#e34948", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUT, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "grid.color": "#e6e6e3", "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

def save(fig, name):
    fig.savefig(os.path.join(HERE, f"{name}.pdf"))
    fig.savefig(os.path.join(HERE, f"{name}.png"))
    plt.close(fig)
    print("ok", name)

# ---------- fig 1 : spectre, variance cumulee reelle vs null ----------
S = json.load(open(os.path.join(RES, "spectrum.json")))
ks = [2, 4, 8, 16, 32, 64]
real = [S["spectre_deltaw"]["cum_var"][str(k)] for k in ks]
null = [S["spectre_null_aleatoire"]["cum_var"][str(k)] for k in ks]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.plot(ks, real, "-o", color=BLUE, lw=2, ms=4, clip_on=False)
ax.plot(ks, null, "--s", color=MUT, lw=1.5, ms=3.5, clip_on=False)
ax.annotate("real adapters", (ks[-1], real[-1]), xytext=(-4, 8),
            textcoords="offset points", ha="right", color=INK)
ax.annotate("random null", (ks[-1], null[-1]), xytext=(-14, -16),
            textcoords="offset points", ha="right", color=MUT)
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("principal components (k)")
ax.set_ylabel("cumulative explained variance")
ax.set_ylim(0, 0.6); ax.grid(axis="y")
ax.minorticks_off()
save(fig, "fig1_spectrum")

# ---------- fig 2 : recuperation fonctionnelle vs k ----------
F = json.load(open(os.path.join(RES, "functional_dim.json")))
panels = [("amazon_polarity_Is_this_product_review_positive", "amazon_polarity"),
          ("wiki_hop_original_choose_best_object_affirmative_1", "wiki_hop")]
fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.0))
for ax, (task, title) in zip(axes, panels):
    d = F[task]["ce"]
    xs = [0, 8, 32, 128, 194]
    ys = [d["k0"], d["k8"], d["k32"], d["k128"], d["full"]]
    ax.axhline(d["base"], color=MUT, lw=1, ls=":")
    ax.axhline(d["orig"], color=MUT, lw=1, ls="--")
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=3.5, clip_on=False)
    ax.set_title(title, color=INK)
    ax.set_xscale("symlog", base=2, linthresh=8)
    ax.set_xticks([0, 8, 32, 128]); ax.set_xticklabels([0, 8, 32, 128])
    ax.minorticks_off(); ax.grid(axis="y")
    ax.text(0.03, d["base"], "base", va="bottom", ha="left", color=MUT, fontsize=6.5,
            transform=ax.get_yaxis_transform())
    ax.text(0.03, d["orig"], "original adapter", va="bottom", ha="left", color=MUT,
            fontsize=6.5, transform=ax.get_yaxis_transform())
axes[0].set_ylabel("validation CE per token")
axes[0].set_ylim(0, 0.9); axes[1].set_ylim(0, 4.6)
fig.supxlabel("projection rank k (leave-one-out)", fontsize=8, y=-0.04)
fig.tight_layout(w_pad=1.5)
save(fig, "fig2_functional")

# ---------- inversion ciblee, depuis poison_flip.json (5 taches, 3 seeds) ----------
PF = json.load(open(os.path.join(RES, "poison_flip.json")))
SEEDS = (0, 1, 2)
def vals(kind, cond, field):
    return np.array([PF[t][f"{kind}_{cond}_s{s}"][field] for t in PF for s in SEEDS])

# ---------- fig 3 : exact-match par tache a 100% d'inversion (money plot) ----------
sh3 = {"wiki_qa_Is_This_True_": "wiki_qa", "qasc_is_correct_1": "qasc",
       "amazon_polarity_Is_this_product_review_positive": "amazon",
       "social_i_qa_Check_if_a_random_answer_is_valid_or_not": "social_iqa",
       "race_middle_Select_the_best_answer": "race*"}
tasks3 = list(PF.keys())
def em_task(kind, cond, task):
    v = [PF[task][f"{kind}_{cond}_s{s}"]["em_val"] for s in SEEDS]
    return np.mean(v), np.std(v)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(len(tasks3)); w = 0.38
for off, kind, col, nm in ((-w / 2, "lora", RED, "LoRA fine-tuning"),
                           (w / 2, "zman", BLUE, "manifold z")):
    m = [em_task(kind, "flip100", t)[0] for t in tasks3]
    s = [em_task(kind, "flip100", t)[1] for t in tasks3]
    ax.bar(x + off, m, w - 0.04, color=col, yerr=s, error_kw={"ecolor": MUT, "lw": 0.8},
           label=nm)
ax.set_xticks(x); ax.set_xticklabels([sh3[t] for t in tasks3], rotation=18, ha="right")
ax.set_ylabel("exact match, 100% labels inverted")
ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2,
          columnspacing=1.4, handlelength=1.2)
ax.text(0.99, -0.42, "* race: adapter pool covers this task poorly (see text)",
        transform=ax.transAxes, ha="right", va="top", fontsize=6, color=MUT)
save(fig, "fig3_poison")

# ---------- fig 4 : separation OOD par loss d'adaptation ----------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
jit = np.random.default_rng(0)
pos = {("lora", "clean"): 0, ("lora", "garbage"): 1, ("zman", "clean"): 2.4, ("zman", "garbage"): 3.4}
for (kind, cond), xp in pos.items():
    v = np.clip(vals(kind, cond, "train_loss"), 1e-4, None)
    col = RED if kind == "lora" else BLUE
    ax.scatter(xp + jit.uniform(-0.08, 0.08, len(v)), v, s=14, color=col,
               edgecolors="white", linewidths=0.5, zorder=3)
ax.set_yscale("log")
ax.set_xticks(list(pos.values()))
ax.set_xticklabels(["clean", "garbage", "clean", "garbage"])
ax.text(0.5, -0.22, "LoRA fine-tuning", transform=ax.get_xaxis_transform(),
        ha="center", color=RED)
ax.text(2.9, -0.22, "manifold z", transform=ax.get_xaxis_transform(),
        ha="center", color=BLUE)
ax.set_ylabel("final adaptation loss (log)")
ax.grid(axis="y")
save(fig, "fig4_ood")

# ---------- fig 5 : oubli sequentiel, CE finale par tache (seed 0) ----------
SEQ = json.load(open(os.path.join(RES, "sequential.json")))
sh = {"wiki_qa_Is_This_True_": "wiki_qa", "qasc_is_correct_1": "qasc",
      "amazon_polarity_Is_this_product_review_positive": "amazon",
      "social_i_qa_Check_if_a_random_answer_is_valid_or_not": "social_iqa",
      "race_middle_Select_the_best_answer": "race"}
order = SEQ["lora_s0"]["order"]
lf = SEQ["lora_s0"]["matrix"][-1]; zf = SEQ["zman_s0"]["matrix"][-1]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(len(order)); w = 0.36
ax.bar(x - w / 2, [lf[t] for t in order], w - 0.04, color=RED, label="LoRA (one continual)")
ax.bar(x + w / 2, [zf[t] for t in order], w - 0.04, color=BLUE, label="manifold z (one continual)")
ax.set_xticks(x); ax.set_xticklabels([sh[t] for t in order], rotation=20, ha="right")
ax.set_ylabel("validation CE after full 5-task chain")
ax.grid(axis="y"); ax.legend(frameon=False, loc="upper center", ncols=1)
save(fig, "fig5_sequential")

# ---------- fig 6 : generateur AE vs PCA, budget de code egal ----------
GEN = json.load(open(os.path.join(RES, "generator.json")))
disc = [t for t, r in GEN.items() if r["ce_orig"] < r["ce_base"] - 0.1]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(len(disc)); w = 0.26
for off, key, col, nm in ((-w, "ce_pca32", MUT, "PCA-32 (linear)"),
                          (0, "ce_ae32", BLUE, "AE-32 (nonlinear)"),
                          (w, "ce_pca64", "#9ec5f4", "PCA-64 (linear)")):
    ax.bar(x + off, [GEN[t][key] for t in disc], w - 0.03, color=col, label=nm)
ax.set_xticks(x); ax.set_xticklabels([sh[t] for t in disc], rotation=20, ha="right")
ax.set_ylabel("validation CE (log)"); ax.set_yscale("log")
ax.grid(axis="y"); ax.legend(frameon=False, fontsize=6.5, loc="upper right")
save(fig, "fig6_generator")

# ---------- fig 7 : attaque adaptative (backdoor), ASR plafond LoRA vs z ----------
AD = json.load(open(os.path.join(RES, "adaptive.json")))
at = {"amazon_polarity_Is_this_product_review_positive": "amazon\n(target unlike pool)",
      "social_i_qa_Check_if_a_random_answer_is_valid_or_not": "social_iqa\n(target common in pool)"}
tks = list(at.keys())
def asr_mean(t, k): return np.mean([AD[t][f"{k}_s{s}"]["asr"] for s in (0, 1, 2)])
def asr_std(t, k): return np.std([AD[t][f"{k}_s{s}"]["asr"] for s in (0, 1, 2)])
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(len(tks)); w = 0.38
for off, key, col, nm in ((-w / 2, "lora_maxasr", RED, "LoRA (unconstrained)"),
                          (w / 2, "zman_maxasr", BLUE, "manifold z")):
    ax.bar(x + off, [asr_mean(t, key) for t in tks], w - 0.04, color=col,
           yerr=[asr_std(t, key) for t in tks], error_kw={"ecolor": MUT, "lw": 0.8}, label=nm)
ax.set_xticks(x); ax.set_xticklabels([at[t] for t in tks], fontsize=7)
ax.set_ylabel("backdoor attack success rate")
ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2,
          columnspacing=1.2, handlelength=1.2, fontsize=6.5)
save(fig, "fig7_adaptive")
print("figures regenerees depuis les JSON")
