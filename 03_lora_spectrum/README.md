# 03 - Experiments at LM scale

All experiments on flan-t5-large (783M) with the 196 public LoRA adapters of
LoraHub (rank 16, q and v projections, 144 modules, Flan/P3 tasks). Every
script writes incremental JSON to `results/` and can be interrupted and resumed.
Figures for the paper are built by `../paper/figures/make_figures.py`.

## Pipeline

| Step | Script | What it does | Output |
|---|---|---|---|
| 0 | `step0_download.py` | download the 196 adapters (~3.7 GB) | `adapters/` |
| 1 | `step1_spectrum.py` | PCA spectrum of the pool in deltaW space | `spectrum.json`, `grams.npz` |
| 2 | `step2_functional_dim.py` | functional recovery vs projection rank | `functional_dim.json` |
| 3 | `step3_screen.py` | screen which adapters beat the base model | `screen.json` |
| 4 | `step4_poison.py` | label-shuffle poisoning | `poison.json` |
| 5 | `step5_poison_flip.py` | targeted label inversion + garbage, 3 seeds | `poison_flip.json` |
| 7 | `step7_sequential.py` | 5-task chains, forgetting, recovery | `sequential.json` |
| 8 | `step8_generator.py` | nonlinear autoencoder vs PCA truncation | `generator.json` |
| 9 | `step9_analysis.py` | aggregate tables and OOD AUROC | `analysis.md` |
| 10 | `step10_holdout.py` | control: dataset-family holdout (leakage) | `holdout.json` |
| 11 | `step11_random_subspace.py` | control: random 128-dim subspace | `random_subspace.json` |
| 12 | `step12_lora_reg.py` | control: LoRA + strong regularization | `lora_reg.json` |
| 13 | `step13_adaptive.py` | adaptive backdoor attack | `adaptive.json` |

## Method note

We compare adapters in gauge-invariant deltaW = B*A space, never the raw (A, B)
factors (which are misaligned, as Text-to-LoRA's appendix D also reports). The
Gram matrix is computed in factored form,
`<dW_i, dW_j> = tr((B_i^T B_j)(A_j A_i^T))`, so no deltaW is ever materialized.
Constrained adaptation optimizes a code z over a leave-one-out kernel-PCA basis
of the pool; only z is trainable, the base model and pool are frozen.

## Key results

- Structure: the pool has real semantic structure (within-family deltaW cosine
  0.204 vs 0.018 across) but is not naively low-dimensional (effective dimension
  129 of 196).
- Functional dimension: where an adapter genuinely helps, projecting it on the
  top-k directions of the other 195 recovers 100% of its function while
  discarding 30 to 38% of its weight norm (degeneracy, functionally inert).
- Poisoning, controls, and adaptive attack: see the paper (`../paper/`).

## Hardware

Steps 0 and 1 run on CPU. Steps 2 to 13 want a GPU; 12 GB in bf16 is enough.

```bash
../.venv/bin/python step0_download.py       # ~3.7 GB, ~20 min
../.venv/bin/python step1_spectrum.py       # ~3 min CPU
# steps 2..13 on GPU
```
