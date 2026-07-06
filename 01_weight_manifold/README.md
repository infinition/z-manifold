# 01 - Weight-manifold adaptation (toy proof of concept)

The seed idea, on a controlled toy problem before the LM-scale experiments in
`../03_lora_spectrum/`.

## Idea

Do not adapt a model in its weight space (N params) but in the latent space z
(8-128 dims) of a generator G trained on a collection of task weights. G defines
a manifold of "valid brains"; online adaptation is navigation on that manifold
(the model and G are frozen, only z moves).

Properties expected and tested here:
- manifold prior gives extreme few-shot,
- geometric impossibility of leaving valid configurations gives anti-forgetting
  and anti-poisoning,
- detectable adaptation failure gives OOD detection,
- a regularized residual on top of G(z) covers OOD (hybrid).

## Protocol

- Task family: y = A*sin(x+phi), A in [0.5, 3.0], phi in [0, pi] (in-dist);
  OOD: A in [4, 5].
- Base model: MLP 1-16-16-1 (321 params), 300-400 networks trained per task.
- Generator G: autoencoder on flattened weights, z = 8 dims, loss =
  0.2 * weight MSE + functional MSE (reconstruct the *function*, not the
  weights; crucial).
- Baselines: full fine-tuning from random init / anchor init / mean.

## Files

- `poc_v1_naive_FAIL.py`: v1, networks trained independently. INSTRUCTIVE
  FAILURE: neuron permutation symmetry destroys the manifold (z-adapt loses
  everywhere, AE functional loss 0.26). `results/results_v1_naive.json`.
- `poc_v2_shared_init.py`: v2, THE FIX: shared init (a pre-trained anchor)
  keeps all networks in one basin, weights aligned, AE functional loss 0.003.
  `results/results_v2.json`.
- v3 split pipeline (1 vCPU, timeouts): `step1_dataset.py` ->
  `step2_train_generator.py` -> `step3_benchmark.py` + `common.py` ->
  `results/results_v3_full.json`. `poc_v3_full_suite_monolithic.py` is the
  monolithic reference.

## Results v3 (median, dense-grid MSE, lower is better)

| Suite | FT (321 params) | Z-Adapt (8 params) | Verdict |
|---|---|---|---|
| Few-shot K=3 | 0.595 | **0.0039** | Z x152 |
| Few-shot K=5 | 0.362 | **0.0040** | Z x91 |
| Few-shot K=10 | 0.016 | **0.0026** | Z x6 |
| Few-shot K=20 | 0.0037 | 0.0030 | tie |
| Forgetting: current task after chain of 5 | 0.191 | **0.023** | Z x8 |
| Forgetting: recover task 1 (K=3) after drift | 3.02 | **0.101** | Z x30 |
| Integrity after chain (off-family residual) | 0.167 | **0.0096** | Z |
| Poison 50% random labels (MSE vs clean truth) | 1.93 | **0.262** | Z x7.4 |
| Integrity after poison | 1.14 | **0.0084** | Z |
| Pure OOD (A in [4,5]) | **0.181** | 1.28 | FT x7 |
| Hybrid z + L2 residual on OOD | 0.181 | **0.055** | beats FT x3.3 |

OOD detection by loss plateau: in-dist 0.0045 vs OOD 0.71, a x158 separation,
simple threshold about 83% accuracy. Storage: one task = 8 floats (32 bytes)
vs 321. Instant rollback.

## Lessons

1. Permutation symmetry is the lock. In raw weight space there is no usable
   manifold. Toy fix: shared init. Scale fix: learn the manifold of DELTAS from
   a common base (LoRA / task vectors), which is the practical regime followed
   in `../03_lora_spectrum/`.
2. The functional loss (reconstruct the function, not the weights) is essential.
3. The hybrid beats FT even on OOD because z gives a smart starting point.
