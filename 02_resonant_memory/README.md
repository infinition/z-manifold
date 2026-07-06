# 02 - Resonant complex-superposition memory (side study)

A self-contained side study, not part of the main subspace-adaptation result.
Kept for completeness.

## Idea

Fixed-size associative memory: N key->value pairs superposed in a single trace
M = sum k_i (*) v_i (element-wise binding, VSA/HRR style). Question: does phase
(unit complex phasors, FHRR) beat real (bipolar, MAP) at equal REAL-parameter
budget (complex D=256 vs real D=512)? And does error-correcting-style
redundancy help?

## Files

- `test1_oneshot_capacity.py`: one-shot recall capacity (unbind + nearest-
  neighbor cleanup, vocab=1000) and erasure robustness with redundancy R=3
  folded into the same field. `results/qmem_results.json`.
- `test2_iterative_recall.py`: iterative recall by successive interference
  cancellation (decode confident items by margin, subtract from the trace,
  repeat; SIC/CDMA analogue). `results/qmem2_results.json`.

## Results

| Test (budget = 512 real numbers, vocab=1000) | Real | Complex |
|---|---|---|
| One-shot N=50 | 0.60 | 0.43 |
| One-shot N>=100 | collapsed | collapsed |
| Iterative N=50 | 0.532 | **0.664 (+25%)** |
| Iterative N>=100 | collapsed | collapsed |
| Redundancy R=3 same field, 0% corruption | 0.14 (R1) | 0.26 (R3, vote) |
| Redundancy R=3, corruption >=20% | no help | no help |

## Lessons

1. Phase only wins coupled to dynamics: static recall is a tie; iterative recall
   is +25% relative (unitary unbinding does not amplify noise per subtraction).
2. Hard information wall: capacity ~ D / (2 ln|vocab|) ~ 35-50 items for 512
   real numbers. Neither phase, nor iteration, nor redundancy breaks it.
3. Redundancy folded into the same field triples crosstalk: an error-correcting
   code needs EXTRA dimensions, not more superposition.
4. Architecture implication: a fixed memory field cannot be verbatim storage.
   Capacity is linear in D. Surprise-gated / selective consolidation is not an
   optimization but a condition of existence. Target: short verbatim buffer +
   consolidated complex resonant field + procedural z.
