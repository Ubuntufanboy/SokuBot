# Data-scaling study — 2026-08-03, RTX 5090

Does more Soku footage make a better world model, and how much training does it
take to use it? Run on 11.55 h drawn from the collection fleet while it was still
filling toward 200 h.

## Setup

* Model: `Config.soku()`, 15.60M parameters, unchanged across every run
* Data: nested subsets so each larger set is a superset of the smaller ones —
  1.03 h ⊂ 2.03 h ⊂ 5.02 h ⊂ 10.01 h, drawn interleaved from hosts A/B/C
* Validation: 1.54 h (32 captures) carved out **first**, disjoint from all
  training sets, decoded once into a fixed uint8 tensor cache of 1,583 windows
  so every model is scored on byte-identical inputs
* `selection.json` records exactly which capture went where

## Metrics

`val_pred` is MSE between predicted and true latents. Both sides are pinned to
unit variance by the projectors' BatchNorm, so it is comparable across models —
but on 15 Hz video it is **not interpretable on its own**, because a predictor
that passes its input through unchanged already scores well. Hence:

    skill = 1 - MSE_model / MSE_identity

where `identity` is "next latent looks like this latent", recomputed per
checkpoint in that checkpoint's own latent space. 0 is a pass-through, 1 is
perfect, negative is worse than copying.

`inv_dyn_auc` asks whether a linear map can recover which buttons were pressed
from a pair of consecutive latents — the Soku stand-in for LeWM's
physical-state probe, since there is no simulator state to regress onto.

## Results

| data | steps | epochs | model | identity | skill | inv-dyn AUC |
|---|---|---|---|---|---|---|
| 1h | 4,000 | 9.48 | 0.0427 | 0.0721 | +0.407 | 0.629 |
| 2h | 4,000 | 4.74 | 0.0434 | 0.0775 | +0.440 | 0.647 |
| 5h | 4,000 | 1.90 | 0.0423 | 0.0752 | +0.438 | 0.658 |
| 10h | 4,000 | 0.95 | 0.0589 | 0.0664 | +0.113 | 0.653 |
| 1h | 16,000 | 37.93 | 0.0699 | 0.0850 | +0.178 | 0.645 |
| **10h** | **16,000** | **3.79** | **0.0427** | **0.0848** | **+0.497** | 0.641 |

## Conclusions

**At fixed compute, more data does not help and eventually hurts.** The four
4,000-step runs are flat from 1h to 5h and collapse at 10h — but that is a
compute limit being misread as a data limit. At 4,000 steps of batch 128 the 1h
run sees 9.48 epochs and the 10h run sees 0.95.

**At matched compute, data is decisively what matters.** Both 16,000-step runs
got identical budgets. The 10h model scores +0.497 against the 1h model's
+0.178 — a **+0.319 skill difference from data alone**. Their identity baselines
are within 0.0002 of each other (0.0848 vs 0.0850), so both latents are equally
dynamic frame-to-frame and this is not an artifact of one facing an easier bar.

**Every dataset size has a compute budget past which it overfits.** 1h peaks at
~4,000 steps (+0.407) and *degrades* to +0.178 by 16,000 as it memorises its
hour. 10h is still improving at 16,000 and is the best model here.

**Do not read the inverse-dynamics AUC as a scaling signal.** The 1h→10h
difference does not survive its bootstrap interval, and a replicate of the 5h
run at identical seed and config differed by 0.015–0.018 AUC — the same size as
the between-condition differences. Single-seed AUC comparisons at this scale are
noise.

## Throughput (`bench_speed.txt`)

253.7 → 67.8 ms/step, 3.74x, from three changes: uint8 frames to the GPU
(bit-identical), gating training diagnostics behind `metrics_every` (every
metric ends in `.item()`, which drains the pipeline), and `torch.compile` with
max-autotune. TF32, fused AdamW and batched SIGReg measured as no-ops here.
Batch 128 is both optimal and the maximum — 192 gives no throughput gain, 256
OOMs.

## Files

* `scaling.json` — the four 4,000-step runs with their per-500-step eval curves
* `scaling_10h_long.json`, `scaling_1h_long.json` — the 16,000-step pair
* `final_eval*.json` — per-checkpoint trivial-predictor baselines and AUC with
  bootstrap intervals
* `selection.json` — capture-level record of the train/val split
* `bench_speed.txt` — raw throughput measurements
