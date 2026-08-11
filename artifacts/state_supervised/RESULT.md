# The block gate passes, for the first time

`artifacts/state_supervised/best.pt` — step 12000, 224 px, `state_coef 0.05`,
warm-started from `wm_225k_bnfix.pt`. Trained on a rented 4090, 2026-08-10.
sha256 verified against the instance before it was released.

## The gate

|  | baseline 225k | this checkpoint |
|---|---|---|
| **block_effect verdict** | corpus 0.0135, model **-0.00028** — fails | corpus 0.0117, model **+0.01712** — BOTH HOLD |
| spatial probe, play | 0.557 | **0.620** |
| spatial probe, full (control) | 0.884 | 0.958 |
| direction sensitivity | 0.66% of latent spread | **2.05%** |
| predictor skill | +0.864 | +0.605 (running) / +0.578 (recalibrated) |

Confirmed on a second checkpoint, step 32000, with an independently rebuilt
bank and refit probe: `world_model_gap +0.01916`. Two checkpoints, two banks,
two probes, same sign.

## What the labels did NOT contain, and improved anyway

The reward probe on the new encoder, held out by replay:

| channel | before | after |
|---|---|---|
| spirit1 / spirit2 | 0.055 / 0.093 | **0.260 / 0.186** |
| cards1 / cards2 | -0.005 / -0.201 | **0.222 / 0.198** |
| hp1 / hp2 | 0.923 / 0.916 | 0.905 / 0.909 |

There is no spirit and no card information in the state labels. Both improved,
which is independent evidence the latent got richer rather than the gate being
fitted.

## The cost, and the three arms it took

Skill fell 0.864 -> 0.605. This is a trade, not a free win.

| arm | config | outcome |
|---|---|---|
| 1 | linear head, `state_coef 1.0` | head could not fit dx (corr +0.19, pred std 0.15 vs true 0.30); skill 0.51 |
| 2 | capacity head, `state_coef 1.0` | corr +0.008, skill **-0.21**, worse than copy-forward |
| 3 | capacity head, **`state_coef 0.05`** | this |

`state_coef 1.0` makes the state term ~90% of the objective and destroys the
predictor. 0.05 puts it in the regime `hud_coef 0.25` occupies, which is the
supervision weight that already worked on this model.

Arm 3 was stopped at step 32000 of 70000 because it was degrading, not
converging: skill 0.584 -> 0.533 -> 0.468 and `l_pred` 0.020 -> 0.030 -> 0.0996.
Step 12000 is the better artifact and is what is saved here. Note step 32000's
saved BatchNorm running stats give skill **-8.90** until recalibrated (+0.346)
— the same trap as docs/BUGS.md 1.

## A mid-run read that was wrong

Around step 16-28k the proxy measurements (a linear dx probe on cached frames)
said position was barely moving and I wrote that the architecture was
implicated and that direct supervision was losing to the unsupervised IDM run
(spatial AUC 0.688). On the actual gate it wins. The proxy disagreed with the
instrument that decides, and the instrument was right.
