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

---

# GRPO on the blocking gym: no signal

`policy_blocking_final.pt` — 20 000 steps, blocking gym (5599 start/side pairs),
warm-started from the corpus action prior, against the gate-passing world model.

Across **201 evaluations**:

    net vs frozen init   mean -0.00203   std 0.02152
                         min  -0.06196   max +0.06749
                         above zero on 49.3% of evals

A coin flip. The policy did not beat its own initialisation on the gym it was
trained on. `policy_best.pt` is therefore **selected on noise** -- it is
whichever eval spiked highest (+0.067), not a better policy, and that is the
checkpoint that went into the live test.

## Live test, agent vs the in-game COM

Agent on P1 (profile `sokubot`), COM on P2, verified by P1 losing health before
the agent was armed.

| round | agent (P1) | COM (P2) |
|---|---|---|
| 1 | 0.87 -> 0.12 | 1.00 -> 0.71 |
| 2 | 0.84 -> 0.51 | 1.00 -> 0.70 |

Round 1 it lost badly, round 2 was roughly even. Input hold rate 8.5-10.4%. Two
rounds against an unknown COM difficulty is a sample of two; round 2 could be
learning, noise, or the COM's script.

**A first attempt at this test was invalid and is recorded because it was
nearly believed.** In Vs Com the computer takes P2, which is where the agent
was, so P1 had nobody at the keyboard and the agent was beating a stationary
character 0.97 to 0.29. The health bar was real; what it described was not what
was claimed.

## What this does and does not say

The world model represents guarding now -- block_effect passes on two
checkpoints with independently rebuilt banks and probes. That was the blocker
and it is cleared. The policy has not learned to use it, and the training
signal says it never started to.

The obvious next question is whether GRPO can move at all on this gym: 5599
pairs from 60 replays is a narrow distribution, and the reward is a probe whose
spirit R2 is 0.26 and combo R2 0.46, read through 4 imagined steps. Before more
GRPO, measure whether the reward can even distinguish blocking from not
blocking on those starts.

The button trace was NOT captured: results/match_vscom.json records scheduling
only, so "does it hold back for seconds" is unanswered. The extractor DLL would
answer it directly from game memory but cannot load on this laptop
(new-WoW64); that measurement needs the .130 sandbox.

---

# Why GRPO could not learn, and a correction to the gate above

The 20 000-step coin flip is not a training bug. The plumbing is healthy --
measured on a real batch: within-group return spread 0.419, actions differing
on 99.8% of entries, advantage std 0.9996, gradient norm 1.108 reaching 9 of 9
tensors. Nothing is dead, stale or detached.

The objective is FLAT in the dimension the mechanic lives in. Forcing the
defender to hold one direction for a whole rollout, on 4096 paired starts with
the same opponent:

    away    reward -0.00125
    toward  reward -0.00100
    none    reward -0.05713

    away - toward = -0.00025 +- 0.00761   (zero)
    away - none   = +0.05588              (7 sem)

Holding ANY direction is worth +0.056. Holding the RIGHT one is worth nothing.
A blocking gym cannot teach blocking through a reward that cannot tell away
from toward, and GRPO correctly reported that there was nothing to learn.

## The gate above is weaker than it was written

`block_effect` reports two contrasts and I quoted one:

| checkpoint | direction vs none | mirror (left<->right) |
|---|---|---|
| baseline | -0.00028 | +0.000014 |
| step 12000 | **+0.01712** | **-0.000221** |
| step 32000 | **+0.01916** | -0.000093 |

The state supervision taught the model that *a direction is being held*. It did
not teach it *which*. The mirror contrast -- swap left and right and see if the
prediction moves -- is flat at every checkpoint, before and after, and that is
the contrast the mechanic actually needs. I reported "the block gate passes"
on the first column without weighting the second, which was in the same JSON I
printed.

This is consistent with the spatial probe, which only moved 0.559 -> 0.620
against a 0.958 ceiling: the model still barely knows which side the opponent
is on. It is the same finding as
memory/position-is-absent-from-the-cls-latent, and the fix is the one already
filed: an encoder that cannot discard position, not another loss term and not
another gym.

## Also corrected: a bug I diagnosed that was not there

An earlier run of the diagnostic showed rollouts holding away scoring 0.112
WORSE than rollouts holding toward, and I called it the bug -- the reward
penalising blocking. Repeating it across six seeds:

    default          -0.042 +- 0.096
    defence-weighted -0.040 +- 0.123

Both indistinguishable from zero, and the single -0.112 was noise from an
unseeded randomly-initialised policy at n~165. I nearly spent an hour of rented
GPU fixing it. The `--damage-dealt`, `--combo` and `--idle` flags added for that
fix are kept, because a defensive weighting is still the right thing for a
defensive drill -- but they are not the bug and they do not change the away vs
toward result (+0.00027 +- 0.00764).

---

# Removing the HUD shortcut did not work either

2400 steps of play-area-mirror augmentation (`state_coef 0.05`,
`mirror_coef 0.20`) from the state-supervised checkpoint:

| checkpoint | play | full (control) | identity |
|---|---|---|---|
| state-supervised | 0.6205 | 0.958 | 0.463 |
| + play-mirror | **0.6047** | 0.950 | 0.479 |

Flat, slightly down. The in-training metric moved a little in the right
direction (`mirror_play` 0.185 -> 0.214 over the run) and skill was barely
touched (0.605 -> 0.586), but the probe did not follow.

The reasoning was sound and the result is still negative: making the HUD
uninformative does not make the encoder pick up the characters. It has now
survived JEPA, inverse dynamics, class-balanced IDM, direct per-frame dx
supervision on 2003 replays, and an augmentation designed specifically to
forbid the shortcut.

That is five objectives against one representation, which is the argument for
changing the representation. ViT-Tiny -> single CLS token -> BatchNorm
projector pools the play area into one 192-d vector, and the fighters are small
and fast where the HUD is large and slow. `full` mirroring reads 0.958, so the
encoder can do horizontal discrimination when the evidence is big; it is the
*characters* it will not keep.
