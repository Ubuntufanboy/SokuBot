# Observational data cannot teach action effects; counterfactual data can

**Status: measured on a validated proxy, 4 seeds per arm, no overlap between
arms. The strongest positive result in the project.**

## Why a proxy, and why this one is allowed to count

`findings/08` established that every remedy attempted inside Soku's
observational corpus failed, and argued the missing ingredient was
interventional data. That argument could not be tested: Soku's counterfactual
pairs need frame-precise input injection, which is a reverse-engineering task
of unknown length (attempted, characterised, unfinished -- see
`memory/soku-input-injection-anatomy`).

So the mechanism was tested on a miniature fighting game we own, where a
counterfactual is one function call: save the state, apply action A, restore,
apply action B. **A toy that is easy where Soku is hard would prove nothing**,
so `sokubot/proxy/validate.py` gates every conclusion on four criteria taken
from the real measurements, and it is allowed to fail:

| criterion | Soku | proxy |
|---|---|---|
| the event is rare | guarding 4.55% | 2.51% |
| most frames non-actionable | 34.8% actionable | 22.3% |
| observational association substantial | +0.0685 | +0.0451 |
| confounding present | unmeasurable | assoc = 1.72x causal |

It rejected four designs before passing:

1. `guarding` set on every away-hold, making it a deterministic function of the
   action -- association 1.000, causal 0.990, no confound possible. This is the
   error the user identified in the real Soku measurement: holding back during
   an attack is not blocking. Fixed by splitting `stance` (holding away,
   unrecorded) from `guarding` (a block that connected, i.e. ACT_RIGHTBLOCK).
2. A horizon mismatch between the two estimators -- a 1-frame association
   compared against a 3-frame intervention -- which made the association look
   *smaller* than the effect it is supposed to overstate.
3. Over-tuning: aggression pushed to 0.9, everyone rushed, nobody blocked, and
   three previously-passing checks broke at once.
4. The opponent's attack startup was hidden from the observation. Soku's state
   is `[2, 33]` and carries the opponent's `hitboxes` and `action_frame`, and
   the startup animation is on screen for several frames. Hiding it
   manufactured a confounder the real game does not have. With it visible the
   measured benefit of counterfactuals fell from 2.8x to 1.55x in raw units --
   half the first result was an artifact of the toy.

What stays hidden is INTENT, which is genuinely unobservable in both.

## The result

Effect of forcing away vs toward on predicted `guarding`, normalised by each
model's own output spread. Truth in the same units is **+0.1678**.

| arm | seed ratios | mean | vs truth |
|---|---|---|---|
| `obs_only` | 0.055 / 0.044 / 0.027 / 0.071 | **+0.049** | **29%** |
| `obs_plus_cf` | 0.276 / 0.233 / 0.177 / 0.148 | **+0.208** | **124%** |

No overlap: the worst counterfactual seed beats the best observational seed by
more than 2x. The counterfactual arm receives NO EXTRA GRADIENT STEPS -- it
swaps 35% of each batch for interventional pairs -- so this is not more
training.

**Observational training recovers 29% of the causal effect. Interventional
training recovers all of it.**

## The normalisation is the finding, again

Read in raw units the same runs say `obs+cf` reaches only 47% of truth, and the
conclusion written from that number was "counterfactuals are necessary but
insufficient -- something else is missing." That was wrong. The model's
`guarding` output is a compressed regression value; ground truth is a
probability difference. Comparing them is a units error, and correcting it
moves the answer from 47% to 124% -- from "still missing something" to
"solved".

This is the **fourth** time the same mistake decided a conclusion here:

  * absolute nats ranked channels by base rate, making blocking look ten times
    less action-dependent than position when the shares are comparable (`07`);
  * mean-error-in-sigma for a 4% flag measured its calibration offset, not its
    predictability (`01`);
  * an observational association was compared against an interventional effect
    and the 4-5x gap was called a defect of the model (`07`);
  * and here.

**An effect is only small relative to something. State the denominator, and
make sure both sides of the comparison are in it.**

## What it rules out

Pricing the rare channel 40x heavier -- the `flagw` hypothesis, which was null
on Soku where no ground truth existed to check it -- moves `obs_only` from 29%
to 21% and `obs+cf` from 124% to 131%. **The objective was never the binding
constraint.** Neither was data volume, reward shaping, move identity, inverse
dynamics or context length. The binding constraint is that the corpus contains
no state played two ways.

## Consequence for the real system

The proxy reproduces Soku's pathologies and says interventional data is both
necessary and sufficient for action sensitivity. That justifies the
reverse-engineering cost that `findings/08` left open, and it is a reversal of
the position taken from the un-normalised numbers an hour earlier.

The remaining work on Soku is one hook: forcing input at
`char_obj + 0x754` (an eight-int `SWRCHARINPUT`, axes signed -- NOT a 16-bit
mask) from a hook that runs BEFORE the character update, rather than from the
`wglSwapBuffers` hook, which is end-of-frame and demonstrably too late. The
replay input buffers found by content search are copies and patching them does
nothing. Everything else -- rootfs, capture, deterministic replay, the search,
the struct write -- works.

## Caveat

One proxy, 1-D, four actions, thirteen channels. It shows the mechanism is real
in a system with Soku's measured pathologies; it does not show the magnitude
carries over. The honest chain is: proxy demonstrates the mechanism ->
that justifies the injection work -> Soku settles it.
