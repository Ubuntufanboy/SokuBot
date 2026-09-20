# The state is not Markov: move identity is absent

**Status: hypothesis under test. The confirming anomaly that motivated it turned
out to be largely a metric artifact -- see "The anomaly, re-measured" below. The
mechanism argument stands on its own; the evidence for it is weaker than this
document originally claimed.**

## What the model conditions on

`StateDynamics.forward` concatenates exactly three things:

```
state    [B,T,2,33]      the 33 channels of data/state.py
proj     [B,T,2,K,7]     projectile slots
actions  [B,T,ticks,20]  both players' buttons
```

The 33 channels are:

```
dx facing guarding wrongblock crushed knockdown airborne dy x y vx vy ax ay
hitboxes hurtboxes hitstop untech action_frame hit_count hp spirit
spirit_delay timestop ground_dashes air_dashes correction combo_rate
combo_hits combo_damage combo_limit proj_n proj_hb
```

`action_frame` is present. **The nominal action id is not.** The model is told
how many frames into a move a character is, and never told *which move*.

The id exists upstream: `read_state` returns it as a fourth array and its
docstring explains it is kept integer on purpose, "because action ids are
nominal: 801 is not 'one more than' 800". It is then never written into the
bank (`state_bank.build` returns S, P, A, E, V where A is buttons) and never
reaches the simulator.

## Why this should break the dynamics

Soku is deterministic given both players' inputs and full internal state. The
33-channel vector is a lossy summary of that state. Whether the next frame
contains a hit, a block, a whiff or a counter depends on the attacker's move --
its hitbox geometry, its high/low/mid property, its active frames -- and on the
defender's stance. Two situations with identical `action_frame`, positions and
buttons can have entirely different outcomes if the moves differ.

A model trained by regression on such a summary cannot represent the branch. It
predicts the conditional mean over every move consistent with the observable
summary, which is a blur. This is a partial-observability failure, not a
capacity or optimisation failure, and no amount of data or parameters fixes it.

## The confirming anomaly

Simulator error rolled forward with the REAL recorded actions, 384 corpus
windows, in units of each channel's own corpus standard deviation:

| step | hp | x | dx | spirit | **guarding** |
|---|---|---|---|---|---|
| 1 | 0.01 | 0.02 | 0.03 | 0.01 | 0.18 |
| 2 | 0.02 | 0.04 | 0.06 | 0.02 | **0.51** |
| 4 | 0.03 | 0.08 | 0.12 | 0.04 | **0.62** |
| 8 | 0.05 | 0.16 | 0.25 | 0.06 | **0.58** |
| 16 | 0.10 | 0.30 | 0.44 | 0.12 | 0.42 |
| 24 | 0.15 | 0.42 | 0.57 | 0.17 | 0.46 |
| 32 | 0.19 | 0.52 | 0.68 | 0.21 | 0.66 |

Every other channel degrades smoothly with horizon, which is what compounding
autoregressive error looks like. `guarding` is *already* at 0.51 sigma by step
2 and then stays flat. That reads as a variable the model cannot predict one
step ahead, and blocking is exactly the mechanic whose outcome depends on move
identity.

## The anomaly, re-measured -- and mostly withdrawn

That reading does not survive a better instrument.

`guarding` has a 3-4% base rate and the simulator emits roughly 0.45 for it at
every horizon. A mean absolute error in units of sigma is therefore dominated
by one constant calibration offset: re-measured on fix5 the number is 2.48
sigma at step 1 and 2.58 at step 32, i.e. flat *and* enormous, which no amount
of prediction quality could move. The metric was measuring the offset.

AUC reads only the ordering, so a constant offset leaves it unchanged. 0.5
means "this output says nothing about when the flag is set". Two simulators,
384 held-out windows, real recorded actions:

| step | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| fix5 `guarding` | 1.000 | 0.991 | 0.997 | 0.865 | 0.819 | 0.723 | **0.555** |
| rawproj `guarding` | 1.000 | 0.997 | 0.992 | 0.893 | 0.847 | 0.642 | **0.456** |
| fix5 `airborne` | 0.990 | 0.979 | 0.927 | 0.794 | 0.793 | 0.745 | 0.731 |

The simulator ranks blocking **perfectly at one step** and decays smoothly to
chance by 32 -- which is compounding autoregressive error, the ordinary
pattern, not the pathology this document was built on. What is genuinely wrong
with `guarding` is calibration, not knowledge: a flag predicted at 0.45 against
a 4% base rate is unusable as a reward signal even though its ranking is
informative, and that is a different defect with a different fix (the
`pos_weight` clamp, already suspected in the trainer's own comments).

So the corrected claim is narrower and more useful:

  * `guarding` is **not** invisible to the simulator at short horizon.
  * It IS lost faster than the kinematic channels -- chance by 32 steps while
    `hp` is still at 0.53 sigma -- so a blocking reward evaluated over a long
    rollout is scoring noise.
  * The two GRPO arms that failed to teach blocking used horizon 8, where AUC
    is 0.87. That is not nothing, and "the world model cannot represent it" is
    no longer a sufficient explanation for why they failed.

The move-identity hypothesis is therefore still worth testing -- the mechanism
argument above does not depend on the anomaly -- but it has lost its
confirming evidence, and if the ablation comes back null there is no longer a
second reason to believe it.

**The general lesson is the one in `findings/06`:** every metric here that
looked like a discovery turned out to be a property of the instrument first.
A rare binary channel scored by mean error measures its base rate; check
ranking separately from calibration before concluding that a variable is
absent from a representation.

## Downstream consequence, already paid for

Two GRPO arms were spent trying to make the agent block, by paying it for
guard events and by reducing the price of damage dealt. Guard rate did not
move (0.114 -> 0.081 over training, i.e. it fell). The conclusion recorded at
the time was "the agent avoids damage by spacing rather than blocking". The
better explanation is that **no reward can teach a mechanic the world model
does not represent**, and this measurement says it does not represent it.

## A confound in the test as run, stated before the result

The move arm differs from the baseline in TWO ways, not one: the id is an
input, and predicting the next id is an extra loss term (weight 1.0, ~5.3 nats
at init against a total loss of 3.5). Both were necessary -- an autoregressive
rollout has nothing to feed itself at step two without the prediction -- but it
means a win is attributable to conditioning, to the auxiliary task, or to
either alone.

The auxiliary task is a plausible cause on its own: next-move accuracy reaches
0.91, so it is a well-posed, learnable objective over a 208-way vocabulary, and
representation learning from exactly that kind of task is the reason it would
help. Disambiguating needs a third arm with the prediction head and no
embedding on the input, which requires decoupling the two behind separate
flags.

Not run yet, deliberately: it is only worth GPU time if the move arm wins. If
it does not, there is nothing to attribute.

## Result: null, at n=2, after two wrong intermediate readings

Three versions of this result were written before the right one. The sequence
is worth recording because each error had a different cause.

**Reading 1 (wrong, bad hardware).** The first A/B ran on a box that was later
found to return three different answers for one deterministic computation. It
said move identity made guarding *worse* at every horizon past two. Void.

**Reading 2 (wrong, n=1).** Re-run on hardware that passes a self-diff, seed 0
said close to the opposite: guarding AUC **+0.053 at h4 and +0.033 at h8** --
precisely the horizons GRPO trains at -- and `block_gain` up 66%, from +0.0087
to +0.0144. Written up as "not refuted, and it lands where it matters."

**Reading 3 (the result).** Seed 1, same config, same corpus cache, flips every
sign:

| move - base | seed 0 | seed 1 |
|---|---|---|
| `skill_h1` | +0.094 | +0.000 |
| `block_gain` | +0.0057 | **-0.0061** |
| guarding AUC @ h4 | +0.053 | **-0.131** |
| guarding AUC @ h8 | +0.033 | **-0.155** |

**At n=2 there is no consistent effect of move identity in either direction.**

### The seed variance is the finding

The between-seed spread is larger than the between-arm difference it was
supposed to establish:

| | seed 0 | seed 1 | spread |
|---|---|---|---|
| `move` `skill_h1` | +0.657 | +0.568 | **0.089** |
| `base` `skill_h1` | +0.562 | +0.567 | 0.005 |
| `base` `skill_h16` | +0.267 | +0.480 | **0.213** |
| `move` `block_gain` | +0.0144 | +0.0055 | **0.009** |
| `base` `block_gain` | +0.0087 | +0.0116 | 0.003 |

Two things follow. First, the claimed 0.094 arm effect on `skill_h1` is the
same size as the move arm's own 0.089 seed spread, so it was never
measurable at n=1.

Second -- and this is the one durable observation -- **the move arm is
consistently the less stable of the two**, by 18x on `skill_h1` spread and 3x
on `block_gain`. That is what conditioning a rollout on its own argmax
predicts: the training signal depends on when the move classifier happens to
become reliable, so runs diverge from each other in a way the baseline's
continuous inputs do not. The mechanism argument below survives; the effect it
was invoked to explain does not.

### What this cost, and the rule adopted

Two full write-ups were produced and retracted before a second seed existed.
The first retraction was forced by hardware; the second would have been avoided
by one extra run. On this project, at these effect sizes, **n=1 is not a
measurement** -- it is a draw from a distribution whose width nobody had
measured. Every arm from here is run at two seeds before it is written down,
and the width is reported alongside the difference.

## A confound in the test as run, stated before the result

The move arm differs from the baseline in TWO ways, not one: the id is an
input, and predicting the next id is an extra loss term (weight 1.0, ~5.3 nats
at init against a total loss of 3.5). Both were necessary -- an autoregressive
rollout has nothing to feed itself at step two without the prediction -- but it
means a win is attributable to conditioning, to the auxiliary task, or to
either alone.

The auxiliary task is a plausible cause on its own: next-move accuracy reaches
0.91, so it is a well-posed, learnable objective over a 208-way vocabulary, and
representation learning from exactly that kind of task is the reason it would
help. Disambiguating needs a third arm with the prediction head and no
embedding on the input, which requires decoupling the two behind separate
flags.

Not run yet, deliberately: it is only worth GPU time if the move arm wins. If
it does not, there is nothing to attribute.

## Result: null on its own prediction, and a mechanism worth more than the hypothesis

**Provisional. Both arms were trained on hardware later found to be computing
non-reproducibly (see `06-metric-pathologies.md`); the rerun is queued on
another box. Every number below is quoted so it can be checked, not believed.**

Two arms, 400 replays, seed 0, 10k steps, differing only in move identity:

| | baseline | +move identity |
|---|---|---|
| `skill_h1` | +0.5835 | **+0.6647** |
| `skill_h4` | +0.5711 | **+0.6166** |
| `skill_h8` | **+0.5121** | +0.4620 |
| `skill_h16` | **+0.3325** | +0.1577 |
| `block_gain` | +0.0092 | +0.0097 |
| next-move accuracy | -- | 0.9352 |

Guarding AUC, the metric the hypothesis was actually about:

| step | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| baseline | 0.997 | 0.959 | 0.952 | **0.955** | 0.857 | 0.813 | **0.782** |
| +move | 0.995 | 0.975 | 0.898 | **0.792** | 0.735 | 0.649 | **0.523** |

The prediction was that conditioning on move identity would make blocking
predictable. It does the opposite everywhere past two steps, and `block_gain`
does not move at all. **The hypothesis is refuted on its own terms.**

### Why it fails, which is the part that generalises

Making a hidden variable observable is not free: the model must now predict it,
because an autoregressive rollout has to supply its own conditioning at step
two. And a *discrete* variable fed back through an argmax fails differently
from a continuous one.

At 0.935 per-step accuracy, the chance of a 16-step rollout containing no wrong
move id is 0.935^16 = 0.34. A continuous channel that is 6.5% off degrades the
trajectory by 6.5%. A move id that is wrong does not degrade the trajectory --
it *relocates* it, onto the dynamics of a different attack. So the error is not
merely larger at long horizon, it is categorically different: the rollout is
now a confident simulation of something that is not happening.

That is exactly the observed shape. Short horizon, where the id is almost
always right, improves substantially (+0.081 at one step). Long horizon, where
at least one id is almost always wrong, collapses (-0.175 at sixteen).

**The transferable claim: adding a latent variable to a world model's input
imposes the cost of predicting it, and for discrete variables that cost is
paid as catastrophic rather than graceful degradation.** The right design is
therefore either to keep the conditioning variable continuous, to feed a
distribution rather than an argmax, or to accept the variable as observable
only within the horizon where its prediction is reliable. This is a general
constraint on "just add the missing state" as a remedy for partial
observability, and it is worth more than the hypothesis that produced it.

Untested variants that follow directly: feed the softmax-weighted embedding
instead of the argmax (graceful degradation by construction), and condition
only the first k steps of a rollout on a predicted id.

## The test

Embed the nominal action id per player and concatenate it to the model input,
change nothing else, retrain, and re-measure the per-step table above. The
prediction is specific and falsifiable: `guarding` step-2 error should fall
substantially while the smoothly-degrading channels move little. If `guarding`
stays at ~0.5 sigma the hypothesis is wrong and the missing variable is
something else (candidates: hitbox geometry rather than identity, or the
opponent's card/meter state, neither of which is in the sidecar either).

Cardinality is manageable: observed ids in the extractor header are in the low
hundreds (crushed 143/145, knockdown 97/98/100), so a small embedding suffices.

## Why this matters beyond the game

If it holds, the general claim is: **the failure of a learned simulator to
support a mechanic can be diagnosed from a per-channel, per-step error table,
and traced to a specific missing conditioning variable** -- rather than being
attributed to horizon, capacity, or reward design, which is where we looked
first and spent two training arms.
