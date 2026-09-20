# The encoder's held-out score does not survive contact with the game

## Setup

At inference the policy may see only pixels and its own inputs. A convolutional
encoder maps a frame pair to the same 33-channel state the world model uses.
It is scored by held-out R^2 with the split taken **by replay**, which is the
standard precaution and turns out to be insufficient.

## The gap

During a real match, the game's own memory was read as a passive instrument --
never routed to the policy -- and logged beside the encoder's output for every
decision. 2084 paired samples.

| channel | held-out corpus R^2 | **live R^2** | mean error |
|---|---|---|---|
| x | +0.895 | **+0.547** | 199 units |
| dx | +0.839 | **-0.103** | 300 units |
| hp | +0.723 | **+0.408** | 1702 HP |
| y | +0.657 | +0.047 | 62 units |

`dx` is worse than predicting the mean. Health is wrong by 17% of a bar.

## It is not a framing or calibration fault

Bias is ~0 on every channel (x +17 units, dx -0.0, hp -6 HP), and refitting an
affine correction recovers almost nothing (x 0.547 -> 0.568, dy -0.640 ->
0.002). A wrong crop or a scale mismatch would appear as a large offset or a
slope that refitting repairs.

What is there instead: large scatter and **slopes well below 1** (x 0.844, dx
0.440, dy 0.053) with near-zero bias -- the signature of a model regressing
toward the mean because it does not recognise its input.

Frame statistics were compared directly (live capture vs corpus frame):
brightness, contrast and gradient energy all in the same range. Not a pipeline
bug. The encoder was trained on 149 replays; the conclusion is ordinary
distribution shift over characters, stages and matchups.

## Two failure modes visible only in the live log

**1. Error scales with camera zoom-out.**

| separation | x error |
|---|---|
| 0-100 u | 117 u |
| 100-250 u | 154 u |
| 250-450 u | 202 u |
| 450+ u | 300 u |

The game zooms out as players separate, so characters shrink and localisation
degrades. Clutter is not the driver (193-207 u regardless of projectile count);
distance is.

**2. Identity tracking fails in an absorbing way.**

The encoder reports the scene left-to-right; which character the agent *is*
must be tracked. Nearest-neighbour association on position was right **64% of
the time with PERFECT inputs** and ~51% at live accuracy, in wrong runs up to
85 decisions (7.1 s). For that share of a match the policy read the opponent's
row as its own.

The mechanism is not noise, it is permanence: one bad association at a crossup
locks onto the wrong character until another crossing flips it back. Two fixes
were implemented and both rejected by measurement -- health as a second cue does
not help while health itself carries 1700 HP of live error, and hysteresis is
*worse than nothing* (50% vs 64%) because it also suppresses the ~25 genuine
side changes per match. Passive identification from the agent's own inputs is
also chance (43-51%): a direction is commanded on only 28% of decisions and
brief taps do not separate the characters.

What works is active re-anchoring -- hold a direction for ~1 s and watch, which
measured 647 units against 127 -- at a cost of ~6% of playing time.

## Projectiles: a representation failure, not a data failure

The projectile block was dropped from the original encoder at R^2 0.042 and
filled with the corpus mean. Measured live: projectiles were present on **75.1%
of steps** with counts 0-8, while the encoder emitted a constant **4.74, standard
deviation 0.0000**. The policy was structurally blind to them, which is exactly
the human's winning strategy (zone from across the screen).

Retrained on 2003 replays -- 13x the coverage that fixed every other channel --
predicting slot-0 coordinates. It peaked at step 4500 and then went backwards:

| | step 4500 | step 9000 |
|---|---|---|
| present | +0.133 | +0.073 |
| hb | +0.179 | +0.091 |
| proj_n | +0.062 | -0.010 |
| dx, dy, vx, closing | ~0 | all negative |

and the **state channels degraded with it** (dy +0.424 -> +0.384, vx +0.047 ->
-0.044): 14 unpredictable outputs consumed capacity and injected gradient noise
into channels that had been improving.

Conclusion: asking a pooled feature vector to regress the coordinate of one of
several objects that may be anywhere on screen is the same error that made
character position hard. The replacement is a spatial occupancy map over
positions relative to the target player, with slot-0 coordinates read off by
soft-argmax so the policy's input format is unchanged.

## Transferable claim

**A by-replay held-out split measures generalisation across replays, not to the
live renderer.** For any pipeline where a perception model is fitted offline
and deployed online, the honest measurement requires logging ground truth
during deployment. Doing so here changed three separate conclusions and
uncovered two failures invisible to the offline score.

## Coverage did not fix it (hypothesis refuted, with confounds)

The natural fix for a distribution-shift failure is more coverage: the encoder
had seen 149 replays and 2003 with video were available. Rebuilt the cache
across all 2003 and retrained.

Both encoders then scored on the SAME held-out split (297 replays held out of
the 2003-replay cache, 2079 samples), ten shared channels:

| channel | old (149 replays) | new (2003 replays) | delta |
|---|---|---|---|
| x | **+0.888** | +0.836 | -0.052 |
| dx | **+0.919** | +0.901 | -0.018 |
| hp | **+0.739** | +0.691 | -0.049 |
| y | **+0.674** | +0.607 | -0.067 |
| airborne | **+0.648** | +0.563 | -0.085 |
| spirit | **+0.578** | +0.539 | -0.039 |
| mean | **+0.5511** | +0.4247 | |

The encoder trained on thirteen times fewer replays is better on every channel,
on replays it mostly never saw. Leakage was checked: 26 of the 297 validation
replays (8.8%) were in the old encoder's training set, which is real but far
too small to account for a 0.126 gap.

**Two confounds, and they are the interesting part.**

1. Disk capacity forced 7 frames per replay instead of 150, so the new cache
   holds 13,874 samples against the old 22,350. This was never "more data" --
   it was **more breadth and 38% less depth**, and depth won.
2. The new model also carries the 14 unlearnable projectile outputs shown above
   to drag the state channels down.

`P_none` (same cache, same samples, projectile block removed) disentangles
them. **It does not recover**: +0.3975, against +0.4069 for the same cache WITH
the projectile block and +0.5511 for the 149-replay encoder. Deleting the
fourteen unlearnable outputs entirely buys nothing -- it is very slightly
*worse* than keeping them -- so the projectile block was not the cause, and
0.154 of the gap is left standing with the confound removed.

At a fixed sample budget, then, **frames-per-replay beats replay-diversity for
this perception task.** 149 replays at 150 frames each beat 2003 replays at 7
frames each, on a split drawn from the 2003. That is the opposite of the
standard intuition and it has a mechanism: consecutive frames of one replay are
nearly identical *as images* but differ in exactly the way the task cares about
-- a character has moved a few pixels and the label has moved with it. Seven
frames from a replay sample the pose manifold at seven points; 150 sample the
local geometry of position itself. Breadth buys new stages, characters and
palettes; depth buys the derivative the regression is actually fitting.

The claim is bounded: it is measured at one budget (~14k samples) on one task,
and the two caches differ in build as well as in shape. What it does refute is
the reflex that a 13x wider corpus must produce a better model.

The recorded lesson stands either way: the coverage hypothesis was tested
against a like-for-like held-out split rather than assumed, and it did not
survive.

## Projectiles as a map, and a null that was mine rather than the model's

Four arms on the same cache, differing only in how projectile position is asked
for and how heavily it is weighted:

| arm | projectile target | weight | mean R^2 | slot-0 error |
|---|---|---|---|---|
| P_none | none | -- | +0.3975 | -- |
| FULL_base | slot coordinates | 1 | +0.4069 | -- |
| FULL_proj3 | slot coordinates | 3 | +0.3771 | -- |
| P_heatmap | occupancy map | 1 | **+0.4082** | 281 u vs 325 u |
| P_heatmap3 | occupancy map | 3 | +0.3952 | 279 u vs 325 u |

Two things separate cleanly here.

**The representation, not the auxiliary task, was the drag.** Coordinate
regression at weight 3 costs 0.030 of state R^2 against no projectile block at
all; the occupancy map at weight 1 *gains* 0.011. Same information, same
budget, same backbone -- the state channels are harmed by being asked for a
coordinate and unharmed by being asked for a map.

**The weight behaves the same way in both formulations.** Tripling it costs
~0.013 whichever target is used, so the auxiliary task competes for capacity
regardless of how well-posed it is. Well-posedness decides whether the trade is
worth making, not whether there is a trade.

### The null was a measurement of my instrument

Both heatmap arms stalled around 280 units against a 325-unit baseline -- 14%
better than "assume the projectile is on top of the player" -- and I was one
step from concluding that projectile position is not recoverable at 224 px and
that the fix is a resolution rebuild.

The head that produced that plateau had **514 parameters**: a single 1x1
convolution from 256 channels to 2 maps. A 1x1 convolution cannot *find*
anything. It reweights channels the backbone already computes, per cell, with
no access to the local neighbourhood that makes a projectile a projectile. The
measurement was of the readout, not of the input, and "the resolution is
insufficient" would have been an expensive conclusion drawn from a probe with
no capacity to detect.

Rebuilt at 443k parameters -- 862 times the capacity -- it moved the number by
**11.9 units**:

| head | parameters | slot-0 error | vs 325 u baseline | mean R^2 |
|---|---|---|---|---|
| P_heatmap | 514 | 281.2 u | -13.4% | +0.4082 |
| **H2_heatmap** | **443,394** | **269.3 u** | **-17.0%** | **+0.4118** |

So capacity was *a* constraint and not *the* constraint. The high-capacity arm
is the best full-corpus encoder measured on any axis -- best mean R^2 of all
six arms, best projectile error -- but a 4.2% reduction in absolute error from
862x the parameters says the plateau near 270 units is a property of the input
or of the task, not of the readout. That conclusion is now founded; the same
words written yesterday would not have been.

### What the plateau probably is, and it is not resolution

Reading the head against its target: `to_map` is a stack of 3x3 convolutions
over a **screen-space** feature map, and `render_targets` renders the
occupancy map in **player-relative** coordinates -- +-600 units around the
player the projectile is flying at. Nothing in between converts one frame to
the other.

A convolution is translation-equivariant in its input frame. Asking it to emit,
at screen cell (i, j), evidence about *relative offset* (i, j) requires
subtracting the player's position, which is a global operation a stack of 3x3
kernels cannot express. It has no access to where the player is.

That predicts exactly what is observed. The camera keeps the midpoint of the
two players near screen centre, so screen position and player-relative position
are *correlated* -- which is why the head reaches 17% better than baseline
rather than 0% -- but the correlation degrades as the players separate and the
view zooms, which is where the live measurement already showed positional error
tripling.

This is the same failure as the two above it in a third costume: the question
was posed in a frame the architecture cannot represent, and the resulting null
was read as a fact about the data. The fix is cheap and does not need a new
cache -- give the head the coordinate grid and the player's own predicted
position as input channels, so "relative" becomes something it can compute
rather than something it must memorise. That is the next arm, and it is a far
rather than something it must memorise.

### It was tested, and it is wrong

`H3_coord`: identical to `H2_heatmap` except for a coordinate grid and the
encoder's own detached predicted player positions as six extra input planes,
450k head parameters against 443k. Slot-0 error came out at **279.7 u, worse
than the 269.3 u it was supposed to beat.** Mean R^2 rose to +0.4179, the best
of any arm, so the state channels were unharmed; the thing the arm was designed
to fix did not move.

The argument was clean and it was still wrong. Recorded as such.

### The measurement that was missing under all of this

Lining the arms up is what makes the real problem visible:

| arm | head | slot-0 error |
|---|---|---|
| P_heatmap | 514 params | 281.2 u |
| P_heatmap3 | 514 params, weight 3 | 278.9 u |
| H2_heatmap | 443k params | **269.3 u** |
| H3_coord | 450k + coordinate frame | 279.7 u |

Four architectures spanning 862x in head capacity, with and without an explicit
coordinate frame, all land in a **269-281 u band** about 15% better than
baseline. The spread between fundamentally different designs is 12 units --
and **not one of these arms was ever repeated at a second seed.**

So the 12 u by which `H2_heatmap` "beat" the linear probe is the same size as
the 10 u by which `H3_coord` "lost", and they cannot both be signal. The
capacity conclusion drawn above is therefore also withdrawn: what the evidence
supports is only that *nothing tried has moved projectile localisation out of
that band*.

This is the quietest pathology in the project. An unmeasured noise floor does
not produce a wrong number; it silently sets the resolution of every comparison
drawn against it, and every ranking reported here has been finer than that
resolution. The fix is one repeat per configuration, which costs most exactly
when the effects are small -- which is exactly when it matters.

What survives a 12 u error bar: the heatmap formulation beats coordinate
regression (+0.4082 vs +0.3771 at weight 3, a 0.031 gap in state R^2), and
`P_none` at +0.3975 establishes depth-over-breadth. Those are the two claims
worth keeping.


The procedural lesson is the one that generalises, and it is the third
instance of the same shape in this project:

  * velocity was called unlearnable from a loss that never funded it;
  * entropy coupling was called the critic's failure until a control exonerated it;
  * projectile position was nearly called resolution-limited from a linear probe.

**Before concluding that a quantity is not present in a representation, verify
that the thing asked to read it had the capacity to.** A negative result from
an underpowered readout is a statement about the readout. This is cheap to
check -- count the parameters in the head and compare against the complexity of
the function it is supposed to express -- and each time it went unchecked here
it pointed at an expensive and wrong next step.
