# Measurements that misled us

A paper about world models should say how its own instruments lie. Each of
these produced a wrong conclusion that survived until a second measurement
contradicted it.

## 1. A mean over channels hid a 4.75x gain on the channel under study

Velocity had been unlearnable across a 44-arm sweep (vx +0.040). An arm that
weighted the velocity loss 5x took it to **+0.190** -- while the reported mean
R^2 moved **+0.0002**, because the gain was paid for out of x (-0.055), y
(-0.059) and dy (-0.066).

The whole sweep was ranked on that mean. Three separate hypotheses about
velocity (wider frame gap, difference-image input, finer feature grid) had
already been tested and rejected against it. Velocity was never unlearnable; it
was **unfunded**, and the ranking statistic was blind to the transfer.

## 2. Marginal maxima across heterogeneous configurations

A selector chose the best learning rate and best width by taking the max over
each factor independently, and picked width 32 -- an artifact, because width 32
was the only width that had been run at the best learning rate. Fixed by
comparing each factor only among arms that agree on everything else.

## 3. Inverting a nonlinearity through its own mean

Proximity was believed to be inert because the agent "already fought at 145-158
units, the distance where hits land". That figure came from inverting
`mean(clamp(1 - sep/300))` to a separation. The function is clamped, so
`mean(f(sep)) != f(mean(sep))`, and rollouts beyond 300 units contribute zero
while dragging the average. True mean separation was **268 units** -- nearly
double, and on the wrong side of the corpus median. The conclusion "behaviour is
already optimal, drop the term" was exactly backwards.

## 4. Absolute error is the wrong test for a group-relative advantage

Rollout `dx` error at horizon 24 (210 units) exceeds the effect being shaped
(~120 units), which reads as disqualifying. But GRPO advantages are computed
within a group sharing a start state, so common bias cancels; the
action-differential is the relevant quantity and it is 217 units at the same
horizon. Absolute and differential give opposite verdicts.

## 5. A held-out split that does not span the deployment distribution

By-replay held-out R^2 0.895 on x became 0.547 live. See `04`.

## 6. Silent failure beats loud failure, every time

Three separate hour-plus losses came from success paths that reported nothing:

- `rc=$?` captured `date`'s exit status because `$(date -Is)` ran first in the
  same string, so 18 crashed training arms all reported success.
- A monitor grepped for `Traceback` and `Killed`; the arms died by SIGBUS with
  neither, so a dead sweep looked identical to a running one.
- A rewritten cache builder crashed on its final line (a variable the rewrite
  had deleted) after 70 minutes of correct extraction, with a 0-byte log.

The common shape: **the failure is at the end, where nothing is watching, and
the monitor matches the happy path only.** The rule adopted -- if this crashed
right now, would my filter emit anything? -- would have caught all three.

## 7. A simulated fix that was worse than doing nothing

Two repairs for the identity tracker were implemented and then *rejected by
replaying them against logged trajectories*: health as a second cue (no help)
and hysteresis (**50% vs 64%**, because it suppresses genuine side changes).
Both looked obviously correct when written. Neither shipped.

## 8. The instrument can be the machine

Every pathology above assumes the hardware computes what it is told to. On
2026-08-18 one box stopped doing that, and it is worth recording because the
failure mode is invisible to every other check in this document.

The same comparison was run three times: `torch.no_grad`, fixed window seed,
identical cached `.npz` input, CPU only, same machine. It is deterministic by
construction and must return the same numbers. It returned three different
answers -- steps 1 and 2 agreeing to the digit, everything past step 4
diverging, one run filling with NaN from step 4 and another clean to step 32.
Alongside it: four `libpython` segfaults in ninety minutes at different
addresses on different cores, and a training arm that computed NaN from a
corpus its twin read cleanly and that measured clean on inspection (0
non-finite cells, deterministic hash).

What makes this dangerous is that **none of the outputs looked wrong**. The
clean run produced a smooth, monotone, entirely plausible error table -- the
kind of result that gets written into a paper. It was only visible by running
the same thing twice and diffing, which nothing in the workflow did, because
determinism is the one property nobody thinks to test.

The cost was not one number. Weights carry the corruption forward, so every
model trained on that box is void and cannot be rescued by re-scoring it
somewhere good: the world model underneath every state-space RL result to date
was built there, along with the reward-structure findings and the
longer-horizon policy arm. All of it has to be re-run rather than re-measured.

**Adopted:** any box, before its results are believed, runs one deterministic
computation twice and diffs it. It costs one repeat and it is the only check
that distinguishes a wrong answer from a right one when the wrong answer is
plausible. It is now built into the chain that produces the replacement
results, not left as something to remember.
