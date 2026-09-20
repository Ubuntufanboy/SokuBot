# Which behaviours are structurally unlearnable at a short horizon

## The setting

Training horizon was 8 decision steps = 667 ms, chosen because kinematic skill
is still well above a no-op there. Three behaviours the agent conspicuously
lacks all have payoffs outside that window.

## Measured, not argued

**Approach.** Rolled from corpus starts, the trained policy sits at 268 units of
separation and closes 7 units inside 8 steps; over 32 steps it closes 48 and
beats the untrained reference (220 vs 241). Holding toward would reach 187.
So approach is a ~2.5 s behaviour being optimised over a 0.67 s window, and the
agent leaves available approach on the table.

The corpus says hits land at a **median separation of 144 units**. The agent
fights at 268. That gap is the passivity a human opponent reported unprompted.

**Winning.** A KO lands inside the rollout on:

| starts | h8 | h24 | h48 |
|---|---|---|---|
| any corpus state | 0.0% | 0.8% | 0.0% |
| either side < 15% hp | 0.0% | 0.8% | 2.7% |
| either side < 8% hp | 0.0% | 2.3% | 2.3% |

A match is ~720 decisions. Even at 48 steps, >97% of rollouts carry no outcome
signal, and **selecting near-death start states barely helps**. A sparse
match-outcome reward is not learnable from rollouts alone at any horizon the
fidelity budget permits.

**Slow mechanics.** Per-gym scores from the same policy: `neutral` +23.4
HP/step, `block_enter` +21.7, down to `combo_extend` +4.8 and `okizeme` +1.6.
The fourteen-fold spread tracks how far past the horizon each mechanic pays.

## Why this is a world-model problem and not a horizon knob

Raising the horizon is bounded by `02-fidelity-decay.md`. The design that
escapes the bind is a **critic with lambda-returns**: imagination is asked to be
accurate only over H steps, and v(s_H) carries everything beyond, fitted across
the 1-3% of rollouts that do terminate. The outcome never has to be *imagined*,
only *predicted*.

## The result, and a diagnosis that was wrong

The critic fits well -- value R^2 -0.04 -> **+0.94** by step 2800, so it does
predict returns. But the first two arms degraded from their initialisation
(+0.9 and +0.3 HP/step, warm-started from a policy worth +10.0 at horizon 8),
with entropy rising 16.5 -> 17.5.

The diagnosis at the time was an advantage-scale/entropy coupling: `ent_alpha`
had been tuned against GRPO's group advantages and was left unchanged when the
advantage became unit-normalised critic advantages. It was a clean story and it
was **wrong**.

The control settles it. Five arms at horizon 24, identical except as noted:

| arm | critic | warm start | best net (h24) |
|---|---|---|---|
| C0 | yes | yes | +0.9 |
| C1 (win x5) | yes | yes | +0.3 |
| C3 CONTROL | **no** | yes | +2.0 |
| **C5** | yes | **NO** | **+2.9** |

C3 has no critic and degrades exactly like the critic arms, which exonerates
the critic and refutes the entropy story. C5 differs from C0 only in dropping
the warm start, and it is the sole arm that *rose* to its best rather than
decaying from its initialisation, with all 11 gyms positive.

**The finding: a policy optimised at horizon 8 occupies a basin that is wrong
at horizon 24.** Every arm initialised there was pulled apart regardless of
advantage estimator or reward; the one that started fresh learned. Transfer
across imagination horizons is not free, and warm-starting across them can be
actively harmful.

**Two caveats that must travel with this.** +2.9 HP/step at h24 is not
comparable to +10.0 at h8 -- per-step damage dilutes over a longer rollout, so
these are different scales, not a regression. And whether an h24 policy plays
BETTER against a human than the h8 one is unmeasured; only a match answers it.

**Kept because it is the kind of error a paper usually omits:** a confident,
mechanistically plausible diagnosis (entropy coupling) survived two failed arms
and would have been "fixed" -- and credited -- had the control not been run
first. The control was the cheapest arm in the set.
