# What "trustworthy horizon" actually means, per channel

## Method

Roll the frozen simulator forward from corpus states using the **real recorded
actions of both players**, and compare against the state that actually
followed. No policy is involved, so every divergence is the model's. Errors are
quoted in units of each channel's own corpus standard deviation, because an
absolute error is meaningless without the spread: 0.05 of a health bar is a
twentieth of the resource, 0.05 of a binary flag is enormous.

384 windows, `fix5` simulator, history 12, decision rate 5 frames (83 ms).

## The table

| step | wall | hp | x | dx | vx | spirit | guarding |
|---|---|---|---|---|---|---|---|
| 1 | 83 ms | 0.01 | 0.02 | 0.03 | 0.27 | 0.01 | 0.18 |
| 8 | 0.67 s | 0.05 | 0.16 | 0.25 | 0.52 | 0.06 | 0.58 |
| 16 | 1.3 s | 0.10 | 0.30 | 0.44 | 0.60 | 0.12 | 0.42 |
| 24 | 2.0 s | 0.15 | 0.42 | 0.57 | 0.66 | 0.17 | 0.46 |
| 32 | 2.7 s | 0.19 | 0.52 | 0.68 | 0.64 | 0.21 | 0.66 |

In game units at the horizons that matter: health error 168 HP at step 8 and
470 HP at 24; x error 64 then 165 units; dx error 94 then 210 units.

## Three distinct failure shapes, which is the point

1. **Smooth compounding** (hp, x, dx, spirit). Error grows roughly with the
   square root of horizon. This is ordinary autoregressive drift and it sets a
   soft budget: health is still usable at 24 steps, which is why the reward
   -- almost entirely a health difference -- survives a longer horizon.
2. **Immediately broken** (guarding, 0.51 sigma by step 2, flat after). Not
   drift. A variable the model cannot predict even one step ahead. See
   `01-missing-move-identity.md`.
3. **Never learned** (vx, 0.27 sigma at step ONE, saturating ~0.6). Velocity is
   poorly predicted from the first step, so it is a representation problem in
   the one-step model rather than a rollout problem.

A single scalar "the model is good to 0.67 s" hides all three. **The per-channel
per-step table is the diagnostic that should be reported**, because which
mechanics are learnable depends on which channels survive, not on an average.

## The corollary that bit us

`dx` reaches 0.57 sigma (210 game units) by step 24. The behaviour we most
wanted to teach at that horizon -- closing distance -- involves changing `dx`
by about 120 units. **The model's error in the quantity exceeded the effect
being optimised.**

This is *not* automatically fatal, and the reason is worth recording: GRPO
scores actions within a group sharing one start state, so error common to the
group cancels in the advantage. What must survive is the DIFFERENTIAL. Measured
directly by driving the same starts three ways:

| step | hold toward | hold away | neutral | controllable range |
|---|---|---|---|---|
| 8 | 207 u | 320 u | 261 u | 113 u |
| 16 | 187 | 372 | 260 | 186 u |
| 24 | 193 | 411 | 264 | 217 u |

Neutral stays flat across 32 steps (261 -> 264), which is a good sanity check
that the model is responding to input rather than drifting. So the simulator
does represent approach; the absolute error is shared bias.

**Method note worth publishing:** absolute rollout error is the wrong statistic
for judging whether a model can support a shaping term under a
group-relative advantage. The action-differential is the right one, and the two
give opposite verdicts here.
