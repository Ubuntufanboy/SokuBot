# What reward shaping bought, and why the rest did nothing

Seven arms, 10k steps each, identical bank and simulator, scored with a FIXED
damage-only reward so `net` measures play rather than the objective trained on.

| arm | net (HP/step) | vs control |
|---|---|---|
| control | +9.0 | - |
| whiff penalty only | +9.0 | +0.0 |
| proximity only | +9.1 | +0.1 |
| combo 0.5 | +9.8 | +0.8 |
| all three | +9.9 | +0.9 |
| all three, doubled | +9.9 | +0.9 |
| **combo 1.0 alone** | **+10.0** | **+1.0** |

## Combo works, and needed a safety check first

`combo` pays for growth in the agent's combo damage, on top of the damage term,
so weight w makes damage landed inside a combo worth ~(1+0.72w) times the same
damage in isolation. It was pinned at 0 in the code until ownership of the
`combo_damage` channel was settled, because if it tracked damage RECEIVED then
paying for its growth would reward being comboed and the run would look healthy
throughout. Measured over 6237 rising edges: the opponent's health fell on
95.9%, the agent's own on 1.5%. Dealer. Safe to weight.

## Proximity is inert, for a measurable reason

Weights were sized against an anchor: damage averaged over all player-steps is
**0.00089 bars/step**. Proximity was set to ~11% of that.

It never changed behaviour -- mean separation over training went 145 -> 165 ->
158 units, i.e. *outward*. The gradient existed (the agent can control 113 units
of separation inside 8 steps, 38% of the term's range) but was worth ~8% of the
damage cost of approaching, since closing distance gets you hit. **Outweighed,
not absent.** See `03` for why the horizon is the binding constraint.

## Whiff is inert and, contrary to expectation, safe

Humans whiff **92.5%** of attack presses (2-step window) and press on 5.5% of
steps, so a whiff penalty lands on ~5% of steps; at 0.01 bars it would cost 57%
of the entire damage signal. The prediction was that any useful size would
suppress attacking. It did not: at 2.5x the penalty, attack rate ROSE (0.111 ->
0.113) and the whiff rate barely moved (-0.00034 -> -0.00028, an 18%
reduction). The policy paid the tax and kept swinging, which says whiffing is
not something it can control at this horizon.

## Transferable claim

Shaping terms should be sized against a measured per-step anchor, and the
useful diagnostic for a failed term is not "it did not improve the score" but
**"what is the controllable range of the quantity it pays for, inside the
horizon, relative to what it costs"**. Both inert terms here have a specific
number attached rather than a shrug.
