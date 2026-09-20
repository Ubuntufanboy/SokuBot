# The action Jacobian is sparse, and the objective never priced it

**Status: measured, on hardware that passes a self-diff. The estimator and the
counterfactual are in `scripts/action_information.py` and
`scripts/action_influence.py`; the numbers below are reproducible from
`~/rl/mv2_corpus.npz`.**

## The question

Every earlier failure in this project was described as the world model being
"bad at" something -- blocking, spellcards, okizeme. That framing has no
denominator. The quantity that matters for RL is not how well the model
predicts the future but how much the future depends on the agent's action, so:

    I_k = I(A_t ; F_{t+k} | F_t)

estimated on the corpus with no world model involved, as the drop in
conditional entropy between a predictor given the state and one given the state
and the action. Same architecture, same budget; only the input differs. A
lower bound: a worse fit understates it.

## What the game makes available

| k (83 ms steps) | I_k, ctx 1 | I_k, ctx 12 | context cost |
|---|---|---|---|
| 1 | 1.784 | 1.388 | 1.29x |
| 2 | 2.533 | 2.203 | 1.15x |
| 4 | **3.030** | 1.951 | 1.55x |
| 8 | 2.933 | 1.445 | **2.03x** |
| 16 | 2.117 | 1.166 | 1.82x |
| 32 | 1.610 | 1.009 | 1.60x |

(actionable frames only; nats)

**I_k peaks at k = 4, not k = 1.** Eighty-three milliseconds after a press
almost nothing has happened; the consequence resolves over the next few steps
and is then washed out. Rollout horizon had been chosen by feel; this is a
measured reason to centre it near 4.

**Conditioning on 12 frames of state destroys up to 2.03x of the action
information, at exactly the horizon GRPO trains at.** A held input is legible
in the state it produces -- a player who has been walking backwards for 44
frames has said so through their position -- so a model with long context reads
intent off the state and the button becomes redundant. Trained at history 2
instead of 12, the simulator's measured response to the stick roughly doubles
(2.2x on the can-act subset, 2.4x on the rollout counterfactual, both at n=2
with between-seed spreads of 0.0007 and 0.0004). **Those two numbers match**:
the model is not irrationally ignoring the button, the context genuinely
destroys the information and the model tracks the loss closely.

Measured over ALL frames the context cost looks like a mere 1.04-1.52x, and
reading it that way indicts the model for something the data does. 62.6% of
frames are in `untech`, `hitstop` or knockdown, where no input changes
anything; averaging over them dilutes every action measurement by about three.

## The normalisation that reverses the obvious conclusion

Absolute nats rank channels by how COMMON they are. Against the uncertainty
that actually remains after the state is known:

| k | `guarding` I | H(guarding\|F) | share | `airborne` share |
|---|---|---|---|---|
| 2 | 0.0066 | 0.0540 | **12.2%** | 39.3% |
| 4 | 0.0139 | 0.0927 | **15.0%** | 21.4% |
| 8 | 0.0059 | 0.1199 | 4.9% | 8.2% |

Blocking is not an information-poor channel. It is a *rare* one: a 4.5% event
that the state already partly predicts leaves 0.093 nats to explain, and the
action explains 15% of it, against airborne's 21%. The claim "position carries
ten times the action information of guarding" -- which this project made, from
the raw nats -- measures base rate, not relevance.

The consequence is about the objective, not the architecture. **A loss that
sums per-channel error prices `guarding` at roughly 0.5% of the binary term
because it is rare.** The model allocates capacity accordingly and is right to.
Nothing stopped it learning to block; nothing asked.

## Why the two obvious remedies failed

Both were run, both at n=2, both null:

  * **Move identity** (`findings/01`) adds context, not action information.
  * **A blocking gym** at 5x enrichment of guard-onset windows moved action
    sensitivity from 0.0599 to 0.0564 -- i.e. not at all -- while costing
    general skill (`skill_h1` -0.127 and -0.044). Enriching *situations*
    cannot manufacture the contrast, because the corpus never shows the same
    situation played two ways.

## The intervention was also being applied too late

Median lead time before a block lands: **44 frames, 733 ms** of holding away
first; only 2.8% of blocks have no prior away-hold. Forcing "away" at a random
state and asking whether guard appears within a few steps therefore intervenes
*after* the causal event in most cases. The attacker's own hitbox is live at
only 53.9% of guard onsets -- the rest are blocks of something else, almost
certainly projectiles, which is the channel the encoder is separately blind to.

## Transferable claim

**Before concluding that a model fails at a mechanic, measure how much the
mechanic depends on the agent at all, and normalise by what was there to
explain.** Both halves matter and this project got both wrong for months: the
unconditioned average diluted every action measurement threefold by including
frames where no input can act, and absolute information ranked channels by
their base rate. The corrected picture moves the diagnosis from "the world
model cannot represent blocking" to "the objective prices blocking at half a
percent and long context launders the button out of the input" -- which implies
completely different work.
