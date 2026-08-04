# Milestone 4 — where GRPO stands, and why

Written at the end of the 2026-08-04 overnight session. Every number here is
reproducible from a script in `scripts/`; none of it is inference from a
training curve.

## The short version

GRPO cannot learn in this world model, and the reason is measured rather than
guessed: **inside the distribution the policy actually plays in, the model is
indifferent to what the policy does.** Fixing that is upstream of the RL, and
upstream of the latent too.

## The chain of measurements

Each one rules out a cheaper explanation than the last.

**1. The world model was action-blind.** `scripts/finetune_action.py` scores how
well the model can say which of four candidate action sequences produced a real
transition. Chance is ln(4) = 1.3863 nats. The pretrained checkpoint scored
**1.3869** — exactly chance. It predicts the next latent at skill +0.864 by
learning the scene's momentum, which is most of the next frame, and never had to
use its action input.

**2. Fine-tuning fixed that, and it was not enough.** A counterfactual objective
took it to **0.118 nats at 0.997 accuracy** while *improving* reward
readability (hp1 +0.816 → +0.820, combo1 +0.397 → +0.493), at a skill cost of
0.864 → 0.819. `ckpt_cf/best.pt`, step 12000. GRPO still did not move.

**3. The action response does not survive contact with realistic play.**
`scripts/action_sensitivity.py` reported 0.235 of a bar of controllable spread —
but by holding one button for 1.6 straight seconds. `scripts/aggression_test.py`
samples 32 rollouts per start from the corpus prior, with a realistic opponent:

| | value |
|---|---|
| damage spread within one start | **0.0112** |
| damage spread across starts | 0.5115 |
| corr(attack rate, damage dealt) | **-0.0001** |

The reward is 45x more determined by the situation than by the policy, and
uncorrelated with attacking. Group-relative advantages remove the situation
term, which leaves 0.011 of action-uncorrelated variation. There is no gradient.

**4. It is not the pooling.** `scripts/pooling_test.py` encodes the same frames
through the same weights twice — once as the [CLS] token, once keeping all 256
patch tokens — and predicts the attack buttons from each. 0.5718 against 0.5783.
Keeping the entire spatial grid buys 0.0065. The detail is not being discarded
at the last step; it is never computed.

**5. Why that makes sense.** The game renders 640x480, capture squashes to
480x480, training downsamples to 224x224, and at patch 14 each of the 256 tokens
covers roughly 40 pixels of the original frame. Whether an attack connects turns
on a few pixels of spacing. So the model can know a button was pressed —
counterfactual accuracy 0.997 — and not whether it landed.

Everything else follows from this one fact. Behaviour cloning caps at 0.565 mean
AUC across two unrelated architectures. GRPO reads +0.000 against its reference
in six configurations. Damage does not correlate with aggression. These are not
separate problems.

## Two auxiliary objectives that were gamed

Worth recording as a pattern rather than two incidents, because a third attempt
would probably repeat it.

*Inverse dynamics* — recover the action from `(z_t, zhat_t)`. Recovering it from
a **real** transition stalled at 0.78 nats by step 600 and stopped improving;
recovery from the **predicted** transition kept falling to 0.53 while prediction
loss degraded sixfold. The predictor was stamping the action into a spare
direction as a watermark and paying accuracy for it.

*Health-space counterfactual* — the same contest judged only on the probe's two
health directions. Accuracy 0.864 within 200 steps, 1.000 by 5400, while skill
fell to **-10.97**. Two dimensions is far too little room to make "encode the
action here" hard.

The full-latent counterfactual resists this because the real next latent carries
no watermark to match, so an action-encoding component moves the true and the
wrong candidates equally far from the target. **Any objective of the form "make
X depend on the action" is hackable unless X is pinned to something the model
cannot also write to.**

## What is built and working

- `sokubot/rl/` — policy with a factored action space (two 3-way categoricals
  plus six buttons, so left+right is unrepresentable rather than merely
  unlikely), imagined arena, group-relative advantages, opponent pool. 27 tests.
- Reward function with KO detection hardened against probe noise, shaping
  weights set from measured term breakdowns, and side symmetry tested.
- `ckpt_cf/best.pt` — the action-aware world model, skill +0.8189, cf 0.118.
- `horizon_best/reward_probe.npz` — the probe calibrated on predictor outputs.
  Fitting on encoder outputs and reading imagined latents gives R^2 of -44.
- Diagnostics: `action_sensitivity`, `aggression_test`, `pooling_test`,
  `hit_prediction_test`, `reward_sensitivity`, `bench_arena`, `eval_bc`.

The GRPO stack runs unchanged the moment the reward becomes controllable. It is
not the part that is broken.

## What would actually unblock it

In order of how much evidence supports them.

**Not resolution.** That was the obvious reading of measurements 4 and 5, and
`scripts/hit_prediction_test.py` was written to price it before spending GPU
hours. It puts four representations through the same head and split on the
objective target the reward is built from -- is the opponent about to lose
health within 0.4 s, base rate 0.1385:

| features | AUC |
|---|---|
| [CLS] latent, 192-d | 0.6205 |
| all 256 patch tokens | 0.6460 |
| CNN from scratch, 224px | 0.6538 |
| same CNN, 448px | **0.6327** |

Doubling the resolution made it *worse*, and a CNN trained from scratch beats
the pretrained encoder by 0.033. Everything caps between 0.62 and 0.65. The
encoder is already extracting close to what a single frame contains, so a
larger or finer-grained encoder is not the fix -- and this was a cheap way to
learn that rather than an expensive one.

**Temporal, most likely.** Whether an attack connects depends on which
animation frame each character occupies and how fast they are moving, and a
still frame shows neither. Every representation above was given exactly one
frame. The encoder is per-frame by construction and history enters only later,
at the predictor, so the latents it consumes may individually lack the motion
that decides a hit.

**Not motion either.** `scripts/temporal_test.py` ran that experiment. Same
target, same head, same by-replay split, base rate 0.1367:

| features | AUC |
|---|---|
| one latent | 0.5961 |
| three latents concatenated | 0.6070 |
| latent plus its two differences | 0.6248 |
| CNN, one frame | 0.5940 |
| CNN, frame stacked with one two steps back | 0.5256 |

Latent history buys 0.029. Giving the CNN a second frame made it *worse*.
Nothing tried -- more pixels, the full patch grid, a purpose-trained CNN, three
frames of history, explicit motion -- anticipates a hit above about 0.62.

**Which points at the decision rate.** `frame_skip = 4` puts decisions at 15 Hz
against a game running at 60. Whether an attack connects is settled inside two
or three frames, so the moment that decides it falls *between* decision steps
and is not in any observation the model receives, at any resolution and with
any amount of history. That would explain every measurement above at once, and
it is a consequence of a choice made at milestone 3 for training cost.

**Not the frame rate either.** `scripts/frame_rate_test.py` encodes every game
frame -- 53,510 latents at 60 Hz -- and asks the same question at a range of
distances:

| lookahead | seconds | base rate | AUC |
|---|---|---|---|
| 2 frames | 0.033 | 0.0153 | 0.5900 |
| 8 | 0.133 | 0.0575 | 0.6194 |
| 16 | 0.267 | 0.1052 | **0.6825** |
| 48 | 0.800 | 0.2296 | 0.6728 |

AUC **rises** with distance. A hit two frames away is *harder* to call than one
0.8 s away, which is the exact reverse of what a short-lived signal discarded by
15 Hz would produce. Sampling more often would not help.

## What that leaves

The pattern across all four experiments is consistent: the latent tracks
something slow and situational -- roughly, that a player is under pressure and
damage is likely soon -- and does not represent the mechanics that decide whether
a particular attack connects. More pixels, the full patch grid, a
purpose-trained CNN, three frames of history, explicit motion and four times the
sampling rate each failed to change that, and they cover every axis available
without changing what is being encoded.

Whether an attack lands is a function of hitboxes, animation frames, invulnerability
windows and exact positions. Those are discrete game state. They are rendered
into pixels only incidentally, and evidently not recoverably at this scale.

**The route worth taking is reading that state directly.** SokuLib exposes
character position, animation state and hit detection, and `dll/src/session.cpp`
already calls into the game's own structures for the replay bypass and reads
`SWRCHARINPUT` for the input log. A world model over a few dozen exact state
variables is a far smaller problem than one over pixels, its rewards need no
probe, and the action-consequence link this whole document is about would be
present by construction rather than hoped for.

That is a redesign of milestone 3's observation space, not a tuning change, and
it is the honest recommendation rather than a fifth attempt at the same thing.

**Reconsider the frame rate.** Decisions run at 15 Hz against a 60 Hz game.
Whether an attack connects is often decided within two or three frames, which is
below one decision step.

## What is not the problem

Optimiser tuning. Six configurations were tried — learning rates from 5e-5 to
3e-4, entropy coefficients 0.003 to 0.02, group sizes 8 and 16, advantage
scaling per-group and per-batch, uniform and corpus-prior initialisation, with
and without a KL anchor to the reference. The failure modes differed
(thrashing at KL 54 and 43% clipping; frozen at KL 0.0002 with entropy pinned at
24.98 of 25.4; collapse to 0.61 entropy) and the result never did: +0.000
against the frozen reference every time. That is what a missing gradient looks
like, and `aggression_test` says why.
