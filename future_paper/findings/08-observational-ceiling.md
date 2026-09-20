# Observational data cannot teach a world model what actions do

**Status: five interventions, two seeds each, two independent instruments. All
null on action sensitivity. Run on hardware that passes a self-diff.**

## The sweep

Each arm is ONE change against the same controls (`mv2/base` seed 0,
`mv3/base` seed 1; 400 replays, 10k steps, history 12, shared corpus cache).
The bar is the controls' own between-seed spread: 0.00296 on `block_gain`,
0.0049 on `skill_h1`.

| arm | what it changed | d_block (s0/s1) | d_skill (s0/s1) |
|---|---|---|---|
| `ctw` | 70% of loss weight onto actionable frames | +0.0042 / +0.0014 | +0.035 / +0.002 |
| `idm` | read the action back out of the prediction | +0.0003 / +0.0026 | -0.008 / +0.014 |
| `flagw` | price flags by entropy, not base rate | -0.0004 / -0.0013 | **+0.016 / +0.031** |
| `hzw` | per-horizon loss, learned uncertainty weights | -0.0011 / -0.0028 | **+0.029 / +0.040** |
| `hist4` | 4 frames of state inside 12 of action | -0.0029 / -0.0069 | -0.010 / +0.005 |

**Nothing clears the bar on action sensitivity.** The interventions span three
different layers -- the objective (`flagw`, `hzw`), the input (`hist4`), the
auxiliary task (`idm`) and the sampling (`ctw`) -- and none of them changes how
much the predicted future depends on the button.

## A second instrument, and why it mattered

`block_gain` is measured inside a 32-step rollout. `state_action_effect` asks
the same question at step 1 with real inputs and nothing fed back. On the four
arms that were null both agree. On the one arm that looked promising they do
not:

| arm | probe delta (s0/s1) | block_gain delta |
|---|---|---|
| `ctw` | +0.0051 / **-0.0142** | +0.0042 / +0.0014 |
| `hist4` | **-0.0268 / -0.0358** | -0.0029 / -0.0069 |

`ctw` was the only arm positive on both seeds by the training metric. A second
measurement of the same quantity flips its sign. Two instruments disagreeing is
how a marginal effect is revealed not to exist -- and the reason to run the
probe rather than accept the number the training loop already prints.

## The most instructive failure

`idm` adds a head that recovers the action from the model's own predicted next
state, so its loss is a variational lower bound on I(A_t ; F_{t+1} | F_t) as
expressed by the model. It reached **98.9% accuracy** while the probe says its
dynamics became *less* action-dependent (-0.0134 / -0.0058).

The bound was satisfied without the mechanism being bought. A predicted state
is 66 numbers, and the state loss constrains some directions far more tightly
than others; the model wrote the action into the slack. **Information present
in an output is not the same as the output being causally shaped by that
information**, and a variational bound cannot tell them apart. This is a
general warning about auxiliary objectives justified by information-theoretic
arguments: the bound certifies decodability, and decodability is cheap.

## What did work, on a different axis

Two arms improved prediction quality by 6-8x the control spread, on both seeds:
`hzw` (+0.029/+0.040) and `flagw` (+0.016/+0.031). Multi-horizon uncertainty
weighting and entropy-priced flags each make a materially better predictor.
They are worth keeping and they are not what was being tested.

That separation is the point. **Prediction quality and action sensitivity are
independent axes, and this project spent a long time optimising the first while
needing the second.**

## The claim

Combined with the earlier nulls -- reward shaping (`05`), the move-identity
ablation (`01`), the blocking gym (`07`) -- every remedy attempted inside
observational data has failed, across reward, architecture, objective, input
and sampling. The one intervention that ever moved the counterfactual was
restricting what the model may condition on (history 2, +2.2x), which removes
a leak rather than adding a signal.

The mechanism is not subtle: **the corpus never contains the same state played
two ways.** No loss over it can manufacture that contrast, because the contrast
is not in it. Action sensitivity is a property of the interventional
distribution, and the training set is observational.

The remaining move is to generate the missing data: save-state, branch, force
divergent inputs, record both futures. That is the one thing this project has
never had, it is expensive, and every measurement of the last two days points
at it.

## Caveat carried forward

`hist4` was implemented by ZEROING the state on older positions, and a zero
state is a valid-looking state the model must learn to distinguish, not an
absent one. So "split histories do not work" is not established -- what is
established is that this implementation is the worst arm in the sweep. The
contrast with history-2, which shortened both streams and helped, points at
total sequence length rather than the split, and that is a cheap untested arm.
