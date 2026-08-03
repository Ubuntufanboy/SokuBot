# SokuBot

Superhuman Soku AI bot trained using LeWorld and AdaJEPA.

A ~15.6M-parameter **action-conditioned latent world model** for Touhou 12.3
*Hisoutensoku*, built from [LeWorldModel](https://arxiv.org/abs/2603.19312) and
[AdaJEPA](https://arxiv.org/abs/2606.32026).

The model watches gameplay frames, learns to predict what happens next **in
latent space** under a given set of button presses, and then plays by searching
for the button sequence whose imagined future best matches a goal. There is no
pixel decoder, no reward signal, and no expert policy: training is purely
self-supervised on offline replays.

Data comes from
[SokuFrameExtractor](https://github.com/Ubuntufanboy/SokuFrameExtractor)
(milestone 2), which captures frame-accurate `(video, inputs)` pairs from replay
files by hooking the game.

## Architecture

```
frame_t (224x224x3) ──ViT-Tiny────▶ z_t (192-d)  ─┐
                     patch 14                     │  causal over the
                     12L x 192 x 3H               │  observation history
                     [CLS] → MLP+BatchNorm        │
                                                  ▼
action chunk ──MLP──▶ c_t ──AdaLN-Zero──▶ Latent Predictor ──▶ zhat_{t+1}
(frame_skip x buttons)                    6L x 384 x 16H, causal
                                          → MLP+BatchNorm

loss:  L = || zhat_{t+1} - z_{t+1} ||^2  +  0.1 * SIGReg(Z)
```

| component | shape | params |
|---|---|---|
| encoder (ViT-Tiny, patch 14) | 192 x 12L x 3H | **5.54M** |
| predictor (causal, AdaLN-Zero) | 384 x 6L x 16H | **9.94M** |
| action encoder | 20x4 → 256 → 384 | 0.12M |
| **total** | | **15.60M** |

**No stop-gradient, no EMA, no target encoder.** Gradients flow through the
prediction target as well as the prediction. Collapse is prevented by SIGReg
(a differentiable normality test that pushes the latent toward `N(0, I)`) plus
the projector's non-affine BatchNorm, exactly as in LeWM.

At test time, AdaJEPA closes the loop: after each executed action the observed
transition becomes a self-supervised target, one gradient step updates only the
final layers of the encoder and predictor, and the planner replans with the
updated model.

## Install

```bash
python -m venv --system-site-packages .venv     # reuse a system torch if present
.venv/bin/pip install -r requirements.txt
```

`pymunk` is pinned below 7.0: `gym_pusht` 0.1.6 calls
`Space.add_collision_handler`, which pymunk 7 removed, and the environment
raises on its first `reset()` otherwise.

## Quick start

```bash
# 1. Smoke test -- offline, CPU, a few minutes. Collects its own PushT data.
python -m sokubot.tests.smoke

# 2. Collect PushT trajectories, then train the paper-sized model.
python -m scripts.collect_pusht --out data/pusht --episodes 2000
python -m sokubot.train --preset pusht --data-root data/pusht --device cuda

# 3. Goal-conditioned planning: frozen model vs AdaJEPA test-time adaptation.
python -m scripts.eval_pusht --ckpt checkpoints/sokubot.pt --data-root data/pusht

# 4. Train on Soku captures (points at SokuFrameExtractor output directories).
python -m sokubot.train --preset soku --data-root /path/to/dataset --device cuda
```

`Config` in `config.py` is the single source of truth; `Config.pusht()`,
`Config.soku()` and `Config.tiny()` are the presets.

## Layout

```
sokubot/
  config.py            all dimensions and hyper-parameters
  model/
    layers.py          attention, MLP, AdaLN-Zero block, MLP+BatchNorm projector
    encoder.py         ViT-Tiny observation encoder
    predictor.py       causal action-conditioned latent predictor
    action_encoder.py  action chunk -> conditioning vector
    world_model.py     the trainable stack + autoregressive rollout
  losses/
    prediction.py      next-embedding MSE (teacher forcing, no stop-grad)
    sigreg.py          Epps-Pulley normality statistic over random projections
  planning/
    cem.py             CEM (paper default) and gradient-descent planners
    adajepa.py         test-time adaptation + plan-execute-adapt-replan loop
  data/
    episode.py         npz episode format, action normalisation
    window.py          sub-trajectory windowing
    pusht.py           PushT env wrapper + block-biased data collection
    soku.py            streaming loader over SokuFrameExtractor captures
  probe.py             linear probe from latent to ground-truth state
  train.py             training loop
  tests/smoke.py       end-to-end verification
scripts/               data collection, evaluation, hyper-parameter sweep
```

## Two things worth knowing before you change anything

**1. SIGReg's sample-size factor decides whether the objective has a global
minimum at collapse.** The classical Epps-Pulley statistic is
`T_n = n * integral |phi_n - phi_0|^2 w dt`. Measured here, SIGReg is ~0.004 on a
true `N(0, I)` sample and ~0.409 on a collapsed one. Without the `n`,
`lambda * SIGReg` tops out at 0.041 while a healthy `L_pred` is ~1.0 — so
collapsing costs 0.04 and saves 1.0, and gradient descent takes that trade. The
first PushT run here collapsed by step 10 for exactly this reason.

The escape is also one-way: at an exactly collapsed latent every projection is 0,
the empirical characteristic function is identically 1, and
`d/dz cos(t*z)|_0 = 0`, so SIGReg's gradient vanishes. It can prevent collapse
but never undo it. `Config.sigreg_scale_n` defaults to `True`.

**2. Effective rank is not a collapse test; the probe is.** Effective rank falls
both when the encoder collapses *and* when it correctly discovers that PushT has
about five degrees of freedom. `probe.py` fits a linear map from the frozen
latent to ground-truth simulator state and reports held-out R² — a collapsed
latent scores ~0 there, unambiguously. This is LeWM's own evaluation (Tab. 1).

## Soku specifics

* **Actions are 20 binary channels** — both players' ten buttons
  (`up down left right a b c d change spell`). The model is conditioned on both,
  which is the strongest possible offline world model.

  *Open issue for the inference loop:* at play time the opponent's buttons are
  not observable. Planning will need either an opponent model, marginalisation
  over the opponent channel, or a model retrained with dropout on those inputs.
  Nothing in milestone 3 depends on resolving this; the live control loop does.

* **Decision rate is 15 Hz** (`frame_skip = 4` at the game's 60 fps). One action
  chunk is 4 ticks x 20 buttons.

* **Video is decoded on the fly, never pre-shredded into arrays.** 200 hours at
  60 fps is 43.2M frames; even decimated to 15 Hz and stored as raw 224x224x3
  uint8 that is ~1.6 TB against the 130 GB of mp4 it came from. `data/soku.py`
  runs one sequential ffmpeg process per capture, decimates inside the filter
  graph, and shuffles through a buffer.

* **Inputs are cross-checked as they are read.** The extractor writes each frame's
  buttons twice — as a bitmask and as ten boolean columns — and `read_actions`
  verifies they agree. A single flipped bit on disk makes them disagree; without
  the check it would silently teach the model a transition that never happened.

## Status

- [x] Milestone 1 — capture tooling refactor ([SokuFrameExtractor](https://github.com/Ubuntufanboy/SokuFrameExtractor))
- [x] Milestone 2 — headless containerised replay collection at fleet scale
- [x] Milestone 3 — LeWM + AdaJEPA model, trained and smoke-tested on PushT
- [ ] Milestone 4 — train on the full Soku corpus; text conditioning
- [ ] Milestone 5 — real-time inference and control loop

## Papers

- LeWorldModel — *Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312))
- AdaJEPA — *An Adaptive Latent World Model* ([arXiv:2606.32026](https://arxiv.org/abs/2606.32026))
- LeJEPA / SIGReg ([arXiv:2511.08544](https://arxiv.org/abs/2511.08544))
- DiT — AdaLN-Zero conditioning ([arXiv:2212.09748](https://arxiv.org/abs/2212.09748))
- DINO-WM ([arXiv:2411.04983](https://arxiv.org/abs/2411.04983))
