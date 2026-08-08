# SokuBot — handoff

**Read this first.** Written 2026-08-06, at the point where the rented GPU was
released and work paused. Everything below is measured, with the script that
measured it named. Where a claim was later found wrong it says so rather than
being quietly deleted, because several of the wrong claims were wrong in ways
worth not repeating.

Companion documents:

* [`ARTIFACTS.md`](ARTIFACTS.md) — every file in `~/K0NTR0L-2/artifacts`, what
  made it, and whether it can be regenerated.
* [`BUGS.md`](BUGS.md) — the eight bugs found, what each one cost, and the
  pattern three of them shared.
* [`MILESTONE4_STATUS.md`](MILESTONE4_STATUS.md) — the earlier evidence chain.
  Layered chronologically with corrections on top, so read this file first.

---

## 1. Where the project actually stands

The goal is an agent that beats the author in an online Hisoutensoku match,
learned from **pixels and controller inputs only** — no memory reading, ever.
See §8 for why that constraint is not negotiable.

The pipeline works end to end, **and now outside simulation too**:

```
training:  captures (mp4 + CSV) → encoder → latent world model → reward probe → GRPO policy
live:      X11 capture → encoder → 1-step world-model compensation → policy → uinput device
```

The best policy is **`artifacts/grpo_G0/policy_best.pt`**, step 3700, at
**`net +0.00186`** on the shared instrument (`scripts/eval_policy.py`, 6144
starts, noise floor 1e-5). It replaced `artifacts/grpo_bounded/policy_best.pt`,
which reads `+0.00148` on that same instrument against its recorded `+0.002146`.
G0 is plain GRPO with no method change at all -- same optimiser, same horizon,
same trust region -- re-run against the current bank and probe, so the 26% gain
is not attributable to a technique and could be the bank, the probe, or
run-to-run variance. **`+0.00186` is the number any new policy must beat.**

The recorded/instrument gap on the old policy (+0.002146 against +0.00148) is
the frozen reference each run builds from wherever its RNG stood, which cannot
be reconstructed — see §2, and never compare two runs' own `evaluate()` numbers.

Both policies beat their own initialisation by a clear margin and, in
world-model units, exceed human damage throughput. Every number in §3 is still measured inside a world model that
inflates damage 2.11× and whose action signal is only trustworthy for about
0.27 s — read them as world-model units, always.

**Milestone 5 is done: the agent has played a human.** On 2026-08-06 it played
the author in Vs Player, Reimu (human) vs Cirno (agent), driving the game itself
from window pixels. The author's summary is the result that matters:

> "It wasn't great or anything (as expected) but it was 100% real genuine
> gameplay. No super weird behaviour. … I honestly wouldn't even know it was an
> AI playing."

```
2121 decisions over 141 s armed
round trip p50 48.2 / p90 50.6 / p99 61.8 ms
0.42% of decisions missed their 66.7 ms slot; scheduled lag constant at 4 ticks
```

Recording and per-decision log: `results/`. The loop is `sokubot/live/`, run by
`scripts/play_match.py` against `scripts/serve_policy.py`. See §10 for what the
match revealed and what to do next.

**The old blocker is gone.** Wine's DirectInput reads real evdev devices, so
nothing injected above the kernel reaches the game — that root cause was right,
but the conclusion drawn from it ("synthetic input is not worth pursuing") was
too broad. A `/dev/uinput` device created *before the game starts* is read
normally. §10 records which kind of device, and why the obvious choice is wrong.

---

## 2. Units — read this before quoting any number

This caused a wrong conclusion that survived several messages, so it is the
first technical thing in the document.

| where | what it reports |
|---|---|
| `train_grpo.evaluate()` | damage **per step** (divides by `alive.sum()`, which is `batch × timesteps`) |
| `human_baseline.py`, `train_q_policy.py` | damage **summed over the H-step window** |

At `--horizon 4` these differ by **exactly 4×**. Comparing them directly
produced the claim "the agent reaches 33% of human throughput" when the true
figure is roughly 133%. The error was caught only because `train_q_policy`
independently reported `prior dealt 0.0239` for the same policy `train_grpo`
called `0.0064` — two numbers for one quantity.

**Everything below is per 0.27 s window (H=4) unless it says per step.**

### `net` also depends on the horizon it was measured at

The same trap, one level down. `net` is damage per alive step **averaged over a
rollout of `cfg.horizon` steps**, and a longer rollout has had longer to blur, so
the number falls with the horizon even for an unchanged policy. Measured on one
policy (`grpo_bounded`, 2048 starts):

| horizon | net |
|---|---|
| 4 | +0.00147 |
| 16 | +0.00092 |

So reading a horizon-16 actor-critic against GRPO's horizon-4 +0.00215 would
have compared a policy change and an instrument change at once and credited the
sum to the policy — a 35% handicap invented by the ruler.

**Use `scripts/eval_policy.py`.** It scores every checkpoint in one run, against
one frozen reference, at whatever horizons you ask for, and prints the reference
against itself first as a zero point (measured: 2e-5, so it resolves the ~0.002
effect with room to spare). Numbers from different training runs' own
`evaluate()` are *not* comparable to each other, because each builds its
reference from wherever its RNG happened to stand.

**The bar for any new policy is +0.00186 at horizon 4 on that instrument**
(`artifacts/grpo_G0`), not the +0.00215 recorded below — that figure is real but
was taken against a reference that cannot be reconstructed, and the same
checkpoint reads +0.00148 here.

---

## 3. The numbers that matter

### Damage throughput, all per 0.27 s window

| | dealt | note |
|---|---|---|
| Human, real game | 0.0121 | probed from real encoder latents |
| Human, through the world model | 0.0255 | same buttons, imagined — **2.11× inflated** |
| Corpus-prior policy (the GRPO baseline) | 0.0256 | human button *frequency*, random *timing* |
| **Best agent, peak** | **0.0364** | `grpo_bounded` step 2700, as P1 |

Source: `scripts/human_baseline.py`, `scripts/train_q_policy.py`.

Two cautions. The world model inflates damage 2.11×, so absolute figures mean
nothing outside it. And "human throughput" is matched by the *prior bot*, which
has human-like button statistics and no timing — throughput is not skill.

### GRPO evaluation, per step

Peak of the best run (`grpo_bounded`, step 2700):

```
net vs frozen init  +0.00215
as P1   dealt +0.0091   taken -0.0058
as P2   dealt +0.0069   taken -0.0060
press rate 0.222   entropy 11.99
```

`net` averages both chairs. **That averaging is load-bearing, not tidiness**:
the reward probe reads `hp1` and `hp2` with a small constant bias, visible as a
perfectly mirrored ±0.0007 net between chairs, and it cancels only when both are
averaged. The same bias makes `human_baseline`'s winner/loser split unusable —
it calls P1 the winner of 199 of 200 replays. Ignore those rows; the throughput
rows are fine.

### How far the world model can be trusted

`scripts/action_effect_test.py` fits `f(start latent, joint actions) → outcome`,
trains on some start states and scores on **states it has never seen**.
Within-start correlation on held-out starts is exactly what a policy gradient
consumes:

| horizon | seconds | return | net damage | dealt | own actions only |
|---|---|---|---|---|---|
| 1 | 0.07 | **+0.547** | +0.556 | +0.598 | +0.247 |
| 4 | 0.27 | +0.317 | +0.267 | +0.221 | +0.140 |
| 16 | 1.07 | +0.094 | +0.078 | +0.128 | +0.058 |

A synthetic control shaped like a real mechanic (attack rate scaled by the
opponent's starting health) is recovered at +0.999 at every horizon, so the
decay is a fact about the world model and not about the fitting budget.

**This is why `--horizon 4`.** It is measured, not taste.

The counterweight, from the same script's leverage table: at one step,
action-driven variance is only **3.2%** of across-state variance. The effect is
*predictable* but *small* — which is why one-step greedy control also fails
(§5).

### Rollout fidelity (`scripts/horizon_ablation.py`, on `ckpt_cf/best_bnfix.pt`)

| h | seconds | cosine to truth | rel. L2 | probed hp1 R² |
|---|---|---|---|---|
| 1 | 0.07 | 0.9963 | 0.072 | 0.789 |
| 4 | 0.27 | 0.9717 | 0.216 | 0.836 |
| 16 | 1.07 | 0.8258 | 0.538 | 0.809 |
| 48 | 3.20 | 0.4170 | 0.925 | 0.535 |

---

## 4. What is settled

**The world model is not invariant to controller inputs.** This was the central
open question. A one-step action→damage rule transfers to completely unseen
states at r ≈ 0.55–0.60. The model learned real, generalising mechanics from
pixels and buttons alone.

**The encoder does represent the characters.** Matched-HUD frames whose
characters are 31.5/255 apart sit at cosine 0.78, against 0.055 for arbitrary
pairs. The six HUD readings explain only 0.0296 of latent variance, and no
single dimension is half-explained (`scripts/what_is_encoded.py`).

**The KO signal is readable — from the announcement, not the health bar.** The
probe's KO detector runs at precision **0.003**, and anchoring its level to
`data/hud.py` moved that to 0.015, i.e. not at all: the error is in the probe's
per-step *deltas*. But the game draws "KNOCK OUT" across half the screen, and a
60k-parameter CNN on a fixed crop reads it. 120 hand labels, out-of-fold, split
**by capture** so nothing scores by memorising a stage, averaged over 5 seeds:

| class | n | recall | precision |
|---|---|---|---|
| none | 84 | 94.0% | 97.5% |
| round | 12 | 100% | 85.7% |
| start | 3 | 100% | 100% |
| down | 8 | 75.0% | 85.7% |
| **knockout** | **8** | **75.0%** | **85.7%** |
| other | 5 | 60.0% | 37.5% |

`none` misread as a banner 6.0%; knockout↔down confused 12.5% each way.

**KO precision 0.857 against the probe's 0.003.** That is what lets `win`/`lose`
stay in the reward — the ±5 term can pay for outcomes rather than for probe
noise. It stays inside the no-memory-reading rule: this reads pixels, and reading
an announcement is a different and far easier problem than inferring the state
that produced it.

Two limits worth stating. 85.7% is 6 of 7 predicted knockouts, so the interval is
wide, and more `knockout`/`down` labels are the cheapest available improvement.
And the classifier only sees candidates that already passed
`harvest_banners.py`'s blue-coverage filter — recall-first by design, with
precision coming from the classifier — so the deployed rate is the product of the
two, not this number alone. Model at `artifacts/banner/`.

**And the latent carries KNOCKOUT, so it works inside imagination — but only
knockout.** A pixel CNN cannot fire in a rollout, where there are no pixels. So
the classifier was used to label 65 599 decision steps across 24 replays, and a
**linear** probe fit on the existing 224 encoder's latents, held out **by
replay**:

| class | AUC (held out) | precision | recall | P @ recall 0.80 | base rate |
|---|---|---|---|---|---|
| **knockout** | **0.947** | **0.803** | 0.487 | 0.190 | 1.54% |
| round | 0.976 | — | — | — | 2.17% |
| down | 0.730 | **0.132** | 0.117 | 0.004 | 0.35% |

**`win`/`lose` can be switched on**, against the health detector's precision of
0.003 — a 268× improvement — with no encoder retraining. Two conditions:

* **Hold the conservative operating point.** Precision collapses to 0.190 at
  recall 0.80. The errors are not symmetric: a missed KO forgoes a bonus, while a
  false KO pays ±5 *and* masks the rest of the trajectory. Recall 0.487 is the
  right trade.
* **Round-end reward stays off.** `down` reads at precision 0.132 from the
  latent, though the *pixel* classifier gets 0.857 — so the information is in the
  frame and simply is not in the latent. That is the case for a supervised banner
  channel alongside `hud_coef`'s in the next world-model run, not for a cleverer
  probe.

AUC alone would have been misleading here, and nearly was: `down` scores AUC
0.730 and precision 0.132. At a 0.35% base rate a good ranking is worth nothing
on its own, which is the health detector's failure exactly.

**At the horizon the baseline trains at, the health detector fires too RARELY,
not too often.** Worth stating plainly because the 0.003-precision figure from
`probe_reliability.py` points the other way and is easy to over-generalise: that
was measured over long windows. In training at `--horizon 4` with
`ko_persist = 3`, a KO needs three consecutive sub-threshold reads inside a
four-step rollout, which essentially never happens. From the runs' own logs:

| | `alive_frac` | ⇒ rollouts terminating |
|---|---|---|
| `grpo_bounded` (health KO) | 0.9999 | ~0.03% |
| arm K (banner KO) | 0.9965 | ~0.93% |

A KO occupies roughly 1% of 0.27 s windows, so the banner sits near the true rate
while the health detector was about **35× too rare**. The consequence is sharper
than "noisy": `win`/`lose` was contributing almost nothing to the baseline at
all, so the agent was very nearly the outcome-indifferent thing that deleting the
term would have produced. The banner is what makes the ±5 term exist.

**And it survives the move into imagination**, which is a separate question and
not a given: the channel is fitted on *encoder* latents but the reward reads it
on *predictor* outputs, and those measurably drift.
`scripts/banner_in_imagination.py`, at threshold 0.20:

| | fire rate | mean | p99 |
|---|---|---|---|
| imagined rollout states | 2.11% | 0.0196 | 0.253 |
| real encoder latents | 1.06% | 0.0173 | 0.217 |
| true `knockout` base rate | 1.44% | | |

Imagination inflates the detector **2×**. Even taking every extra fire as
spurious, imagined precision is ~0.42 against the health detector's 0.003. Worth
re-running whenever the world model changes — a reward term that is silently
zero and a reward term that is absent produce the same training curve.

The threshold itself is not 0.5, which is what it looks like it should be. The
channel is a ridge fit to a 1.4% positive class, so its output compresses toward
zero and never reaches a half; at 0.5 the detector fires **zero** times. Measured
operating points are in `RewardConfig.ko_banner_threshold`.

**Bounded logits stop the collapse.** Four GRPO runs died identically — entropy
to 0.000, press rate 0.53, KL-to-reference 10¹⁴. Squashing logits to (−6, 6)
made that impossible and produced the best result yet:

| | four earlier runs | with bound |
|---|---|---|
| peak net | +0.00162 | **+0.00215** |
| min entropy | **0.000** | 10.362 |
| max KL | 108.7 | **1.90** |
| klref | 10¹⁴ | 25–34 |

The entropy floor never engaged in that run (α stayed inert at 0.018), which
confirms entropy collapse was a *symptom* of the unbounded runaway rather than
the disease.

---

## 5. What was tried and did not work

Recorded because each cost real time and the reasoning is worth not repeating.

**Run the control arm first, or nothing else is interpretable.** Two arms
(critic, banner KO) both landed near +0.0009 against a baseline of +0.00215, and
two changes that different landing in the same place says the limiter is
something they *share*. They shared a bank (150 replays from `build_hud_bank`)
and a probe (`gate_base`, 8 channels) that the recorded baseline never used.

Scoring the baseline **checkpoint** on the new instrument validates the
*evaluation*; it says nothing about the *training* pipeline. The control arm —
plain GRPO, baseline settings, this bank and probe — reached **+0.00191**, which
tracks the recorded +0.00215 closely and makes every other arm attributable:

| arm (own reference, horizon 4) | best `net` | vs control |
|---|---|---|
| baseline, as recorded | +0.00215 | — |
| **G0 — control: plain GRPO, this bank + probe** | **+0.00191** | — |
| K — banner KO at ±5 | +0.00140 | −27% |
| H — critic bootstrap | +0.00091 | −52% |

Without G0 there were two live explanations and no way to separate them. It cost
1.4 h and it is the difference between a result and a rumour.

**The verdict, all arms on one instrument (6144 starts, noise floor 1e-5):**

| arm | net (h4) | net (h16) | outcome vs control (h16) |
|---|---|---|---|
| **G0 — plain GRPO, current bank+probe** | **+0.00186** | +0.00093 | +0.277% |
| grpo — previous best | +0.00148 | +0.00096 | +0.488% |
| K2 — banner KO at ±1 | +0.00142 | +0.00073 | +0.431% |
| K — banner KO at ±5 | +0.00138 | +0.00070 | +0.374% |
| H — critic bootstrap | +0.00095 | +0.00046 | +0.334% |
| control | +0.00001 | −0.00000 | 0 by construction |

Two things are settled and one is not. The critic is **half** the control's
score, 90× the noise floor — not ambiguous. And the banner arms score *lower*
damage but *higher* outcome than G0 at both horizons, consistently, which is the
trade a finishing reward should produce. That last one is **not established**:
the control deviates 0.098pp at h16 while the G0↔K2 gap is 0.154pp, only ~1.5×
the noise. Separating them needs far more KO events than 6144 starts contain.

**A critic as GRPO's baseline — null, and the measurement says it had to be.**
Phase 2's premise was that a critic amortises the baseline over the batch,
freeing the factor of `group_size` GRPO spends on variance reduction to buy that
many more distinct starts. Arm A ran it at horizon 4 for 1100 steps and `net`
never left ±0.00005.

`scripts/baseline_quality.py` (256 starts × 16 rollouts, horizon 4):

| | value |
|---|---|
| var(λ-return), total | 8.51e-01 |
| …within a group (same start) — **the action-driven part** | 1.02e-03 |
| …between starts | 8.50e-01 |
| action share of total | **0.12%** |
| critic R² on the across-state mean return | +0.385 |
| noise per sample, critic vs group baseline | **21.8×** |

Only 0.12% of return variance is anything the policy did. A group mean removes
the other 99.88% *exactly*, because every rollout in the group shares the state;
a critic removes it only as well as it predicts, and the 61% it misses is still
hundreds of times the signal. Eight times the start coverage does not pay for
21.8× the noise per sample.

The fix is not to drop the critic but to stop using it as the baseline:
`--group 8` restores the exact conditional baseline, and the critic keeps
supplying `v(s_H)` inside the λ-return, which is the job it is actually good at.
**Baseline and bootstrap are two jobs and only one of them wanted a learned
function.**

**Horizon 16 — not available on this world model.** The other half of Phase 2
was that a λ-return lets imagination be short while credit reaches long. True in
principle, but `scripts/ood_calibration.py` measures how long the rollout stays
in distribution at all, under the corpus's *own* actions:

| h | cosine to truth | % inside the 99% band | ‖z‖ |
|---|---|---|---|
| 1 | 0.9961 | 100% | 14.05 |
| 2 | 0.9895 | 88.9% | 13.75 |
| 3 | 0.9813 | 41.8% | 13.45 |
| 4 | 0.9717 | 6.1% | 13.17 |
| 16 | 0.8238 | 0% | 10.88 |

Under *policy* actions the guard cuts at ~1.5 steps. Note the norm **shrinks**
(14.05 → 10.88) while Mahalanobis² rises 11×: the rollout is not inflating, it is
leaking into directions the corpus never occupies. That rules out the cheap fix —
rescaling the latent does nothing. It also means the guard is not miscalibrated:
at h=1 it flags nothing at cosine 0.9961, so it is measuring real drift.

**FF-JEPA terminal value — null, three independent runs.** The action-free
latent planner (arXiv:2606.09311) works well as a *model*: it beats the flat
autoregressive rollout at every horizon (+0.91 vs +0.86 at H=4, +0.51 vs +0.40
at H=64) and predicts health change over the interval at +0.72 (H=4) and +0.59
(H=16). But as a GRPO reward bootstrap it never helped:

| | peak net |
|---|---|
| floor only | +0.00162 |
| floor + FF-JEPA value | +0.00102 |

Structural reason: the planner is **action-free**, so its value is mostly a
state baseline — and GRPO already subtracts the group mean. It contributes
little signal and some noise, and appears to *accelerate* the entropy collapse
(min entropy 1.76 vs 3.89). FF-JEPA answers "where to aim"; our bottleneck is
"which actions get me there". Good model, wrong job.
See `scripts/subgoal_test.py`, `scripts/train_planner.py`.

**One-step Q / MPC — clean null.** `scripts/train_q_policy.py`, 20k steps,
greedy argmax over 32 prior-sampled candidates. `gain_net` oscillated around
zero (+0.00015, −0.00011, +0.00010, …). Consistent with the 3.2% leverage
figure: the effect is predictable but too small for candidate selection to move
the outcome.

**Behaviour cloning — abandoned earlier.** The BC head reached 0.910 accuracy
against a 0.896 base rate, indistinguishable from always predicting "no button".

**Entropy bonus, fixed coefficient.** At 0.02 it pinned the policy at 24.98 of a
possible 25.4 nats for 625 steps. A bonus and a floor are different objectives.

**KL-to-reference as the collapse remedy.** At 0.05 it loses; at 0.3 the policy
never moves (+0.00002 over 1000 steps). The asymptotics explain it: the policy
collapses by becoming *confident*, so the log-ratio runs to −∞, where the true
KL is only linear and its restoring gradient is a constant 1 however far gone
the policy is.

---

## 5b. Rebuilding everything after a new world model

A new encoder invalidates every derived artifact, and the order matters because
each step's fingerprint check refuses the previous step's output if it is stale.
Run it exactly like this:

```bash
WM=~/ckpt448/best.pt          # the new model

# 1. bank — latents mean nothing except relative to the encoder that made them.
#    Keep --replays and --seed identical to the old bank: build_hud_bank walks
#    the shuffled manifest and numbers what it keeps 0..N-1, so the same seed
#    reproduces the same replay ORDER, which is what lets the banner labels
#    below be reused instead of re-decoded (~25 min).
python -m scripts.build_hud_bank --ckpt $WM --replays 150 --seed 0 \
    --corpus ~/corpus --out ~/bank448.npz

# 2. reward probe — fit on *predictor outputs*, so it is invalidated by a
#    predictor change as well as an encoder one.
python -m scripts.horizon_ablation --ckpt $WM --corpus ~/corpus \
    --out ~/gate448                       # writes reward_probe.npz

# 3. banner channel — reuses ~/blp_labels.npz, which is per-frame pixel labels
#    and therefore independent of the encoder. Only the latents changed.
python -m scripts.fit_banner_channel --probe ~/gate448/reward_probe.npz \
    --bank ~/bank448.npz --labels ~/blp_labels.npz \
    --out ~/gate448/reward_probe_banner.npz

# 4. re-check the two things that do not transfer across encoders
python -m scripts.banner_in_imagination --wm $WM \
    --probe ~/gate448/reward_probe_banner.npz --bank ~/bank448.npz
python -m scripts.ood_calibration --wm $WM --bank ~/bank448.npz

# 5. re-measure the baseline ON THE NEW INSTRUMENT before believing any new
#    number. +0.00147 is a property of the 224 model, not a constant.
python -m scripts.eval_policy --wm $WM --probe ~/gate448/reward_probe_banner.npz \
    --bank ~/bank448.npz --policy grpo=~/sokubot-art/policy_best.pt --horizons 4
```

Step 5 is the one most likely to be skipped and the one that matters most. Every
policy comparison in this document is against a reference and a probe built from
one specific encoder; carrying `+0.00147` across a retrain would repeat exactly
the mistake §2 exists to prevent.

Two things worth re-deriving rather than assuming at 448: whether `down` becomes
readable from the latent (it is not at 224, precision 0.132, which is the case
for a supervised banner channel), and whether `spirit` becomes decodable at all —
at 224 it sits at R² 0.036/0.015, and the 5×6 px gauges arriving as ~3 px is the
reason the resolution was raised in the first place.

---

## 6. Getting back up to speed on a fresh box

Nothing is lost. The corpus is on HuggingFace and **public — no token needed**.

```bash
# 0. box: any GPU with >=12 GB and >=240 GB disk. The workload used 5 GB VRAM;
#    a 3060 runs it at ~1/4 the speed of a 5090. Disk is the real constraint.
git clone https://github.com/Ubuntufanboy/SokuBot.git && cd SokuBot
pip install -r requirements.txt opencv-python-headless huggingface_hub

# 1. corpus — 63.5 GB, 2003 captures, ~3 min at 370 MB/s.
#    Shard `a` alone gives 94.75 h train / 3.01 h val, which is plenty.
python -m scripts.fetch_corpus --out /root/corpus --repos Smashlytics/soku-frames-a
python -m scripts.split_corpus --corpus /root/corpus          # -> train/ and val/
python -m scripts.build_val_cache --manifest-root /root/corpus/val \
       --out /root/corpus/val.pt                              # ~5 min

# 2. world model — upload from artifacts/, do NOT retrain. See §7.
#    ckpt_cf/best_bnfix.pt is the one to use.

# 3. derived, in this order
python -m scripts.build_bank --ckpt /root/ckpt_cf/best_bnfix.pt \
       --out /root/bank_bnfix.npz                             # ~10 min
python -m scripts.horizon_ablation --ckpt /root/ckpt_cf/best_bnfix.pt \
       --out /root/horizon_bnfix                              # ~20 min, writes reward_probe.npz

# 4. the run that produced the best policy (~1.2 h on a 5090, 40k steps)
python -m scripts.train_grpo \
  --wm /root/ckpt_cf/best_bnfix.pt \
  --probe /root/horizon_bnfix/reward_probe.npz \
  --horizon 4 --steps 40000 --kl-ref-coef 0.05 \
  --entropy-floor-frac 0.8 --replay-share 0.3 --out /root/grpo_bounded
```

**Sync artifacts off the box regularly.** A vast.ai instance vanished mid-run
and only luck brought it back:

```bash
tools/sync_artifacts.sh <host> <port>          # pulls to ~/K0NTR0L-2/artifacts
```

For a final teardown, stream one tar instead — per-file `scp` took ten minutes
for 481 MB where a single tar moved 633 MB in under one:

```bash
ssh -i ~/.ssh/id_ed25519_ai -p PORT root@HOST 'cd /root && tar cf - <paths>' > box.tar
```

**Operational notes learned the hard way.** Never use `pgrep -f` or `pkill -f`
with a pattern that appears in your own command line — it matches itself, which
killed three SSH sessions and once reported a dead training run as `RUNNING` for
forty minutes. Write the PID to a file and use `kill -0 $PID`.

---

## 7. Do not retrain the world model, and check one thing first

`artifacts/ckpt/best.pt` is 225k steps at skill **+0.8642**, and
`artifacts/ckpt_cf/best_bnfix.pt` is that model counterfactually fine-tuned to
be action-aware (action discrimination 1.3869 nats → 0.118, chance is 1.3863)
at skill **+0.7948**. Both are safe locally. Retraining costs 8+ hours and buys
nothing.

**SETTLED 2026-08-07 — 225k wins, so nothing downstream moves.** This was open
across two sessions. `artifacts/final/ckpt/sokubot.pt` is step 320000, the end of
a completed cosine decay, and the worry was that `best.pt`, selected on skill
mid-schedule, might be the worse model. It is not:

| checkpoint | skill, as saved | skill, BN recalibrated |
|---|---|---|
| `ckpt/best.pt` (225k) | +0.8787 | **+0.8789** |
| `final/ckpt/sokubot.pt` (320k) | +0.8709 | +0.8733 |

The last 95k steps of the schedule made the model slightly *worse* on held-out
data. `ckpt/best.pt` stays the base, `ckpt_cf/best_bnfix.pt` stays the model
everything uses, and the bank, probe and policies all stay valid. No rebuild.

**Both were measured in one sitting rather than comparing 320k against the
+0.8642 recorded in this document — and that is the only reason the answer is
right.** On this val set `best.pt` scores +0.8789, not +0.8642:
`build_val_cache` samples 2048 windows, and a different sample moves the
absolute figure by more than the gap being tested. Against the *documented*
number, 320k would have appeared to win by +0.009 and triggered a full rebuild
of the counterfactual fine-tune, the bank and the probe, for nothing. Two
checkpoints, one instrument, one sitting — the same discipline `BUGS.md` opens
with, applied to a number rather than to an artifact.

Reproduce (split with `--seed 20260803`, which regenerates the original
94.75 h / 3.01 h split exactly):

```bash
python -m scripts.eval_ckpt --ckpt ~/sokubot-art/wm_225k_best.pt  --val ~/corpus/val.pt
python -m scripts.eval_ckpt --ckpt ~/sokubot-art/wm_320k_final.pt --val ~/corpus/val.pt
```

### Retraining at 448: halve the learning rate, and read the eval as noisy

The 224 → 448 move (because captures are 480×480 and 224 threw away 4.6× the
pixels — spirit hexagons are 5×6 px and arrive as ~3) needs one change that is
not obvious, and one habit.

**`--lr 1e-4`, not the 2e-4 that was healthy at 224.** At patch 14, 448 px is
**1024 patches against 256**, so the same learning rate arrives four times
hotter. Measured at step 4000, warm-started from `wm_cf_bnfix.pt`, everything
else identical:

| | val | skill |
|---|---|---|
| `--lr 2e-4` | 0.2657 | **−3.08** |
| `--lr 1e-4` | 0.0229 | **+0.6571** |

The 2e-4 run reproduced the signature of the earlier run that diverged at 5e-4 —
training loss falling while held-out loss bounced — which is what identified it.

**The eval is noisy; do not read a trend off three points.** Held-out skill at
lr 1e-4 went +0.657 → +0.304 → +0.024 → +0.650 → +0.508 → −0.543 → +0.606. The
first three of those look like a clean monotone collapse and are not; a restart
on that reading would have thrown away a healthy run. Only 512 val windows, plus
a BatchNorm recalibration per eval, so the sampling error is large. Judge on the
best-checkpoint envelope over many evals, and note that `on_eval` already keeps
`best.pt` by skill, so a bad eval costs nothing.

**Warm-starting works.** 224/225 tensors transfer; only `encoder.pos_embed` is
re-gridded (16×16 → 32×32, bicubic) and `hud_head` is fresh. That is why skill
starts near +0.65 rather than climbing from zero as the original run did
(+0.1975 at step 5000).

---

## 8. The constraint that shapes everything

**No memory reading. Ever, for anything the agent learns from.**

The premise is a world model needing no surgical access to the game. The plan is
to **crowdsource gameplay** with a controlled recording program, and
crowdsourced players will not run a memory-reading mod. `SokuFrameExtractor`'s
DLL injection exists only to validate the surrounding ecosystem on a game where
ground truth happens to be available. It is scaffolding, not the interface.

Memory reading is acceptable for *evaluation* ground truth and for validating an
instrument, provided nothing the agent learns from depends on it.

When something looks unlearnable from pixels, the response is the scientific
method — state a falsifiable hypothesis, test it, report what it ruled out — not
a change of observation space. That discipline is what found the BatchNorm bug
after "the world model is invariant to inputs" had been accepted as a fact for
weeks.

---

## 9. Next steps, in order

1. ~~**Evaluate `sokubot.pt` (320k)** against `best.pt` (225k).~~ **Done
   2026-08-07: 225k wins (+0.8789 vs +0.8733), nothing downstream changes.** §7.
2. **Chase the two behaviours the live match exposed** — §10. Both are policy or
   reward questions, which is the first time that has been the honest place to
   point.
3. **Finetune against CPU bots in the real game.** Now unblocked: the live loop
   can run unattended (`scripts/play_match.py --launch --seek-battle`), which
   makes real-environment episodes collectable rather than hypothetical. The
   first learning signal that is not self-referential.

**What would raise the ceiling, if it needs raising.** Four runs peaked between
+0.0016 and +0.00215. The binding constraint is most likely the world model's
0.27 s trustworthy horizon rather than the optimiser — the action signal falls
from r=0.55 at one step to r=0.09 at sixteen. Improving *that* is a world-model
problem (rollout fidelity, or a stochastic latent so multi-step futures stop
collapsing to a blur), not a GRPO problem.

---

## 10. What the first live match established

### The loop, and the four things settled by measurement

Full rationale lives next to each decision in `sokubot/live/`; the short form:

* **Inference must be local.** The author's uplink is persistently jittery *at
  idle* — 26–239 ms to Google with the link at 0 KB/s, against 2 ms to their own
  gateway. The tail is the ISP hop every packet crosses, so no relay helps.
  Cloud: 65% of decisions missed their slot. LAN box: 0.42%.
* **Compress at 480×480, not 224.** Latent cosine to the training chain is
  0.9899 for a 480 JPEG downscaled server-side, 0.9152 for a 224 JPEG. One
  ordinary gameplay step is 0.9723 — so compressing at 224 costs *more than a
  whole step of real play*. JPEG artifacts at 224 sit at the same scale as the
  encoder's 14 px patches; at 480 the downscale averages them out.
* **The corpus really is stored vertically flipped.** Confirmed by eye against
  HF shard A-0001, not taken on trust from `data/hud.py`. Live frames are
  flipped to match.
* **A DirectInput joystick makes Soku's menus unusable under Wine** — the cursor
  scrolls continuously once the game receives any joystick input, independent of
  axis range, POV hat, and VID/PID. The author sees the same with physical
  controllers. The agent therefore uses a *keyboard* bound to keys the human's
  profile does not hold (`profile/sokubot.pf`).

### Two behaviours worth chasing, in priority order

**It gives up when its health gets low.** — **RESOLVED 2026-08-07, and both
candidate mechanisms are wrong.** The reward down there is noise.

`scripts/probe_reliability.py`, on held-out replays:

| true health | n | bias | noise |
|---|---|---|---|
| 0.02–0.04 | 294 | **+0.154** | **0.251** |
| 0.25–0.40 | 5214 | −0.011 | 0.123 |
| 0.80–1.01 | 15862 | −0.002 | 0.137 |

The probe is at its worst exactly where the agent gives up — double the noise
and a large positive bias. And the KO detector built on it barely works at all:

| | true KO rate | detector fires | precision | recall |
|---|---|---|---|---|
| encoder latents | 0.016% | 0.727% | **0.003** | 0.143 |
| imagined latents | 0.016% | 0.310% | **0.000** | 0.000 |

It fires 20–45× more often than KOs happen and essentially every one is false,
each paying `win`/`lose` = ±5 against damage terms worth ~0.1 and masking out
every later step. **So `win`/`lose` should be 0**, the same call already made for
`crush` and spell-cost: terms computed from unreadable state get switched off,
not tuned. No threshold retuning saves precision 0.003.

Both hypotheses in the original note are dead. The `alive`-mask starvation story
does not matter if the signal being starved was noise, and the scarcity story is
simply false — low-health states are **8.5%** of frames, not rare.

**It is not the HUD reader's fault, and that was checked rather than assumed.**
48 blind-annotated frames (`scripts/make_hud_annotation.py`, stratified to
oversample low health) put `data/hud.py` at **MAE 0.012 for health and 0.025 for
spirit**, and 0.012/0.013 in the low-health band specifically. The labels are
sound; the 224 px downsample is what destroys the information before the encoder
ever sees it — five six-pixel spirit hexagons become about three pixels.

That is why the fix is architectural: carry HUD state explicitly
(`sokubot/model/augmented.py`) instead of hoping a probe recovers it. With health
as a state channel at MAE 0.012 rather than a probe readout at 0.25 noise, the
KO detector becomes meaningful and the endgame stops being trained on noise.

**It presses too much.** The trained policy sits at a 0.113–0.130 press rate
against the corpus prior's 0.094 (`scripts/build_action_prior.py`, 361k frames).
Human players spam too, so this is mild — but it is a measurable drift from the
reference the whole evaluation is anchored to, and worth knowing whether it is
the entropy floor, the bounded logits, or a genuine strategy.

### What the card channel turned out to be worth (2026-08-07)

Nothing. Added to the probe's targets to see whether the spellcard reward could
be switched on, and measured:

| channel | ceiling (real latents) | calibrated, h = 1 / 4 / 16 |
|---|---|---|
| hp1 / hp2 | +0.878 / +0.896 | +0.78 / +0.83 / +0.81 |
| cards1 | +0.156 | +0.031 / +0.063 / +0.094 |
| cards2 | +0.051 | **−0.067 / −0.024** / +0.008 |

`cards2` is *negative* — worse than predicting the mean. The two strips are
geometrically symmetric, so +0.156 against +0.051 is noise straddling the 0.15
inclusion cutoff rather than signal. `spell_cost_min = 1e9` stays.

Note this moved `usable_horizon` from 5 to 0 without the model changing:
that figure is a mean over whichever targets clear the cutoff, so admitting a
near-zero channel drags it. It is not comparable across different target sets.

### A caution about instruments, from this session

Two measurement mistakes here cost hours, and both had the same shape: **an
instrument that could not see the failure reported success.**

* Mean absolute frame difference could not distinguish Soku's animated sky from
  a cursor crossing eight rows.
* Sampling that cursor twice, ten seconds apart, *aliased* onto the same row
  while it was cycling through all twelve entries every few seconds — reporting
  STABLE twice, confidently, on a screen that was looping.

Sample continuously and count transitions. And when the human at the machine
contradicts the measurement, the measurement is the thing to doubt first: the
author said "I have this problem with real controllers too" early on, and taking
that at face value would have skipped four wrong hypotheses.
