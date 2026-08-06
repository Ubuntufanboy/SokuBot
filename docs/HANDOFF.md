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

The pipeline works end to end **inside simulation**:

```
captures (mp4 + CSV) → encoder → latent world model → reward probe → GRPO policy
```

The best policy is `artifacts/grpo_bounded/policy_best.pt`, step 2700,
`net +0.002146`. It beats its own initialisation by a clear margin and, in
world-model units, exceeds human damage throughput. **That is not the same as
beating a person**, and the gap between those two statements is the honest
summary of the project's state. Every number is measured inside a world model
that inflates damage 2.11× and whose action signal is only trustworthy for about
0.27 s.

**The blocker is actuation, and it is not a GPU problem.** Wine's DirectInput
reads real evdev devices, so synthetic keystrokes never reach the game. Until
that is solved the agent cannot press a button in Soku, cannot be tested against
a CPU bot or a human, and every number stays self-referential.

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

**Unresolved, five minutes to settle:** `artifacts/final/ckpt/sokubot.pt` is
**step 320000** — the final checkpoint of the full run, never evaluated, where
everything to date has used `best.pt` from step 225000. `best.pt` was selected
on skill mid-schedule; the 320k model is the end of a completed cosine decay and
may be better. Check it before any further work:

```bash
python -m scripts.eval_ckpt --ckpt /root/ckpt/sokubot.pt --val /root/corpus/val.pt
```

If it beats +0.8642, redo the counterfactual fine-tune from it and rebuild the
bank and probe — everything downstream is keyed to the world model's weights.

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

1. **Evaluate `sokubot.pt` (320k)** against `best.pt` (225k). Five minutes,
   possibly invalidates the base of everything else. §7.
2. **Solve actuation.** Wine's DirectInput reads real evdev, so synthetic input
   never reaches Soku. Local work, no GPU. Until this is done nothing can be
   measured outside the simulator, and the project's central question — does it
   beat a person — is unanswerable.
3. **Then finetune against CPU bots in the real game.** The first measurement
   that is not self-referential.
4. Optional, only if simulator work resumes: the policy still drifts down from
   its peak (+0.00215 at step 2700 → +0.00117 at 40k). Not a collapse any more,
   just decay. `policy_best.pt` captures the peak, so this is a nicety.

**What would raise the ceiling, if it needs raising.** Four runs peaked between
+0.0016 and +0.00215. The binding constraint is most likely the world model's
0.27 s trustworthy horizon rather than the optimiser — the action signal falls
from r=0.55 at one step to r=0.09 at sixteen. Improving *that* is a world-model
problem (rollout fidelity, or a stochastic latent so multi-step futures stop
collapsing to a blur), not a GRPO problem.
