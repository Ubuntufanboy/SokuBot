# Bugs found, what they cost, and the pattern three of them shared

Written 2026-08-06. Every one of these was live in the repo and producing
numbers that looked plausible. None of them raised an exception until it was
looked for specifically.

---

## The pattern worth internalising

Three separate bugs were the same shape: **an artifact whose recorded metric
described something other than the artifact itself.**

* A checkpoint whose `skill` field belonged to a model that was never saved.
* A reward probe fit in one model's latent space, applied to another's.
* Two scripts reporting the same quantity in units that differed by 4×.

In every case the number was real, the artifact was real, and the *association
between them* was wrong. Nothing crashes when that happens. The only defences
that worked were (a) stamping artifacts with the identity of what produced them,
and (b) measuring the same quantity two independent ways and noticing they
disagreed.

Both are now in the code. `scripts/eval_ckpt.py:assert_predictor_sane` refuses a
predictor that cannot beat copy-forward; `horizon_ablation` stamps the reward
probe with the world model's fingerprint and `train_grpo` refuses a mismatch;
`build_bank` stamps the bank.

---

## 1. The checkpoint that was never saved — the big one

**`ckpt_cf/best.pt` had a one-step skill of −6.15 while its own blob recorded
+0.8189.** Its predictor was seven times worse than copying the previous latent
and doing nothing.

`scripts/finetune_action.py` scored a recalibrated *copy* and saved the original:

```python
def measure() -> dict:
    probe = _copy.deepcopy(model)
    recalibrate_bn(probe, val)      # BatchNorm stats re-estimated here
    ev = skill_eval(probe, val, cfg)
    return ev                       # ...and the copy is discarded
...
torch.save({"model": model.state_dict(), ...})   # the un-recalibrated model
```

The `--min-skill 0.80` gate worked perfectly. It just admitted a set of weights
that were never the ones written to disk.

**Why this fine-tune specifically.** The counterfactual objective pushes
deliberately *wrong* actions through the network as negatives, so BatchNorm's
running statistics absorb a distribution the model never sees at eval. Ordinary
training keeps them close — `ckpt/best.pt` measures +0.8689 as saved against
+0.8642 recalibrated — which is why the same bug in `train_full.py` was
invisible.

**Cost:** weeks. It was the default for `aggression_test`, `action_effect_test`,
`latent_drift_test`, and the `horizon_ablation` run that produced the reward
probe GRPO used. It generated a whole false picture: "the latent mean-collapses",
"the rollout is worse than copy-forward at h=1", "the probe reads encoder latents
2.6× over-dispersed". All artifacts of one file.

**Fix:** `finetune_action.py` and `train_full.py` now save the scored copy.
`assert_predictor_sane` catches it on load.

---

## 2. The reward probe fit to a different model

`train_grpo.py` defaults to `--wm /root/ckpt/best.pt` (healthy). Its reward probe
came from a `horizon_ablation` run whose own JSON records
`"ckpt": "/root/ckpt_cf/best.pt"` — the broken one.

**Every reward GRPO ever saw came from a linear map fit to a different model's
latent space.** A probe is a linear map out of one particular space and means
nothing in another. Nothing failed; the rewards were simply noise, and GRPO
faithfully optimised noise for weeks.

**Fix:** the probe is stamped with the world model's fingerprint and `train_grpo`
raises on a mismatch, warning when a probe predates the stamp.

---

## 3. The units error

`train_grpo.evaluate()` divides by `alive.sum()` — `batch × timesteps` — so it
reports damage **per step**. `human_baseline.py` and `train_q_policy.py` sum over
the H-step window. At `--horizon 4` they differ by exactly 4×.

They were compared directly, producing "the agent reaches 33% of human
throughput" when the truth is roughly **133%**. A conclusion inverted, and it
survived several messages before being caught.

**How it was caught:** `train_q_policy` independently reported `prior dealt
0.0239` for the same policy `train_grpo` called `0.0064`. Two numbers for one
quantity is what exposed it — not review, not reasoning.

**Lesson:** when a second script measures something the first already measures,
that redundancy is the point. Do not "simplify" it away.

---

## 4. Unbounded policy logits

Four GRPO runs died identically: entropy to **0.000**, press rate 0.53 against a
human's 0.097, KL-to-reference **10¹⁴**. Nothing stopped the policy from becoming
arbitrarily confident, and a policy whose log-probabilities have run away has
left the region the world model was trained on — it is optimising noise.

Three separate controls were added chasing this before the cause was identified:
an entropy floor, a KL target, LR decay. All were arguing with a runaway instead
of preventing one.

**Fix:** logits squashed to (−6, 6) with `tanh` before any distribution is built.
Minimum achievable entropy goes from 0.0000 to 0.4168, and the log-ratio to the
reference is bounded by ~64·L instead of reaching 10¹⁴. `tanh` rather than a hard
clamp because a clamp has no gradient outside its bounds — the same mistake as
bug 6.

**Result:** best peak of the project (+0.00215) and no collapse. The entropy
floor never engaged, confirming it had been treating a symptom.

**Bonus catch:** the bound was first set to 4.0, and the check in
`set_action_prior` rejected it — the corpus needs a logit of **−5.33** for its
rarest button (pressed 0.48% of the time, almost certainly spellcard). Below that
the bound cannot express human play, and would have silently compressed the
reference policy every evaluation is measured against. Two lines of validation
caught it on first contact.

---

## 5. The KL early-stop that guarded nothing

Added to stop the trust region blowing out. Written as:

```python
if _ep > 0 and stats["kl"] > gcfg.target_kl:
```

At `_ep = 0`, `logp == old_logp` by construction, so KL is always ~0 there. The
`_ep > 0` condition meant **the first epoch was never guarded** — every batch
still took one completely unconstrained step. The log showed
`early-stops: 5938/6001`, which looked like the guard working hard; it was only
ever suppressing the second update.

Diagnosed but **not yet fixed** — bug 4's bound made the run survive anyway. If
GRPO work resumes, fix this properly.

---

## 6. The KL anchor with no gradient outside its fence

`kl_ref` was computed from a hard-clamped log ratio. A hard clamp has **exactly
zero gradient beyond its bounds**, so the anchor stopped existing precisely when
the policy escaped — a fence whose gate opens outward.

**Fix:** `_anchored_kl` keeps the exponential clamped (that is what overflows)
and adds a linear term outside, so the penalty grows without bound and its
gradient keeps pointing home. This moved the collapse from step ~1200 to ~2000
but did not prevent it; bug 4 was the real cause.

---

## 7. Integrator wind-up in the entropy floor

The floor's multiplier decayed to its −10 lower clamp during the thousands of
steps the policy spent healthy, and Adam moves `log_alpha` by about `entropy_lr`
per step regardless of violation size. Recovery took ~500 steps — slower than the
collapse it existed to catch. One run sat at entropy 3.89 against a floor of 8.59
with α still 0.000; another overshot to α 54.6.

**Fix:** bounds set from measurements — lower clamp −4 (α never below 0.018,
still inert), upper 0.0 (α ≤ 1.0, because α = 1.0 already drove entropy from
10.76 to the 25.40 maximum in fifty steps, and 54.6 was never sensible), lr 0.05
so the full range takes 80 steps. Verified by simulation against the entropy
trace that broke it.

---

## 8. Small ones that still cost time

**`train_full.py` never created its checkpoint directory.** `on_eval` writes
`best.pt` into `--ckpt-dir`, but the only code that created that directory was
`save_checkpoint`, which runs when training *finishes*. Against a fresh dir the
run always died at the first eval — 18 minutes in, eval computed and discarded.
Invisible until then because `/root/ckpt` already existed on the old box.

**A flag that did nothing.** `--grad-accum` was added to the help text before
checking whether `train()` supports accumulation. It does not. Removed rather
than shipped as a silent no-op.

**`git pull` aborted, run proceeded on stale code.** Untracked files blocked the
merge; the launch command ignored the failure. Caught only because α started at
0.007 = exp(−5) — the *old* default — instead of exp(−4). Verify the code
version from a value it prints, not from the fact that you ran `git pull`.

**`pgrep -f` / `pkill -f` matching their own command line.** Killed three SSH
sessions, and once reported a dead training run as `RUNNING` for forty minutes
because the monitoring loop's own command line contained the pattern. Use a PID
file and `kill -0 $PID`.
