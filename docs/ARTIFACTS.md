# Artifact inventory — `~/K0NTR0L-2/artifacts`

1.1 GB, 154 files, pulled off the rented 5090 before it was released
(2026-08-06). Everything here was verified to load. Nothing on that box was left
behind except things reproducible in under ~20 minutes, listed at the bottom.

Layout note: files pulled in the first sync sit at the top level
(`ckpt/`, `grpo_*/`, `horizon_*/`); the final teardown sweep landed under
`final/`. Same provenance, different sync pass.

---

## World models — the irreplaceable ones

All are `LeWorldModel`: ViT-Tiny encoder (patch 14, 12L, 3H, dim 192) → `[CLS]`
→ MLP+BatchNorm projector; predictor is a causal transformer (6L, 16H) with
AdaLN-Zero action conditioning. 15.6 M params. `latent_dim 192, history 3,
seq_len 4, frame_skip 4, action_dim 20, action_ticks 4, image_size 224`.

`skill = 1 − MSE_model / MSE_identity`, where identity is copy-the-previous-latent.

| file | step | skill | what it is |
|---|---|---|---|
| `ckpt/best.pt` | 225000 | **+0.8642** | Best-by-skill of the 320k-step run on the full corpus. The strongest plain predictor. |
| `ckpt/best_bnfix.pt` | 225000 | +0.8642 | Same, BatchNorm recalibrated. Barely differs — ordinary training keeps the stats close. |
| `final/ckpt/sokubot.pt` | **320000** | **unmeasured** | **Final** checkpoint of the same run. Never evaluated. End of a completed cosine decay, so plausibly better than `best.pt`. **Evaluate this first** — see `HANDOFF.md` §7. |
| `ckpt_cf/best.pt` | 12000 | **−6.15** as saved | **Broken artifact.** Blob claims +0.8189. Kept only as the evidence for `BUGS.md` §1. Do not use. |
| `ckpt_cf/best_bnfix.pt` | 12000 | **+0.7948** | **The one everything uses.** `ckpt_cf/best.pt` with BatchNorm recalibrated. Counterfactually fine-tuned, so it is action-*aware*: discrimination 0.118 nats against a 1.3863 chance baseline, where the plain model sits at 1.3869 (i.e. exactly chance). Fingerprint `886abb89c8b8e770`. |
| `final/ckpt_cf/sokubot_cf.pt` | 15000 | unmeasured | Final fine-tune step. Not BN-recalibrated, so probably shares bug 1 — recalibrate before trusting. |
| `ckpt_10h/best.pt` | 16000 | +0.5191 | Trained on a 10 h subset. The only checkpoint that survived locally when the first box vanished; the reason a full retrain was avoided. |
| `final/ref_best_10h.pt` | 16000 | unmeasured | Sibling of the above. |
| `final/cf_snapshot.pt` | 4000 | unmeasured | Mid-fine-tune snapshot. |
| `final/ckpt_health/sokubot_cf.pt` | 6000 | unmeasured | From an abandoned health-objective experiment. |
| `final/ckpt_idm_smoke/sokubot_idm.pt` | 40 | — | Inverse-dynamics smoke test. Disposable. |

**Action awareness matters more than skill here.** The plain 225k model predicts
better but is at chance on "which action produced this transition". The
fine-tuned one trades skill 0.86 → 0.79 for discrimination 1.39 → 0.118 nats.
GRPO needs the latter.

---

## Policies (`SokuPolicy`, 0.85 M params)

Factored action space: two 3-way categoricals (lr, ud) plus six Bernoullis,
because the game stores `lr`/`ud` as signed ints, making left+right
unrepresentable.

| file | step | net | notes |
|---|---|---|---|
| **`grpo_bounded/policy_best.pt`** | **2700** | **+0.002146** | **Best of the project.** First run with bounded logits; first that did not collapse. |
| `long_floor/policy_best.pt` | 3800 | +0.001622 | Entropy floor, unbounded logits. Collapsed after. |
| `grpo_h4c/policy_best.pt` | 1800 | +0.001517 | Fixed KL anchor, unbounded. Collapsed. |
| `long_value/policy_best.pt` | 1600 | +0.001024 | Floor + FF-JEPA terminal value. The value arm of the A/B — worse, and collapsed faster. |
| `grpo_h4b/policy_best.pt` | 200 | +0.000058 | `kl_ref_coef 0.3`. Frozen solid; kept as the "anchor too strong" data point. |

`net` is **per step**, averaged over both chairs. Multiply by 4 for per-window
figures. See `HANDOFF.md` §2 — this cost a wrong conclusion once.

---

## Reward probes

Linear maps from latent → `[hp1, hp2, spirit1, spirit2, combo1, combo2]`.
**A probe is only valid for the model it was fit on.**

| file | alpha | fit against |
|---|---|---|
| **`horizon_bnfix/reward_probe.npz`** | 100 | `ckpt_cf/best_bnfix.pt`, predictor outputs. **The correct current one.** |
| `horizon_bnfix/reward_probe_encoder.npz` | 10000 | same model, encoder outputs. For reading *real* latents (`human_baseline`'s `real` arm). |
| `horizon_best/reward_probe*.npz` | 1000 / 10000 | **Fit on the broken `ckpt_cf/best.pt`.** Poisoned every GRPO run before the fix. Kept as evidence. |
| `horizon2/`, `horizon_cf/` | 1 / 1000 | older, superseded. |

None carry a fingerprint stamp — they predate it. Anything regenerated from now
on will be stamped and checked by `train_grpo`.

**Known limitation:** the probe reads `hp1` and `hp2` with a small constant bias,
visible as a mirrored ±0.0007 net between chairs. It cancels when both chairs are
averaged (which `train_grpo.evaluate` does) but makes any single-chair or
winner/loser analysis unreliable.

Spirit is not usable at all: R² 0.036 / 0.015. Five six-pixel hexagons do not
survive the downsample. Guard-crush and spell-cost rewards are switched off
rather than computed from noise.

---

## Other models

| file | what |
|---|---|
| `planner_h4.pt` | FF-JEPA latent planner, H=4, ctx=1. Latent skill +0.8926, health-delta skill +0.6942, keyed to world model `886abb89c8b8e770`. Good model; **did not help GRPO** — see `HANDOFF.md` §5. |
| `final/heads/heads.pt` | Decoder heads, step 200k, 95 MB. The decoder that plateaued at L1 0.13 and produced frames with no visible characters — which was misread as evidence the *latent* had no characters. It was the decoder's failure. |
| `final/heads/heads_v1_linear_cond.pt` | Earlier linear-conditioned decoder variant. |
| `final/qpolicy/q.pt` | Q(z, a) → one-step damage, 2.7 M params, 20k steps. Clean null as a controller. |

---

## Logs and measurement outputs

Run logs as JSON (`*/log.json`) and raw stdout (`final/*.log`). The JSON is the
useful one — every eval row, entropy, KL, α, press rate, per-term rewards.

Measurement JSON worth keeping:

* `action_effect_h{1,2,4,8,16}.json`, `action_effect_fix_h{1,4,16}.json` — the
  action-signal-versus-horizon tables. `_fix_` are on the repaired checkpoint and
  are the valid ones.
* `subgoal_test.json`, `subgoal_probe.json` — FF-JEPA ablation.
* `human_baseline.json` — the human throughput baseline.
* `horizon*/horizon.json` — rollout fidelity curves.
* `what_is_encoded.json` — the matched-HUD test that proved characters are
  encoded.
* `train_log.json` — the 320k-step world model training curve.

## Media

`final/banner{,2,3}.mp4`, `final/review.mp4`, `final/spell{,2}.mp4`,
`final/decoded.png`, `final/combo_check.png`, `final/rise.png`,
`final/spelltext.png`, `horizon*/horizon.png`. Showcase and diagnostic renders —
expensive to regenerate, cheap to store.

---

## Deliberately not kept

| what | why | cost to rebuild |
|---|---|---|
| The corpus | Public on HuggingFace: `Smashlytics/soku-frames-{a,b,c}`, no token | ~3 min for 63.5 GB |
| `bank*.npz` | Derived from corpus + a specific checkpoint | ~10 min |
| `corpus/val.pt` | Derived | ~5 min |
| `hf_cache/` | Download cache | — |
| `*_smoke` checkpoints | Smoke tests, disposable by definition | — |
| `horizon_*/cache/` | Decoded HUD traces | minutes |
| The git stash on the box | Verified superseded — both changes (`LinearProbe`/`fit_ridge`, `assert_predictor_sane`) already committed. Saved as `final/stash_superseded.patch` anyway | — |

---

## Integrity check

```bash
python - <<'EOF'
import torch, glob, os
A = os.path.expanduser("~/K0NTR0L-2/artifacts")
for p in sorted(glob.glob(A + "/**/*.pt", recursive=True)):
    try:
        b = torch.load(p, map_location="cpu", weights_only=False)
        print(f"OK   {os.path.relpath(p, A):44} step={b.get('step')}")
    except Exception as e:
        print(f"FAIL {os.path.relpath(p, A):44} {type(e).__name__}")
EOF
```
