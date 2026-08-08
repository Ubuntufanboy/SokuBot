"""Teach the predictor to consume its own output, on cached latents.

    python -m scripts.finetune_predictor --ckpt ~/art/best_bnfix.pt \
        --bank ~/bank.npz --out ~/ckpt_roll --steps 20000 --plan 16

WHAT THIS ATTACKS
-----------------
`docs/HANDOFF.md` section 9 names the world model's trustworthy horizon as the
binding constraint on everything: the action->outcome correlation a policy
gradient consumes falls from r=0.55 at one step to 0.32 at four to 0.09 at
sixteen, and four GRPO runs plateaued in a narrow band regardless of the
optimiser. `losses/prediction.py` explains the mechanism -- training is
teacher-forced, so the predictor has never once been asked to consume its own
output, while at play time it consumes nothing else.

WHY THE ENCODER IS FROZEN, AND WHAT THAT BUYS
---------------------------------------------
The predictor consumes *latents*, not pixels. With the encoder frozen there is
no video decode and no encoder forward in the loop, so this trains from a cached
bank at a speed unrelated to the 63 GB of mp4 the bank came from -- which is what
makes it affordable on a GTX 1650 rather than a rented 5090.

It also keeps the latent space fixed, so:

  * the **bank** stays valid (it is encoder latents; `build_bank` stamps it with
    the encoder's fingerprint and would otherwise have to be rebuilt),
  * the **encoder-side probe** (`reward_probe_encoder.npz`) stays valid,
  * **SIGReg is unnecessary** -- the targets are fixed encoder outputs, so there
    is no shared representation for predictor and target to collapse into. The
    anti-collapse machinery in `train_full.py` exists because that loss trains
    both sides; this one does not.

What does *not* stay valid is the **calibrated reward probe**
(`horizon_bnfix/reward_probe.npz`, alpha 100), because that one is fit on
*predictor outputs* and this changes them. Refit it with
`scripts/horizon_ablation.py --ckpt <the new checkpoint>`; `train_grpo` refuses a
probe whose fingerprint does not match its world model, so forgetting is loud
rather than silent.

THE BATCHNORM TRAP, WHICH THIS RUN IS EXACTLY SHAPED TO FALL INTO
------------------------------------------------------------------
`docs/BUGS.md` section 1: the counterfactual fine-tune pushed deliberately
off-distribution activations through the projector's BatchNorm, its running
statistics absorbed a distribution the model never sees at eval, and the saved
checkpoint had one-step skill -6.15 while its blob claimed +0.82. Nothing threw.

An autoregressive rollout is the same hazard by construction: steps 2..P feed the
predictor its own outputs, which is precisely a distribution the encoder never
produces. So BatchNorm is held in **eval mode for the whole run** -- running
statistics frozen and used for normalisation, never updated. Training and rollout
are then numerically identical, the saved weights need no recalibration, and the
failure mode simply cannot occur. The rest of the model still trains normally;
only the BatchNorm statistics are pinned.

Belt and braces: the checkpoint is gated on `assert_predictor_sane` before it is
written, so a run that destroys the predictor cannot produce an artifact.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.losses.prediction import rollout_loss
from sokubot.model.augmented import (HUD_CHANNELS, AugmentedPredictor,
                                     AugmentedWorldModel)
from sokubot.model.world_model import LeWorldModel
from scripts.eval_ckpt import assert_predictor_sane, predictor_skill


def freeze_batchnorm(model: torch.nn.Module) -> int:
    """Put every BatchNorm in eval mode and stop it tracking running stats.

    Called after every `model.train()`, because `train()` recurses and would
    otherwise switch them back on. Returns how many were pinned, so a model that
    silently stops having BatchNorms does not pass this by doing nothing.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.track_running_stats = False
            n += 1
    return n


@torch.no_grad()
def augmented_skill(model, Z, A, Hd, ep, cfg, n: int = 8192, seed: int = 0) -> float:
    """`eval_ckpt.predictor_skill` for the augmented predictor.

    Same quantity and same formula -- one-step `1 - MSE_model / MSE_identity` on
    the *latent* channels -- but the augmented predictor needs the HUD half of
    the state to run at all, so it cannot be fed through the base helper.

    This exists rather than skipping the gate under `--hud`. `assert_predictor_sane`
    is the line that would have caught `ckpt_cf/best.pt` shipping at skill -6.15
    while its own blob claimed +0.82 (`docs/BUGS.md` section 1), and an arm that
    quietly opts out of the check is the arm that gets to write a broken artifact.
    """
    dev = next(model.parameters()).device
    H = cfg.history
    same = np.zeros(len(ep), dtype=bool)
    same[: len(ep) - (H + 1)] = ep[: len(ep) - (H + 1)] == ep[H + 1 :]
    ok = np.flatnonzero(same)
    ok = ok[ok >= H - 1]
    idx = torch.from_numpy(np.random.default_rng(seed).choice(
        ok, size=min(n, len(ok)), replace=False)).to(dev)
    off = torch.arange(H, device=dev) - (H - 1)
    se_m = se_i = 0.0
    for s in range(0, len(idx), 4096):
        base = idx[s : s + 4096]
        zw = Z[base[:, None] + off[None, :]].float()
        hw = Hd[base[:, None] + off[None, :]].float()
        acts = A[base[:, None] + off[None, :]].float()
        zhat, _ = model.predictor(zw, hw, model.action_encoder(acts))
        zt = Z[base + 1].float()
        se_m += float(((zhat[:, -1] - zt) ** 2).mean(1).sum())
        se_i += float(((zw[:, -1] - zt) ** 2).mean(1).sum())
    return 1.0 - se_m / se_i


def valid_starts(ep: np.ndarray, history: int, plan: int) -> np.ndarray:
    """Indices whose context and full rollout stay inside one replay."""
    ok = []
    edges = np.flatnonzero(np.diff(ep)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ep)]):
        a, b = lo + history - 1, hi - plan - 1
        if b > a:
            ok.append(np.arange(a, b))
    if not ok:
        raise SystemExit(
            f"no start states survive history {history} + plan {plan}; the bank's "
            f"replays are too short")
    return np.concatenate(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True,
                    help="npz from scripts.train_grpo.build_bank: z, a, ep, fingerprint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--plan", type=int, default=16,
                    help="rollout length in decision steps (15 Hz), so 16 = 1.07 s")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="which steps of the rollout are scored. 1 must be in "
                         "here: it is the horizon everything else is anchored to "
                         "and the one the live loop actually uses.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of *replays* held out; splitting by row would "
                         "put neighbouring near-duplicate frames on both sides")
    ap.add_argument("--hud", action="store_true",
                    help="ARM B: carry data/hud.py's eight channels as explicit "
                         "world-model state (sokubot/model/augmented.py) instead "
                         "of hoping a probe recovers them from the latent. "
                         "Requires a bank built by scripts.build_hud_bank. The "
                         "augmented predictor is zero-initialised on the new "
                         "weights, so it starts numerically identical to arm A "
                         "and any difference is learned rather than an artefact "
                         "of a different initialisation.")
    ap.add_argument("--hud-coef", type=float, default=1.0,
                    help="weight on the HUD rollout loss. Both terms are already "
                         "normalised by their own copy-forward baseline, so 1.0 "
                         "means 'these matter equally', not 'these have equal "
                         "magnitude'.")
    ap.add_argument("--train-action-encoder", action="store_true",
                    help="also train the 0.12M action encoder. Off by default: it "
                         "defines the conditioning space the counterfactual "
                         "fine-tune made action-aware, and that property is the "
                         "one thing GRPO depends on most.")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--min-skill", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if 1 not in a.horizons:
        raise SystemExit("--horizons must include 1; see the flag's help")
    if max(a.horizons) > a.plan:
        raise SystemExit(f"--horizons up to {max(a.horizons)} exceed --plan {a.plan}")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    model = LeWorldModel(cfg).to(a.device)
    model.load_state_dict(blob["model"])

    d = np.load(a.bank)
    Z, A, ep = d["z"], d["a"], d["ep"]
    bank_fp = str(d["fingerprint"]) if "fingerprint" in d.files else None
    Hd = None
    if a.hud:
        if "hud" not in d.files:
            raise SystemExit(
                f"--hud needs a bank carrying HUD state, and {a.bank} has none. "
                f"Build one with scripts.build_hud_bank --ckpt {a.ckpt}.")
        Hd = d["hud"]
        chans = tuple(str(x) for x in d["hud_channels"])
        if chans != HUD_CHANNELS:
            raise SystemExit(
                f"bank HUD channels {chans} do not match the model's "
                f"{HUD_CHANNELS}. These are matched positionally, so a mismatch "
                f"would train the predictor to forecast health in the spirit "
                f"slot without anything failing.")

    # The bank is encoder latents. Encoding them with a different encoder than
    # the one loaded here would train the predictor to map between two unrelated
    # spaces -- which trains happily and means nothing. `build_bank` stamps the
    # bank with the *whole model's* fingerprint, so this compares against that.
    from scripts.train_grpo import encoder_fingerprint, model_fingerprint
    fp, enc_fp = model_fingerprint(model), encoder_fingerprint(model)
    bank_enc = str(d["encoder_fingerprint"]) if "encoder_fingerprint" in d.files else None
    if bank_enc is not None:
        if bank_enc != enc_fp:
            raise SystemExit(
                f"bank was encoded by encoder {bank_enc} but --ckpt's encoder is "
                f"{enc_fp}. Latents mean nothing except relative to the encoder "
                f"that produced them; rebuild the bank.")
    elif bank_fp is None:
        print(f"WARNING: {a.bank} carries no fingerprint and cannot be checked "
              f"against this checkpoint ({fp}).", flush=True)
    elif bank_fp != fp:
        # An older bank stamped with the whole-model fingerprint. A predictor
        # fine-tune changes that legitimately, so fall back to comparing the
        # encoder, which is the part the bank actually depends on.
        base = torch.load(blob.get("base_ckpt", a.ckpt), map_location="cpu",
                          weights_only=False) if blob.get("base_ckpt") else None
        ok = False
        if base is not None:
            probe_m = LeWorldModel(base["cfg"])
            probe_m.load_state_dict(base["model"])
            ok = (model_fingerprint(probe_m) == bank_fp
                  and encoder_fingerprint(probe_m) == enc_fp)
        if not ok:
            raise SystemExit(
                f"bank was encoded by {bank_fp} but --ckpt is {fp}, and the "
                f"encoder could not be shown to match. Rebuild the bank.")
        print(f"note: bank is stamped with the whole-model fingerprint {bank_fp}, "
              f"which a predictor fine-tune changes. Its encoder matches this "
              f"checkpoint's ({enc_fp}), so the latents are valid.", flush=True)

    n_ep = int(ep.max()) + 1
    val_ep = rng.choice(n_ep, size=max(1, int(n_ep * a.val_frac)), replace=False)
    is_val = np.isin(ep, val_ep)
    starts = valid_starts(ep, cfg.history, a.plan)
    tr_starts = starts[~is_val[starts]]
    va_starts = starts[is_val[starts]]
    print(f"bank {len(Z)} latents, {n_ep} replays | "
          f"{len(tr_starts)} train / {len(va_starts)} val starts "
          f"({len(val_ep)} held-out replays)", flush=True)

    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)
    Ht = torch.from_numpy(Hd).to(a.device) if Hd is not None else None

    if a.hud:
        # Wrap the trained predictor. The new input columns are zero, so this is
        # numerically the same model until it learns otherwise -- which is what
        # makes the A/B against arm A attributable.
        prior = torch.from_numpy(Hd.astype(np.float32).mean(0))
        model.predictor = AugmentedPredictor.from_pretrained(
            model.predictor, cfg, hud_prior=prior).to(a.device)
        model = AugmentedWorldModel(model, model.predictor).to(a.device)
        print("ARM B: HUD-augmented state, "
              f"{cfg.latent_dim} latent + {len(HUD_CHANNELS)} hud channels | "
              "prior " + " ".join(f"{c} {float(p):.3f}"
                                  for c, p in zip(HUD_CHANNELS, prior)), flush=True)

    # Only the dynamics train. The encoder defines the latent space that the
    # bank, the probe and the policy are all expressed in.
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = list(model.predictor.parameters())
    if a.train_action_encoder:
        train_params += list(model.action_encoder.parameters())
    for p in train_params:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in train_params)
    print(f"training {n_train/1e6:.2f}M params "
          f"({'predictor + action encoder' if a.train_action_encoder else 'predictor only'})",
          flush=True)

    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.steps), eta_min=a.lr * 0.05)

    H, P = cfg.history, a.plan
    off = torch.arange(H, device=a.device) - (H - 1)
    plan_off = torch.arange(P, device=a.device)

    def batch_from(idx: torch.Tensor):
        """-> (z_ctx, a_hist, a_plan, z_true) plus (hud_ctx, hud_true) under --hud."""
        z_ctx = Zt[idx[:, None] + off[None, :]].float()
        a_hist = At[idx[:, None] + off[None, :-1]].float()
        a_plan = At[idx[:, None] + plan_off[None, :]].float()
        z_true = Zt[idx[:, None] + 1 + plan_off[None, :]].float()
        if Ht is None:
            return z_ctx, a_hist, a_plan, z_true, None, None
        h_ctx = Ht[idx[:, None] + off[None, :]].float()
        h_true = Ht[idx[:, None] + 1 + plan_off[None, :]].float()
        return z_ctx, a_hist, a_plan, z_true, h_ctx, h_true

    def step_loss(batch):
        """One forward pass and its loss, identical in shape for both arms.

        The HUD term is normalised by its own copy-forward baseline, exactly like
        the latent term, so the two are commensurable without hand-weighting: a
        value of 1.0 in either means "no better than assuming nothing changed".
        Health barely moves between two 15 Hz frames, so copy-forward is a strong
        baseline here and beating it is a real claim.
        """
        z_ctx, a_hist, a_plan, z_true, h_ctx, h_true = batch
        if h_ctx is None:
            zhat = model.rollout(z_ctx, a_plan, a_hist)
            loss, rep = rollout_loss(zhat, z_true, z_ctx[:, -1], a.horizons)
            return loss, rep, {}
        zhat, hhat = model.rollout(z_ctx, h_ctx, a_plan, a_hist)
        loss, rep = rollout_loss(zhat, z_true, z_ctx[:, -1], a.horizons)
        hloss, hrep = rollout_loss(hhat, h_true, h_ctx[:, -1], a.horizons)
        return loss + a.hud_coef * hloss, rep, hrep

    @torch.no_grad()
    def validate() -> tuple[dict, dict]:
        model.eval()
        rows, hrows, n = {}, {}, 0
        for s in range(0, min(len(va_starts), 4096), a.batch):
            idx = torch.from_numpy(va_starts[s : s + a.batch]).to(a.device)
            if len(idx) < 8:
                break
            _, rep, hrep = step_loss(batch_from(idx))
            for h, v in rep.items():
                rows[h] = rows.get(h, 0.0) + v
            for h, v in hrep.items():
                hrows[h] = hrows.get(h, 0.0) + v
            n += 1
        m = max(n, 1)
        return ({h: v / m for h, v in rows.items()},
                {h: v / m for h, v in hrows.items()})

    # Skill and validation *before* any update, so every later number has a
    # same-instrument reference rather than one quoted from a document.
    model.eval()
    skill = (lambda: augmented_skill(model, Zt, At, Ht, ep, cfg)) if a.hud else \
            (lambda: predictor_skill(model, Zt, At, ep, cfg))
    sk0 = skill()
    va0, hva0 = validate()
    print(f"before: one-step skill {sk0:+.4f} | val rel-err " +
          " ".join(f"h{h} {v:.4f}" for h, v in sorted(va0.items())), flush=True)
    if hva0:
        print("        hud rel-err " +
              " ".join(f"h{h} {v:.4f}" for h, v in sorted(hva0.items())), flush=True)

    n_bn = freeze_batchnorm(model)
    print(f"pinned {n_bn} BatchNorm modules to eval mode for the whole run "
          f"(docs/BUGS.md section 1)", flush=True)
    if n_bn == 0:
        raise SystemExit(
            "found no BatchNorm to pin -- the projector is supposed to end in "
            "one, so either the model changed or the wrong object was passed")

    hist, t0 = [], time.time()
    best = float("inf")
    for step in range(1, a.steps + 1):
        model.train()
        freeze_batchnorm(model)          # train() re-enables them; undo that
        idx = torch.from_numpy(rng.choice(tr_starts, size=a.batch)).to(a.device)
        loss, rep, hrep = step_loss(batch_from(idx))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(train_params, a.grad_clip)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()

        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "grad_norm": float(gn), "lr": float(sched.get_last_lr()[0]),
                   **{f"h{h}": v for h, v in rep.items()},
                   "elapsed_h": (time.time() - t0) / 3600}
            if step % a.eval_every == 0 or step == 1:
                va, hva = validate()
                rec.update({f"val_h{h}": v for h, v in va.items()})
                rec.update({f"val_hud_h{h}": v for h, v in hva.items()})
                mean_val = float(np.mean(list(va.values())))
                rec["val_mean"] = mean_val
                if mean_val < best:
                    best = mean_val
                    torch.save({"model": model.state_dict(), "cfg": cfg,
                                "step": step, "val_mean": mean_val,
                                "val": va, "base_ckpt": str(a.ckpt),
                                "horizons": a.horizons, "plan": P},
                               a.out / "predictor_best.pt")
                print(f"  [val] step {step:6d} | mean {mean_val:.4f} | " +
                      " ".join(f"h{h} {v:.4f}" for h, v in sorted(va.items())) +
                      ("  | hud " + " ".join(f"h{h} {v:.4f}"
                                             for h, v in sorted(hva.items()))
                       if hva else ""), flush=True)
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            print(f"step {step:6d} | loss {rec['loss']:.4f} | " +
                  " ".join(f"h{h} {rep[h]:.3f}" for h in sorted(rep)) +
                  f" | gn {rec['grad_norm']:.2f} | {rec['elapsed_h']:.2f}h",
                  flush=True)

    # ---- gate the artifact, then write it ----
    model.eval()
    # Same gate for both arms; only the way skill is computed differs.
    sk = skill()
    if sk < a.min_skill:
        raise SystemExit(
            f"fine-tuned predictor has one-step skill {sk:+.4f}, at or below the "
            f"floor {a.min_skill:+.2f}: its predictor is no better than copying "
            f"the previous latent, so any rollout through it is noise. Refusing "
            f"to write the checkpoint (docs/BUGS.md section 1).")
    va, hva = validate()
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": a.steps,
                "one_step_skill": sk, "val": va, "val_hud": hva,
                "base_ckpt": str(a.ckpt), "hud": bool(a.hud),
                "hud_channels": list(HUD_CHANNELS) if a.hud else None,
                "horizons": a.horizons, "plan": P}, a.out / "predictor_final.pt")
    print(f"\nafter: one-step skill {sk:+.4f} (was {sk0:+.4f})")
    for h in sorted(va):
        print(f"  h{h:<3d} rel-err {va[h]:.4f}  (was {va0[h]:.4f})")
    print(f"\n-> {a.out}")
    print("Next: refit the calibrated reward probe against these weights --\n"
          f"  python -m scripts.horizon_ablation --ckpt {a.out}/predictor_best.pt\n"
          "then re-run scripts.action_effect_test. The gate is whether the "
          "action->outcome correlation at h=4/16 beats +0.32/+0.09.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
