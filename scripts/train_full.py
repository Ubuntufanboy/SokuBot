"""The full-corpus training run, with periodic held-out evaluation.

    python -m scripts.train_full --corpus /root/corpus --steps 320000

`sokubot.train`'s CLI trains and checkpoints but never looks at held-out data.
Over six hours that is too long to be flying blind: a collapse, or a loader that
silently starts repeating one capture, looks identical to healthy training from
the loss curve alone.

Evaluation runs through `train()`'s callback, inside one continuous run --
calling train() in chunks would rebuild the optimiser and restart the learning
rate schedule at every eval point.

Metrics per eval, all on the same cached val windows the model never trains on:

  val_pred   MSE against the true next latent
  identity   MSE of "next latent looks like this latent" -- the number that
             makes val_pred meaningful, because at 15 Hz a pass-through
             predictor already scores well
  skill      1 - val_pred/identity; 0 is pass-through, 1 is perfect
  inv_dyn    can a linear probe recover the buttons from a latent pair
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.soku import ACTION_COLUMNS, build_soku_dataset
from sokubot.losses import prediction_loss
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import inverse_dynamics_probe
from sokubot.train import save_checkpoint, set_seed, train


@torch.no_grad()
def evaluate(model: LeWorldModel, cache: dict, cfg: Config, batch: int = 64) -> dict:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]

    tot = ident = 0.0
    seen = 0
    zt, zn, acts = [], [], []
    for i in range(0, len(O), batch):
        obs = O[i : i + batch].to(device, non_blocking=True)
        a = A[i : i + batch].to(device).float()
        out = model(obs, a)
        z, tgt = out.z, out.z[:, 1:]
        n = obs.shape[0]
        tot += float(prediction_loss(out.zhat, z).item()) * n
        ident += float(F.mse_loss(z[:, :-1], tgt).item()) * n
        seen += n
        zt.append(z[:, :-1].reshape(-1, cfg.latent_dim).cpu().numpy())
        zn.append(tgt.reshape(-1, cfg.latent_dim).cpu().numpy())
        acts.append(a[:, :-1].amax(dim=2).reshape(-1, cfg.action_dim).cpu().numpy())

    if was_training:
        model.train()
    vp, idm = tot / seen, ident / seen
    probe = inverse_dynamics_probe(
        np.concatenate(zt), np.concatenate(zn),
        (np.concatenate(acts) > 0.5).astype(np.float32), names=list(ACTION_COLUMNS))
    return {"val_pred": vp, "identity": idm,
            "skill": 1.0 - vp / idm if idm > 0 else float("nan"),
            "inv_dyn_auc": probe.auc_mean, "n": seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--steps", type=int, default=320_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/root/ckpt"))
    ap.add_argument("--log", type=Path, default=Path("/root/train_log.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="peak LR. The 5e-4 tuned on 16k-step runs diverged at "
                         "75k of a 320k-step schedule -- see module docstring.")
    ap.add_argument("--warmup", type=int, default=5_000)
    args = ap.parse_args()

    cfg = Config.soku(device=args.device, batch_size=args.batch_size,
                      num_workers=args.num_workers, total_steps=args.steps,
                      warmup_steps=args.warmup, lr=args.lr, seed=args.seed)
    cache = torch.load(args.corpus / "val.pt", map_location="cpu", weights_only=False)
    print(f"val cache: {cache['obs'].shape[0]} windows", flush=True)

    set_seed(args.seed)
    model = LeWorldModel(cfg)
    rep = model.param_report()
    print("params: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in rep.items()), flush=True)

    ds = build_soku_dataset(cfg, [str(args.corpus / "train")],
                            shuffle_buffer=4096, seed=args.seed)

    hours = sum(json.loads(l)["frames"] for l in
                (args.corpus / "train" / "manifest.jsonl").read_text().splitlines()) / 60 / 3600
    windows = hours * 3600 * 15
    print(f"train: {hours:.2f} h ~= {windows/1e6:.2f}M windows | "
          f"{args.steps} steps x {cfg.batch_size} = "
          f"{args.steps*cfg.batch_size/windows:.2f} epochs", flush=True)

    curve, t0 = [], time.time()

    def on_eval(m, step, hist):
        # Evaluate a copy with BatchNorm statistics re-estimated against the
        # current weights. With a long cosine schedule the learning rate stays
        # near maximum for tens of thousands of steps, and the running stats lag
        # far enough behind that held-out loss becomes meaningless -- this run
        # reported val 1.1431 at step 30k while training loss was 0.058. The
        # copy keeps training's own running stats untouched.
        from scripts.eval_ckpt import recalibrate_bn
        probe_model = copy.deepcopy(m)
        recalibrate_bn(probe_model, cache)
        ev = evaluate(probe_model, cache, cfg)
        del probe_model
        ev["step"] = step
        tail = hist[-20:] if hist else []
        ev["train_pred"] = float(np.mean([h["l_pred"] for h in tail])) if tail else float("nan")
        ev["latent_var"] = float(np.mean([h["latent_var"] for h in tail])) if tail else float("nan")
        el = time.time() - t0
        ev["elapsed_h"] = el / 3600
        ev["eta_h"] = (el / max(step, 1)) * (args.steps - step) / 3600
        # Divergence guard. BatchNorm pins each latent dimension to unit
        # variance, so two *uncorrelated* latents can differ by at most ~2.0 in
        # mean-squared terms. identity above that is structurally impossible and
        # means the normalisation has broken down -- pre-BN variance has fallen
        # to the order of eps, so BatchNorm has stopped normalising and started
        # amplifying. The first run hit identity 7.68 at step 85k.
        #
        # This has to be checked explicitly because `skill` does not catch it:
        # skill is a ratio, and an exploding latent inflates numerator and
        # denominator together. During that divergence skill *rose* from +0.35
        # to +0.69 while absolute prediction error got 6x worse.
        ev["diverged"] = bool(ev["identity"] > 2.0 or ev["latent_var"] < 0.85)
        if ev["diverged"]:
            print(f"  *** DIVERGENCE at step {step}: identity {ev['identity']:.3f} "
                  f"(max ~2.0), latent_var {ev['latent_var']:.3f} (want ~1.0) ***",
                  flush=True)
        curve.append(ev)
        args.log.write_text(json.dumps(curve, indent=2))

        # Keep the best model, not just the most recent. The first run
        # overwrote one file every 5k steps, so when it diverged at 75k the
        # healthy step-65k model was already gone.
        healthy = [c for c in curve if not c["diverged"]]
        if healthy and ev is healthy[-1] and ev["skill"] >= max(c["skill"] for c in healthy):
            torch.save({"model": m.state_dict(), "cfg": cfg, "step": step,
                        "eval": ev}, Path(args.ckpt_dir) / "best.pt")
            print(f"  saved best.pt (skill {ev['skill']:+.4f} at step {step})", flush=True)
        print(f"  [eval] step {step:6d} | train {ev['train_pred']:.4f} "
              f"| val {ev['val_pred']:.4f} | identity {ev['identity']:.4f} "
              f"| skill {ev['skill']:+.4f} | AUC {ev['inv_dyn_auc']:.4f} "
              f"| var {ev['latent_var']:.3f} | {ev['elapsed_h']:.2f}h elapsed, "
              f"{ev['eta_h']:.2f}h left", flush=True)

    train(cfg, ds, steps=args.steps, model=model, verbose=True, log_every=500,
          ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
          callback=on_eval, callback_every=args.eval_every)

    save_checkpoint(model, cfg, args.ckpt_dir, args.steps)
    if curve:
        b = max(curve, key=lambda r: r["skill"])
        print(f"\nbest skill {b['skill']:+.4f} at step {b['step']} "
              f"(val {b['val_pred']:.4f}, AUC {b['inv_dyn_auc']:.4f})")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {args.ckpt_dir}")


if __name__ == "__main__":
    main()
