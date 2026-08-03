"""Data-scaling study: train the same model on 1h / 2h / 5h / 10h and compare.

    python -m scripts.scaling_run --exp /root/exp --steps 3000 --out runs/scaling.json

Design:

* **Fixed compute.** Every run takes the same number of optimiser steps at the
  same batch size. Otherwise "more data" and "more gradient updates" move
  together and the curve cannot separate them.
* **Fixed initialisation.** Same seed, so the four models start identical and
  every difference is attributable to what they were shown.
* **Fixed evaluation.** All four are scored on one cached, byte-identical set of
  held-out windows from captures none of them trained on.

Reading the numbers
-------------------
`val_pred` is MSE between predicted and true latents. Both sides are pinned to
zero mean and unit variance per dimension by the projectors' BatchNorm, so the
scale is anchored and comparable across models:

    MSE = 2 - 2 * corr(zhat, z)

which makes `corr = 1 - MSE/2` the interpretable number -- 0 is a model whose
predictions are unrelated to the truth, 1 is perfect. Note the predictor cannot
hedge toward the mean to lower its MSE, because its own output is BatchNormed to
unit variance too; a useless predictor scores MSE 2.0, not 1.0.

`inv_dyn_auc` asks whether a linear map can recover which buttons were pressed
from a pair of consecutive latents. A world model whose latent does not encode
what the action did cannot support planning, since planning is that question
inverted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.soku import ACTION_COLUMNS, build_soku_dataset
from sokubot.losses import prediction_loss
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import inverse_dynamics_probe
from sokubot.train import save_checkpoint, set_seed, train


# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: LeWorldModel, cache: dict, cfg: Config, batch: int = 64) -> dict:
    """Held-out prediction quality on the cached windows."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]

    tot, seen = 0.0, 0
    z_t, z_next, act = [], [], []
    for i in range(0, len(O), batch):
        obs = O[i : i + batch].to(device).float().div_(255.0)
        a = A[i : i + batch].to(device).float()
        out = model(obs, a)
        n = obs.shape[0]
        tot += float(prediction_loss(out.zhat, out.z).item()) * n
        seen += n
        # Latent pairs for the inverse-dynamics probe: (z_t, z_{t+1}) and the
        # action chunk applied between them, flattened over time.
        z = out.z
        z_t.append(z[:, :-1].reshape(-1, cfg.latent_dim).cpu().numpy())
        z_next.append(z[:, 1:].reshape(-1, cfg.latent_dim).cpu().numpy())
        act.append(a[:, :-1].amax(dim=2).reshape(-1, cfg.action_dim).cpu().numpy())

    if was_training:
        model.train()

    mse = tot / max(1, seen)
    probe = inverse_dynamics_probe(
        np.concatenate(z_t), np.concatenate(z_next),
        (np.concatenate(act) > 0.5).astype(np.float32),
        names=list(ACTION_COLUMNS),
    )
    return {
        "val_pred": mse,
        "val_corr": 1.0 - mse / 2.0,
        "inv_dyn_auc": probe.auc_mean,
        "inv_dyn_auc_per_button": probe.auc,
        "n_windows": seen,
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", default="/root/exp")
    ap.add_argument("--sizes", default="1,2,5,10")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/scaling.json")
    ap.add_argument("--ckpt-dir", default=None)
    args = ap.parse_args()

    exp = Path(args.exp)
    cache = torch.load(exp / "val.pt", map_location="cpu", weights_only=False)
    print(f"val cache: {cache['obs'].shape[0]} windows")

    sizes = [float(s) for s in args.sizes.split(",")]
    results = []

    for size in sizes:
        tag = f"train_{size:g}h"
        root = exp / tag
        cfg = Config.soku(
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            total_steps=args.steps,
            warmup_steps=min(200, args.steps // 10),
            seed=args.seed,
        )
        n_caps = sum(1 for _ in (root / "manifest.jsonl").read_text().splitlines())
        print(f"\n=== {tag} ({n_caps} captures) ===")

        # Identical initialisation for every run.
        set_seed(args.seed)
        model = LeWorldModel(cfg)

        ds = build_soku_dataset(cfg, [str(root)], shuffle_buffer=1024, seed=args.seed)

        # Evaluate through a callback inside one continuous run. Calling train()
        # in chunks would rebuild the optimiser and restart the LR schedule at
        # every eval point, turning the schedule into a sawtooth and throwing
        # away Adam's moments each time.
        curve = []
        t0 = time.time()

        def on_eval(m, step, hist):
            ev = evaluate(m, cache, cfg)
            ev["step"] = step
            ev["train_pred"] = float(np.mean([h["l_pred"] for h in hist[-50:]]))
            ev["latent_var"] = float(np.mean([h["latent_var"] for h in hist[-50:]]))
            curve.append(ev)
            print(f"  [{tag}] step {step:5d} | train_pred {ev['train_pred']:.4f} "
                  f"| val_pred {ev['val_pred']:.4f} | corr {ev['val_corr']:+.4f} "
                  f"| invdyn AUC {ev['inv_dyn_auc']:.4f} | var {ev['latent_var']:.3f}",
                  flush=True)

        model, hist = train(cfg, ds, steps=args.steps, model=model,
                            log_every=max(50, args.steps // 20), verbose=True,
                            callback=on_eval, callback_every=args.eval_every)
        if not curve:
            on_eval(model, args.steps, hist)

        final = curve[-1]
        results.append({
            "hours": size,
            "captures": n_caps,
            "steps": args.steps,
            "wall_seconds": time.time() - t0,
            "curve": curve,
            **{k: final[k] for k in
               ("val_pred", "val_corr", "inv_dyn_auc", "train_pred", "latent_var")},
        })
        if args.ckpt_dir:
            save_checkpoint(model, cfg, Path(args.ckpt_dir) / tag, args.steps)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print(f"{'hours':>7} {'caps':>5} {'train_pred':>11} {'val_pred':>9} "
          f"{'val_corr':>9} {'invdyn AUC':>11} {'gap':>7}")
    for r in results:
        gap = r["val_pred"] - r["train_pred"]
        print(f"{r['hours']:7.0f} {r['captures']:5d} {r['train_pred']:11.4f} "
              f"{r['val_pred']:9.4f} {r['val_corr']:+9.4f} {r['inv_dyn_auc']:11.4f} "
              f"{gap:+7.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
