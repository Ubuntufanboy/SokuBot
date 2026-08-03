"""Score trained checkpoints against the trivial predictors.

    python -m scripts.baselines --ckpt-dir /root/ckpt --val /root/exp/val.pt

WHY THIS EXISTS
---------------
A latent world model on video can score a very high next-step correlation
without having learned any dynamics at all, because at a 15 Hz decision rate
consecutive frames are nearly the same image. The predictor only has to pass its
input through. `val_pred` alone cannot tell that apart from a model that has
learned how the game moves, so every reported number needs the trivial
predictors next to it:

  mean      zhat = 0. The projector's BatchNorm pins the target to zero mean and
            unit variance, so this scores MSE = 1.0 by construction, for any
            model. It is the "knows nothing" line.

  identity  zhat = z_t, i.e. "next frame looks like this frame". This is the
            line that matters. Beating `mean` is trivial on video; beating
            `identity` is the claim that the model learned dynamics.

  model     the trained predictor.

The honest summary statistic is the fraction of the identity baseline's error
that the model removes:

    skill = 1 - MSE_model / MSE_identity

which is 0 for a pure pass-through and 1 for perfect prediction. A negative value
means the model is worse than copying the previous frame.

Encoders differ between runs, so `identity` is recomputed per checkpoint in that
checkpoint's own latent space. Comparing one model's MSE against another model's
identity baseline would be meaningless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel


@torch.no_grad()
def score(model: LeWorldModel, cache: dict, cfg: Config, batch: int = 64) -> dict:
    model.eval()
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]

    tot_model = tot_ident = tot_mean = 0.0
    tot_persist = 0.0
    seen = 0
    for i in range(0, len(O), batch):
        obs = O[i : i + batch].to(device).float().div_(255.0)
        a = A[i : i + batch].to(device).float()
        out = model(obs, a)
        z, zhat = out.z, out.zhat
        tgt = z[:, 1:]                       # true next latents
        n = obs.shape[0]

        tot_model += float(F.mse_loss(zhat[:, :-1], tgt).item()) * n
        tot_ident += float(F.mse_loss(z[:, :-1], tgt).item()) * n
        tot_mean += float(F.mse_loss(torch.zeros_like(tgt), tgt).item()) * n
        # Linear extrapolation z_t + (z_t - z_{t-1}): a constant-velocity
        # predictor, the next-hardest thing to beat after copying.
        if z.shape[1] >= 3:
            vel = z[:, 1:-1] + (z[:, 1:-1] - z[:, :-2])
            tot_persist += float(F.mse_loss(vel, z[:, 2:]).item()) * n
        seen += n

    d = {
        "model": tot_model / seen,
        "identity": tot_ident / seen,
        "mean": tot_mean / seen,
        "n": seen,
    }
    if tot_persist:
        d["const_velocity"] = tot_persist / seen
    d["skill_vs_identity"] = 1.0 - d["model"] / d["identity"] if d["identity"] > 0 else float("nan")
    d["skill_vs_mean"] = 1.0 - d["model"] / d["mean"] if d["mean"] > 0 else float("nan")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default="/root/ckpt")
    ap.add_argument("--val", default="/root/exp/val.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/baselines.json")
    args = ap.parse_args()

    cache = torch.load(args.val, map_location="cpu", weights_only=False)
    ckpts = sorted(Path(args.ckpt_dir).glob("train_*h/sokubot.pt"),
                   key=lambda p: float(p.parent.name.split("_")[1].rstrip("h")))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.ckpt_dir}")

    rows = []
    print(f"{'run':>10} {'model':>9} {'identity':>9} {'const-vel':>10} {'mean':>7} "
          f"{'skill vs id':>12}")
    for p in ckpts:
        blob = torch.load(p, map_location=args.device, weights_only=False)
        cfg: Config = blob["cfg"]
        cfg.device = args.device
        m = LeWorldModel(cfg).to(args.device)
        m.load_state_dict(blob["model"])
        s = score(m, cache, cfg)
        s["run"] = p.parent.name
        s["hours"] = float(p.parent.name.split("_")[1].rstrip("h"))
        rows.append(s)
        print(f"{s['run']:>10} {s['model']:9.4f} {s['identity']:9.4f} "
              f"{s.get('const_velocity', float('nan')):10.4f} {s['mean']:7.4f} "
              f"{s['skill_vs_identity']:+12.4f}")
        del m
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    print("\nskill vs identity is the number that matters: 0 means the predictor")
    print("only copies the current frame, 1 means perfect. `mean` should read")
    print("~1.0 everywhere -- BatchNorm fixes it there, so it is a sanity check")
    print("on the evaluation rather than a competitive baseline.")


if __name__ == "__main__":
    main()
