"""Definitive per-checkpoint evaluation, with uncertainty.

    python -m scripts.final_eval --ckpt-dir /root/ckpt --val /root/exp/val.pt

Produces, for every trained checkpoint, on one shared held-out cache:

  * MSE for the model and for the trivial predictors (copy-last-frame,
    constant-velocity, predict-the-mean)
  * skill = 1 - MSE_model / MSE_identity
  * inverse-dynamics AUC with a bootstrap 95% confidence interval

The confidence interval is the point of this script. The differences this study
is trying to resolve are around 0.02 AUC, and the probe's test split holds ~950
windows of which only ~10% have any given button pressed -- so the sampling
error is the same order as the effect. Reporting the difference without the
interval would be reporting noise as a finding.

Bootstrap resamples the *evaluation* rows (the probe is refit on a fixed train
split each time), which measures how much the estimate moves with the particular
held-out windows drawn. It does not capture variation from retraining the world
model itself; that would need several seeds per data size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.soku import ACTION_COLUMNS
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import _auc


@torch.no_grad()
def collect(model: LeWorldModel, cache: dict, cfg: Config, batch: int = 64):
    model.eval()
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]

    sums = {"model": 0.0, "identity": 0.0, "mean": 0.0, "const_velocity": 0.0}
    seen = 0
    zt, zn, acts = [], [], []
    for i in range(0, len(O), batch):
        obs = O[i : i + batch].to(device).float().div_(255.0)
        a = A[i : i + batch].to(device).float()
        out = model(obs, a)
        z, zhat = out.z, out.zhat
        tgt, n = z[:, 1:], obs.shape[0]

        sums["model"] += float(F.mse_loss(zhat[:, :-1], tgt).item()) * n
        sums["identity"] += float(F.mse_loss(z[:, :-1], tgt).item()) * n
        sums["mean"] += float(F.mse_loss(torch.zeros_like(tgt), tgt).item()) * n
        if z.shape[1] >= 3:
            vel = z[:, 1:-1] + (z[:, 1:-1] - z[:, :-2])
            sums["const_velocity"] += float(F.mse_loss(vel, z[:, 2:]).item()) * n
        seen += n

        zt.append(z[:, :-1].reshape(-1, cfg.latent_dim).cpu().numpy())
        zn.append(z[:, 1:].reshape(-1, cfg.latent_dim).cpu().numpy())
        acts.append(a[:, :-1].amax(dim=2).reshape(-1, cfg.action_dim).cpu().numpy())

    mses = {k: v / seen for k, v in sums.items()}
    return mses, seen, (np.concatenate(zt), np.concatenate(zn),
                        (np.concatenate(acts) > 0.5).astype(np.float32))


def probe_with_ci(zt, zn, acts, n_boot=200, train_frac=0.8, alpha=1e-2, seed=0):
    """Fit the inverse-dynamics probe once; bootstrap the held-out AUC."""
    x = np.concatenate([zt, zn, zn - zt], axis=1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_tr = int(len(x) * train_frac)
    tr, te = perm[:n_tr], perm[n_tr:]

    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    Xtr = np.concatenate([(x[tr] - mu) / sd, np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([(x[te] - mu) / sd, np.ones((len(te), 1))], axis=1)
    reg = alpha * np.eye(Xtr.shape[1]); reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ acts[tr])
    pred, Yte = Xte @ W, acts[te]

    def mean_auc(idx):
        vals = [_auc(pred[idx, j], Yte[idx, j]) for j in range(Yte.shape[1])]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    point = mean_auc(np.arange(len(te)))
    boots = np.array([mean_auc(rng.integers(0, len(te), len(te))) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    per = {n: _auc(pred[:, j], Yte[:, j]) for j, n in enumerate(ACTION_COLUMNS)}
    return {"auc": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_test": len(te), "per_button": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default="/root/ckpt")
    ap.add_argument("--val", default="/root/exp/val.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="runs/final_eval.json")
    args = ap.parse_args()

    cache = torch.load(args.val, map_location="cpu", weights_only=False)
    ckpts = sorted(Path(args.ckpt_dir).glob("train_*h/sokubot.pt"),
                   key=lambda p: float(p.parent.name.split("_")[1].rstrip("h")))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.ckpt_dir}")

    rows = []
    for p in ckpts:
        blob = torch.load(p, map_location=args.device, weights_only=False)
        cfg: Config = blob["cfg"]; cfg.device = args.device
        m = LeWorldModel(cfg).to(args.device)
        m.load_state_dict(blob["model"])
        mses, n, (zt, zn, acts) = collect(m, cache, cfg)
        pr = probe_with_ci(zt, zn, acts, n_boot=args.boot)
        rows.append({
            "run": p.parent.name,
            "hours": float(p.parent.name.split("_")[1].rstrip("h")),
            **mses,
            "skill": 1.0 - mses["model"] / mses["identity"],
            "inv_dyn_auc": pr["auc"], "auc_ci": [pr["ci_lo"], pr["ci_hi"]],
            "auc_per_button": pr["per_button"],
            "n_windows": n, "n_probe_test": pr["n_test"],
        })
        del m; torch.cuda.empty_cache()

    print(f"{'hours':>6} {'model':>8} {'identity':>9} {'const-v':>8} {'mean':>7} "
          f"{'skill':>8} {'invdyn AUC (95% CI)':>24}")
    for r in rows:
        print(f"{r['hours']:6.0f} {r['model']:8.4f} {r['identity']:9.4f} "
              f"{r['const_velocity']:8.4f} {r['mean']:7.4f} {r['skill']:+8.4f} "
              f"{r['inv_dyn_auc']:9.4f} [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}]")

    # Is the smallest-to-largest AUC change bigger than the noise?
    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        sep = b["auc_ci"][0] > a["auc_ci"][1]
        print(f"\n{a['hours']:.0f}h vs {b['hours']:.0f}h inverse-dynamics AUC: "
              f"{a['inv_dyn_auc']:.4f} -> {b['inv_dyn_auc']:.4f} "
              f"({'CIs disjoint -- real' if sep else 'CIs OVERLAP -- not resolved'})")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
