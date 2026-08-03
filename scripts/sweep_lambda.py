"""Bisection-style sweep over the SIGReg weight, scored by latent probing.

LeWM states that lambda is "the only effective hyperparameter to tune" and
suggests finding it by search. This does that search and scores each candidate
the way the paper evaluates representations -- a linear probe from the frozen
latent onto ground-truth state -- rather than by training loss, which is not
comparable across lambda values (a larger lambda simply adds a larger term).

    python -m scripts.sweep_lambda --data-root data/pusht-smoke --steps 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.episode import Episode
from sokubot.data.window import EpisodeWindowDataset
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import probe_model
from sokubot.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pusht-smoke")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lambdas", default="0.0,0.003,0.01,0.03,0.1")
    ap.add_argument("--threads", type=int, default=3,
                    help="cap torch threads; this runs on a shared workstation")
    ap.add_argument("--out", default="runs/lambda_sweep.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = Path(args.data_root)
    eps = [Episode.load(p) for p in sorted(root.glob("*.npz"))]
    print(f"{len(eps)} episodes, {sum(len(e) for e in eps)} frames")

    base = Config.tiny(total_steps=args.steps, warmup_steps=5)
    baseline = probe_model(LeWorldModel(base), eps, base, max_frames=1500)
    print(f"random-init baseline: {baseline}")

    rows = [{"lambda": None, "tag": "random-init", "r2_mean": baseline.r2_mean,
             "r2": baseline.r2}]

    for lam in [float(x) for x in args.lambdas.split(",")]:
        cfg = Config.tiny(total_steps=args.steps, warmup_steps=5, lambda_sigreg=lam)
        ds = EpisodeWindowDataset(cfg, root)
        model, hist = train(cfg, ds, steps=args.steps, verbose=False)
        pr = probe_model(model, eps, cfg, max_frames=1500)
        tail = hist[-10:]
        row = {
            "lambda": lam,
            "r2_mean": pr.r2_mean,
            "r2": pr.r2,
            "l_pred": float(np.mean([h["l_pred"] for h in tail])),
            "latent_var": float(np.mean([h["latent_var"] for h in tail])),
            "eff_rank": float(np.mean([h["eff_rank"] for h in tail])),
        }
        rows.append(row)
        print(f"lambda {lam:<7} | R2 {pr.r2_mean:6.3f} | L_pred {row['l_pred']:6.4f} "
              f"| var {row['latent_var']:5.3f} | erank {row['eff_rank']:6.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")

    scored = [r for r in rows if r["lambda"] is not None]
    best = max(scored, key=lambda r: r["r2_mean"])
    print(f"best lambda by probe R2: {best['lambda']} (R2 {best['r2_mean']:.3f})")


if __name__ == "__main__":
    main()
