"""Goal-conditioned planning evaluation on PushT: frozen model vs AdaJEPA.

    python -m scripts.eval_pusht --ckpt checkpoints/sokubot.pt \
        --data-root data/pusht --episodes 50

Reproduces the comparison both papers report. Following LeWM App. F, the start
state is sampled from a dataset trajectory and the goal is the state 25
environment steps later in the *same* trajectory, which guarantees the goal is
reachable and consistent with the dynamics. AdaJEPA Sec. 4.1 caps MPC at 20
replanning steps and executes one action chunk per step.

Each configuration is run on identical seeds so the two columns are paired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.episode import Episode
from sokubot.data.pusht import PushTRunner
from sokubot.model.world_model import LeWorldModel
from sokubot.planning.adajepa import plan_and_adapt


def load_model(ckpt: str, device: str):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = device
    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True, help="episodes to draw goals from")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20, help="MPC replanning steps")
    ap.add_argument("--goal-offset", type=int, default=5,
                    help="model steps ahead for the goal (5 x frame_skip env steps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/eval_pusht.json")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    paths = sorted(Path(args.data_root).glob("*.npz"))
    if not paths:
        raise SystemExit(f"no episodes under {args.data_root}")
    rng = np.random.default_rng(0)

    rows = []
    for mode in ("frozen", "adajepa"):
        finals, deltas, successes = [], [], []
        for i in range(args.episodes):
            ep = Episode.load(paths[int(rng.integers(len(paths))) % len(paths)])
            t = int(rng.integers(0, max(1, len(ep) - args.goal_offset)))
            goal = ep.frames[min(t + args.goal_offset, len(ep) - 1)]

            runner = PushTRunner(cfg, seed=1000 + i)
            res = plan_and_adapt(
                runner, model, cfg, goal,
                max_steps=args.max_steps,
                adapt=(mode == "adajepa"),
                success_fn=lambda info: bool(info.get("is_success", False)),
            )
            runner.close()
            finals.append(res.goal_distance[-1])
            deltas.append(res.goal_distance[0] - res.goal_distance[-1])
            successes.append(res.success)

        row = {
            "mode": mode,
            "episodes": args.episodes,
            "final_latent_distance": float(np.mean(finals)),
            "distance_reduced": float(np.mean(deltas)),
            "success_rate": float(np.mean(successes)),
        }
        rows.append(row)
        print(f"{mode:9s} | final d(goal) {row['final_latent_distance']:.4f} "
              f"| reduced {row['distance_reduced']:+.4f} "
              f"| success {row['success_rate']:.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
