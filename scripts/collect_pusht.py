"""Collect offline PushT trajectories for training.

    python -m scripts.collect_pusht --out data/pusht --episodes 2000

Uses the block-biased random policy described in LeWM App. E. Uniform-random
actions almost never make contact with the T, and a world model trained on them
learns that the block is furniture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sokubot.config import Config
from sokubot.data.pusht import collect_episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/pusht")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=60, help="model steps per episode")
    ap.add_argument("--block-bias", type=float, default=0.8)
    ap.add_argument("--image-size", type=int, default=None,
                    help="override the model input resolution (default: config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="use the CPU smoke config")
    args = ap.parse_args()

    cfg = Config.tiny() if args.tiny else Config.pusht()
    if args.image_size:
        cfg.image_size = args.image_size

    out = Path(args.out)
    print(f"collecting {args.episodes} episodes at {cfg.image_size}px, "
          f"frame-skip {cfg.frame_skip} -> {out}")
    n = collect_episodes(
        cfg, out,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        block_bias=args.block_bias,
        seed=args.seed,
    )
    frames = n * args.max_steps
    print(f"wrote {n} episodes (~{frames} decision frames, "
          f"~{frames * cfg.frame_skip} env steps)")


if __name__ == "__main__":
    main()
