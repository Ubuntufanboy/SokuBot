"""Materialise a fixed validation set as a tensor cache.

    python -m scripts.build_val_cache --manifest-root /root/exp/val --out /root/exp/val.pt

Every model in the scaling study must be scored on **byte-identical** inputs.
The Soku loader is an IterableDataset that decodes video on the fly and shuffles
through a buffer, so two passes over it do not produce the same windows in the
same order. That is right for training and useless for comparing models: any
difference between two runs would be partly the eval draw.

So the val windows are decoded once, here, and stored as uint8 (the same dtype
they had in the video, so the round trip is exact -- `frames_to_chw` divides by
255 and this multiplies back, and both are exact in float32 for 0..255).

Windows are taken with a stride so the cache spans many moments across many
captures rather than a few seconds of a few matches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sokubot.config import Config
from sokubot.data.soku import build_soku_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest-root", default="/root/exp/val")
    ap.add_argument("--out", default="/root/exp/val.pt")
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=17,
                    help="keep every Nth window within a capture; prime, to avoid "
                         "locking onto any periodicity in the game's animation")
    args = ap.parse_args()

    cfg = Config.soku()
    ds = build_soku_dataset(cfg, [args.manifest_root], shuffle_buffer=1, stride=args.stride)

    obs, acts, n = [], [], 0
    for sample in ds:
        obs.append((sample["obs"] * 255.0).round().to(torch.uint8))
        acts.append(sample["actions"].to(torch.uint8))
        n += 1
        if n >= args.windows:
            break
        if n % 256 == 0:
            print(f"  {n}/{args.windows}")

    if not obs:
        raise SystemExit(f"no windows decoded from {args.manifest_root}")

    O = torch.stack(obs)           # [N, T, 3, S, S] uint8
    A = torch.stack(acts)          # [N, T, ticks, 20] uint8
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": O, "actions": A, "cfg": cfg, "stride": args.stride}, out)

    press = A.float().mean().item()
    print(f"wrote {out}: {O.shape[0]} windows, obs {tuple(O.shape)} "
          f"({O.numel()/1e9:.2f} GB uint8), mean button press rate {press:.3f}")


if __name__ == "__main__":
    main()
