"""Split a fetched corpus manifest into disjoint train and validation sets.

    python -m scripts.split_corpus --corpus /root/corpus --val-hours 3

Writes `train/manifest.jsonl` and `val/manifest.jsonl` next to the corpus, both
pointing at the same capture directories -- these are views, not copies.

The split is at **capture** granularity, not window granularity. Two windows
from the same replay share characters, stage, players and most of their frames,
so splitting by window would put near-duplicates on both sides and report a
validation loss that is really a training loss. A replay is the smallest unit
that is plausibly independent.

Validation is drawn by shuffling with a fixed seed, so re-running gives the same
split and a resumed training run keeps its held-out set honest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--val-hours", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    src = args.corpus / "manifest.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no entries in {src}")

    random.Random(args.seed).shuffle(rows)

    val, val_frames = [], 0.0
    budget = args.val_hours * 3600 * 60
    for r in rows:
        if val_frames >= budget:
            break
        val.append(r)
        val_frames += r.get("frames") or 0
    val_ids = {r["replay_id"] for r in val}
    train = [r for r in rows if r["replay_id"] not in val_ids]

    for name, rs in (("train", train), ("val", val)):
        d = args.corpus / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        h = sum(r.get("frames") or 0 for r in rs) / 60 / 3600
        print(f"{name:5s}: {len(rs):5d} captures, {h:7.2f} h -> {d}/manifest.jsonl")

    assert not (val_ids & {r["replay_id"] for r in train}), "split is not disjoint"
    print(f"\ndisjoint: OK (seed {args.seed})")


if __name__ == "__main__":
    main()
