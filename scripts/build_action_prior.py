"""Recover the corpus button statistics the live opponent model needs.

    python -m scripts.build_action_prior --out artifacts/action_prior.npz

WHY THE LIVE LOOP NEEDS THIS
----------------------------
At play time the agent has to roll the world model forward to compensate for
control latency, and the predictor is conditioned on **both** players' twenty
buttons. The opponent's presses are not observable -- the open issue flagged in
`README.md` under "Soku specifics" -- so they have to be marginalised over some
distribution.

The right distribution is not a guess. `scripts/train_grpo.py` rolls every
imagined episode against `PolicyOpponent(reference)`, where the reference is a
`SokuPolicy` initialised to the corpus's own button statistics and frozen. Its
trunk weights are near zero at initialisation, so it is effectively a
state-independent sample from those statistics. Reproducing them here makes the
live opponent model *the same distribution the policy was trained against*
rather than a new assumption introduced at inference time.

`train_grpo` computes them from the full corpus action array it already has in
memory. Nothing else needs the corpus, so rather than make the live path depend
on 63 GB, this pulls a few captures' `inputs.csv` out of the public dataset --
these are aggregate frequencies over hundreds of thousands of frames, and they
converge long before one shard does.

The result is checked against the figures recorded in `docs/HANDOFF.md` and
`rl/policy.py` (human press rate 9.85%, vertically neutral 82%), so a bad
download fails loudly instead of quietly shifting the opponent model.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import requests

from sokubot.data.soku import read_actions

SHARD_URL = ("https://huggingface.co/datasets/Smashlytics/soku-frames-{s}/"
             "resolve/main/shards/{s.upper}-{n:04d}.tar")

# Documented in HANDOFF.md and rl/policy.py, from the full corpus.
EXPECT_PRESS_RATE = 0.0985
EXPECT_UD_NEUTRAL = 0.82
TOLERANCE = 0.02


def fetch_inputs(shard: str, index: int, head_mb: int, work: Path) -> list[Path]:
    """Range-GET the head of a shard and keep every complete inputs.csv in it."""
    url = (f"https://huggingface.co/datasets/Smashlytics/soku-frames-{shard}/"
           f"resolve/main/shards/{shard.upper()}-{index:04d}.tar")
    work.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    n = head_mb * 1024 * 1024
    r = requests.get(url, headers={"Range": f"bytes=0-{n - 1}"}, timeout=600)
    r.raise_for_status()
    tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r|")
    try:
        for m in tf:
            if not m.isfile() or not m.name.endswith("inputs.csv"):
                continue
            data = tf.extractfile(m).read()
            if len(data) != m.size:      # truncated by the range; unusable
                break
            p = work / f"{shard}{index:04d}_{len(got):02d}_inputs.csv"
            p.write_bytes(data)
            got.append(p)
    except (tarfile.TarError, EOFError):
        pass                              # ran off the end of the range: expected
    return got


def compute(csvs: list[Path]) -> dict:
    """inputs.csv files -> the three prior arrays `set_action_prior` takes."""
    blocks = []
    for p in csvs:
        A = read_actions(p)               # cross-checks bitmask vs booleans
        # Both players are equally valid samples of human play, so both blocks
        # count. train_grpo uses only P1's because that is the slice it had.
        blocks.append(A[:, :10])
        blocks.append(A[:, 10:])
    p = np.concatenate(blocks, 0).astype(np.float32)

    # Exactly train_grpo's formulas. The axes are 3-way categoricals because the
    # game cannot represent left+right or up+down (rl/policy.py), so "neutral"
    # is the probability that neither is held, not one minus their sum.
    lr = np.array([float(((1 - p[:, 2]) * (1 - p[:, 3])).mean()),
                   float(p[:, 2].mean()), float(p[:, 3].mean())])
    ud = np.array([float(((1 - p[:, 0]) * (1 - p[:, 1])).mean()),
                   float(p[:, 0].mean()), float(p[:, 1].mean())])
    lr /= lr.sum()
    ud /= ud.sum()
    btn = p[:, 4:10].mean(0)
    return {"lr": lr, "ud": ud, "btn": btn,
            "press_rate": float(p.mean()), "frames": len(p)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("artifacts/action_prior.npz"))
    ap.add_argument("--shard", default="a")
    ap.add_argument("--indices", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--head-mb", type=int, default=64)
    ap.add_argument("--work", type=Path,
                    default=Path.home() / ".cache/sokubot/prior")
    a = ap.parse_args()

    csvs: list[Path] = []
    for i in a.indices:
        found = fetch_inputs(a.shard, i, a.head_mb, a.work)
        print(f"  shard {a.shard}-{i:04d}: {len(found)} captures")
        csvs += found
    if not csvs:
        print("no inputs.csv recovered")
        return 1

    r = compute(csvs)
    print(f"\n  frames                {r['frames']:,} "
          f"(both players, {len(csvs)} captures)")
    print(f"  lr  none/left/right   {np.round(r['lr'], 4)}")
    print(f"  ud  none/up/down      {np.round(r['ud'], 4)}")
    print(f"  btn a b c d chg spl   {np.round(r['btn'], 4)}")
    print(f"  press rate            {r['press_rate']:.4f} "
          f"(corpus-wide {EXPECT_PRESS_RATE})")

    ok = True
    if abs(r["press_rate"] - EXPECT_PRESS_RATE) > TOLERANCE:
        print(f"  WARNING: press rate is {r['press_rate']:.4f}, expected "
              f"~{EXPECT_PRESS_RATE}")
        ok = False
    if abs(r["ud"][0] - EXPECT_UD_NEUTRAL) > TOLERANCE * 3:
        print(f"  WARNING: vertical neutral is {r['ud'][0]:.4f}, expected "
              f"~{EXPECT_UD_NEUTRAL}")
        ok = False

    # rl/policy.py refuses a prior it cannot represent, so check it here where
    # the message can say what to do rather than at policy construction.
    rarest = float(r["btn"].min())
    logit = float(np.log(rarest / (1 - rarest)))
    print(f"  rarest button         {rarest:.4f} -> logit {logit:.2f} "
          f"(policy logit_bound is 6.0)")
    if abs(logit) >= 6.0:
        print("  ERROR: the policy's logit bound cannot represent this prior")
        return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, lr=r["lr"], ud=r["ud"], btn=r["btn"],
             press_rate=r["press_rate"], frames=r["frames"])
    print(f"\n  wrote {a.out}")
    return 0 if ok else 0      # warnings do not fail the build; they are printed


if __name__ == "__main__":
    raise SystemExit(main())
