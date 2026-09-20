"""Interaction density of a capture set, against a human reference.

`soku-selfplay-collector-density` is the reason this exists. The first
self-play corpus was tuned until its guarding rate looked human and still
trained a strictly worse model, because it had 28% of the human attack rate and
sat 304px apart: most of it was two characters walking around, which is
predictable from state alone. A single rate is not a calibration. The vector is.

Rates are per PLAYER-FRAME -- both players contribute -- so they are directly
comparable between corpora of different lengths. Separation is reported as the
median absolute `dx` in world units.

Pass `--ref` a set of human replay captures and the same numbers are computed
there rather than quoted from a memory, so the comparison is measured on the
corpora actually in hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, STAGE_SPAN, read_state   # noqa: E402

ROWS = ("hitboxes", "hitstop", "guarding", "wrongblock")


def measure(dirs: list[Path], limit: int) -> tuple[dict, int, int]:
    acc = {k: [] for k in ROWS}
    seps, n_frames, n_caps = [], 0, 0
    for d in dirs[:limit]:
        sc = d / "state.csv.gz"
        if not sc.exists():
            sc = d / "inputs.csv.gz"
        if not sc.exists():
            continue
        try:
            st, _p, _a, valid = read_state(sc)
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        if ok.sum() < 60:
            continue
        s = st[ok]
        for k in ROWS:
            v = s[:, :, CH[k]]
            # hitboxes/hitstop are counts; guarding/wrongblock are flags. Either
            # way the question is "was it happening on this player-frame".
            acc[k].append((v > 0).astype(np.float64).ravel())
        seps.append(np.abs(s[:, 0, CH["dx"]]) * STAGE_SPAN)
        n_frames += int(ok.sum())
        n_caps += 1
    out = {k: float(np.concatenate(v).mean()) if v else float("nan")
           for k, v in acc.items()}
    out["separation"] = (float(np.median(np.concatenate(seps)))
                         if seps else float("nan"))
    return out, n_frames, n_caps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", type=Path, nargs="+", required=True,
                    help="capture roots to measure")
    ap.add_argument("--ref", type=Path, nargs="*", default=None,
                    help="human replay capture roots, for the reference column")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()

    def expand(roots):
        out = []
        for r in roots or []:
            for w in sorted(r.glob("w*")):
                out += [d for d in sorted(w.iterdir()) if d.is_dir()]
            out += [d for d in sorted(r.iterdir())
                    if d.is_dir() and not d.name.startswith("w")
                    and not d.name.startswith(".")]
        return out

    got, nf, nc = measure(expand(a.dirs), a.limit)
    ref = None
    if a.ref:
        ref, rf, rc = measure(expand(a.ref), a.limit)

    print(f"measured   {nc} captures, {nf} frames")
    if ref:
        print(f"reference  {rc} captures, {rf} frames")
    print()
    w = 14
    hdr = f"{'':<18}{'measured':>{w}}"
    if ref:
        hdr += f"{'human ref':>{w}}{'ratio':>{w}}"
    print(hdr)
    for k in (*ROWS, "separation"):
        line = f"{k:<18}{got[k]:>{w}.4f}"
        if ref:
            r = ref[k]
            line += f"{r:>{w}.4f}"
            line += f"{got[k] / r:>{w-1}.2f}x" if r else f"{'--':>{w}}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
