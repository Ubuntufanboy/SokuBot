"""Compare a counterfactual PAIR of captures: same replay, one forced window.

WHAT A PAIR HAS TO BE
---------------------
`findings/08` established that no loss over observational data can teach a world
model what actions do, because the corpus never contains one state played two
ways. `findings/09` showed on a validated proxy that interventional data both is
necessary and suffices (29% of the true causal effect observationally, 124%
with counterfactual pairs). This is the instrument that produces those pairs in
the real game.

For the pair to mean anything the two runs must be IDENTICAL before the forced
window. Under self-play they are not: the behaviour policy's mode schedule
advances per hook call while the window is keyed to the battle frame, and the
battle frame freezes during hitstop, so the clocks slip -- measured at 6
disagreeing frames out of 671 before the window, which is not a controlled
comparison. Under BATTLE_SUBMODE_REPLAY there is no policy at all and both
players' inputs come from the recorded stream, so identity before the window is
a property of the vehicle rather than something to be tuned toward.

THE FIRST NUMBER TO READ IS THE PRE-WINDOW DIFF
-----------------------------------------------
If it is not zero, nothing below it is a counterfactual and the effect estimate
is meaningless. That check is printed first and on its own, deliberately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, STATE_CHANNELS, STAGE_SPAN, read_state  # noqa: E402


def load(d: Path):
    sc = d / "state.csv.gz"
    if not sc.exists():
        sc = d / "inputs.csv.gz"
    if not sc.exists():
        raise SystemExit(f"no sidecar in {d}")
    st, _p, _a, valid = read_state(sc)
    return st, valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="control capture dir")
    ap.add_argument("--b", type=Path, required=True, help="forced capture dir")
    ap.add_argument("--frame", type=int, default=None, help="SFE_CF_FRAME used")
    ap.add_argument("--len", type=int, default=None, help="SFE_CF_LEN used")
    ap.add_argument("--player", type=int, default=1)
    a = ap.parse_args()

    A, va = load(a.a)
    B, vb = load(a.b)
    n = min(len(A), len(B))
    print(f"control {len(A)} rows, forced {len(B)} rows, comparing {n}")
    A, B = A[:n], B[:n]

    # Per-frame equality over every channel of both players. float32 straight
    # out of the same build, so exact equality is the right test -- a tolerance
    # here would hide precisely the drift this is looking for.
    same = (A == B).reshape(n, -1).all(1)
    diff_idx = np.flatnonzero(~same)

    print(f"\nframes differing anywhere : {len(diff_idx)} / {n}")
    if len(diff_idx):
        print(f"first differing frame     : {int(diff_idx[0])}")

    if a.frame is None:
        print("\nno --frame given; this is a pure determinism check")
        print("VERDICT: " + ("IDENTICAL — the vehicle is deterministic"
                             if len(diff_idx) == 0 else
                             f"NOT identical ({len(diff_idx)} frames) — not a "
                             f"controlled pair"))
        return 0 if len(diff_idx) == 0 else 1

    w0 = a.frame
    w1 = a.frame + (a.len or 30)
    pre = diff_idx[diff_idx < w0]
    print(f"\nwindow                    : {w0}..{w1 - 1}")
    print(f"PRE-WINDOW differing      : {len(pre)}   <- must be 0")
    if len(pre):
        print(f"  first at {int(pre[0])}; the pair is NOT controlled and the "
              f"effect below is not causal")

    # What the intervention did. `guarding` is the channel every blocking number
    # in this project is quoted in.
    i = a.player - 1
    g = CH["guarding"]
    post = slice(w0, n)
    ga = (A[post, i, g] > 0.5)
    gb = (B[post, i, g] > 0.5)
    print(f"\nfrom the window onward ({n - w0} frames), player {a.player}:")
    print(f"  P(guarding) control {ga.mean():.4f}   forced {gb.mean():.4f}   "
          f"effect {gb.mean() - ga.mean():+.4f}")
    hp = CH["hp"]
    print(f"  final hp  control {A[-1, i, hp]:.4f}   forced {B[-1, i, hp]:.4f}")
    dxa = np.abs(A[post, 0, CH['dx']]).mean() * STAGE_SPAN
    dxb = np.abs(B[post, 0, CH['dx']]).mean() * STAGE_SPAN
    print(f"  mean separation  control {dxa:.0f}u   forced {dxb:.0f}u")

    if len(diff_idx):
        first_in = diff_idx[diff_idx >= w0]
        if len(first_in):
            print(f"\ndivergence begins at frame {int(first_in[0])} "
                  f"({int(first_in[0]) - w0} frames into the window)")
            ch = np.flatnonzero(~(A[int(first_in[0])] == B[int(first_in[0])]).all(0))
            print("  channels first differing: "
                  + ", ".join(STATE_CHANNELS[c] for c in ch[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
