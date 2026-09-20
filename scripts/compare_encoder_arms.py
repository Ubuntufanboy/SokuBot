"""Compare encoder arms with their seed spread, not against each other's best.

    python -m scripts.compare_encoder_arms \
        --control ~/rl/enc_control_s0.pt ~/rl/enc_control_s1.pt \
        --arm char=~/rl/enc_char_s0.pt,~/rl/enc_char_s1.pt

WHY THIS SCRIPT AND NOT A GLANCE AT TWO NUMBERS
-----------------------------------------------
`future_paper/findings/04` closes on the quietest pathology in the project:
four projectile architectures spanning 862x in head capacity all landed inside
a 269-281 unit band, **not one of them repeated at a second seed**, so the 12 u
by which the best "won" and the 10 u by which another "lost" were the same size
and neither was signal. An unmeasured noise floor does not produce a wrong
number -- it silently sets the resolution of every comparison drawn against it.

So this refuses to report a delta without a spread to judge it by. With one
seed per arm it prints the delta and says the comparison is unresolved, rather
than ranking.

WHAT IT COMPARES
----------------
The character head is an AUXILIARY task, and auxiliary tasks compete for
capacity whatever their form -- measured on this encoder, tripling the
projectile weight cost ~0.013 of state R2 whether the target was coordinates or
a heatmap. The question is therefore not "does the head work" but "what did it
cost the channels the policy actually eats", so the state R2 is reported beside
the identity numbers and not underneath them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load(path: Path) -> dict:
    d = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    r2 = np.asarray(d["r2"])
    sup = list(d["supervised"])
    n_state = int(d["n_state"])
    per = r2[:len(sup)]
    return {
        "path": Path(path).name,
        "mean_state": float(np.mean(r2[:n_state])),
        "per": {n: float(v) for n, v in zip(sup, per)},
        "n_char": int(d.get("n_char", 0)),
        "char_acc": float(d.get("char_acc", float("nan"))),
        "decide_acc": float(d.get("decide_acc", float("nan"))),
        "id_acc": float(d.get("id_acc", float("nan"))),
        "step": int(d.get("step", -1)),
    }


def agg(runs: list[dict], key: str) -> tuple[float, float]:
    v = np.array([r[key] for r in runs], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.max() - v.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", type=Path, nargs="+", required=True,
                    help="one checkpoint per seed")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=P1,P2",
                    help="repeatable; comma-separated checkpoints, one per seed")
    ap.add_argument("--channels", nargs="+",
                    default=["x", "dx", "y", "hp", "spirit", "airborne"])
    a = ap.parse_args()

    ctrl = [load(p) for p in a.control]
    arms = {}
    for spec in a.arm:
        name, _, paths = spec.partition("=")
        arms[name] = [load(Path(p)) for p in paths.split(",") if p]

    c_mean, c_spread = agg(ctrl, "mean_state")
    print(f"control: {len(ctrl)} seed(s), mean state R2 {c_mean:+.4f}, "
          f"seed spread {c_spread:.4f}")
    if len(ctrl) < 2:
        print("  WARNING: one seed. There is no noise floor, so every delta "
              "below is unresolved by construction.")
    print()
    hdr = f"{'arm':10s} {'meanR2':>9s} {'d(ctrl)':>9s} {'spread':>8s} " \
          f"{'verdict':>12s} {'char':>7s} {'decide':>7s}"
    print(hdr); print("-" * len(hdr))
    print(f"{'control':10s} {c_mean:+9.4f} {'--':>9s} {c_spread:8.4f} "
          f"{'--':>12s} {'--':>7s} {'--':>7s}")
    for name, runs in arms.items():
        m, sp = agg(runs, "mean_state")
        d = m - c_mean
        # The floor is the LARGER of the two spreads: a delta smaller than the
        # run-to-run variation of either side is not distinguishable from it.
        floor = max(sp, c_spread)
        if not np.isfinite(floor) or floor == 0:
            verdict = "no floor"
        elif abs(d) < floor:
            verdict = "WITHIN NOISE"
        else:
            verdict = f"{abs(d) / floor:.1f}x floor"
        ca, _ = agg(runs, "char_acc")
        da, _ = agg(runs, "decide_acc")
        print(f"{name:10s} {m:+9.4f} {d:+9.4f} {sp:8.4f} {verdict:>12s} "
              f"{ca:7.3f} {da:7.3f}")

    print("\nper-channel state R2 (control -> arm), the channels the policy eats:")
    for ch in a.channels:
        row = f"  {ch:<10s} {np.mean([r['per'].get(ch, np.nan) for r in ctrl]):+.4f}"
        for name, runs in arms.items():
            row += f"   {name}={np.mean([r['per'].get(ch, np.nan) for r in runs]):+.4f}"
        print(row)

    print("\nidentity:")
    print(f"  p1_left accuracy (control) {agg(ctrl, 'id_acc')[0]:.4f}  "
          f"<- chance by construction; nothing in the play area names Player 1")
    for name, runs in arms.items():
        ca, cs = agg(runs, "char_acc")
        da, ds = agg(runs, "decide_acc")
        if not np.isfinite(da):
            continue
        print(f"  {name}: per-row character {ca:.4f} (spread {cs:.4f}), "
              f"IDENTITY DECISION {da:.4f} (spread {ds:.4f})")
        print(f"      compare: live nearest-neighbour tracker 0.51, "
              f"perfect-input tracker 0.64, and the 3 s probe this replaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
