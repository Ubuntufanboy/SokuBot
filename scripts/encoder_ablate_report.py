"""Rank the encoder ablations and say what each factor bought.

    python -m scripts.encoder_ablate_report --enc ~/SokuBot-lab/enc

Reports the mean against a NAMED baseline arm rather than against the best, so
"did this factor help" has a fixed reference. Per-channel deltas are shown for
the channels the sweep was run to fix -- a mean R^2 that rises while `vx` stays
at zero has improved the easy channels and answered nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANNELS = ("x", "y", "dx", "dy", "vx", "vy", "hp", "spirit", "airborne",
            "timestop")


def load(enc: Path) -> list[dict]:
    out = []
    for j in sorted(enc.glob("*.pt.json")):
        try:
            d = json.loads(j.read_text())
        except (ValueError, OSError):
            continue
        best = None
        for rec in d.get("log", []):
            if best is None or rec["mean_r2"] > best["mean_r2"]:
                best = rec
        if best is None:
            continue
        d["best_rec"] = best
        out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--enc", type=Path, default=Path("~/SokuBot-lab/enc"))
    ap.add_argument("--baseline", default="B00_base")
    ap.add_argument("--prev", type=float, default=0.359,
                    help="encoder2's best mean state R2, the number to beat")
    a = ap.parse_args()
    arms = load(a.enc.expanduser())
    if not arms:
        print(f"no results in {a.enc}")
        return 1
    base = next((d for d in arms if d["arm"] == a.baseline), None)
    b = base["best_rec"] if base else None

    arms.sort(key=lambda d: -d["best_mean_r2"])
    print(f"{len(arms)} arms | baseline {a.baseline} | encoder2 was "
          f"{a.prev:+.4f}\n")
    hdr = (f"{'arm':<20}{'mean':>8}{'vs base':>9}{'step':>7}"
           + "".join(f"{c:>9}" for c in ("vx", "vy", "x", "dx", "hp")))
    print(hdr)
    print("-" * len(hdr))
    for d in arms:
        r = d["best_rec"]
        dv = (f"{r['mean_r2'] - b['mean_r2']:+8.4f}" if b else "        -")
        # A best step at the very end means the arm was still improving and the
        # schedule, not the factor, is what limited it.
        tail = "  <- still rising" if d["best_step"] >= 0.9 * d["steps"] else ""
        print(f"{d['arm']:<20}{r['mean_r2']:+8.4f}{dv}{d['best_step']:>7}"
              + "".join(f"{r.get('r2_' + c, float('nan')):+9.3f}"
                        for c in ("vx", "vy", "x", "dx", "hp")) + tail)

    print(f"\n--- best arm, all channels ---")
    top = arms[0]
    r = top["best_rec"]
    print(f"{top['arm']}  mean {r['mean_r2']:+.4f} at step {top['best_step']} "
          f"| delta {top['delta']} input {top['input']} downs {top['downs']} "
          f"grid {top['grid']} width {top['width']} params "
          f"{top['params']/1e6:.2f}M")
    for c in CHANNELS:
        v = r.get(f"r2_{c}", float("nan"))
        mark = "  (still below its own corpus mean)" if v <= 0 else ""
        print(f"    {c:<10} {v:+.4f}{mark}")
    usable = [c for c in CHANNELS if r.get(f"r2_{c}", 0) > 0]
    print(f"  {len(usable)}/{len(CHANNELS)} channels beat their corpus mean: "
          f"{', '.join(usable)}")

    if b:
        print("\n--- what each factor bought, against the baseline ---")
        for d in arms:
            if d["arm"] == a.baseline:
                continue
            dm = d["best_rec"]["mean_r2"] - b["mean_r2"]
            dvx = (d["best_rec"].get("r2_vx", 0) - b.get("r2_vx", 0))
            dvy = (d["best_rec"].get("r2_vy", 0) - b.get("r2_vy", 0))
            verdict = "HELPS" if dm > 0.01 else ("hurts" if dm < -0.01 else "flat")
            print(f"  {d['arm']:<20} mean {dm:+.4f}  vel {(dvx+dvy)/2:+.4f}"
                  f"   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
