"""Put several state-GRPO runs side by side, per gym.

    python -m scripts.compare_state_runs a.json b.json c.json --names sym def raw

Consumes the JSON `scripts/eval_state_policy.py` writes, so the numbers being
compared are the ones that were measured rather than re-derived, and each run
was scored against ITS OWN frozen reference inside ITS OWN simulator.

WHAT IS AND IS NOT COMPARABLE ACROSS THESE COLUMNS
--------------------------------------------------
`net` is comparable: every run is re-scored with the same fixed symmetric
damage reward, whatever it was trained on, so a run trained with a defensive
weighting is not credited or penalised for that at evaluation time.

`guard` is NOT comparable across runs that used different simulators. It is the
simulator's own predicted `guarding` flag, and two world models have different
base rates for it -- rawproj predicts 0.184 where fix5 predicts 0.234 in the
same forced-away test. So it is printed as a DIFFERENCE from that run's own
reference, which is the only form that means the same thing in both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FULL_HP = 10000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--metric", default="net",
                    choices=("net", "dealt", "taken", "guard", "press"))
    a = ap.parse_args()
    data = [json.loads(p.read_text()) for p in a.runs]
    names = a.names or [p.parent.name or p.stem for p in a.runs]
    if len(names) != len(data):
        raise SystemExit(f"{len(names)} names for {len(data)} runs")

    def value(d, gym):
        v = d["gyms"].get(gym)
        if v is None:
            return None
        if a.metric == "net":
            return v["net"] * FULL_HP
        if a.metric in ("dealt", "taken"):
            return (v["agent"][a.metric] - v["reference"][a.metric]) * FULL_HP
        return v["agent"][a.metric] - v["reference"][a.metric]

    gyms = sorted({g for d in data for g in d["gyms"]},
                  key=lambda g: -(value(data[0], g) or -1e9))
    unit = "HP/step" if a.metric in ("net", "dealt", "taken") else "vs own ref"
    print(f"\n  {a.metric} ({unit}); each run against its own frozen reference")
    print(f"  {'gym':<20} " + " ".join(f"{n:>10}" for n in names))
    for g in gyms:
        cells = []
        for d in data:
            v = value(d, g)
            cells.append("         -" if v is None else f"{v:+10.2f}")
        print(f"  {g:<20} " + " ".join(cells))
    print(f"\n  {'noise floor':<20} " +
          " ".join(f"{d['noise_floor_net']*FULL_HP:+10.2f}" for d in data))
    print(f"  {'step':<20} " + " ".join(f"{d['step']:>10d}" for d in data))
    if a.metric == "net":
        print("\n  A difference between columns is only meaningful against the "
              "noise floor row.\n  Runs on different simulators are comparable "
              "in `net` but not in absolute\n  damage: see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
