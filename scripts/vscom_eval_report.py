"""Tables for a vs-COM evaluation grid (ops/amarel/vscom_eval.slurm output).

    python -m scripts.vscom_eval_report ~/sokubot-runs/vscom-eval

One JSON per (arm, level, COM character), named <arm>_L<level>_c<char>.json. Prints, per arm and
level: matches and rounds won with Wilson 95% intervals (pooled over the 20 characters, each with
the same number of matches, so the pool is a uniform mixture of opponents), milestone 2 as the user
set it, and the per-character round win rates -- where a learned one-opponent trick shows up.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from scripts.vscom_play import wilson

NAMES = ["Reimu", "Marisa", "Sakuya", "Alice", "Patchouli", "Youmu", "Remilia", "Yuyuko",
         "Yukari", "Suika", "Reisen", "Aya", "Komachi", "Iku", "Tenshi", "Sanae", "Cirno",
         "Meiling", "Utsuho", "Suwako"]


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:])[0]).expanduser()
    cells: dict = defaultdict(dict)
    errors = 0
    for f in sorted(root.glob("*_L*_c*.json")):
        m = re.match(r"(.+)_L(\d)_c(\d+)\.json$", f.name)
        if not m:
            continue
        j = json.loads(f.read_text())
        cells[(m.group(1), int(m.group(2)))][int(m.group(3))] = j
        errors += len(j.get("errors", []))
    for (arm, level), by_char in sorted(cells.items()):
        M = sum(j["matches"] for j in by_char.values())
        Mw = sum(j["match_wins"] for j in by_char.values())
        R = sum(j["rounds"] for j in by_char.values())
        Rw = sum(j["round_wins"] for j in by_char.values())
        H = sum(j["wins_with_20pct_hp"] for j in by_char.values())
        lo, hi = wilson(Mw, M)
        rlo, rhi = wilson(Rw, R)
        print(f"\n=== {arm} vs the {'Lunatic' if level == 3 else 'Normal' if level == 1 else level} COM"
              f" ({len(by_char)} characters)")
        print(f"  matches won {Mw}/{M} = {Mw / max(M, 1):.3f} [{lo:.3f}, {hi:.3f}] | rounds won "
              f"{Rw}/{R} = {Rw / max(R, 1):.3f} [{rlo:.3f}, {rhi:.3f}] | wins with >= 20% HP "
              f"{H}/{Mw} | milestone 2 met: {bool(M and Mw / M >= 0.95 and Mw and H / Mw >= 0.8)}")
        row = []
        for c in sorted(by_char):
            j = by_char[c]
            row.append(f"{NAMES[c][:7]:>7} {j['round_wins']:>2}/{j['rounds']:<2}")
        for k in range(0, len(row), 5):
            print("   " + " | ".join(row[k:k + 5]))
    print(f"\nerrors across all cells: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
