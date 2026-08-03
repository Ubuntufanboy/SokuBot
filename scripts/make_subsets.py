"""Turn a transferred capture pool into nested train subsets + a fixed val set.

    python -m scripts.make_subsets --data /root/data --lists /root/lists --out /root/exp

Writes one ``manifest.jsonl`` per split, all pointing at the *same* capture
directories. The subsets are views, not copies: the 10h manifest lists every
capture the 5h one does plus more, and nothing is duplicated on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Where each (host, dataset root) landed under --data.
SOURCE_DIR = {
    ("A", "/root/dataset2"): "A2",
    ("A", "/root/dataset"): "A1",
    ("B", "/root/dataset"): "B",
    ("C", "/root/dataset"): "C",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="/root/data")
    ap.add_argument("--lists", default="/root/lists")
    ap.add_argument("--out", default="/root/exp")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    sel = json.loads((Path(args.lists) / "selection.json").read_text())
    caps = sel["captures"]

    def entry(name: str):
        c = caps[name]
        sub = SOURCE_DIR[(c["host"], c["root"])]
        d = data / sub / c["rel"]
        if not (d / "video.mp4").exists() or not (d / "inputs.csv").exists():
            return None
        return {"replay_id": name, "status": "ok",
                "video": str(d / "video.mp4"), "frames": c["frames"]}

    def write(split: str, names) -> tuple[int, float, int]:
        rows, missing = [], 0
        for n in names:
            e = entry(n)
            if e is None:
                missing += 1
                continue
            rows.append(e)
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        hours = sum(r["frames"] for r in rows) / 60 / 3600
        return len(rows), hours, missing

    n, h, m = write("val", sel["val"])
    print(f"val       : {n:4d} captures, {h:5.2f} h" + (f"  ({m} missing)" if m else ""))

    for size, names in sorted(sel["train"].items(), key=lambda kv: float(kv[0])):
        n, h, m = write(f"train_{float(size):g}h", names)
        print(f"train {float(size):4.1f}h: {n:4d} captures, {h:5.2f} h"
              + (f"  ({m} missing)" if m else ""))

    # A capture must never appear in both -- the whole comparison rests on it.
    val_set = set(sel["val"])
    for size, names in sel["train"].items():
        overlap = val_set & set(names)
        if overlap:
            raise SystemExit(f"FATAL: {len(overlap)} captures in both val and train_{size}h")
    print("\nval/train disjoint: OK")


if __name__ == "__main__":
    main()
