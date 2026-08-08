"""Join human annotations against `data/hud.py` and decide what to build next.

    python -m scripts.score_hud_annotation --dir ~/hud_annot

This is the second half of `scripts/make_hud_annotation.py`. That script writes
blind sheets plus an answer key; a human fills `annotations.csv`; this compares
them and prints which of two mutually exclusive worlds we are in.

THE DECISION THIS MAKES
-----------------------
Spirit probes out of the latent at R^2 0.05 and the reward's KO detector runs at
precision 0.003. Two explanations, opposite consequences:

  (a) `hud.py` is RIGHT and the 224 px downsample destroys the signal.
      -> the labels are fine, the bottleneck is the encoder's input, and the fix
         is to feed the HUD at native resolution as extra world-model state.
  (b) `hud.py` is WRONG.
      -> the probe has been fit against noise and no architecture change helps;
         the reader has to be fixed or learned first.

Only a human looking at the pixels separates them, which is what the sheets are.

HOW AGREEMENT IS JUDGED
-----------------------
Correlation alone would be generous: a reader that is right about the *shape* of
the gauge but wrong about its *scale* still correlates near 1.0 while producing
labels that are systematically off. So this reports correlation, bias and mean
absolute error together, and separately over the low-health band, which is the
regime `scripts/probe_reliability.py` singled out.

Spirit is annotated in lit hexagons (0 to 5) and `hud.py` reports a fraction of
the gauge, so the human's count is divided by SPIRIT_N before comparison; health
is annotated as a percentage and divided by 100. Getting either conversion wrong
would manufacture a disagreement, so both are asserted to land in [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sokubot.data.hud import SPIRIT_N

LOW_HEALTH = 0.16          # the band probe_reliability found the probe worst in


def load_annotations(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as fh:
        for row in csv.DictReader(r for r in fh if not r.lstrip().startswith("#")):
            try:
                i = int(row["id"])
            except (TypeError, ValueError):
                continue
            vals = {}
            for k in ("hp1", "hp2", "spirit1", "spirit2"):
                v = (row.get(k) or "").strip()
                if not v:
                    continue
                try:
                    vals[k] = float(v)
                except ValueError:
                    # The header tells the annotator that "unreadable" is a real
                    # answer, so some of them write the word. That is the right
                    # answer expressed the obvious way, and crashing on it would
                    # throw away a whole session over a wording choice this file
                    # invited. Anything non-numeric means the same as blank.
                    print(f"  frame {i} {k}: {v!r} read as unreadable (blank)")
            if vals:
                out[i] = vals
    return out


def agreement(human: np.ndarray, machine: np.ndarray) -> dict:
    n = len(human)
    if n < 3:
        return {"n": n}
    d = machine - human
    r = float(np.corrcoef(human, machine)[0, 1]) if human.std() > 1e-9 else float("nan")
    return {"n": n, "r": r, "bias": float(d.mean()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())),
            "human_std": float(human.std()), "machine_std": float(machine.std())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=Path("hud_annot"))
    ap.add_argument("--mae-ok", type=float, default=0.10,
                    help="mean absolute error below which a channel is called "
                         "trustworthy, in gauge fractions")
    a = ap.parse_args()

    key = {p["id"]: p for p in json.loads((a.dir / "answer_key.json").read_text())}
    ann = load_annotations(a.dir / "annotations.csv")
    if not ann:
        raise SystemExit(
            f"no filled rows in {a.dir}/annotations.csv -- every cell is blank, so "
            f"there is nothing to compare. Fill it in and re-run.")
    print(f"{len(ann)} of {len(key)} frames annotated\n")

    # Human units -> hud.py units. Two different things can go wrong here and
    # they need opposite treatment: a *unit mix-up* (health written as 0-1, or
    # spirit as a percentage) silently manufactures a disagreement and must be
    # caught, while *edge rounding* (0 written as -0.4, or a full bar as 101) is
    # a human reading a ruler and must not throw away the session. So the scale
    # is inferred per column from its whole range, then values are clamped.
    scale = {}
    for ch, full in (("hp1", 100.0), ("hp2", 100.0),
                     ("spirit1", float(SPIRIT_N)), ("spirit2", float(SPIRIT_N))):
        vals = [v[ch] for v in ann.values() if ch in v]
        if not vals:
            scale[ch] = full
            continue
        hi = max(vals)
        if hi <= 1.5 and full > 1.5:
            print(f"  NOTE: {ch} tops out at {hi:g}, so it was written as a "
                  f"fraction rather than 0-{full:g}. Reading it that way.")
            scale[ch] = 1.0
        elif hi > full * 1.5:
            raise SystemExit(
                f"{ch} reaches {hi:g}, well past the expected maximum of "
                f"{full:g}. Health is a percentage 0-100 and spirit is lit "
                f"hexagons 0-{SPIRIT_N}; comparing mixed units would invent a "
                f"disagreement that is not real. Fix the column and re-run.")
        else:
            scale[ch] = full

    pairs: dict[str, tuple[list, list]] = {k: ([], []) for k in
                                           ("hp1", "hp2", "spirit1", "spirit2")}
    low: dict[str, tuple[list, list]] = {k: ([], []) for k in ("hp1", "hp2")}
    blanks = {k: 0 for k in pairs}
    for i, vals in ann.items():
        k = key.get(i)
        if k is None:
            print(f"  annotation for unknown frame id {i}, skipped")
            continue
        for ch in pairs:
            if ch not in vals:
                blanks[ch] += 1
                continue
            h = vals[ch] / scale[ch]
            if not -0.1 <= h <= 1.1:
                raise SystemExit(
                    f"frame {i} {ch}={vals[ch]} converts to {h:.3f}, far outside "
                    f"[0,1] even after inferring the column's scale. Check that "
                    f"row.")
            h = min(max(h, 0.0), 1.0)          # edge rounding, not a mix-up
            pairs[ch][0].append(h)
            pairs[ch][1].append(k[ch])
            if ch.startswith("hp") and k[ch] <= LOW_HEALTH:
                low[ch][0].append(h)
                low[ch][1].append(k[ch])

    print(f"{'channel':<10} {'n':>4} {'r':>7} {'bias':>8} {'mae':>7} {'rmse':>7}"
          f"  {'blank':>5}")
    verdict = {}
    for ch, (h, m) in pairs.items():
        s = agreement(np.array(h), np.array(m))
        verdict[ch] = s
        if s.get("n", 0) < 3:
            print(f"{ch:<10} {s['n']:>4}   too few annotations to judge")
            continue
        print(f"{ch:<10} {s['n']:>4} {s['r']:>7.3f} {s['bias']:>+8.3f} "
              f"{s['mae']:>7.3f} {s['rmse']:>7.3f}  {blanks[ch]:>5}")

    print("\nhealth, restricted to hud.py <= %.2f (where the probe is worst):"
          % LOW_HEALTH)
    for ch, (h, m) in low.items():
        s = agreement(np.array(h), np.array(m))
        verdict[ch + "_low"] = s
        if s.get("n", 0) < 3:
            print(f"  {ch}: only {s['n']} frames, not enough to judge")
        else:
            print(f"  {ch}: n {s['n']} r {s['r']:+.3f} bias {s['bias']:+.3f} "
                  f"mae {s['mae']:.3f}")

    (a.dir / "agreement.json").write_text(json.dumps(verdict, indent=1))

    # ---- the actual decision ----
    def ok(ch: str) -> bool:
        s = verdict.get(ch, {})
        return s.get("n", 0) >= 3 and s.get("mae", 9) <= a.mae_ok

    sp = [c for c in ("spirit1", "spirit2") if ok(c)]
    hp_low_ok = all(ok(c + "_low") or verdict.get(c + "_low", {}).get("n", 0) < 3
                    for c in ("hp1", "hp2"))
    print("\n" + "=" * 70)
    if len(sp) == 2:
        print("SPIRIT: hud.py AGREES with the human. The labels are sound, so the\n"
              "  R^2 0.05 is the 224 px downsample destroying the gauge before the\n"
              "  encoder sees it -- an input-resolution problem, not a reading one.\n"
              "  -> Build the HUD-augmented latent. A learned reader would add\n"
              "     nothing, because hud.py is already right.")
    elif sp:
        print(f"SPIRIT: hud.py agrees on {sp} but not the other chair. A reader that\n"
              "  works on one side and not the mirrored other is broken, not noisy\n"
              "  -> inspect the disagreeing chair's geometry before building on it.")
    else:
        print("SPIRIT: hud.py DISAGREES with the human. The probe has been fit\n"
              "  against bad labels, so no architecture change rescues it.\n"
              "  -> Fix or learn the reader FIRST; augmenting with wrong state\n"
              "     just teaches the predictor to forecast noise.")
    if not hp_low_ok:
        print("\nHEALTH AT LOW VALUES: hud.py disagrees with the human down there.\n"
              "  That pushes the give-up behaviour further upstream than the probe:\n"
              "  the *labels* were wrong near zero, so every downstream measurement\n"
              "  in that regime -- including the KO detector -- inherited it.")
    else:
        print("\nHEALTH AT LOW VALUES: hud.py agrees with the human, so the labels\n"
              "  are sound and the probe's 0.25 noise there is the representation's\n"
              "  failure, not the reader's.")
    print("=" * 70)
    print(f"\n-> {a.dir}/agreement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
