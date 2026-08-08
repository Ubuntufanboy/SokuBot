"""Build a blind annotation sheet to check `data/hud.py` against a human.

    python -m scripts.make_hud_annotation --corpus ~/corpus --out ~/hud_annot

WHY THIS EXISTS
---------------
Every label the reward probe is fit against comes from `data/hud.py`, a stack of
hand-tuned colour thresholds that has never been checked against a person. Two
of its channels are known to be unusable downstream -- spirit probes at R^2 0.05,
cards at 0.156/0.051 -- and there are two very different explanations:

  (a) the 224 px downsample destroys them before the encoder ever sees them, or
  (b) `hud.py`'s readings are wrong, so the probe is being fit against noise.

Nobody has distinguished these, and they imply opposite work. Under (a) the fix
is architectural (feed the HUD at native resolution); under (b) no architecture
helps, because the targets themselves are wrong.

`scripts/probe_reliability.py` added a third question with more urgency than
either: the probe is at its worst precisely at low health (noise 0.25 against
0.13 elsewhere, bias +0.154 in the 0.02-0.04 band), and the KO detector built on
it fires 20-45x too often at precision ~0.003. If `hud.py` is *also* unreliable
near zero, then the low-health labels were never right and the whole regime has
been trained on noise.

THE SHEET IS BLIND, AND THAT IS THE POINT
------------------------------------------
`hud.py`'s reading is deliberately **not** drawn on the sheet. Showing it would
anchor the annotator, and an anchored annotator agreeing with the instrument is
not evidence the instrument works. The readings go to a separate answer key that
`scripts/score_hud_annotation.py` joins against afterwards.

Frames are stratified by `hud.py`'s health reading with low health deliberately
oversampled, because that is the band the measurement is about. The stratifier
uses the very instrument under test, which is fine: it only decides *which*
frames to look at, and being wrong there costs coverage, not correctness.

WHAT IS DRAWN
-------------
Health bars get a tick ruler at 10% intervals derived from `HP_FULL_PX`, so
eyeballing a fraction is reading off a scale rather than guessing. Spirit
hexagons are cropped and magnified so lit/unlit is obvious. Both HUD strips
(health at the top of the frame, spirit and cards at the bottom) are stacked
into one tile per frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokubot.config import Config
from sokubot.data.hud import (CARD_ROWS, FILL_ROWS, HP_FULL_PX, P1_CARD_X,
                              SPIRIT_CY_EVEN, SPIRIT_CY_ODD,
                              P1_HP_X, P1_SPIRIT_X, P2_CARD_X, P2_HP_X,
                              P2_SPIRIT_X, read_trace)
from scripts.horizon_ablation import capture_paths, decode_hud

# Health rows plus a margin, and the whole spirit/card block at the bottom.
TOP = (FILL_ROWS[0] - 8, FILL_ROWS[1] + 8)
BOT = (CARD_ROWS[0] - 4, CARD_ROWS[1] + 4)
SCALE = 3
# Strata over hud.py's health reading. Low health is oversampled on purpose:
# it is 8.5% of frames but the entire question.
STRATA = [(0.00, 0.06, 10), (0.06, 0.16, 10), (0.16, 0.35, 8),
          (0.35, 0.65, 6), (0.65, 1.01, 6)]


HP_ROWS = (FILL_ROWS[0] - 3, FILL_ROWS[1] + 3)
SPIRIT_ROWS = (SPIRIT_CY_EVEN - 12, SPIRIT_CY_ODD + 12)
LABEL_W = 132


def _row(crop: np.ndarray, label: str, scale: int, ruler: bool) -> np.ndarray:
    """One labelled, magnified HUD region, optionally with a 10-tick ruler."""
    import cv2
    img = cv2.resize(crop, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    if ruler:
        # A dedicated strip *below* the bar rather than ticks drawn over it. The
        # first version drew 1 px ticks on the frame itself and they were
        # effectively invisible at this magnification, which defeats the point:
        # the ruler exists so a fraction is read off a scale instead of guessed.
        strip = np.zeros((26, w, 3), np.uint8)
        for k in range(11):
            x = min(int(k * w / 10.0), w - 1)
            tall = k % 5 == 0
            cv2.line(strip, (x, 0), (x, 14 if tall else 8), (80, 220, 255),
                     2 if tall else 1)
            if tall:
                cv2.putText(strip, f"{k*10}", (max(x - 10, 2), 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 255), 1,
                            cv2.LINE_AA)
        img = np.concatenate([img, strip], axis=0)
        h = img.shape[0]
    pad = np.zeros((h, LABEL_W, 3), np.uint8)
    cv2.putText(pad, label, (6, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([pad, img], axis=1)


def tile(frame480: np.ndarray) -> np.ndarray:
    """One frame -> four labelled regions: each player's health bar and spirit gauge.

    Deliberately *not* the whole HUD strip. The first version cropped both full
    strips, which dragged in character portraits, the weather widget and the
    round timer -- none of which is being annotated, all of which competes for
    attention, and the card stock, which `scripts/horizon_ablation.py` has
    already shown to be undecodable (calibrated R^2 +0.03 for cards1 and
    *negative* for cards2). Asking for annotations on a channel already ruled out
    spends the scarcest resource here, which is the annotator.
    """
    import cv2
    f = frame480[::-1]                      # corpus is stored vertically flipped
    rows = [
        _row(f[HP_ROWS[0]:HP_ROWS[1], P1_HP_X[0]:P1_HP_X[1]], "P1 HP", 5, True),
        _row(f[HP_ROWS[0]:HP_ROWS[1], P2_HP_X[0]:P2_HP_X[1]], "P2 HP", 5, True),
        _row(f[SPIRIT_ROWS[0]:SPIRIT_ROWS[1],
               P1_SPIRIT_X[0]:P1_SPIRIT_X[1]], "P1 SPIRIT", 5, False),
        _row(f[SPIRIT_ROWS[0]:SPIRIT_ROWS[1],
               P2_SPIRIT_X[0]:P2_SPIRIT_X[1]], "P2 SPIRIT", 5, False),
    ]
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.concatenate(
                [r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)], axis=1)
        out += [r, np.zeros((3, w, 3), np.uint8)]
    return np.concatenate(out, axis=0)


def sheet(tiles: list[np.ndarray], labels: list[str], cols: int = 1) -> np.ndarray:
    import cv2
    out = []
    for t, lab in zip(tiles, labels):
        strip = np.zeros((26, t.shape[1], 3), np.uint8)
        cv2.putText(strip, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.concatenate([strip, t,
                                   np.zeros((10, t.shape[1], 3), np.uint8)], 0))
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--out", type=Path, default=Path("hud_annot"))
    ap.add_argument("--captures", type=int, default=10)
    ap.add_argument("--total", type=int, default=48,
                    help="frames to actually put on the sheets, balanced across "
                         "the health strata. 0 keeps everything collected.")
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="source frames per capture; 0 decodes the whole thing. "
                         "Do NOT lower this to save memory: `decode_hud` takes "
                         "the *first* N frames, and health only gets low at the "
                         "END of a round -- capping at 2400 (40 s) produced a "
                         "sheet whose lowest health was 0.20, with nothing in "
                         "the band the sheet exists to measure. The memory cost "
                         "is ~4 GB transient for one capture and is freed before "
                         "the next, which is what the .copy() below is for.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    import cv2

    manifest = a.corpus / "val" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rng.shuffle(rows)

    picked = []
    for r in rows[: a.captures]:
        try:
            video, _ = capture_paths(r, manifest)
            hud = decode_hud(str(video), a.max_frames)
            if len(hud) < 200:
                continue
            tr = read_trace(hud, smooth=3)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}")
            continue
        ok = ~(tr.healing | tr.flash | tr.clamped)
        for lo, hi, n in STRATA:
            # Either player in the band; the tile shows both bars anyway.
            m = ok & (((tr.hp1 >= lo) & (tr.hp1 < hi)) |
                      ((tr.hp2 >= lo) & (tr.hp2 < hi)))
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            take = rng.choice(idx, size=min(n, len(idx)), replace=False)
            for i in take:
                picked.append({
                    "capture": r["replay_id"], "frame": int(i),
                    "hp1": float(tr.hp1[i]), "hp2": float(tr.hp2[i]),
                    "spirit1": float(tr.spirit1[i]), "spirit2": float(tr.spirit2[i]),
                    "cards1": float(tr.cards1[i]), "cards2": float(tr.cards2[i]),
                    # .copy() is load-bearing. `hud[i]` is a *view* into the
                    # whole decoded capture, so keeping one alive keeps all
                    # 6000 native-resolution frames -- about 4 GB -- alive with
                    # it. Retaining forty such views per capture OOM-killed this
                    # script on the third capture, silently: the kernel kills the
                    # process, so there is no traceback and the log simply stops.
                    "_img": np.array(hud[i], copy=True),
                })
        del hud, tr
        print(f"  {r['replay_id']}: {len(picked)} cumulative", flush=True)

    # Trim to the requested budget, keeping the strata balanced rather than
    # taking a flat random sample: a flat sample would restore the corpus's own
    # health distribution, and the low-health band -- 8.5% of frames and the
    # entire reason for this sheet -- would nearly vanish from it.
    if a.total and len(picked) > a.total:
        by_band: dict[int, list] = {}
        for p in picked:
            hp = min(p["hp1"], p["hp2"])
            b = next((i for i, (lo, hi, _) in enumerate(STRATA) if lo <= hp < hi),
                     len(STRATA) - 1)
            by_band.setdefault(b, []).append(p)
        quota, kept = a.total // max(len(by_band), 1), []
        for b, items in sorted(by_band.items()):
            rng.shuffle(items)
            kept += items[:quota]
        # Any shortfall (a thin band) is topped up from whatever is left over.
        left = [p for p in picked if p not in kept]
        rng.shuffle(left)
        picked = kept + left[: max(0, a.total - len(kept))]

    rng.shuffle(picked)                      # so strata are not visually grouped
    for n, p in enumerate(picked):
        p["id"] = n

    tiles = [tile(p.pop("_img")) for p in picked]
    n_sheet = 0
    for s in range(0, len(tiles), a.per_sheet):
        chunk = tiles[s : s + a.per_sheet]
        labs = [f"#{picked[s + k]['id']:03d}" for k in range(len(chunk))]
        cv2.imwrite(str(a.out / f"sheet_{n_sheet:02d}.png"),
                    sheet(chunk, labs)[:, :, ::-1])
        n_sheet += 1

    # The answer key is written separately and is NOT on the sheets.
    (a.out / "answer_key.json").write_text(json.dumps(picked, indent=1))
    with (a.out / "annotations.csv").open("w") as fh:
        fh.write("# Fill in what you SEE. Leave a cell BLANK if you cannot tell --\n"
                 "# 'unreadable' is a real answer and is more useful than a guess.\n"
                 "#\n"
                 "# hp1, hp2   : the LENGTH of the YELLOW segment as a percentage,\n"
                 "#              read off the ruler under each bar (0-100).\n"
                 "#              Measure the length, not where it sits: the bars\n"
                 "#              deplete outward from the centre, so the yellow is\n"
                 "#              not always flush with the left edge.\n"
                 "#              RED is combo damage in progress, NOT health --\n"
                 "#              do not include it.\n"
                 "#\n"
                 "# spirit1,2  : number of LIT (blue) hexagons out of 5.\n"
                 "#              Dark/black = spent. Purple = partly recovered,\n"
                 "#              count it as 0.5. So: 0, 0.5, 1, 1.5 ... 5\n"
                 "#\n"
                 "# Cards are deliberately not annotated: they were already\n"
                 "# measured as undecodable, so your time is better spent here.\n"
                 "id,hp1,hp2,spirit1,spirit2\n")
        for p in picked:
            fh.write(f"{p['id']},,,,\n")

    # Report what actually landed in each band, and complain loudly if the band
    # this sheet exists to measure is empty. The first run of this script wrote
    # 48 tidy frames and printed nothing wrong while containing zero frames below
    # health 0.20 -- the sample was capped to the first 40 s of each capture, and
    # health only falls at the end of a round. A sheet that cannot see the
    # low-health regime is indistinguishable from one that can, unless it says so.
    got = {i: 0 for i in range(len(STRATA))}
    for p in picked:
        hp = min(p["hp1"], p["hp2"])
        for i, (lo, hi, _) in enumerate(STRATA):
            if lo <= hp < hi:
                got[i] += 1
                break
    print("\nhealth strata actually on the sheets (by the lower of the two bars):")
    for i, (lo, hi, want) in enumerate(STRATA):
        flag = "  <-- EMPTY" if got[i] == 0 else ""
        print(f"   {lo:.2f}-{hi:.2f}  {got[i]:3d} frames{flag}")
    if got[0] == 0:
        print("\nWARNING: no frames below health "
              f"{STRATA[0][1]:.2f}. The low-health band is the point of this "
              "sheet -- scripts/probe_reliability.py measured the probe's noise "
              "at 0.25 there against 0.13 elsewhere. Re-run with --max-frames 0 "
              "and more --captures; a sheet without this band cannot answer the "
              "question it was made for.")

    print(f"\n{len(picked)} frames over {n_sheet} sheets -> {a.out}")
    print("Sheets are BLIND: hud.py's readings are in answer_key.json and are "
          "deliberately not drawn, so agreeing with it is evidence rather than "
          "anchoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
