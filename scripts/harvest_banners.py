"""Harvest round-boundary banner candidates and build a labelling sheet.

    python -m scripts.harvest_banners --corpus ~/corpus --out ~/banners --captures 12

WHY A CLASSIFIER RATHER THAN MORE COLOUR THRESHOLDS
----------------------------------------------------
The reward's KO signal is currently inferred from health crossing 0.06, which
`scripts/anchored_ko_test.py` shows is hopeless: the probe reads health 8.6x
worse than assuming it never changes, and no arm detects a real KO at all.

But the game *announces* the event. It draws a huge red kanji plus blue outlined
text across the centre of the screen -- and it does the same for the start of a
round. Measured on the live clip, blue-text coverage in the centre band is 0.369
for KNOCK OUT, 0.113 for DOWN and 0.011 a second earlier. That is a 33x
separation, so finding *a* banner is easy.

Telling them apart is not. The same style is used for at least ROUND 1, START,
DOWN and KNOCK OUT, and coverage alone conflates them -- ROUND 1 measured 0.24 on
the corpus, above DOWN's 0.11. Separating four sprite families across every stage,
character and weather by hand-tuned colour rules is exactly the work that left
`hud.py`'s spellcard reader marked "BLOCKED BY CAPTURE RESOLUTION". A small CNN
on a fixed crop is the right tool, and it needs a few dozen labels.

WHAT THIS PRODUCES
------------------
Candidates are found by blue-text coverage, which over-triggers on purpose: a
cheap recall-first filter, with the classifier doing precision. Each sheet shows
the full frame (so the banner is legible in context) and the band crop the model
will actually see, with the coverage score.

The score is printed, unlike in `make_hud_annotation.py` where the instrument's
reading was hidden. That sheet was validating an existing reader, where showing
its answer would have anchored the annotator. Here nothing is being validated --
the score is only how the candidate was found, and it is the *label* that is new
information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokubot.config import Config
from scripts.horizon_ablation import capture_paths, decode_hud

# The band is defined ONCE, in display orientation -- the way a person sees the
# frame -- and converted for the stored orientation where it is needed. The first
# version kept a single constant in *stored* coordinates and used it for both,
# which scored the right region and cropped the mirror of it: the saved crops
# were of the health bars at the top of the screen while the score came from the
# banner at the bottom. Everything looked plausible, and the labelling sheet
# would have trained a classifier on the wrong pixels.
#
# Measured on the live 640x480 clip: KNOCK OUT text spans y 350-420, START sits
# lower at ~424, so the band runs 300-465 with margin. The corpus is squashed to
# 480 wide, hence x * 0.75.
FRAME = 480
DISP_Y = (300, 465)                       # as seen
DISP_X = (int(60 * 0.75), int(580 * 0.75))
# Captures are stored vertically flipped (data/hud.py flips before reading).
STORE_Y = (FRAME - DISP_Y[1], FRAME - DISP_Y[0])
BAND_X = DISP_X
MIN_SCORE = 0.05
MIN_GAP = 30                             # frames; one label per banner event


def banner_score(frames: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Fraction of the centre band that is saturated blue outlined text.

    Chunked, and that is not an optimisation. Slicing the band out of a whole
    capture and casting it to int16 materialises
    ``n_frames x 165 x 390 x 3 x 2`` bytes -- about 7 GB for a long replay, on
    top of the 5 GB the decode already holds, before the per-channel differences
    add their own int16 temporaries. That OOM-killed this script at 15 GB with no
    traceback, because the kernel kills the process rather than raising.

    This is the fourth bug of this exact shape in this codebase. The rule the
    others also violate: anything that touches every frame of a capture at once
    needs to be chunked, however cheap the per-frame arithmetic looks.
    """
    out = np.empty(len(frames), dtype=np.float32)
    for i in range(0, len(frames), chunk):
        b = frames[i : i + chunk, STORE_Y[0]:STORE_Y[1],
                   BAND_X[0]:BAND_X[1]].astype(np.int16)
        r, g, bl = b[..., 0], b[..., 1], b[..., 2]
        out[i : i + chunk] = ((bl > 140) & ((bl - r) > 50)
                              & ((bl - g) > 30)).mean(axis=(1, 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--out", type=Path, default=Path("banners"))
    ap.add_argument("--captures", type=int, default=12)
    ap.add_argument("--per-capture", type=int, default=0,
                    help="cap candidates per capture; 0 takes all that survive "
                         "--min-run. Capping randomly was throwing away most of "
                         "the real banners: each capture holds 2-3 DOWNs and a "
                         "KNOCK OUT, and a random six were mostly flicker.")
    ap.add_argument("--min-run", type=int, default=40,
                    help="frames the detection must persist. Real banners hold "
                         "for ~2 s (120 frames) while gameplay flashes flicker: "
                         "measured over 84 candidates, mean score rises from "
                         "0.096 at runs of 1-5 frames to 0.283 at 60-200, and "
                         "4 of the 10 longest were real banners against roughly "
                         "none of the short ones.")
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "crops").mkdir(exist_ok=True)
    import cv2
    rng = np.random.default_rng(a.seed)

    man = a.corpus / "val" / "manifest.jsonl"
    rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    rng.shuffle(rows)

    picked = []
    for r in rows[: a.captures]:
        try:
            video, _ = capture_paths(r, man)
            hud = decode_hud(str(video), 0)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}")
            continue
        s = banner_score(hud)
        hi = np.flatnonzero(s > MIN_SCORE)
        if len(hi):
            groups = [g for g in np.split(hi, np.flatnonzero(np.diff(hi) > MIN_GAP) + 1)
                      if len(g) >= a.min_run]
            rng.shuffle(groups)
            for gp in (groups if a.per_capture <= 0 else groups[: a.per_capture]):
                i = int(gp[int(np.argmax(s[gp]))])       # peak of the event
                picked.append({
                    "capture": r["replay_id"], "frame": int(i),
                    "score": float(s[i]), "run": int(len(gp)),
                    # .copy(): a view keeps the whole 4.7 GB capture alive, which
                    # OOM-killed two earlier scripts silently.
                    "_img": np.array(hud[i][::-1], copy=True),
                })
        print(f"  {r['replay_id'][:12]}: {len(picked)} cumulative", flush=True)
        del hud, s

    if not picked:
        raise SystemExit("no banner candidates found; lower --min-score")
    rng.shuffle(picked)
    for n, p in enumerate(picked):
        p["id"] = n

    for s0 in range(0, len(picked), a.per_sheet):
        tiles = []
        for p in picked[s0 : s0 + a.per_sheet]:
            img = p["_img"]
            full = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
            crop = img[DISP_Y[0]:DISP_Y[1], BAND_X[0]:BAND_X[1]]
            crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / crop.shape[1])),
                              interpolation=cv2.INTER_NEAREST)
            pad = np.zeros((max(0, 300 - crop.shape[0]), 300, 3), np.uint8)
            col = np.concatenate([full, crop, pad], axis=0)[:380]
            lab = np.zeros((26, 300, 3), np.uint8)
            cv2.putText(lab, f"#{p['id']:03d}  s={p['score']:.2f}", (5, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(np.concatenate([lab, col], axis=0))
        cv2.imwrite(str(a.out / f"sheet_{s0 // a.per_sheet:02d}.png"),
                    np.concatenate(tiles, axis=1)[:, :, ::-1])

    for p in picked:
        img = p.pop("_img")
        np.save(a.out / "crops" / f"{p['id']:03d}.npy",
                img[DISP_Y[0]:DISP_Y[1], BAND_X[0]:BAND_X[1]])
    (a.out / "candidates.json").write_text(json.dumps(picked, indent=1))
    with (a.out / "labels.csv").open("w") as fh:
        fh.write("# One row per candidate. Write ONE of:\n"
                 "#   none      - no banner, the detector misfired\n"
                 "#   round     - ROUND 1 / ROUND 2 ... (start of a round)\n"
                 "#   start     - START\n"
                 "#   down      - DOWN (a knockdown; the match continues)\n"
                 "#   knockout  - KNOCK OUT (the match is over)\n"
                 "#   other     - a banner none of the above describes\n"
                 "# The top image is the whole frame, the bottom is the crop the\n"
                 "# model will see. Leave blank if you genuinely cannot tell.\n"
                 "id,label\n")
        for p in picked:
            fh.write(f"{p['id']},\n")
    print(f"\n{len(picked)} candidates over "
          f"{(len(picked) + a.per_sheet - 1)//a.per_sheet} sheets -> {a.out}")
    print("Crops are saved as .npy so the classifier trains on exactly the "
          "pixels the sheet showed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
