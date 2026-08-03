"""Render an expert-review video: tracked HUD stats overlaid on real gameplay.

    python -m scripts.review_video --corpus /root/corpus --out /root/review.mp4

Clips are chosen where the extractor is *least* confident, not where it looks
good, so a reviewer's time buys the most correction per minute. Every clip is
captioned with what to check and why it was picked.

Selection criteria, in descending order of doubt:

  repaired    a reading the flash filter rewrote. Did it kill a phantom drop, or
              erase real damage?
  regen       health rising mid-round. Calm weather grants a healing spotlight,
              so this may be correct -- or a misread.
  combo       the red-life channel, which the wiki confirms is "damage the
              current combo has done" but which has never been checked visually.
  ko          round boundaries, which delimit everything else.
  damage      the largest ordinary damage events, as a control.

Frames are flipped upright for viewing: captures are stored vertically flipped
(see sokubot.data.hud).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sokubot.data.hud import (FILL_ROWS, P1_HP_X, P2_HP_X, P1_SPIRIT_X,
                              P2_SPIRIT_X, SPIRIT_ROWS, damage_events, read_trace)

W = H = 480
PANEL = 132
FPS = 60


def decode(path: str) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True)
    n = len(p.stdout) // (W * H * 3)
    return np.frombuffer(p.stdout[:n * W * H * 3], dtype=np.uint8).reshape(n, H, W, 3)


def pick_clips(t, d1, d2, n, clip_len, per_kind=2):
    """Return [(start, kind, note)] over the moments with the most doubt."""
    half = clip_len // 2
    out, taken = [], []

    def add(idx, kind, note):
        for i in idx:
            i = int(i)
            if any(abs(i - j) < clip_len for j in taken):
                continue
            if i < half or i > n - half:
                continue
            taken.append(i)
            out.append((i - half, kind, note))
            return True
        return False

    rep = np.where(t.flash)[0]
    if len(rep):
        groups = np.split(rep, np.where(np.diff(rep) > 30)[0] + 1)
        groups.sort(key=len, reverse=True)
        for g in groups[:per_kind]:
            add([int(np.median(g))], "REPAIRED",
                "flash filter rewrote health here - phantom drop, or real damage erased?")

    hp = t.hp1
    rise = np.diff(hp)
    rise[t.healing[1:]] = 0
    for i in np.argsort(rise)[::-1][:per_kind * 6]:
        if rise[i] > 0.004 and add([i], "REGEN",
                                   "P1 health RISING mid-round - Calm weather spotlight, or misread?"):
            break

    for i in np.argsort(t.combo1)[::-1][:per_kind * 6]:
        if add([i], "COMBO", "large RED life on P1 - does the red match the combo just landed?"):
            if sum(1 for o in out if o[1] == "COMBO") >= per_kind:
                break

    ko = np.where(np.diff(t.healing.astype(int)) == 1)[0]
    for i in ko[:per_kind]:
        add([int(i)], "KO", "round boundary - is this really a KO/round reset?")

    for i in np.argsort(d1)[::-1][:per_kind * 4]:
        if add([i], "DAMAGE", "largest damage events - control, should be uncontroversial"):
            if sum(1 for o in out if o[1] == "DAMAGE") >= per_kind:
                break
    out.sort()
    return out


def draw(frame, t, d1, d2, i, kind, note, replay):
    """frame: uint8 [480,480,3] already upright. Returns a taller annotated frame."""
    img = Image.new("RGB", (W, H + PANEL), (14, 12, 20))
    img.paste(Image.fromarray(frame), (0, 0))
    dr = ImageDraw.Draw(img)

    # mark the regions the extractor actually reads
    dr.rectangle([P1_HP_X[0], FILL_ROWS[0], P1_HP_X[1], FILL_ROWS[1]], outline=(0, 255, 120))
    dr.rectangle([P2_HP_X[0], FILL_ROWS[0], P2_HP_X[1], FILL_ROWS[1]], outline=(0, 255, 120))
    dr.rectangle([P1_SPIRIT_X[0], SPIRIT_ROWS[0], P1_SPIRIT_X[1], SPIRIT_ROWS[1]],
                 outline=(90, 170, 255))
    dr.rectangle([P2_SPIRIT_X[0], SPIRIT_ROWS[0], P2_SPIRIT_X[1], SPIRIT_ROWS[1]],
                 outline=(90, 170, 255))

    y0 = H + 6
    dr.text((8, y0), f"{kind}   {replay}   frame {i}", fill=(217, 164, 65))
    dr.text((8, y0 + 14), note[:78], fill=(180, 175, 195))

    def bar(y, label, val, col, extra=""):
        dr.text((8, y), f"{label} {val:5.3f}{extra}", fill=(235, 229, 242))
        x0, wdt = 150, 300
        dr.rectangle([x0, y + 3, x0 + wdt, y + 12], outline=(70, 62, 88))
        if val > 0:
            dr.rectangle([x0, y + 3, x0 + int(wdt * min(1, val)), y + 12], fill=col)

    bar(y0 + 34, "P1 life  ", t.hp1[i], (230, 190, 40))
    bar(y0 + 50, "P1 red   ", t.combo1[i], (200, 60, 70))
    bar(y0 + 66, "P2 life  ", t.hp2[i], (230, 190, 40))
    bar(y0 + 82, "P2 red   ", t.combo2[i], (200, 60, 70))
    bar(y0 + 98, "P1 spirit", t.spirit1[i], (80, 150, 235))
    bar(y0 + 114, "P2 spirit", t.spirit2[i], (80, 150, 235))

    flags = []
    if t.flash[i]:
        flags.append("REPAIRED")
    if t.healing[i]:
        flags.append("ROUND-RESET")
    if d1[i] > 0:
        flags.append(f"dmg P1 -{d1[i]:.3f}")
    if d2[i] > 0:
        flags.append(f"dmg P2 -{d2[i]:.3f}")
    if flags:
        dr.text((300, y0 + 34), "  ".join(flags)[:26], fill=(255, 110, 110))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--out", type=Path, default=Path("/root/review.mp4"))
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--clip-seconds", type=float, default=7.0)
    ap.add_argument("--target-mb", type=float, default=24.0)
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.corpus / "manifest.jsonl").read_text().splitlines()]
    rows = rows[7:7 + args.replays]
    clip_len = int(args.clip_seconds * FPS)

    frames_out = []
    for r in rows:
        fr = decode(r["video"])
        t = read_trace(fr, smooth=3)
        d1, d2 = damage_events(t, 1), damage_events(t, 2)
        n = len(fr)
        for start, kind, note in pick_clips(t, d1, d2, n, clip_len):
            up = fr[start:start + clip_len][:, ::-1]
            for k in range(len(up)):
                frames_out.append(draw(up[k], t, d1, d2, start + k, kind, note,
                                       r["replay_id"][:12]))
            if len(frames_out) / FPS >= args.seconds:
                break
        if len(frames_out) / FPS >= args.seconds:
            break

    secs = len(frames_out) / FPS
    kbps = int((args.target_mb * 8192) / max(secs, 1) * 0.92)
    print(f"{len(frames_out)} frames = {secs:.0f}s -> {kbps} kbit/s", flush=True)

    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H + PANEL}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k",
         "-maxrate", f"{int(kbps*1.3)}k", "-bufsize", f"{kbps*2}k",
         "-pix_fmt", "yuv420p", str(args.out)], stdin=subprocess.PIPE)
    for f in frames_out:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} — {secs:.0f}s, {mb:.1f} MB")


if __name__ == "__main__":
    main()
