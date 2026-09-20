"""Does reading the health bar directly beat asking the encoder for it?

The encoder sees 224x224 downscaled from 640x480. At that size the health bar
is a thin strip and the spirit hexagons are six pixels wide, so it cannot read
either -- measured on two real matches it is beaten by a constant that predicts
the MEAN of both bars, and it explains 8.6% / 0.0% of the variance in the
difference between the players, which is the only part that says who is ahead.

`hud.read_frame` reads the bar where it is, at full resolution. This scores it
against the capture's own state sidecar, on the SAME frames, and reports both
metrics so the comparison is like for like:

    R2 per row      what the encoder was gated on. Flattered by the +0.73
                    correlation between the two players' bars: predicting their
                    average scores ~0.87 here while knowing nothing about
                    either player.
    R2 on p1 - p2   the discriminative part. This is the number that matters
                    and the one nobody was looking at.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.hud import detect_flip, read_frame          # noqa: E402
from sokubot.data.state import CH, read_state                 # noqa: E402

W = H = 480


def decode(path: Path, max_frames: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    n = len(out) // (W * H * 3)
    return np.frombuffer(out, np.uint8, n * W * H * 3).reshape(n, H, W, 3)


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss = ((true - true.mean()) ** 2).sum()
    return float(1 - ((pred - true) ** 2).sum() / max(ss, 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--captures", type=int, default=6)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    dirs = sorted({p.parent for p in a.corpus.rglob("video.mp4")})
    if not dirs:
        print(f"no captures under {a.corpus}")
        return 2
    rng = np.random.default_rng(0)
    dirs = [dirs[i] for i in rng.choice(len(dirs), min(a.captures, len(dirs)),
                                        replace=False)]

    rows, flips = [], []
    for d in dirs:
        sc = d / "state.csv.gz"
        if not sc.exists():
            continue
        try:
            st, _pr, _ac, valid = read_state(sc)
        except (ValueError, OSError) as e:
            print(f"  {d.name}: state unreadable ({e})")
            continue
        fr = decode(d / "video.mp4", a.frames)
        n = min(len(fr), len(st), a.frames)
        if n < 200:
            print(f"  {d.name}: only {n} frames")
            continue
        # Sample ACROSS the clip: the opening frames are loading and fades.
        probe = fr[np.linspace(0, n - 1, min(n, 60), dtype=int)]
        flip = detect_flip(probe)
        if flip is None:
            print(f"  {d.name}: HUD not found in EITHER orientation -- skipped")
            continue
        flips.append(flip)
        hp = np.stack([read_frame(fr[i], flip)[0] for i in range(n)])
        sp = np.stack([read_frame(fr[i], flip)[1] for i in range(n)])
        # Frames are selected by TRUTH, never by what the reader said, or the
        # score would be measuring its own confidence. A capture opens on a
        # loading screen and closes on the KO heal; in both the bar is not a
        # bar, and neither reader nor encoder is being asked about those.
        live = (st[:n, 0, CH["hp"]] > 0.02) & (st[:n, 1, CH["hp"]] > 0.02)
        ok = valid[:n] & live
        t_hp = st[:n, :, CH["hp"]][ok]
        t_sp = st[:n, :, CH["spirit"]][ok]
        rows.append((d.name, hp[ok], sp[ok], t_hp, t_sp))
        print(f"  {d.name}: {ok.sum()}/{n} frames in a live round, "
              f"flip={flip}", flush=True)

    if not rows:
        print("nothing readable")
        return 3
    print(f"\norientation: {sum(flips)}/{len(flips)} captures read bottom-up")

    P_hp = np.concatenate([r[1] for r in rows])
    P_sp = np.concatenate([r[2] for r in rows])
    T_hp = np.concatenate([r[3] for r in rows])
    T_sp = np.concatenate([r[4] for r in rows])

    print(f"\n{len(T_hp)} frames from {len(rows)} captures\n")
    print(f"{'channel':<9}{'R2 per row':>12}{'R2 if avg':>12}"
          f"{'R2 on p1-p2':>13}{'median err':>12}")
    res = {}
    for name, P, T in (("hp", P_hp, T_hp), ("spirit", P_sp, T_sp)):
        avg = np.tile(T.mean(1, keepdims=True), (1, 2))
        d_p, d_t = P[:, 0] - P[:, 1], T[:, 0] - T[:, 1]
        e = float(np.median(np.abs(P - T)))
        res[name] = {"r2_row": r2(P, T), "r2_avg_baseline": r2(avg, T),
                     "r2_diff": r2(d_p, d_t), "median_abs_err": e,
                     "n": int(len(T))}
        print(f"{name:<9}{res[name]['r2_row']:>12.3f}"
              f"{res[name]['r2_avg_baseline']:>12.3f}"
              f"{res[name]['r2_diff']:>13.3f}{e:>12.4f}")

    print("\nThe encoder, measured live on the same two quantities:")
    print(f"{'hp':<9}{0.560:>12.3f}{0.693:>12.3f}{0.086:>13.3f}")
    print(f"{'spirit':<9}{0.377:>12.3f}{0.565:>12.3f}{0.000:>13.3f}")

    if a.out:
        a.out.write_text(json.dumps(
            {"captures": [r[0] for r in rows], "flip_bottom_up": sum(flips),
             "results": res}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
