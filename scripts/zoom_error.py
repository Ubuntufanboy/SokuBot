"""Is the encoder's positional error a LOCALISATION failure or a ZOOM failure?

`future_paper/findings/04` measured live `x` error rising with player
separation -- 117 u at 0-100 u apart, 300 u at 450+ -- and attributed it to
characters shrinking as the camera pulls back. That is a mechanism worth
separating, because the two readings imply completely different fixes:

  * If the encoder localises to a roughly constant error IN PIXELS and the
    world-unit error grows only because each pixel is worth more world units at
    zoom-out, then the network is not getting worse at seeing. The world figure
    is a projection artefact, and the fix is to predict SCREEN coordinates --
    which is what a spatial soft-argmax head can actually do -- and recover the
    camera scale separately, as one global scalar per frame rather than as an
    implicit multiply folded into every position.

  * If the pixel error itself grows with zoom-out, the model genuinely loses
    small characters and the fix is resolution or architecture.

Nothing distinguished these before, because the corpus had no camera. It does
now (`sokubot/data/state.py:CAMERA_COLUMNS`), and `to_screen` is validated to
pixels, so the same prediction can be scored in both spaces at once.

Errors are reported against the TRUE camera rect. That is deliberate: the
question is how well the encoder placed the character on screen given where the
camera actually was, not whether it also inferred the camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import (CH, GAME_W, STAGE_SPAN, read_camera,  # noqa: E402
                                read_state, to_screen, valid_camera)
from sokubot.live.visionstate import VisionState                      # noqa: E402
from scripts.train_encoder import SUPERVISED                          # noqa: E402


def _np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--per-replay", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import cv2

    rng = np.random.default_rng(a.seed)
    dirs = []
    for w in sorted(a.corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    if not dirs:
        dirs = [d for d in sorted(a.corpus.iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    dirs = [d for d in dirs if (d / "video.mp4").exists()
            and (d / "state.csv.gz").exists()]
    pick = [dirs[i] for i in rng.choice(len(dirs), min(a.replays, len(dirs)),
                                        replace=False)]

    vss = [VisionState.load(c, a.device) for c in a.ckpt]
    delta, size = vss[0].delta, vss[0].size
    print(f"{len(pick)} replays, delta {delta}, size {size}", flush=True)

    X, S, CAMS = [], [], []
    for n, d in enumerate(pick):
        try:
            st, _p, _act, valid = read_state(d / "state.csv.gz")
        except (ValueError, OSError):
            continue
        cam = read_camera(d / "state.csv.gz")
        if len(cam) < len(st):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1) & valid_camera(cam)[:len(st)]
        idx = np.flatnonzero(ok)
        idx = idx[idx >= delta]
        if len(idx) < a.per_replay:
            continue
        rows = np.sort(rng.choice(idx, a.per_replay, replace=False))

        cap = cv2.VideoCapture(str(d / "video.mp4"))
        frames, keep = [], []
        for f in rows:
            pair = []
            for off in (delta, 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - off)
                got, fr = cap.read()
                if not got:
                    break
                fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
                pair.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(pair) != 2:
                continue
            frames.append(np.concatenate(pair, axis=2).transpose(2, 0, 1))
            keep.append(f)
        cap.release()
        if not frames:
            continue
        keep = np.array(keep)
        s_sel = st[keep]
        # Same left/right ordering the encoder was trained to emit.
        p1_left = (s_sel[:, 0, CH["dx"]] > 0)
        order = np.where(p1_left[:, None], np.array([0, 1]), np.array([1, 0]))
        take = np.take_along_axis(s_sel, order[:, :, None], axis=1)
        X.append(np.stack(frames))
        S.append(take.astype(np.float32))
        CAMS.append(cam[keep])
        print(f"  [{n}] {d.name}: {len(frames)}", flush=True)

    X = np.concatenate(X); S = np.concatenate(S); CAMS = np.concatenate(CAMS)
    print(f"\n{len(X)} frames\n", flush=True)

    sup = [CH[n] for n in SUPERVISED]
    Y = S[:, :, sup].reshape(len(S), -1)
    ix, iy = SUPERVISED.index("x"), SUPERVISED.index("y")
    ns = len(SUPERVISED)

    # Camera width in world units: the whole point. A wide rect is a pulled-back
    # camera, where one pixel buys more world units.
    from sokubot.data.state import CAM
    width = CAMS[:, CAM["cam_right"]] - CAMS[:, CAM["cam_left"]]

    for vs, cpath in zip(vss, a.ckpt):
        mu = np.asarray(_np(vs.mu), np.float32)
        sd = np.asarray(_np(vs.sd), np.float32)
        preds = []
        with torch.no_grad():
            for j in range(0, len(X), 128):
                xb = torch.from_numpy(X[j:j+128]).to(a.device).float().div(255.0)
                out = vs.net(xb, return_feat=True)
                o = out[0] if isinstance(out, tuple) else out
                preds.append(o[:, :vs.n_state].cpu().numpy())
        pred = np.concatenate(preds) * sd[None, :vs.n_state] + mu[None, :vs.n_state]

        pw = np.stack([pred[:, r * ns + ix] for r in (0, 1)], 1) * STAGE_SPAN
        tw = np.stack([Y[:, r * ns + ix] for r in (0, 1)], 1) * STAGE_SPAN
        pwy = np.stack([pred[:, r * ns + iy] for r in (0, 1)], 1) * STAGE_SPAN
        twy = np.stack([Y[:, r * ns + iy] for r in (0, 1)], 1) * STAGE_SPAN

        psx, _ = to_screen(pw, pwy, CAMS)
        tsx, _ = to_screen(tw, twy, CAMS)
        # Screen x is normalised to [-1, 1] across the frame, so half the game's
        # width converts it to pixels of the 640px render.
        err_world = np.abs(pw - tw).mean(1)
        err_pix = np.abs(psx - tsx).mean(1) * (GAME_W / 2.0)

        print(f"=== {cpath.name} " + "=" * 40)
        q = np.quantile(width, [0, .25, .5, .75, 1.0])
        print(f"{'camera width (world u)':<26} {'n':>6} {'world err':>10} "
              f"{'screen err (px)':>16}")
        for lo, hi in zip(q[:-1], q[1:]):
            m = (width >= lo) & (width <= hi if hi == q[-1] else width < hi)
            if m.sum() == 0:
                continue
            print(f"  {lo:7.0f} - {hi:7.0f} {'':<8} {int(m.sum()):6d} "
                  f"{err_world[m].mean():10.1f} {err_pix[m].mean():16.2f}")
        rw = np.corrcoef(width, err_world)[0, 1]
        rp = np.corrcoef(width, err_pix)[0, 1]
        print(f"  corr(width, world err) {rw:+.3f}   "
              f"corr(width, screen err) {rp:+.3f}")
        wide, narrow = width >= q[3], width <= q[1]
        print(f"  widest quartile vs narrowest: "
              f"world x{err_world[wide].mean() / max(err_world[narrow].mean(), 1e-9):.2f}, "
              f"screen x{err_pix[wide].mean() / max(err_pix[narrow].mean(), 1e-9):.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
