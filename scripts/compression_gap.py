"""Is the encoder brittle to the one step the live path does not reproduce?

THE ASYMMETRY
-------------
Every corpus frame went through ``libx264 -preset veryfast -crf 26 -pix_fmt
yuv420p`` (``SokuFrameExtractor/runner/encode.py``). ``sokubot/live/capture.py``
documents the corpus chain in its own docstring --

    game backbuffer 640x480 BGRA
      -> vflip, scale=480:480:flags=lanczos   -> h264 crf 26
      -> decode, scale=224:224 -> rgb24, /255

-- and then reproduces every step of it EXCEPT the encode: the live grabber
emits ``-f rawvideo -pix_fmt rgb24`` straight out of the filter graph. So the
encoder trained exclusively on frames carrying DCT quantisation and 4:2:0
chroma subsampling, and at inference it is handed full-chroma, unquantised RGB.

That is a systematic train/deploy domain gap, and no offline metric in this
repo can see it, because every offline metric reads back the same compressed
corpus the model trained on. It would degrade position, identity and small
bright objects at once -- which is the observed live failure set.

TWO WAYS TO RUN IT
------------------
**Exact (point `--corpus` at a `runner.collect --lossless` tree).** The capture
writes an RGB master with no quantisation and no chroma subsampling, and this
script derives the corpus rendering from that same master with
`runner/encode.py`'s arguments verbatim. Both arms are then the SAME game
frames, the state csv indexes both without any alignment step, and the `base`
row is the live-like rendering while `crf26` is the training-like one. The
delta is the imaging gap, measured in the correct direction.

**Sensitivity (point `--corpus` at the ordinary corpus).** A compressed frame
cannot be uncompressed, so here the only available move is to push the corpus
one more generation through its own codec and see whether the readouts move.
That runs the WRONG WAY -- more compression, where live has less -- and is only
symmetric if the response is locally smooth, which is an assumption rather than
a result. Read it as "the axis matters / does not matter", never as a size
estimate. Its one virtue is that `chroma` isolates the 4:2:0 subsample from the
DCT quantisation, which the exact run cannot separate.

In the sensitivity mode the frames come from the same replays the cache was
built from, so some are training frames: the absolute R2 per arm is optimistic
and only the PAIRED DELTA on identical frames should be read. The exact mode
should be pointed at replays the encoder never saw, so that memorising the
training rendering cannot masquerade as robustness to it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, read_state          # noqa: E402
from sokubot.live.visionstate import VisionState       # noqa: E402
from scripts.train_encoder import SUPERVISED, _decide_acc  # noqa: E402


def _np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def reencode(src: Path, dst: Path, crf: int) -> None:
    """One pass through the corpus's own encoder settings, frame count intact.

    These are `runner/encode.py:build_command`'s non-VAAPI arguments verbatim,
    VBV cap included. The cap is not a detail: `-maxrate 1500k` over a 2 s
    window only engages in the busiest scenes, so the corpus is quality-starved
    exactly where the screen is full of projectiles and effects. That is a
    content-correlated degradation, and it is not present live at all.

    ``-fps_mode passthrough`` for the same reason ``data/soku.py`` needs
    ``-vsync 0``: otherwise ffmpeg may duplicate frames to honour a nominal
    rate and the indices stop lining up with the state csv.
    """
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-maxrate", "1500k", "-bufsize", "3000k", "-g", "120",
         "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(dst)],
        check=True)


def chroma_roundtrip(bgr: np.ndarray) -> np.ndarray:
    """RGB -> 4:2:0 -> RGB, the colour half of what h264 does, on its own.

    Isolating it matters: identity is a colour task (the roster is separated far
    more by palette than by silhouette at 224px), so if the gap is chroma rather
    than DCT ringing, the fix is a nearly free filter rather than a re-encode.
    """
    import cv2
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420),
                        cv2.COLOR_YUV2BGR_I420)


def grab(video: Path, rows: np.ndarray, delta: int, size: int,
         chroma: bool) -> np.ndarray | None:
    """The exact frame preparation `train_encoder.build` uses, arm-transformed.

    The arm transform is applied at 480, i.e. where the corpus compression
    happened, before the downscale -- applying it after would measure a
    different thing entirely.
    """
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = []
    for f in rows:
        pair = []
        for off in (delta, 0):              # older first, then current
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - off)
            got, fr = cap.read()
            if not got:
                break
            if chroma:
                fr = chroma_roundtrip(fr)
            fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
            pair.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if len(pair) != 2:
            out.append(None)
            continue
        out.append(np.concatenate(pair, axis=2).transpose(2, 0, 1))
    cap.release()
    if any(o is None for o in out):
        return None
    return np.stack(out)


@torch.no_grad()
def readouts(vs: VisionState, X: np.ndarray, batch: int = 128):
    """-> (state prediction [N, n_state], p1_left logit [N], char logits or None)."""
    pred, idl, chl = [], [], []
    for j in range(0, len(X), batch):
        xb = torch.from_numpy(X[j:j + batch]).to(vs.device).float().div(255.0)
        out = vs.net(xb, return_feat=True)
        o, feat = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        pred.append(o[:, :vs.n_state].cpu())
        idl.append(o[:, vs.n_state + vs.n_proj].cpu())
        if vs.chead is not None:
            chl.append(vs.chead(feat)[0].cpu())
    return (torch.cat(pred).numpy(), torch.cat(idl).numpy(),
            torch.cat(chl).numpy() if chl else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--chars", type=Path, default=None,
                    help="corpus_chars.json; enables the character readout")
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True,
                    help="two seeds, so a finding is not one model's quirk")
    ap.add_argument("--replays", type=int, default=12)
    ap.add_argument("--per-replay", type=int, default=60)
    ap.add_argument("--crfs", type=int, nargs="*", default=[26, 35, 45])
    ap.add_argument("--deltas", type=int, nargs="*", default=None,
                    help="measure PAIR SPACING instead of compression: "
                         "build the frame pair at these gaps rather than "
                         "the checkpoint's trained delta. The live loop "
                         "cannot guarantee the training gap under load, and "
                         "a pair at the wrong spacing is a different input "
                         "distribution presented to a net that cannot tell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import json
    chars = json.loads(a.chars.read_text()) if a.chars else None
    rng = np.random.default_rng(a.seed)
    dirs = []
    for w in sorted(a.corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    if not dirs:                    # a flat `runner.collect --out` tree
        dirs = [d for d in sorted(a.corpus.iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    dirs = [d for d in dirs if (d / "video.mp4").exists()
            and (d / "state.csv.gz").exists()
            and (chars is None or d.name in chars)]
    pick_dirs = [dirs[i] for i in
                 rng.choice(len(dirs), min(a.replays, len(dirs)), replace=False)]
    print(f"{len(dirs)} replays available, sampling {len(pick_dirs)}", flush=True)

    vss = [VisionState.load(c, a.device) for c in a.ckpt]
    v0 = vss[0]
    delta, size = v0.delta, v0.size
    sup = [CH[n] for n in SUPERVISED]
    print(f"  delta {delta}, size {size}, n_state {v0.n_state}, "
          f"char head {v0.char_head}", flush=True)

    if a.deltas:
        arms = [f"d{d}" for d in a.deltas]
        if delta not in a.deltas:
            arms = [f"d{delta}"] + arms       # the trained gap is the base
    else:
        arms = ["base", "chroma"] + [f"crf{c}" for c in a.crfs]
    Xs: dict[str, list] = {k: [] for k in arms}
    Ss, Ls, Cs = [], [], []

    tmp = Path(tempfile.mkdtemp(prefix="cgap-"))
    for n, d in enumerate(pick_dirs):
        try:
            st, _pr, _act, valid = read_state(d / "state.csv.gz")
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        idx = np.flatnonzero(ok)
        need = max([int(x[1:]) for x in arms]) if a.deltas else delta
        idx = idx[idx >= need]
        if len(idx) < a.per_replay:
            continue
        rows = np.sort(rng.choice(idx, a.per_replay, replace=False))

        # Every arm must see the SAME rows, so a replay that fails to decode in
        # any one arm is dropped from all of them -- an unpaired frame would
        # turn a decode failure into a fake effect.
        got = {}
        ok_all = True
        for arm in arms:
            if a.deltas:
                g = grab(d / "video.mp4", rows, int(arm[1:]), size, False)
                if g is None:
                    ok_all = False
                    break
                got[arm] = g
                continue
            if arm == "base":
                vid, chroma = d / "video.mp4", False
            elif arm == "chroma":
                vid, chroma = d / "video.mp4", True
            else:
                vid = tmp / f"{d.name}-{arm}.mp4"
                if not vid.exists():
                    reencode(d / "video.mp4", vid, int(arm[3:]))
                chroma = False
            g = grab(vid, rows, delta, size, chroma)
            if g is None:
                ok_all = False
                break
            got[arm] = g
        for f in tmp.glob(f"{d.name}-*.mp4"):
            f.unlink()
        if not ok_all:
            print(f"  [{n}] {d.name}: decode failed, dropped", flush=True)
            continue

        s_sel = st[rows]
        p1_left = (s_sel[:, 0, CH["dx"]] > 0)
        order = np.where(p1_left[:, None], np.array([0, 1]), np.array([1, 0]))
        take = np.take_along_axis(s_sel, order[:, :, None], axis=1)
        for arm in arms:
            Xs[arm].append(got[arm])
        Ss.append(take.astype(np.float32))
        Ls.append(p1_left.astype(np.float32))
        if chars is not None:
            lab = chars[d.name]
            pr_ = np.array([lab["p1_char"], lab["p2_char"]], np.int64)
            Cs.append(np.where(p1_left[:, None], pr_[None, :],
                               pr_[::-1][None, :]))
        print(f"  [{n}] {d.name}: {len(rows)} frames", flush=True)

    if not Ss:
        print("no replays survived", file=sys.stderr)
        return 1
    S = np.concatenate(Ss)
    L = np.concatenate(Ls)
    C = np.concatenate(Cs) if Cs else None
    Y = S[:, :, sup].reshape(len(S), -1)
    print(f"\n{len(Y)} paired frames across {len(Ss)} replays\n", flush=True)

    # A CHANNEL WITH NO VARIANCE IS NOT A CHANNEL.
    #
    # `timestop` and `airborne` are near-constant over a short sample, and R2
    # divides by their variance. The first run of this script clamped sst to
    # 1e-9 and reported a mean R2 of -54824: one degenerate denominator
    # swamping nineteen real channels. Score only the outputs whose target
    # actually moves, and say which were dropped.
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    live = sst > 1e-3 * max(float(sst.max()), 1e-9)
    names = [f"{n}{r}" for r in (0, 1) for n in SUPERVISED]
    dead = [names[i] for i in np.flatnonzero(~live)]
    print(f"scoring {int(live.sum())}/{len(live)} outputs; "
          f"no variance in: {', '.join(dead) if dead else 'none'}\n", flush=True)

    def r2_vec(pred: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        y = Y if rows is None else Y[rows]
        p = pred if rows is None else pred[rows]
        ss = ((y - y.mean(0)) ** 2).sum(0)
        return 1 - ((p - y) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)

    def summarise(pred, idl, chl, rows=None):
        r2 = r2_vec(pred, rows)
        m = float(np.nanmean(r2[live]))
        l = L if rows is None else L[rows]
        il = idl if rows is None else idl[rows]
        idacc = float(((il > 0).astype(np.float32) == l).mean())
        ch = de = float("nan")
        if chl is not None and C is not None:
            c = C if rows is None else C[rows]
            cl = chl if rows is None else chl[rows]
            ch = float((cl.argmax(-1) == c).mean())
            de = _decide_acc(torch.from_numpy(cl), torch.from_numpy(c))
        return m, r2, idacc, ch, de

    boot = np.random.default_rng(a.seed + 1)
    n = len(Y)
    draws = [boot.integers(0, n, n) for _ in range(400)]

    for vs, cpath in zip(vss, a.ckpt):
        mu = np.asarray(_np(vs.mu), np.float32)
        sd = np.asarray(_np(vs.sd), np.float32)
        print(f"=== {cpath.name} " + "=" * 46)
        cache = {}
        for arm in arms:
            pred, idl, chl = readouts(vs, np.concatenate(Xs[arm]))
            cache[arm] = (pred * sd[None, :vs.n_state] + mu[None, :vs.n_state],
                          idl, chl)
        print(f"{'arm':<8} {'meanR2':>8} {'d(base)':>18} {'x':>7} {'dx':>7} "
              f"{'id':>6} {'char':>6} {'d(char)':>17} {'decide':>7}")
        b_m, b_r2, b_id, b_ch, b_de = summarise(*cache[arms[0]])
        ix, idx_ = SUPERVISED.index("x"), SUPERVISED.index("dx")
        for arm in arms:
            m, r2, idacc, ch, de = summarise(*cache[arm])
            if arm == arms[0]:
                dm = dch = "--"
            else:
                # PAIRED bootstrap: the same resampled frames score both arms,
                # so frame-sampling noise cancels instead of being counted
                # twice. The interval is on the DIFFERENCE, which is the only
                # quantity this design can speak to.
                dms, dchs = [], []
                for r in draws:
                    am, _, _, ac, _ = summarise(*cache[arm], rows=r)
                    bm, _, _, bc, _ = summarise(*cache[arms[0]], rows=r)
                    dms.append(am - bm); dchs.append(ac - bc)
                lo, hi = np.percentile(dms, [2.5, 97.5])
                dm = f"{m-b_m:+.4f}[{lo:+.4f},{hi:+.4f}]"
                lo, hi = np.percentile(dchs, [2.5, 97.5])
                dch = f"{ch-b_ch:+.3f}[{lo:+.3f},{hi:+.3f}]"
            print(f"{arm:<8} {m:+8.4f} {dm:>18} {r2[ix]:+7.3f} "
                  f"{r2[idx_]:+7.3f} {idacc:6.3f} {ch:6.3f} {dch:>17} {de:7.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
