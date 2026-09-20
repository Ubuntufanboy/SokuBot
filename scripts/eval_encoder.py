"""Score one or more encoder checkpoints on an arbitrary capture set.

The comparison this exists for: a REPLAY-trained encoder and a VERSUS-trained
encoder, both scored on held-out VERSUS frames. The corpus is 100% replay
footage and deployment is 100% Versus play, and that difference is the one no
experiment on the replay corpus can reach -- every other candidate (codec,
geometry, pair spacing, character coverage) has been measured and killed.

Nothing here is a by-replay split of a single corpus. The eval captures must be
ones NEITHER model trained on, held out by directory, because `findings/04`'s
whole lesson is that a by-replay split inside one corpus does not measure
transfer to a different data-generating process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, read_state          # noqa: E402
from sokubot.live.visionstate import VisionState       # noqa: E402
from scripts.train_encoder import SUPERVISED, _decide_acc  # noqa: E402


def _np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", type=Path, nargs="+", required=True,
                    help="capture roots to evaluate on")
    ap.add_argument("--only", type=Path, default=None,
                    help="text file of capture directory names to restrict to; "
                         "this is how a held-out split is pinned so both models "
                         "see exactly the same frames")
    ap.add_argument("--chars", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--per-replay", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import cv2

    keep_names = None
    if a.only:
        keep_names = {ln.strip() for ln in a.only.read_text().split() if ln.strip()}
    chars = json.loads(a.chars.read_text()) if a.chars else None

    dirs = []
    for r in a.dirs:
        for w in sorted(r.glob("w*")):
            dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
        dirs += [d for d in sorted(r.iterdir())
                 if d.is_dir() and not d.name.startswith("w")
                 and not d.name.startswith(".")]
    dirs = [d for d in dirs if (d / "video.mp4").exists()
            and ((d / "state.csv.gz").exists() or (d / "inputs.csv.gz").exists())]
    if keep_names is not None:
        dirs = [d for d in dirs if d.name in keep_names]
    if chars is not None:
        dirs = [d for d in dirs if d.name in chars]

    rng = np.random.default_rng(a.seed)
    if len(dirs) > a.replays:
        dirs = [dirs[i] for i in
                sorted(rng.choice(len(dirs), a.replays, replace=False))]

    vss = [VisionState.load(c, a.device) for c in a.ckpt]
    delta, size = vss[0].delta, vss[0].size
    print(f"{len(dirs)} captures, delta {delta}, size {size}", flush=True)

    X, S, L, C = [], [], [], []
    for n, d in enumerate(dirs):
        sc = d / "state.csv.gz"
        if not sc.exists():
            sc = d / "inputs.csv.gz"
        try:
            st, _p, _act, valid = read_state(sc)
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
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
        p1_left = (s_sel[:, 0, CH["dx"]] > 0)
        order = np.where(p1_left[:, None], np.array([0, 1]), np.array([1, 0]))
        take = np.take_along_axis(s_sel, order[:, :, None], axis=1)
        X.append(np.stack(frames))
        S.append(take.astype(np.float32))
        L.append(p1_left.astype(np.float32))
        if chars is not None:
            lab = chars[d.name]
            pr = np.array([lab["p1_char"], lab["p2_char"]], np.int64)
            C.append(np.where(p1_left[:, None], pr[None, :], pr[::-1][None, :]))
        print(f"  [{n}] {d.name}: {len(frames)}", flush=True)

    X = np.concatenate(X); S = np.concatenate(S); L = np.concatenate(L)
    C = np.concatenate(C) if C else None
    Y = S[:, :, [CH[n] for n in SUPERVISED]].reshape(len(S), -1)
    print(f"\n{len(X)} frames from {len(dirs)} captures\n", flush=True)

    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    live = sst > 1e-3 * max(float(sst.max()), 1e-9)
    names = [f"{n}{r}" for r in (0, 1) for n in SUPERVISED]
    dead = [names[i] for i in np.flatnonzero(~live)]
    print(f"scoring {int(live.sum())}/{len(live)} outputs; "
          f"no variance in: {', '.join(dead) if dead else 'none'}\n")

    boot = np.random.default_rng(a.seed + 1)
    draws = [boot.integers(0, len(Y), len(Y)) for _ in range(400)]

    print(f"{'checkpoint':<22} {'meanR2':>8} {'x':>7} {'dx':>7} {'y':>7} "
          f"{'hp':>7} {'id':>6} {'char':>6} {'decide':>7}")
    for vs, cpath in zip(vss, a.ckpt):
        mu = np.asarray(_np(vs.mu), np.float32)
        sd = np.asarray(_np(vs.sd), np.float32)
        pr, idl, chl = [], [], []
        with torch.no_grad():
            for j in range(0, len(X), 128):
                xb = torch.from_numpy(X[j:j+128]).to(a.device).float().div(255.0)
                out = vs.net(xb, return_feat=True)
                o, feat = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
                pr.append(o[:, :vs.n_state].cpu().numpy())
                idl.append(o[:, vs.n_state + vs.n_proj].cpu().numpy())
                if vs.chead is not None:
                    chl.append(vs.chead(feat)[0].cpu().numpy())
        pred = np.concatenate(pr) * sd[None, :vs.n_state] + mu[None, :vs.n_state]
        idl = np.concatenate(idl)
        chl = np.concatenate(chl) if chl else None
        r2 = 1 - ((pred - Y) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
        m = float(np.nanmean(r2[live]))
        ms = [float(np.nanmean((1 - ((pred[r] - Y[r]) ** 2).sum(0) /
              np.maximum(((Y[r] - Y[r].mean(0)) ** 2).sum(0), 1e-9))[live]))
              for r in draws[:120]]
        lo, hi = np.percentile(ms, [2.5, 97.5])
        idacc = float(((idl > 0).astype(np.float32) == L).mean())
        ch = de = float("nan")
        if chl is not None and C is not None:
            ch = float((chl.argmax(-1) == C).mean())
            de = _decide_acc(torch.from_numpy(chl), torch.from_numpy(C))
        g = lambda n: r2[SUPERVISED.index(n)]
        print(f"{cpath.name:<22} {m:+8.4f} {g('x'):+7.3f} {g('dx'):+7.3f} "
              f"{g('y'):+7.3f} {g('hp'):+7.3f} {idacc:6.3f} {ch:6.3f} {de:7.3f}")
        print(f"{'':<22} 95% CI [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
