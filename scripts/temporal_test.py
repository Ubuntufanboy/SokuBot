"""Is the missing signal motion? Same target, single frames against history.

    python -m scripts.temporal_test --ckpt /root/ckpt_cf/best.pt

scripts/hit_prediction_test asked four single-frame representations whether the
opponent is about to lose health within 0.4 s. They all landed between 0.6205
and 0.6538 AUC, and doubling the input resolution made it *worse*, so the
encoder is already extracting close to what one frame contains and more pixels
is not the answer.

What a still frame cannot show is motion. Whether an attack connects depends on
which animation frame each character occupies and how fast they are closing, and
both are properties of a sequence. Every representation measured so far saw
exactly one frame; the encoder is per-frame by construction, and history only
enters later at the predictor.

Five arms, same target, same head, same by-replay split:

  cls1      one latent -- the 0.6205 baseline, repeated here for comparability
  cls3      three consecutive latents concatenated
  cls3d     the current latent plus its two differences, which is the same
            information as cls3 in a form a linear layer reaches more easily
  pix1      a CNN on one 224px frame
  pixdiff   the same CNN on six channels: the frame stacked with the one two
            steps earlier. Stacked rather than differenced because these are
            uint8 and a subtraction would wrap, and because the network can
            form whatever comparison it wants from both.

If the history arms clear the single-frame arms by a wide margin, motion is what
the representation is missing and the fix is to give the encoder access to it --
frame stacking, or a difference channel -- which is far cheaper than more
resolution. If they do not, whether a hit lands is not predictable from what is
on screen at 15 Hz, and the decision rate itself is the thing to revisit: a hit
is often decided inside two or three frames of a 60 Hz game, which is less than
one decision step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.hud import read_trace
from sokubot.data.soku import decode_frames
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import _auc
from scripts.hit_prediction_test import SmallCNN, encode_feats, fit_and_score
from scripts.horizon_ablation import capture_paths, decode_hud


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=16)
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--gap", type=int, default=2, help="steps back for the difference")
    ap.add_argument("--max-frames", type=int, default=5400)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/temporal_test.json"))
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    np.random.default_rng(0).shuffle(rows)

    C, S, Y, ep = [], [], [], []
    g = a.gap
    for k, r in enumerate(rows[: a.replays]):
        video, inputs = capture_paths(r, manifest)
        hud = decode_hud(str(video), a.max_frames)
        tr = read_trace(hud, smooth=3)
        del hud
        frames = list(decode_frames(video, cfg.image_size, cfg.frame_skip))
        D = min(len(frames), len(tr.hp1) // cfg.frame_skip)
        D = min(D, a.max_frames // cfg.frame_skip) - a.lookahead - 1
        if D < 64 + g:
            continue
        arr = np.stack(frames[:D])
        c, _ = encode_feats(wm, arr, a.device)
        idx = np.arange(D) * cfg.frame_skip
        hp2 = tr.hp2[idx]
        fut = tr.hp2[np.minimum(idx + a.lookahead * cfg.frame_skip, len(tr.hp2) - 1)]
        y = ((hp2 - fut) > a.drop).astype(np.float32)
        # Drop the first `g` steps of each replay: they have no history to look
        # back on, and padding them would put a fake "no motion" case in both
        # splits and flatter the history arms.
        C.append(c[g:]); S.append(arr[g:]); Y.append(torch.from_numpy(y[g:]))
        # Keep the lagged views aligned by construction rather than by index
        # arithmetic later, which is where this kind of test usually goes wrong.
        C[-1] = torch.stack([c[g:], c[g - 1 : -1], c[: -g]], dim=1)  # [N,3,latent]
        S[-1] = np.concatenate([arr[g:], arr[: -g]], axis=-1)        # [N,H,W,6]
        ep.append(np.full(len(Y[-1]), k, dtype=np.int32))
        if (k + 1) % 4 == 0:
            print(f"   {k+1}/{a.replays}", flush=True)

    Ch = torch.cat(C)                       # [N, 3, latent]
    Sx = np.concatenate(S)                  # [N, H, W, 6]
    Yt = torch.cat(Y)
    E = np.concatenate(ep)
    uniq = np.unique(E)
    val_eps = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    base = float(Yt.mean())
    dev = a.device
    pw = torch.tensor([(1 - base) / max(base, 1e-4)], device=dev).clamp(max=50)
    Yd = Yt.to(dev)
    print(f"\n{len(tr_i)} train / {len(va_i)} val, base rate {base:.4f}\n")

    def vec(X):
        Xd = X.to(dev)
        def f(split, n):
            pool = tr_i if split == "train" else va_i
            sel = torch.from_numpy(np.random.choice(pool, n)).to(dev)
            return Xd[sel], Yd[sel], pw
        return f

    def pix(chans):
        def f(split, n):
            pool = tr_i if split == "train" else va_i
            sel = np.random.choice(pool, n)
            x = torch.from_numpy(np.ascontiguousarray(Sx[sel][..., :chans])).to(dev)
            return (x.permute(0, 3, 1, 2).float().div_(255.0),
                    Yd[torch.from_numpy(sel).to(dev)], pw)
        return f

    head = lambda d: nn.Sequential(nn.Linear(d, 512), nn.GELU(),
                                   nn.Linear(512, 512), nn.GELU(),
                                   nn.Linear(512, 1)).to(dev)
    L = Ch.shape[-1]
    cls1 = Ch[:, 0]
    cls3 = Ch.reshape(len(Ch), -1)
    cls3d = torch.cat([Ch[:, 0], Ch[:, 0] - Ch[:, 1], Ch[:, 0] - Ch[:, 2]], dim=-1)

    res = {}
    for name, X in (("cls1", cls1), ("cls3", cls3), ("cls3d", cls3d)):
        res[name] = fit_and_score(vec(X), len(tr_i), len(va_i), dev,
                                  head(X.shape[1]), a.steps)
        print(f"{name:8s} AUC {res[name]:.4f}")
    res["pix1"] = fit_and_score(pix(3), len(tr_i), len(va_i), dev,
                                SmallCNN().to(dev), a.steps, bs=128)
    print(f"{'pix1':8s} AUC {res['pix1']:.4f}")
    cnn6 = SmallCNN().to(dev)
    cnn6.net[0] = nn.Conv2d(6, 32, 5, 2, 2).to(dev)
    res["pixdiff"] = fit_and_score(pix(6), len(tr_i), len(va_i), dev, cnn6,
                                   a.steps, bs=128)
    print(f"{'pixdiff':8s} AUC {res['pixdiff']:.4f}")

    lat_gain = max(res["cls3"], res["cls3d"]) - res["cls1"]
    pix_gain = res["pixdiff"] - res["pix1"]
    print()
    print(f"latent history over one latent : {lat_gain:+.4f}")
    print(f"frame difference over one frame: {pix_gain:+.4f}")
    if max(lat_gain, pix_gain) > 0.05:
        v = ("motion is the missing signal; give the encoder history and the "
             "representation problem is a cheap fix, not a retrain at scale")
    elif max(res.values()) < 0.70:
        v = ("no representation tried anticipates a hit well, with or without "
             "motion; 15 Hz decisions may be too coarse for a 60 Hz game")
    else:
        v = "history helps only marginally; the ceiling is elsewhere"
    print(f"VERDICT: {v}")
    res["base_rate"] = base
    a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
