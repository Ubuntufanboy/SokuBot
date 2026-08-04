"""How far ahead is a hit visible, and does 15 Hz throw that away?

    python -m scripts.frame_rate_test --ckpt /root/ckpt_cf/best.pt

Every representation tried caps near 0.62 AUC at "is the opponent about to lose
health within 0.4 s": more pixels, the whole patch grid, a purpose-trained CNN,
three frames of latent history, explicit motion. One explanation covers all of
them. `frame_skip = 4` puts decisions at 15 Hz against a 60 Hz game, and whether
an attack connects is settled inside two or three frames -- so the moment that
decides it falls between decision steps and is in no observation the model gets.

That predicts something specific and testable without retraining anything: a hit
should be much more predictable a few frames ahead than a few tenths of a second
ahead. Frames are encoded at full rate here, one latent per game frame, and the
same head is asked the same question at a range of lookaheads.

  sharp decay with distance   the signal is there and short-lived, so 15 Hz
                              is discarding it and the decision rate is the
                              blocker
  flat and low everywhere     a hit is not visible in these captures at any
                              timescale, and a pixel world model is the wrong
                              instrument regardless of frame rate

Base rate moves with the lookahead -- a hit within two frames is much rarer than
one within twenty-four -- which is exactly why this reports AUC, which does not
care, alongside the rate itself so the reader can see what is being asked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sokubot.config import Config
from sokubot.data.hud import read_trace
from sokubot.data.soku import decode_frames
from sokubot.model.world_model import LeWorldModel
from scripts.hit_prediction_test import encode_feats, fit_and_score
from scripts.horizon_ablation import capture_paths, decode_hud


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--lookaheads", type=int, nargs="+",
                    default=[2, 4, 8, 16, 24, 48], help="in game frames at 60 Hz")
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--drops", type=float, nargs="*", default=None,
                    help="sweep the damage threshold at a fixed lookahead "
                         "instead of sweeping the lookahead. One health bar is "
                         "189 px, so 0.01 is about two pixels -- plausibly "
                         "inside the HUD reader's own jitter, which would cap "
                         "AUC for every representation and look exactly like a "
                         "representation failure")
    ap.add_argument("--fixed-lookahead", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=5400)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/frame_rate_test.json"))
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

    C, HP, ep = [], [], []
    for k, r in enumerate(rows[: a.replays]):
        video, inputs = capture_paths(r, manifest)
        hud = decode_hud(str(video), a.max_frames)
        tr = read_trace(hud, smooth=3)
        del hud
        # stride 1: one latent per game frame, which is what 15 Hz discards.
        frames = list(decode_frames(video, cfg.image_size, 1))
        D = min(len(frames), len(tr.hp1), a.max_frames) - max(a.lookaheads) - 1
        if D < 256:
            continue
        c, _ = encode_feats(wm, np.stack(frames[:D]), a.device)
        C.append(c)
        HP.append(tr.hp2[:D + max(a.lookaheads) + 1])
        ep.append(np.full(D, k, dtype=np.int32))
        if (k + 1) % 3 == 0:
            print(f"   {k+1}/{a.replays}", flush=True)

    X = torch.cat(C)
    E = np.concatenate(ep)
    uniq = np.unique(E)
    val_eps = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    dev = a.device
    Xd = X.to(dev)
    print(f"\n{len(X)} frames at 60 Hz, {len(tr_i)} train / {len(va_i)} val, "
          f"split by replay\n")
    print(f"{'lookahead':>10} {'drop':>8} {'base rate':>10} {'AUC':>8}")
    print("-" * 40)

    # Either sweep is the same loop over (lookahead, threshold) pairs.
    sweep = ([(a.fixed_lookahead, d) for d in a.drops] if a.drops
             else [(L, a.drop) for L in a.lookaheads])
    res = {}
    for L, drop in sweep:
        # Targets are rebuilt per replay so a lookahead never reads across a
        # boundary into the next capture.
        parts = []
        for hp in HP:
            n = len(hp) - max(a.lookaheads) - 1
            cur = hp[:n]
            fut = hp[L : L + n]
            parts.append(((cur - fut) > drop).astype(np.float32))
        Y = torch.from_numpy(np.concatenate(parts))
        base = float(Y[tr_i].mean())
        Yd = Y.to(dev)
        pw = torch.tensor([(1 - base) / max(base, 1e-4)], device=dev).clamp(max=50)

        def make(split, n, Yd=Yd, pw=pw):
            pool = tr_i if split == "train" else va_i
            sel = torch.from_numpy(np.random.choice(pool, n)).to(dev)
            return Xd[sel], Yd[sel], pw

        head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(),
                             nn.Linear(512, 512), nn.GELU(),
                             nn.Linear(512, 1)).to(dev)
        auc = fit_and_score(make, len(tr_i), len(va_i), dev, head, a.steps)
        key = f"L{L}_d{drop:g}"
        res[key] = {"auc": auc, "base_rate": base, "seconds": L / 60.0,
                    "lookahead": L, "drop": drop}
        print(f"{L:10d} {drop:8.3f} {base:10.4f} {auc:8.4f}")

    aucs = [res[f"L{L}_d{d:g}"]["auc"] for L, d in sweep]
    near, far = aucs[0], aucs[-1]
    print("-" * 40)
    if a.drops:
        print(f"threshold {sweep[0][1]:g} -> {sweep[-1][1]:g}: "
              f"AUC {near:.4f} -> {far:.4f} ({far - near:+.4f})")
        if far - near > 0.08:
            print("VERDICT: the earlier ceiling was the label, not the "
                  "representation -- a two-pixel threshold was scoring HUD "
                  "jitter, and real hits are far more predictable")
        else:
            print("VERDICT: the ceiling holds at every threshold, so it is "
                  "the representation and not label noise")
        a.out.write_text(json.dumps(res, indent=2))
        return 0
    print(f"nearest lookahead {a.lookaheads[0]} frames: {near:.4f}")
    print(f"farthest {a.lookaheads[-1]} frames        : {far:.4f}")
    print(f"decay across the sweep                : {near - far:+.4f}")
    if near > 0.75:
        v = ("a hit is clearly visible a few frames ahead and 15 Hz discards "
             "it; the decision rate is the blocker and a finer frame_skip is "
             "the fix")
    elif near - far > 0.10:
        v = ("the signal is short-lived but weak even close up; a finer step "
             "would help and may not be sufficient on its own")
    else:
        v = ("a hit is not visible at any timescale tried, so the frame rate "
             "is not the blocker and pixels are the wrong instrument")
    print(f"VERDICT: {v}")
    a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
