"""Can the representation see a hit coming? And would more resolution help?

    python -m scripts.hit_prediction_test --ckpt /root/ckpt_cf/best.pt

Behaviour cloning is a poor way to ask this. Predicting which button a person
presses next has a low ceiling whatever the representation, because people are
not deterministic -- both BC attempts landed near 0.565 AUC and that number may
say more about human variability than about the encoder.

This asks something objective instead, and it is the exact quantity the reward
is built from: given what is on screen now, is the opponent about to lose
health? A representation that cannot answer that cannot support a policy that
seeks damage, and one that can is a representation GRPO could work with.

Four feature sets, same head, same split, same target:

  cls      the 192-d latent the model actually uses
  patch    all 256 patch tokens pooled three ways, same encoder weights
  pixels   a small CNN trained from scratch on the same 224x224 frames
  pixels2x the same CNN on 448x448 frames

The last two are the ones that matter. cls and patch differ only in pooling and
pooling_test already showed that gap is 0.0065. If the CNN on the same frames
does much better than the encoder, the encoder's training objective is what lost
the information; if 448 does much better than 224, resolution is, and a retrain
at higher resolution is justified by more than a hunch. If none of them beat
chance by much, whether a hit is about to land is not visible in single frames
at all and the fix is temporal, not spatial.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from scripts.horizon_ablation import capture_paths, decode_hud


class SmallCNN(nn.Module):
    """Deliberately modest: this is a question about the features, not a race."""

    def __init__(self, out: int = 1, width: int = 32):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, 2, 2), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.GroupNorm(8, c * 2), nn.GELU(),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.GroupNorm(8, c * 4), nn.GELU(),
            nn.Conv2d(c * 4, c * 4, 3, 2, 1), nn.GroupNorm(8, c * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c * 4, out),
        )

    def forward(self, x):
        return self.net(x)


def fit_and_score(make_batch, n_train, n_val, dev, model, steps, lr=3e-4, bs=256):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_vl, best_auc = 1e9, 0.5
    for step in range(1, steps + 1):
        x, y, pw = make_batch("train", bs)
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y,
                                                  pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                model.eval()
                xs, ys, pw = make_batch("val", min(n_val, 8192))
                lo = model(xs).squeeze(-1)
                vl = float(F.binary_cross_entropy_with_logits(lo, ys, pos_weight=pw))
                if vl < best_vl:
                    best_vl = vl
                    best_auc = _auc(torch.sigmoid(lo).cpu().numpy(), ys.cpu().numpy())
                model.train()
    return best_auc


@torch.no_grad()
def encode_feats(wm, obs, device, batch=128):
    enc = wm.encoder
    cls_o, pat_o = [], []
    for i in range(0, len(obs), batch):
        x = torch.from_numpy(np.ascontiguousarray(obs[i : i + batch]))
        x = x.permute(0, 3, 1, 2).to(device).float().div_(255.0)
        B = x.shape[0]
        tok = torch.cat([enc.cls_token.expand(B, -1, -1), enc.patch_embed(x)], 1)
        tok = tok + enc.pos_embed
        for blk in enc.blocks:
            tok = blk(tok)
        tok = enc.norm(tok)
        cls_o.append(enc.projector(tok[:, 0]).float().cpu())
        p = tok[:, 1:]
        pat_o.append(torch.cat([p.mean(1), p.amax(1), p.std(1)], -1).float().cpu())
    return torch.cat(cls_o), torch.cat(pat_o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--lookahead", type=int, default=6, help="decision steps")
    ap.add_argument("--drop", type=float, default=0.01, help="bar fraction = a hit")
    ap.add_argument("--max-frames", type=int, default=7200)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/hit_prediction.json"))
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

    C, P, Px, Y, ep = [], [], [], [], []
    for k, r in enumerate(rows[: a.replays]):
        video, inputs = capture_paths(r, manifest)
        hud = decode_hud(str(video), a.max_frames)
        tr = read_trace(hud, smooth=3)
        del hud
        small = list(decode_frames(video, cfg.image_size, cfg.frame_skip))
        big = list(decode_frames(video, cfg.image_size * 2, cfg.frame_skip))
        D = min(len(small), len(big), len(tr.hp1) // cfg.frame_skip)
        D = min(D, a.max_frames // cfg.frame_skip) - a.lookahead - 1
        if D < 64:
            continue
        idx = np.arange(D) * cfg.frame_skip
        hp2 = tr.hp2[idx]
        fut = tr.hp2[np.minimum(idx + a.lookahead * cfg.frame_skip, len(tr.hp2) - 1)]
        y = ((hp2 - fut) > a.drop).astype(np.float32)      # P2 about to be hit
        c, p = encode_feats(wm, np.stack(small[:D]), a.device)
        C.append(c); P.append(p); Y.append(torch.from_numpy(y))
        Px.append((np.stack(small[:D]), np.stack(big[:D])))
        ep.append(np.full(D, k, dtype=np.int32))
        if (k + 1) % 5 == 0:
            print(f"   {k+1}/{a.replays}", flush=True)

    Xc, Xp, Yt = torch.cat(C), torch.cat(P), torch.cat(Y)
    S1 = np.concatenate([s for s, _ in Px]); S2 = np.concatenate([b for _, b in Px])
    E = np.concatenate(ep)
    uniq = np.unique(E)
    val_eps = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr_i = np.flatnonzero(~is_val)
    va_i = np.flatnonzero(is_val)
    base = float(Yt.mean())
    pw = torch.tensor([(1 - base) / max(base, 1e-4)], device=a.device).clamp(max=50)
    print(f"\n{len(tr_i)} train / {len(va_i)} val frames, split by replay")
    print(f"positive rate (a hit lands within {a.lookahead} steps "
          f"= {a.lookahead*cfg.frame_skip/60:.2f}s): {base:.4f}\n")

    dev = a.device
    Yd = Yt.to(dev)

    def vec_batch(X):
        Xd = X.to(dev)
        def f(split, n):
            pool = tr_i if split == "train" else va_i
            sel = torch.from_numpy(np.random.choice(pool, n)).to(dev)
            return Xd[sel], Yd[sel], pw
        return f

    def pix_batch(S):
        def f(split, n):
            pool = tr_i if split == "train" else va_i
            sel = np.random.choice(pool, n)
            x = torch.from_numpy(np.ascontiguousarray(S[sel])).to(dev)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            return x, Yd[torch.from_numpy(sel).to(dev)], pw
        return f

    res = {}
    head = lambda d: nn.Sequential(nn.Linear(d, 512), nn.GELU(),
                                   nn.Linear(512, 512), nn.GELU(),
                                   nn.Linear(512, 1)).to(dev)
    res["cls"] = fit_and_score(vec_batch(Xc), len(tr_i), len(va_i), dev,
                               head(Xc.shape[1]), a.steps)
    print(f"cls       AUC {res['cls']:.4f}")
    res["patch"] = fit_and_score(vec_batch(Xp), len(tr_i), len(va_i), dev,
                                 head(Xp.shape[1]), a.steps)
    print(f"patch     AUC {res['patch']:.4f}")
    res["pixels"] = fit_and_score(pix_batch(S1), len(tr_i), len(va_i), dev,
                                  SmallCNN().to(dev), a.steps, bs=128)
    print(f"pixels    AUC {res['pixels']:.4f}   ({cfg.image_size}px)")
    res["pixels2x"] = fit_and_score(pix_batch(S2), len(tr_i), len(va_i), dev,
                                    SmallCNN().to(dev), a.steps, bs=64)
    print(f"pixels2x  AUC {res['pixels2x']:.4f}   ({cfg.image_size*2}px)")

    print()
    gain_enc = res["pixels"] - res["cls"]
    gain_res = res["pixels2x"] - res["pixels"]
    print(f"CNN over encoder at the same resolution : {gain_enc:+.4f}")
    print(f"doubling resolution for the same CNN    : {gain_res:+.4f}")
    if max(res.values()) < 0.60:
        v = ("no representation sees a hit coming from a single frame; the "
             "missing signal is temporal, not spatial, and more resolution "
             "will not supply it")
    elif gain_res > 0.05:
        v = "resolution is the binding constraint; retrain the encoder larger"
    elif gain_enc > 0.05:
        v = ("the frames carry it and the encoder's objective discarded it; "
             "retrain the encoder with this as an auxiliary target")
    else:
        v = "the encoder is already extracting what these frames contain"
    print(f"VERDICT: {v}")
    res["base_rate"] = base
    a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
