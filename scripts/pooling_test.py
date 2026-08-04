"""Is the control information in the encoder, or thrown away by [CLS] pooling?

    python -m scripts.pooling_test --ckpt /root/ckpt_cf/best.pt

Everything measured tonight points at the same suspect. The world model predicts
the next latent well and cannot say whether an attack connected: across 32
realistic rollouts from one start, damage varies by 0.011 and correlates with
attacking at -0.0001. Behaviour cloning from the latent tops out at 0.565 mean
AUC across two unrelated architectures. Whether an attack lands is a question
about spacing and timing, and the latent is one 192-dimensional vector taken
from the [CLS] token of a ViT over a 224x224 frame.

But "the representation is too small" and "the pooling discards it" are
different diagnoses with different fixes, and the evidence so far cannot tell
them apart. This does.

The same frames are encoded twice: once as the model uses them, the [CLS] token
after the projector, and once keeping all 256 patch tokens. A small head then
predicts the action from each. Patch tokens carry the same computation -- the
same weights, the same layers -- and differ only in what survives the last step.

  patch >> cls   the ViT computes the spatial detail and pooling throws it
                 away, so the fix is a latent that keeps spatial structure and
                 the encoder does not need retraining from scratch
  patch ~ cls    the detail was never computed, so the fix is upstream:
                 resolution, patch size, or encoder capacity

Judged by AUC on held-out replays, against the 0.565 the [CLS] latent manages.
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
from sokubot.data.soku import decode_frames, read_actions
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import _auc
from scripts.horizon_ablation import capture_paths

BUTTONS = ("a", "b", "c", "d")          # the ones that decide whether a hit lands


@torch.no_grad()
def encode_both(model: LeWorldModel, obs: np.ndarray, device: str, batch: int = 128):
    """-> (cls [N, latent], patch [N, tokens, enc_dim]) from one forward pass."""
    enc = model.encoder
    cls_out, patch_out = [], []
    for i in range(0, len(obs), batch):
        x = torch.from_numpy(np.ascontiguousarray(obs[i : i + batch]))
        x = x.permute(0, 3, 1, 2).to(device).float().div_(255.0)
        tok = enc.patch_embed(x) if hasattr(enc, "patch_embed") else None
        if tok is None:                          # fall back to the public path
            raise SystemExit("encoder has no patch_embed; inspect model/encoder.py")
        B = x.shape[0]
        tok = torch.cat([enc.cls_token.expand(B, -1, -1), tok], dim=1) + enc.pos_embed
        for blk in enc.blocks:
            tok = blk(tok)
        tok = enc.norm(tok)
        cls_out.append(enc.projector(tok[:, 0]).float().cpu())
        # Mean and max over patches keeps the head small while still exposing
        # what pooling to a single token discards.
        p = tok[:, 1:]
        patch_out.append(torch.cat([p.mean(1), p.amax(1), p.std(1)], -1).float().cpu())
    return torch.cat(cls_out), torch.cat(patch_out)


def train_head(X, Y, tr, va, dev, steps=3000, width=512, lr=3e-4):
    """Same head on both feature sets, so only the features differ."""
    head = nn.Sequential(nn.Linear(X.shape[1], width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    rates = Y[tr].mean(0).clamp_min(1e-4)
    pw = ((1 - rates) / rates).clamp(max=50.0).to(dev)
    best, best_auc = 1e9, None
    for step in range(1, steps + 1):
        sel = tr[torch.randint(len(tr), (2048,))]
        loss = F.binary_cross_entropy_with_logits(head(X[sel].to(dev)),
                                                  Y[sel].to(dev), pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                p = torch.sigmoid(head(X[va].to(dev))).cpu().numpy()
                vl = float(F.binary_cross_entropy_with_logits(
                    head(X[va].to(dev)), Y[va].to(dev), pos_weight=pw))
            if vl < best:
                best = vl
                t = Y[va].numpy()
                best_auc = {b: _auc(p[:, i], t[:, i]) for i, b in enumerate(BUTTONS)}
    return best_auc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/pooling_test.json"))
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

    C, P, Yl, ep = [], [], [], []
    for k, r in enumerate(rows[: a.replays]):
        video, inputs = capture_paths(r, manifest)
        obs = list(decode_frames(video, cfg.image_size, cfg.frame_skip))
        raw = read_actions(inputs)
        D = min(len(obs), len(raw) // cfg.frame_skip)
        if D < 64:
            continue
        c, p = encode_both(wm, np.stack(obs[:D]), a.device)
        chunks = np.stack([raw[d * cfg.frame_skip : (d + 1) * cfg.frame_skip]
                           for d in range(D)])
        y = torch.from_numpy((chunks[:, :, 4:8].max(1) > 0.5).astype(np.float32))
        C.append(c); P.append(p); Yl.append(y)
        ep.append(np.full(D, k, dtype=np.int32))
        if (k + 1) % 8 == 0:
            print(f"   {k+1}/{a.replays}", flush=True)

    Xc, Xp, Y = torch.cat(C), torch.cat(P), torch.cat(Yl)
    E = np.concatenate(ep)
    uniq = np.unique(E)
    val_eps = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_val = np.array([e in val_eps for e in E])
    tr = torch.from_numpy(np.flatnonzero(~is_val))
    va = torch.from_numpy(np.flatnonzero(is_val))
    print(f"\ncls features {tuple(Xc.shape)}, patch features {tuple(Xp.shape)}")
    print(f"{len(tr)} train / {len(va)} val frames, split by replay\n")

    res = {}
    for name, X in (("cls", Xc), ("patch", Xp)):
        auc = train_head(X, Y, tr, va, a.device)
        res[name] = auc
        print(f"{name:6s} " + "  ".join(f"{b} {auc[b]:.3f}" for b in BUTTONS)
              + f"   mean {np.mean(list(auc.values())):.4f}")

    mc = float(np.mean(list(res["cls"].values())))
    mp = float(np.mean(list(res["patch"].values())))
    print()
    print(f"cls {mc:.4f} -> patch {mp:.4f}   ({mp - mc:+.4f})")
    if mp - mc > 0.05:
        v = ("the ViT computes spatial detail that [CLS] pooling discards; a "
             "latent keeping spatial structure should fix control without "
             "retraining the encoder from scratch")
    else:
        v = ("the detail is not in the encoder either; the fix is upstream -- "
             "resolution, patch size or capacity -- and needs a real retrain")
    print(f"VERDICT: {v}")
    a.out.write_text(json.dumps({"cls": res["cls"], "patch": res["patch"],
                                 "cls_mean": mc, "patch_mean": mp}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
