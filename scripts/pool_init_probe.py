"""Is position linearly readable off an UNTRAINED encoder, on REAL frames?

    python -m scripts.pool_init_probe --corpus ~/corpus --captures 8

WHY THIS RUNS BEFORE RENTING ANYTHING
-------------------------------------
`SpatialPool` makes position a coordinate rather than a summary, and on a
synthetic scene with one bright square a linear probe reads it at AUC 0.9999
with no training at all. That is not the question. The question is whether the
same holds on a frame containing two fighters, a HUD, a background, weather
and up to 213 projectile objects -- where "which cell is brightest" is not the
fighters.

If the spatial pool starts high here, the architecture is doing what it claims
on real input and the remaining work is ordinary training. If it starts at
chance, the idea is wrong and it cost an hour on a box we already have rather
than a night on one we rented.

No gradients are taken. Both pools are evaluated at random init from the same
seed, so the only difference between the arms is the pooling.
"""
from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.encoder import ViTEncoder
from scripts.train_full import _mirror_probe_auc, _play_band
from scripts.horizon_ablation import capture_paths, load_replay


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--captures", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg0 = dataclasses.replace(Config(), image_size=a.image_size)
    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    np.random.default_rng(a.seed).shuffle(rows)

    frames = []
    for r in rows:
        if len(frames) >= a.captures:
            break
        try:
            video, inputs = capture_paths(r, manifest)
            obs, _chunks, _lab = load_replay(video, inputs, cfg0, a.max_frames,
                                             a.corpus / ".pool_probe_cache")
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}")
            continue
        frames.append(torch.as_tensor(np.ascontiguousarray(obs[:a.max_frames])))
    if not frames:
        raise SystemExit("no captures loaded")
    O = torch.cat(frames)
    # load_replay yields NHWC uint8; the encoder wants NCHW. Getting this wrong
    # does not raise until a conv deep inside, complaining about channel counts
    # in a way that says nothing about layout.
    if O.ndim == 4 and O.shape[-1] in (1, 3):
        O = O.permute(0, 3, 1, 2).contiguous()
    print(f"{len(O)} real frames from {len(frames)} captures at {a.image_size} px\n")

    lo, hi = _play_band(a.image_size)
    print(f"{'pool':>9} {'probe_play':>11} {'probe_full':>11} {'identity':>10}")
    out = {}
    for pool in ("cls", "spatial"):
        torch.manual_seed(a.seed)         # same init draw for both arms
        cfg = dataclasses.replace(cfg0, encoder_pool=pool)
        enc = ViTEncoder(cfg).to(a.device).eval()
        b0, bp, bf = [], [], []
        with torch.no_grad():
            for i in range(0, len(O), 32):
                o = O[i:i + 32].to(a.device)
                mp = o.clone()
                mp[..., lo:hi, :] = torch.flip(mp[..., lo:hi, :], dims=[-1])
                b0.append(enc(o).float().cpu())
                bp.append(enc(mp).float().cpu())
                bf.append(enc(torch.flip(o, dims=[-1])).float().cpu())
        m = _mirror_probe_auc(torch.cat(b0), torch.cat(bp), torch.cat(bf))
        out[pool] = m
        print(f"{pool:>9} {m['probe_play']:>11.4f} {m['probe_full']:>11.4f} "
              f"{m['probe_identity']:>10.4f}")
    print("\nprobe_play is the gate. Chance is 0.500; the trained [CLS] model "
          "reached 0.553\nafter twelve hours, and the ceiling is ~0.958.")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
