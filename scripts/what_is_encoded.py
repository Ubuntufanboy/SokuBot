"""Does the latent encode the characters, or only the HUD?

    python -m scripts.what_is_encoded --ckpt /root/ckpt/best.pt

THE HYPOTHESIS
--------------
Probe quality across the HUD sorts by how predictable each quantity is over
time, not by how large or bright it is on screen:

    health          slow, monotone, rarely changes      R^2 0.82-0.86
    combo red       appears and decays over ~1 s        R^2 0.40
    spirit          fast, action-driven, fluctuates     R^2 0.03
    characters      fastest, most chaotic               absent from decodes

That ordering is what the training objective would produce on its own. The loss
is prediction error plus SIGReg, and SIGReg constrains the latent's *shape* --
isotropic, non-degenerate -- while saying nothing about its *content*. The
encoder is free to choose what to represent, and anything it declines to encode
costs it nothing while making the prediction term easier. Character positions
move chaotically across a 67 ms step; the weather timer is a counter. Encoding
the second and discarding the first lowers the loss.

If that is right, the representation is not missing detail because the input
lacks it -- resolution, patch grid, motion and sampling rate were all ruled out
already -- but because the objective is paid to remove it.

FOUR MEASUREMENTS
-----------------
1. matched-HUD pairs   Frames from one replay with near-identical HUD readings
                       but characters in different places. If the latents are
                       nearly identical anyway, the encoder is not looking at
                       the characters. This is the direct test.

2. random-init control The same pairs through an *untrained* encoder of the
                       same architecture. This is what separates "the
                       architecture cannot represent position" from "training
                       removed it". An untrained ViT is a random projection of
                       the pixels and will happily distinguish two different
                       images. If the trained encoder separates these pairs
                       *less* than the random one does, training actively
                       destroyed the information, and no amount of resolution
                       or frame rate was ever going to help.

3. HUD-explains-latent Regress every latent dimension on the six HUD readings.
                       The fraction of latent variance they account for is how
                       much of this representation is just the HUD.

4. predictability      For each HUD quantity, how well a linear model predicts
                       it one step ahead, plotted against its probe R^2. The
                       hypothesis says these should be strongly related. If
                       they are not, the story above is wrong.

Pixel distance inside the play area is reported alongside, to confirm the
characters really did move in the pairs being compared -- otherwise "the latents
are similar" would be the correct answer rather than a finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.hud import FILL_ROWS, read_trace
from sokubot.data.soku import decode_frames
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import fit_ridge
from scripts.horizon_ablation import capture_paths, decode_hud, encode_all

TARGETS = ("hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2")
# The HUD occupies the top and bottom of the frame; the characters fight in
# between. Rows are in 224-space after the encoder's resize.
PLAY_TOP, PLAY_BOTTOM = 60, 200


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    b = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=5400)
    ap.add_argument("--hud-tol", type=float, default=0.002,
                    help="max HUD difference for a pair to count as matched")
    ap.add_argument("--min-pixel-move", type=float, default=8.0,
                    help="min mean abs pixel difference in the play area, so the "
                         "characters demonstrably moved")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/what_is_encoded.json"))
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    torch.manual_seed(0)
    rnd = LeWorldModel(cfg).to(a.device)     # same architecture, never trained
    rnd.eval()

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    np.random.default_rng(0).shuffle(rows)

    Zt, Zr, H, F, ep = [], [], [], [], []
    for k, r in enumerate(rows[: a.replays]):
        video, inputs = capture_paths(r, manifest)
        hud = decode_hud(str(video), a.max_frames)
        tr = read_trace(hud, smooth=3)
        del hud
        frames = np.stack(list(decode_frames(video, cfg.image_size, cfg.frame_skip)))
        D = min(len(frames), len(tr.hp1) // cfg.frame_skip, a.max_frames // cfg.frame_skip)
        if D < 128:
            continue
        frames = frames[:D]
        idx = np.arange(D) * cfg.frame_skip
        Zt.append(encode_all(wm, frames, a.device))
        Zr.append(encode_all(rnd, frames, a.device))
        H.append(np.stack([getattr(tr, t)[idx] for t in TARGETS], axis=1))
        # Grey play-area only, so pixel distance measures the fight and not the
        # HUD that the pair was matched on.
        F.append(frames[:, PLAY_TOP:PLAY_BOTTOM].mean(axis=-1).astype(np.float32))
        ep.append(np.full(D, k, dtype=np.int32))
        if (k + 1) % 4 == 0:
            print(f"   {k+1}/{a.replays}", flush=True)

    ZT, ZR = np.concatenate(Zt), np.concatenate(Zr)
    HH = np.concatenate(H).astype(np.float32)
    FF = np.concatenate(F)
    E = np.concatenate(ep)
    print(f"\n{len(ZT)} frames from {len(Zt)} replays\n")

    # ---- 1 & 2: matched-HUD pairs -----------------------------------------
    rng = np.random.default_rng(0)
    ia, ib, moved = [], [], []
    tries = 0
    while len(ia) < a.pairs and tries < a.pairs * 400:
        tries += 1
        i = int(rng.integers(len(ZT)))
        j = int(rng.integers(len(ZT)))
        if E[i] != E[j] or i == j:
            continue                               # same replay: same characters
        if np.abs(HH[i] - HH[j]).max() > a.hud_tol:
            continue                               # HUD must match
        d = float(np.abs(FF[i] - FF[j]).mean())
        if d < a.min_pixel_move:
            continue                               # characters must have moved
        ia.append(i); ib.append(j); moved.append(d)
    if len(ia) < 50:
        raise SystemExit(f"only {len(ia)} matched pairs; loosen --hud-tol")
    ia, ib = np.array(ia), np.array(ib)

    # Unmatched control: random pairs from the same replay, any HUD.
    ca = rng.integers(0, len(ZT), size=len(ia))
    cb = rng.integers(0, len(ZT), size=len(ia))
    keep = E[ca] == E[cb]
    ca, cb = ca[keep], cb[keep]

    r_trained = cosine_rows(ZT[ia], ZT[ib])
    r_random = cosine_rows(ZR[ia], ZR[ib])
    r_ctrl_t = cosine_rows(ZT[ca], ZT[cb])
    r_ctrl_r = cosine_rows(ZR[ca], ZR[cb])

    print(f"{len(ia)} matched-HUD pairs, mean play-area pixel difference "
          f"{np.mean(moved):.1f}/255\n")
    print(f"{'':22} {'matched HUD':>12} {'any pair':>10}")
    print("-" * 46)
    print(f"{'trained encoder':22} {r_trained.mean():12.4f} {r_ctrl_t.mean():10.4f}")
    print(f"{'random-init encoder':22} {r_random.mean():12.4f} {r_ctrl_r.mean():10.4f}")

    # ---- 3: how much of the latent is just the HUD ------------------------
    n = len(ZT)
    perm = rng.permutation(n)
    tr_i, te_i = perm[: int(n * 0.8)], perm[int(n * 0.8) :]
    fit = fit_ridge(HH[tr_i], ZT[tr_i], names=[f"z{i}" for i in range(ZT.shape[1])],
                    alpha=1.0)
    r2 = np.array(list(fit.r2(HH[te_i], ZT[te_i]).values()))
    r2 = r2[np.isfinite(r2)]

    # ---- 4: predictability against probe quality --------------------------
    print()
    print(f"{'quantity':10} {'1-step predictability':>22} {'probe R^2 (recorded)':>22}")
    print("-" * 56)
    recorded = {"hp1": 0.820, "hp2": 0.863, "spirit1": 0.036, "spirit2": 0.015,
                "combo1": 0.493, "combo2": 0.422}
    pred = {}
    for c, name in enumerate(TARGETS):
        x = HH[:-1, c : c + 1]
        y = HH[1:, c]
        same = E[:-1] == E[1:]
        x, y = x[same], y[same]
        m = int(len(x) * 0.8)
        p = fit_ridge(x[:m], y[:m, None], names=[name], alpha=1.0)
        v = list(p.r2(x[m:], y[m:, None]).values())[0]
        pred[name] = float(v)
        print(f"{name:10} {v:22.4f} {recorded[name]:22.3f}")

    print()
    print(f"latent variance explained by the six HUD readings: "
          f"{r2.mean():.4f} mean, {np.median(r2):.4f} median")
    print(f"dimensions over 0.5 explained: {(r2 > 0.5).sum()}/{len(r2)}")

    gap = float(r_random.mean() - r_trained.mean())
    print()
    if r_trained.mean() > 0.9 and gap < 0:
        v = ("matched-HUD frames are near-identical in the trained latent and "
             "*more* separable in an untrained one: training removed the "
             "characters, and the objective is the cause")
    elif r_trained.mean() > 0.9:
        v = ("matched-HUD frames are near-identical in the latent, so the "
             "characters are not encoded; compare against the random control "
             "to see whether training or the architecture is responsible")
    else:
        v = ("the latent does separate matched-HUD frames, so the characters "
             "are encoded to some degree and the hypothesis is wrong")
    print(f"VERDICT: {v}")

    a.out.write_text(json.dumps({
        "matched_trained": float(r_trained.mean()),
        "matched_random": float(r_random.mean()),
        "control_trained": float(r_ctrl_t.mean()),
        "control_random": float(r_ctrl_r.mean()),
        "n_pairs": int(len(ia)), "mean_pixel_move": float(np.mean(moved)),
        "hud_explains_latent_mean": float(r2.mean()),
        "hud_explains_latent_median": float(np.median(r2)),
        "predictability": pred, "probe_r2_recorded": recorded,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
