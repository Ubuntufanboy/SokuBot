"""Where does an imagined rollout go, and is it anywhere the encoder would put it?

    python -m scripts.latent_drift_test --ckpt /root/ckpt_cf/best.pt

The horizon ablation already records that a rollout decorrelates from the truth
it should be tracking: cosine between the imagined latent and the real one falls
0.86 -> 0.32 over sixteen steps, and relative L2 error passes 1.0 by step 24 --
worse than predicting the zero vector. What it does not say is *why*, and the
two candidate reasons want opposite fixes.

  ACCUMULATED ERROR   Each step is a little wrong; the errors compound. The
                      trajectory stays somewhere the encoder might have put a
                      real frame, it is just the wrong frame. Shorter horizons
                      and multi-step training help.

  MEAN COLLAPSE       The predictor is trained with MSE, whose optimum under a
                      genuinely multi-modal future is the *average* of that
                      future -- a blur that corresponds to no game state at
                      all. Feeding it back as though it were a sample walks the
                      rollout toward the corpus mean and off the manifold the
                      encoder occupies. No amount of horizon tuning fixes this;
                      the latent has to become stochastic.

The two are distinguishable. Mean collapse has a signature that accumulated
error does not: shrinkage. A conditional mean is systematically *smaller* than
the samples it averages, so the rollout's latents lose norm, lose per-dimension
spread, and move closer to the corpus mean with every step. Accumulated error
preserves scale and wanders.

Rollouts here use the actions that were really pressed, from the capture the
start was drawn from, so the comparison against the real future latent is
apples to apples and nothing is attributable to the policy.

Every measurement is reported against two baselines, because a number without
one means nothing here:

  identity    copy the start latent forward and never update it. A model that
              cannot beat this has learned no dynamics at all.
  real        the same statistic computed on true encoder latents, which is
              what "correct" looks like for spread and distance-to-mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--bank", type=Path, default=Path("/root/bank_best.npz"))
    ap.add_argument("--starts", type=int, default=8192)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/latent_drift.json"))
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(a.device)
    A = torch.from_numpy(bank["a"]).to(a.device)
    ep = bank["ep"]
    H, T = cfg.history, a.horizon

    # Starts whose whole window and whole future stay inside one capture.
    same = np.zeros(len(ep), dtype=bool)
    same[: len(ep) - (H + T)] = (ep[: len(ep) - (H + T)] == ep[H + T :])
    ok = np.flatnonzero(same)
    ok = ok[ok >= H - 1]
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(ok, size=min(a.starts, len(ok)),
                                      replace=False)).to(a.device)
    print(f"{len(idx)} starts, horizon {T} ({T * cfg.frame_skip / 60:.2f} s)\n")

    mu = Z.float().mean(0)                     # corpus mean latent
    off = torch.arange(H, device=a.device) - (H - 1)

    acc = {k: torch.zeros(T, dtype=torch.float64, device=a.device)
           for k in ("cos", "cos_id", "nrm", "nrm_real", "d_mu", "d_mu_real",
                     "rel_l2", "rel_l2_id")}
    sum_hat = torch.zeros(T, cfg.latent_dim, dtype=torch.float64, device=a.device)
    sq_hat = torch.zeros_like(sum_hat)
    sum_real = torch.zeros_like(sum_hat)
    sq_real = torch.zeros_like(sum_hat)
    n = 0

    for s in range(0, len(idx), a.chunk):
        base = idx[s : s + a.chunk]
        B = len(base)
        z_win = Z[base[:, None] + off[None, :]].float()
        z_start = z_win[:, -1].clone()
        a_win = A[base[:, None] + off[None, :]][:, :-1].float()
        for t in range(T):
            # The buttons really pressed at this step, from the same capture.
            act = A[base + t].float()
            zhat = wm.predictor(z_win, wm.action_encoder(
                torch.cat([a_win, act[:, None]], dim=1)))[:, -1]
            ztrue = Z[base + t + 1].float()

            def cos(x, y):
                return torch.nn.functional.cosine_similarity(x, y, dim=1).double()

            acc["cos"][t] += cos(zhat, ztrue).sum()
            acc["cos_id"][t] += cos(z_start, ztrue).sum()
            acc["nrm"][t] += zhat.norm(dim=1).double().sum()
            acc["nrm_real"][t] += ztrue.norm(dim=1).double().sum()
            acc["d_mu"][t] += (zhat - mu).norm(dim=1).double().sum()
            acc["d_mu_real"][t] += (ztrue - mu).norm(dim=1).double().sum()
            den = ztrue.norm(dim=1).double().clamp_min(1e-8)
            acc["rel_l2"][t] += ((zhat - ztrue).norm(dim=1).double() / den).sum()
            acc["rel_l2_id"][t] += ((z_start - ztrue).norm(dim=1).double() / den).sum()
            sum_hat[t] += zhat.double().sum(0)
            sq_hat[t] += (zhat.double() ** 2).sum(0)
            sum_real[t] += ztrue.double().sum(0)
            sq_real[t] += (ztrue.double() ** 2).sum(0)

            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            if H > 1:
                a_win = torch.cat([a_win[:, 1:], act[:, None]], dim=1)
        n += B
        print(f"   {n}/{len(idx)}", flush=True)

    for k in acc:
        acc[k] /= n
    std_hat = (sq_hat / n - (sum_hat / n) ** 2).clamp_min(0).sqrt().mean(1)
    std_real = (sq_real / n - (sum_real / n) ** 2).clamp_min(0).sqrt().mean(1)

    print(f"\n{'h':>3} {'sec':>6} {'cos':>7} {'cos id':>7} {'relL2':>7} "
          f"{'relL2 id':>9} {'norm/real':>10} {'spread/real':>12} {'dmu/real':>9}")
    print("-" * 82)
    rows = []
    for t in range(T):
        r = {"h": t + 1, "seconds": (t + 1) * cfg.frame_skip / 60.0,
             "cos": float(acc["cos"][t]), "cos_identity": float(acc["cos_id"][t]),
             "rel_l2": float(acc["rel_l2"][t]),
             "rel_l2_identity": float(acc["rel_l2_id"][t]),
             "norm_ratio": float(acc["nrm"][t] / acc["nrm_real"][t]),
             "spread_ratio": float(std_hat[t] / std_real[t]),
             "dmu_ratio": float(acc["d_mu"][t] / acc["d_mu_real"][t])}
        rows.append(r)
        if r["h"] in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            print(f"{r['h']:>3} {r['seconds']:6.2f} {r['cos']:7.4f} "
                  f"{r['cos_identity']:7.4f} {r['rel_l2']:7.4f} "
                  f"{r['rel_l2_identity']:9.4f} {r['norm_ratio']:10.4f} "
                  f"{r['spread_ratio']:12.4f} {r['dmu_ratio']:9.4f}")

    last = rows[min(15, T - 1)]
    print()
    print(f"at h={last['h']} ({last['seconds']:.2f} s, the GRPO horizon):")
    print(f"  cosine to truth      {last['cos']:.4f}   "
          f"(copy-the-start baseline {last['cos_identity']:.4f})")
    print(f"  relative L2          {last['rel_l2']:.4f}   "
          f"(copy-the-start baseline {last['rel_l2_identity']:.4f})")
    print(f"  norm vs real         {last['norm_ratio']:.4f}")
    print(f"  per-dim spread       {last['spread_ratio']:.4f}")
    print(f"  distance to mean     {last['dmu_ratio']:.4f}")
    print()
    shrunk = last["spread_ratio"] < 0.7 or last["dmu_ratio"] < 0.7
    beats_id = last["rel_l2"] < last["rel_l2_identity"]
    if shrunk and not beats_id:
        v = ("the rollout collapses toward the corpus mean and does not even "
             "beat copying the start latent: MSE mean-collapse, and the latent "
             "has to become stochastic")
    elif shrunk:
        v = ("the rollout beats the identity baseline but shrinks steadily "
             "toward the corpus mean -- a conditional-mean blur, which is what "
             "MSE gives on a multi-modal future")
    elif not beats_id:
        v = ("the rollout keeps its scale but is no better than copying the "
             "start: the predictor has learned little usable dynamics at this "
             "horizon")
    else:
        v = ("the rollout keeps its scale and beats the identity baseline, so "
             "the drift is accumulated error rather than mean collapse")
    print(f"VERDICT: {v}")
    a.out.write_text(json.dumps({"ckpt": str(a.ckpt), "starts": n,
                                 "curve": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
