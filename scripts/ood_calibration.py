"""Is the OOD guard measuring drift, or measuring "this is a prediction"?

    python -m scripts.ood_calibration --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --bank ~/bank_hud.npz --horizon 16

WHAT PROMPTED THIS
------------------
The first actor-critic run logged ``ood keep 0.035``: the guard truncated 96.5%
of every imagined rollout. Since truncation drops lambda to 0 at the cut, the
lambda-return degenerates to a one-step TD target almost everywhere -- so the
effective horizon collapses to 1, which is *shorter* than the GRPO baseline's 4
and destroys the entire reason for having a critic.

Before touching a threshold, the question has to be answered properly, because
the two possible causes call for opposite responses:

  (a) The guard is right, and imagined rollouts really do leave the corpus
      manifold within a step or two. Then the longer horizon is not available at
      all and the actor-critic's premise is wrong.

  (b) The guard is miscalibrated. It is fitted on **encoder** latents -- what
      real frames produce -- but scores **predictor** outputs. A prediction is
      systematically contracted toward the mean relative to an observation, even
      when it is an excellent prediction, so a two-sided typicality test flags it
      from the "too typical" side immediately. The guard would then be detecting
      the difference between a prediction and an observation, which is not drift.

THE MEASUREMENT THAT SEPARATES THEM
-----------------------------------
Roll the world model forward from real corpus starts under **the corpus's own
true actions**, and report, at every step, both the OOD verdict and the cosine
similarity to the true latent the encoder produced for that frame.

Cosine is the arbitrator. If step 1 is flagged at cosine 0.996 -- a prediction so
good it is what the whole model is built on -- the flag cannot be about leaving
the manifold, and (b) holds. If the flag rate tracks a genuine fall in cosine,
(a) holds.

The second fit reported here is the alternative calibration: the band taken over
**one-step predictor outputs** instead of encoder latents. That does not "bake in
the drift being measured", which is the objection in `rl/ood.py::fit`, because a
one-step prediction contains no compounded error -- it only removes the constant
contraction that every prediction has. Errors accumulated over sixteen steps
still show against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.model.loading import load_world_model
from sokubot.rl.ood import LatentOOD, OODConfig
from scripts.train_grpo import model_fingerprint, valid_starts


def roll_true(wm, Zt, At, idx, cfg, T, device):
    """Rollout under the corpus's own actions, plus the true latents to match."""
    off = torch.arange(cfg.history, device=device) - (cfg.history - 1)
    steps = torch.arange(T, device=device)
    z_ctx = Zt[idx[:, None] + off[None, :]].float()
    a_hist = At[idx[:, None] + off[None, :-1]].float()
    a_plan = At[idx[:, None] + steps[None, :]].float()
    with torch.no_grad():
        z_roll = wm.rollout(z_ctx, a_plan, a_hist)          # [B, T, D]
    z_true = Zt[idx[:, None] + 1 + steps[None, :]].float()  # [B, T, D]
    return z_roll, z_true


def report(tag: str, ood: LatentOOD, z_roll, z_true, out: dict,
           ref_norm: float | None = None) -> None:
    B, T, _ = z_roll.shape
    with torch.no_grad():
        s = ood(z_roll.reshape(B * T, -1)).view(B, T)
        excess, hard = ood.flags(z_roll.reshape(B * T, -1))
        hard = hard.view(B, T).float()
        cos = F.cosine_similarity(z_roll, z_true, dim=-1)
        # Norm is here to separate two very different failures that the
        # Mahalanobis score cannot tell apart. A rollout can leave the corpus
        # distribution by pointing somewhere new (cosine falls) or by running off
        # along the direction it already had (norm grows, cosine barely moves).
        # The second is a scale defect with a cheap fix; the first is not.
        nrm = z_roll.norm(dim=-1)
        nrm_true = z_true.norm(dim=-1)
    lo, hi = float(ood.lo), float(ood.hi)
    kept = (hard.cumsum(dim=1) == 0).float().mean(dim=0)
    print(f"\n=== {tag} | band [{lo:.1f}, {hi:.1f}] ===")
    print("  h  median  %below  %above  %hard  %kept   cosine   |z|   |z_true|")
    rows = []
    for t in range(T):
        st = s[:, t]
        row = {"h": t + 1, "median": float(st.median()),
               "below": float((st < lo).float().mean()),
               "above": float((st > hi).float().mean()),
               "hard": float(hard[:, t].mean()),
               "kept": float(kept[t]), "cosine": float(cos[:, t].mean()),
               "norm": float(nrm[:, t].mean()),
               "norm_true": float(nrm_true[:, t].mean())}
        rows.append(row)
        if t < 4 or (t + 1) % 4 == 0:
            print(f" {row['h']:2d}  {row['median']:6.1f}  {row['below']:6.1%} "
                  f" {row['above']:6.1%} {row['hard']:6.1%} {row['kept']:6.1%}  "
                  f"{row['cosine']:.4f}  {row['norm']:5.2f}  {row['norm_true']:5.2f}")
    out[tag] = {"lo": lo, "hi": hi, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--starts", type=int, default=4096)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--out", type=Path, default=Path("ood_calibration.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    wm, cfg, _ = load_world_model(a.wm, a.device)
    fp = model_fingerprint(wm)
    b = np.load(a.bank.expanduser())
    if "fingerprint" in b.files and str(b["fingerprint"]) != fp:
        raise SystemExit(f"bank was encoded by {b['fingerprint']}, model is {fp}")
    Z, A, E = b["z"], b["a"], b["ep"]
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    starts = valid_starts(E, cfg.history, a.horizon)
    rng = np.random.default_rng(a.seed)
    idx = torch.from_numpy(rng.choice(starts, size=min(a.starts, len(starts)),
                                      replace=False)).to(a.device)
    print(f"{len(idx)} rollouts x {a.horizon} steps under the corpus's own "
          f"actions\n", flush=True)

    z_roll, z_true = roll_true(wm, Zt, At, idx, cfg, a.horizon, a.device)
    out = {"wm": str(a.wm), "fingerprint": fp, "n": int(len(idx)),
           "horizon": a.horizon, "quantile": a.quantile}

    cfg_o = OODConfig(quantile=a.quantile)
    enc = LatentOOD(cfg.latent_dim, cfg_o).to(a.device)
    r = enc.fit(Zt.float())
    print(f"encoder fit: median {r['median']:.1f} vs D {r['dim']} | "
          f"latent var {r['latent_var']:.4f}")
    report("fit on encoder latents", enc, z_roll, z_true, out)

    # The alternative: calibrate against what a *good prediction* looks like.
    # One step only, from the same starts, under the true actions -- so it
    # contains the contraction every prediction has and none of the compounding
    # this is trying to detect.
    one = z_roll[:, 0]
    pred = LatentOOD(cfg.latent_dim, cfg_o).to(a.device)
    r2 = pred.fit(one)
    print(f"\none-step predictor fit: median {r2['median']:.1f} | "
          f"latent var {r2['latent_var']:.4f}")
    report("fit on one-step predictions", pred, z_roll, z_true, out)

    rows = out["fit on encoder latents"]["rows"]
    e = rows[0]
    last = rows[-1]
    print("\n" + "=" * 68)
    # Which way does the rollout leave the distribution? If the norm inflates
    # while cosine holds up, the drift is mostly scale and a renormalisation is
    # worth testing before concluding the horizon is unavailable.
    grow = last["norm"] / max(e["norm"], 1e-9)
    print(f"over {a.horizon} steps: |z| x{grow:.2f} "
          f"({e['norm']:.2f} -> {last['norm']:.2f}, truth stays near "
          f"{last['norm_true']:.2f}) while cosine falls "
          f"{e['cosine']:.4f} -> {last['cosine']:.4f}")
    if grow > 1.3 and last["cosine"] > 0.75:
        print("  The rollout is drifting mostly by INFLATING, not by pointing\n"
              "  somewhere else. That is a scale defect, and rescaling the\n"
              "  latent back to the corpus norm each step is worth an A/B\n"
              "  before accepting a short horizon as a fact.")
    if e["hard"] > 0.5 and e["cosine"] > 0.95:
        print(f"MISCALIBRATED. At h=1 the encoder-fitted guard hard-flags "
              f"{e['hard']:.1%} of states whose cosine to the truth is "
              f"{e['cosine']:.4f}. A one-step prediction that accurate has not "
              f"left the manifold; the guard is separating predictions from "
              f"observations, not in-distribution from out. Fit it on "
              f"predictions.")
    elif e["hard"] < 0.1:
        print(f"CALIBRATED at h=1 ({e['hard']:.1%} flagged, cosine "
              f"{e['cosine']:.4f}). Later truncation is measuring real drift, so "
              f"the short effective horizon is a fact about the world model "
              f"rather than about the threshold.")
    else:
        print(f"AMBIGUOUS: h=1 flags {e['hard']:.1%} at cosine "
              f"{e['cosine']:.4f}. Read the per-step table before changing "
              f"anything.")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
