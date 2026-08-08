"""Does the banner channel still fire once the rollout is imagined?

    python -m scripts.banner_in_imagination \
        --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --probe ~/gate_base/reward_probe_banner.npz --bank ~/bank_hud.npz

THE GAP THIS CLOSES
-------------------
`scripts/banner_latent_probe.py` fits and scores the `ko_banner` channel on
**encoder** latents -- what real frames produce. The reward reads it on
**predictor** outputs, which are a different distribution: `ood_calibration.py`
measures imagined latents leaving the corpus band by h=3, with the norm falling
from 14.05 to 10.88 over sixteen steps.

So a channel with precision 0.844 on real frames can still be useless in the
only place it is used. Two ways, opposite in sign:

  * it never fires, `win`/`lose` is identically zero, and the run reads as a
    reward-shaping null when it is really a transfer failure;
  * it fires constantly, and the +-5 term goes back to being the noise the
    health detector was.

Neither is visible in the training log, because a reward term that is silently
zero and a reward term that is absent produce the same curve.

WHAT IS REPORTED
----------------
The channel's firing rate on imagined rollout states against its rate on real
encoder latents, at `RewardConfig.ko_banner_threshold`. Real frames are the
reference: the true `knockout` base rate is about 1.4%, and the channel fires at
roughly that rate on them. The ratio between the two is what matters -- it is the
factor by which imagination inflates the detector, and it bounds how much worse
precision can get in the place the reward actually lives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.model.loading import load_world_model
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead
from sokubot.rl.reward import KO_BANNER, RewardConfig
from scripts.eval_policy import RCFG, build_reference
from scripts.train_grpo import model_fingerprint, valid_starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--starts", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("banner_in_imagination.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    wm, cfg, _ = load_world_model(a.wm, a.device)
    fp = model_fingerprint(wm)
    d = np.load(a.probe.expanduser(), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    if "ko_banner" not in names:
        raise SystemExit(f"{a.probe} has no ko_banner channel: {names}")
    if names.index("ko_banner") != KO_BANNER:
        raise SystemExit(
            f"ko_banner sits at {names.index('ko_banner')} but the reward reads "
            f"index {KO_BANNER} positionally")
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=names)

    b = np.load(a.bank.expanduser())
    if "fingerprint" in b.files and str(b["fingerprint"]) != fp:
        raise SystemExit(f"bank encoded by {b['fingerprint']}, model is {fp}")
    Z, A, E = b["z"], b["a"], b["ep"]
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    starts = valid_starts(E, cfg.history, a.horizon)
    idx = torch.from_numpy(
        np.random.default_rng(a.seed).choice(starts, size=a.starts)).to(a.device)
    pol = build_reference(A, cfg, a.device)
    arena = ImaginedArena(wm, ProbeHead(probe).to(a.device),
                          GRPOConfig(horizon=a.horizon, reward=RCFG),
                          cfg.history, cfg.action_ticks)
    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    with torch.no_grad():
        tr = arena.rollout(Zt[idx[:, None] + off[None, :]].float(),
                           At[idx[:, None] + off[None, :-1]].float(),
                           torch.randint(0, 2, (len(idx),), device=a.device),
                           pol, PolicyOpponent(pol))
        imag = tr["states"][..., KO_BANNER].flatten().float().cpu()
    real = torch.from_numpy(
        probe.predict(Z.astype(np.float32))[:, KO_BANNER]).float()

    thr = RewardConfig().ko_banner_threshold
    fi = float((imag >= thr).float().mean())
    fr = float((real >= thr).float().mean())
    ratio = fi / max(fr, 1e-12)
    print(f"threshold {thr}\n")
    print(f"  imagined rollout states  mean {imag.mean():.4f}  "
          f"p99 {imag.quantile(0.99):.4f}  fires {fi:.3%}")
    print(f"  real encoder latents     mean {real.mean():.4f}  "
          f"p99 {real.quantile(0.99):.4f}  fires {fr:.3%}")
    print(f"\n  imagination inflates the detector {ratio:.2f}x")
    print("=" * 68)
    if fi < 1e-4:
        print("The channel never fires in imagination. `win`/`lose` would be "
              "identically\nzero and the run would look like a reward-shaping "
              "null. Do not trust any\narm using --ko-source banner until this "
              "is fixed.")
    elif ratio > 5:
        print(f"{ratio:.1f}x over-firing. Precision in imagination is at most "
              f"1/{ratio:.1f} of the\n0.844 measured on real frames, so the +-5 "
              f"term is drifting back toward the\nnoise the health detector was. "
              f"Raise ko_banner_threshold.")
    else:
        print(f"Fires at {fi:.2%} against {fr:.2%} on real frames -- "
              f"{ratio:.1f}x. If every extra\nfire were spurious, imagined "
              f"precision would still be about "
              f"{0.844 / max(ratio, 1.0):.2f},\nagainst the health detector's "
              f"0.003. The term is live and worth paying.")
    a.out.write_text(json.dumps(
        {"wm": str(a.wm), "fingerprint": fp, "threshold": thr,
         "fire_imagined": fi, "fire_real": fr, "ratio": ratio,
         "mean_imagined": float(imag.mean()), "mean_real": float(real.mean())},
        indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
