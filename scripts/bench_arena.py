"""How much faster is imagination than the game it replaces?

    python -m scripts.bench_arena --ckpt /root/ckpt/best.pt

Starting GRPO from random weights is only affordable if the imagined
environment is enormously faster than real Hisoutensoku. Real play produces 15
decisions per second per instance (60 fps decimated by frame_skip). Imagination
produces one decision per rollout step for every member of the batch at once,
so the comparison is decisions per wall-clock second.

This measures the whole loop -- policy sample, opponent sample, action jitter,
predictor step, probe read, reward -- not just the predictor, because the parts
around the predictor are what usually turn out to dominate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import (GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead,
                             group_advantages, grpo_loss)
from sokubot.rl.policy import SokuPolicy

REAL_DECISIONS_PER_SEC = 15.0          # 60 fps / frame_skip 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--batch", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    if a.probe and a.probe.exists():
        d = np.load(a.probe, allow_pickle=True)
        probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                            W=d["W"], names=list(d["names"]))
    else:
        rng = np.random.default_rng(0)
        probe = LinearProbe(zmu=np.zeros(cfg.latent_dim), zsd=np.ones(cfg.latent_dim),
                            ymu=np.zeros(6), ysd=np.ones(6),
                            W=rng.normal(size=(cfg.latent_dim + 1, 6)) * 0.01,
                            names=["hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2"])

    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    print(f"policy {sum(p.numel() for p in policy.parameters())/1e6:.2f}M | "
          f"horizon {a.horizon} steps = {a.horizon*cfg.frame_skip/60:.2f}s of game time")
    print(f"{'batch':>7} {'roll s':>8} {'update s':>9} {'decisions/s':>13} "
          f"{'x real-time':>12} {'game-h / wall-h':>16}")

    for B in a.batch:
        gcfg = GRPOConfig(horizon=a.horizon, group_size=8)
        arena = ImaginedArena(wm, ProbeHead(probe).to(a.device), gcfg,
                              cfg.history, cfg.action_ticks)
        z_ctx = torch.randn(B, cfg.history, cfg.latent_dim, device=a.device)
        a_hist = torch.zeros(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                             device=a.device)
        side = torch.randint(0, 2, (B,), device=a.device)

        for _ in range(2):                       # warm up kernels and allocator
            arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(policy))
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(a.iters):
            traj = arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(policy))
        torch.cuda.synchronize()
        t_roll = (time.time() - t0) / a.iters

        adv = group_advantages(traj["reward"], traj["alive"], gcfg.group_size, gcfg.gamma)
        flat_side = side[:, None].expand(B, a.horizon).reshape(-1)
        obs = traj["obs"].reshape(-1, cfg.history, cfg.latent_dim)
        acts = traj["mine"].reshape(-1, cfg.action_ticks, 10)
        with torch.no_grad():
            old_logp = policy.log_prob_of(obs, flat_side, acts)[0].view(B, a.horizon)

        t0 = time.time()
        for _ in range(a.iters):
            loss, _ = grpo_loss(policy, traj, adv, old_logp, gcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t_upd = (time.time() - t0) / a.iters

        dps = B * a.horizon / (t_roll + t_upd)
        # The speedup is dimensionless, so game-hours per wall-hour equals it.
        speedup = dps / REAL_DECISIONS_PER_SEC
        print(f"{B:7d} {t_roll:8.3f} {t_upd:9.3f} {dps:13,.0f} "
              f"{speedup:11,.0f}x {speedup:16,.0f}")

    print(f"\nreal Soku is {REAL_DECISIONS_PER_SEC:.0f} decisions/s per instance "
          f"(60 fps / frame_skip {cfg.frame_skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
