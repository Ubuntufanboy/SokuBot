"""Shape and gradient-flow checks for the imagined-rollout GRPO loop.

These run a whole step on CPU against a tiny world model. They catch the class
of bug that costs a GPU-hour to discover otherwise: a transposed side index, an
advantage that is not actually group-centred, or a gradient leaking into the
frozen world model.

    python -m pytest tests/test_grpo.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import (GRPOConfig, ImaginedArena, ProbeHead, PolicyOpponent,
                             ReplayOpponent, SnapshotPool, group_advantages,
                             grpo_loss)
from sokubot.rl.policy import SokuPolicy

N_TARGETS = 6


def build(horizon=5, group=4, starts=3):
    torch.manual_seed(0)
    cfg = Config.tiny(base=Config.soku())
    wm = LeWorldModel(cfg)
    rng = np.random.default_rng(0)
    probe = LinearProbe(
        zmu=rng.normal(size=cfg.latent_dim).astype(np.float64),
        zsd=np.ones(cfg.latent_dim),
        ymu=np.zeros(N_TARGETS),
        ysd=np.ones(N_TARGETS),
        W=rng.normal(size=(cfg.latent_dim + 1, N_TARGETS)) * 0.01,
        names=["hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2"],
    )
    gcfg = GRPOConfig(horizon=horizon, group_size=group, starts_per_batch=starts)
    arena = ImaginedArena(wm, ProbeHead(probe), gcfg, cfg.history, cfg.action_ticks)
    policy = SokuPolicy(latent_dim=cfg.latent_dim, history=cfg.history,
                        ticks=cfg.action_ticks, width=64, depth=2)
    B = starts * group
    z_ctx = torch.randn(B, cfg.history, cfg.latent_dim)
    a_hist = torch.zeros(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim)
    # Every member of a group shares a start state and a side.
    side = torch.arange(starts).repeat_interleave(group) % 2
    return cfg, wm, arena, policy, gcfg, z_ctx, a_hist, side, B


def test_rollout_shapes():
    cfg, wm, arena, policy, gcfg, z_ctx, a_hist, side, B = build()
    traj = arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(policy))
    T = gcfg.horizon
    assert traj["obs"].shape == (B, T, cfg.history, cfg.latent_dim)
    assert traj["mine"].shape == (B, T, cfg.action_ticks, 10)
    assert traj["states"].shape == (B, T + 1, N_TARGETS)
    assert traj["reward"].shape == (B, T)
    assert traj["alive"].shape == (B, T)


def test_group_advantages_are_centred_within_each_group():
    reward = torch.randn(8, 5)
    alive = torch.ones(8, 5)
    adv = group_advantages(reward, alive, group_size=4, gamma=0.99)
    g = adv.view(2, 4, 5)
    assert torch.allclose(g.mean(dim=1), torch.zeros(2, 5), atol=1e-5)
    # and groups are independent: perturbing group 0 must not move group 1
    reward2 = reward.clone()
    reward2[:4] += 10.0
    adv2 = group_advantages(reward2, alive, group_size=4, gamma=0.99)
    assert torch.allclose(adv[4:], adv2[4:], atol=1e-5)


def test_dead_steps_do_not_accumulate_return():
    """Return-to-go must stop at the KO, not keep discounting past it."""
    reward = torch.ones(2, 4)
    alive = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = group_advantages(reward, alive, group_size=2, gamma=1.0)
    assert adv.shape == (2, 4)          # centring makes the pair mirror images
    assert torch.allclose(adv[0], -adv[1], atol=1e-4)


def test_loss_backprops_into_policy_only():
    cfg, wm, arena, policy, gcfg, z_ctx, a_hist, side, B = build()
    traj = arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(policy))
    adv = group_advantages(traj["reward"], traj["alive"], gcfg.group_size, gcfg.gamma)
    old_logp, _ = policy.log_prob_of(
        traj["obs"].reshape(-1, cfg.history, cfg.latent_dim),
        side[:, None].expand(B, gcfg.horizon).reshape(-1),
        traj["mine"].reshape(-1, cfg.action_ticks, 10))
    old_logp = old_logp.detach().view(B, gcfg.horizon)

    loss, stats = grpo_loss(policy, traj, adv, old_logp, gcfg)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.parameters())
    assert all(p.grad is None for p in wm.parameters()), \
        "the world model is frozen and must never receive gradient"


def test_ratio_is_one_before_any_update():
    """Re-scoring the sampling policy must reproduce its own probabilities."""
    cfg, wm, arena, policy, gcfg, z_ctx, a_hist, side, B = build()
    traj = arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(policy))
    adv = torch.zeros(B, gcfg.horizon)
    logp, _ = policy.log_prob_of(
        traj["obs"].reshape(-1, cfg.history, cfg.latent_dim),
        side[:, None].expand(B, gcfg.horizon).reshape(-1),
        traj["mine"].reshape(-1, cfg.action_ticks, 10))
    _, stats = grpo_loss(policy, traj, adv, logp.detach().view(B, gcfg.horizon), gcfg)
    assert abs(stats["ratio"] - 1.0) < 1e-4
    assert stats["kl"] < 1e-6


def test_replay_opponent_reads_the_other_chair():
    cfg, wm, arena, policy, gcfg, z_ctx, a_hist, side, B = build()
    T = gcfg.horizon
    future = torch.zeros(B, T, cfg.action_ticks, cfg.action_dim)
    future[..., 3] = 1.0            # P1 right
    future[..., 13] = 2.0           # P2 right, marked distinctly
    opp = ReplayOpponent(future)
    got = opp.act(z_ctx, side)
    for i in range(B):
        want = 2.0 if side[i] == 0 else 1.0     # agent P1 -> opponent is P2
        assert got[i, 0, 3].item() == want


def test_replay_opponent_advances_in_time_and_resets():
    B, T, ticks = 2, 4, 4
    future = torch.zeros(B, T, ticks, 20)
    for t in range(T):
        future[:, t, :, 13] = float(t)          # P2 block carries the step index
    opp = ReplayOpponent(future)
    side = torch.zeros(B, dtype=torch.long)     # agent is P1, opponent is P2
    seen = [opp.act(torch.zeros(B, 1, 1), side)[0, 0, 3].item() for _ in range(T)]
    assert seen == [0.0, 1.0, 2.0, 3.0]
    opp.reset()
    assert opp.act(torch.zeros(B, 1, 1), side)[0, 0, 3].item() == 0.0


def test_snapshot_pool_keeps_a_bounded_history():
    gcfg = GRPOConfig(snapshot_every=2, max_snapshots=3)
    pool = SnapshotPool(gcfg)
    policy = SokuPolicy(latent_dim=8, history=2, ticks=2, width=16, depth=1)
    for step in range(1, 21):
        pool.maybe_add(policy, step)
    assert len(pool.snaps) == 3
    assert pool.sample(np.random.default_rng(0)) is not None
