"""Damage avoided, and the two ways the comparison can be quietly invalidated."""

from __future__ import annotations

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.counterfactual import anneal, damage_avoided
from sokubot.rl.grpo import GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead
from sokubot.rl.policy import SokuPolicy

NAMES = ["hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2",
         "cards1", "cards2"]


def _arena(horizon=3, jitter=1.0):
    cfg = Config.tiny(base=Config.soku())
    wm = LeWorldModel(cfg).eval()
    D = cfg.latent_dim
    rng = np.random.default_rng(0)
    probe = LinearProbe(zmu=np.zeros(D, np.float32), zsd=np.ones(D, np.float32),
                        ymu=np.zeros(len(NAMES), np.float32),
                        ysd=np.ones(len(NAMES), np.float32),
                        W=(rng.standard_normal((D + 1, len(NAMES))) * 0.01
                           ).astype(np.float32),
                        names=list(NAMES))
    gcfg = GRPOConfig(horizon=horizon)
    gcfg.jitter_sigma = jitter
    return cfg, ImaginedArena(wm, ProbeHead(probe), gcfg, cfg.history,
                              cfg.action_ticks)


def _start(cfg, B=6):
    torch.manual_seed(0)
    return (torch.randn(B, cfg.history, cfg.latent_dim),
            torch.rand(B, cfg.history - 1, cfg.action_ticks,
                       cfg.action_dim).round(),
            torch.zeros(B, dtype=torch.long))


def test_shape_and_finiteness():
    cfg, arena = _arena()
    zc, ah, side = _start(cfg)
    pol = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks)
    tr = arena.rollout(zc, ah, side, pol, PolicyOpponent(pol))
    av = damage_avoided(arena, zc, ah, side, tr)
    assert av.shape == tr["terms"]["taken"].shape
    assert torch.isfinite(av).all()


def test_jitter_is_restored_even_though_it_is_switched_off_inside():
    """The counterfactual disables jitter so the recorded opponent actions are
    replayed exactly. If it failed to restore it, every subsequent rollout in
    training would silently become deterministic."""
    cfg, arena = _arena(jitter=1.0)
    zc, ah, side = _start(cfg)
    pol = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks)
    tr = arena.rollout(zc, ah, side, pol, PolicyOpponent(pol))
    damage_avoided(arena, zc, ah, side, tr)
    assert arena.cfg.jitter_sigma == 1.0


def test_a_null_agent_avoids_nothing():
    """The term must be exactly zero for a policy that does nothing, or it could
    be farmed by idling -- which is the opposite of the lesson."""
    from sokubot.rl.counterfactual import _Frozen
    cfg, arena = _arena(jitter=0.0)
    zc, ah, side = _start(cfg)
    B, T = zc.shape[0], arena.cfg.horizon
    null = torch.zeros(B, T + 1, cfg.action_ticks, 10)
    pol = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks)
    tr = arena.rollout(zc, ah, side, _Frozen(null), PolicyOpponent(pol))
    av = damage_avoided(arena, zc, ah, side, tr)
    assert torch.allclose(av, torch.zeros_like(av), atol=1e-6)


def test_the_opponent_is_pinned_across_the_two_arms():
    """The whole comparison rests on the opponent doing the same thing in both
    arms. Running it twice from one trajectory must give the same answer; if the
    opponent were re-rolled or re-jittered it would not."""
    cfg, arena = _arena(jitter=1.0)
    zc, ah, side = _start(cfg)
    pol = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks)
    tr = arena.rollout(zc, ah, side, pol, PolicyOpponent(pol))
    a1 = damage_avoided(arena, zc, ah, side, tr)
    a2 = damage_avoided(arena, zc, ah, side, tr)
    assert torch.allclose(a1, a2, atol=1e-6)


def test_anneal_halves_on_schedule():
    assert anneal(0, 1.0, 1000) == 1.0
    assert abs(anneal(1000, 1.0, 1000) - 0.5) < 1e-9
    assert abs(anneal(2000, 1.0, 1000) - 0.25) < 1e-9
    assert anneal(5000, 0.0, 1000) == 0.0
    assert anneal(5000, 1.0, 0) == 1.0          # 0 half-life disables decay
