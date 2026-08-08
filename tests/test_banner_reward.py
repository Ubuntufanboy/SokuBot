"""The banner KO detector, and the guarantee that the old one is untouched."""

from __future__ import annotations

import pytest
import torch

from sokubot.rl.reward import (KO_BANNER, RewardConfig, banner_ko_masks,
                               compute_rewards, ko_mask, ko_masks, terminal_mask)


def _states(B, T, banner=None, hp1=None, hp2=None):
    """[B, T+1, 9] probe output with everything mid-range unless overridden."""
    s = torch.full((B, T + 1, KO_BANNER + 1), 0.5)
    s[..., KO_BANNER] = 0.0
    if banner is not None:
        s[..., KO_BANNER] = banner
    if hp1 is not None:
        s[..., 0] = hp1
    if hp2 is not None:
        s[..., 1] = hp2
    return s


def test_health_source_is_bit_identical_to_the_old_path():
    """Every recorded number was measured with the health detector.

    `ko_masks` was introduced as a dispatch point, so the health branch has to
    return exactly what the two direct `ko_mask` calls used to.
    """
    torch.manual_seed(0)
    B, T = 16, 8
    s = torch.rand(B, T + 1, KO_BANNER + 1)
    side = torch.randint(0, 2, (B,))
    cfg = RewardConfig()
    assert cfg.ko_source == "health", "the default must stay reproducible"
    from sokubot.rl.reward import _sides
    mine_hp, thr_hp, _, _ = _sides(s, side)
    me, them = ko_masks(s, side, cfg)
    assert torch.equal(me, ko_mask(mine_hp, cfg))
    assert torch.equal(them, ko_mask(thr_hp, cfg))


def test_banner_fires_only_when_lit_long_enough():
    cfg = RewardConfig(ko_source="banner", ko_persist=3, ko_margin=0.1)
    B, T = 1, 8
    # Lit for two steps only — one short of ko_persist.
    ban = torch.zeros(B, T + 1)
    ban[0, 3:5] = 1.0
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.9), hp2=torch.tensor(0.1))
    me, them = banner_ko_masks(s, torch.zeros(B, dtype=torch.long), cfg)
    assert not me.any() and not them.any()

    ban[0, 3:7] = 1.0                      # now four consecutive
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.9), hp2=torch.tensor(0.1))
    me, them = banner_ko_masks(s, torch.zeros(B, dtype=torch.long), cfg)
    assert them.any() and not me.any(), "P1 has more health, so P2 lost"


def test_the_winner_comes_from_the_health_difference_not_its_level():
    """The point of the split: an absolute level the probe cannot resolve is
    replaced by a comparison, which survives a large common-mode error."""
    cfg = RewardConfig(ko_source="banner", ko_persist=1, ko_margin=0.1)
    B, T = 1, 4
    ban = torch.ones(B, T + 1)
    ban[0, 0] = 0.0                        # started clear
    side = torch.zeros(B, dtype=torch.long)

    # Both readings are wrong by +0.4 -- far beyond the health detector's 0.06
    # threshold -- but their *difference* is intact.
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.45), hp2=torch.tensor(0.85))
    me, them = banner_ko_masks(s, side, cfg)
    assert me.any() and not them.any(), "P1 is lower, so P1 lost"

    s = _states(B, T, banner=ban, hp1=torch.tensor(0.85), hp2=torch.tensor(0.45))
    me, them = banner_ko_masks(s, side, cfg)
    assert them.any() and not me.any()


def test_an_unattributable_ko_pays_nothing():
    """A coin flip at +-5 is worth less than nothing."""
    cfg = RewardConfig(ko_source="banner", ko_persist=1, ko_margin=0.10)
    B, T = 1, 4
    ban = torch.ones(B, T + 1)
    ban[0, 0] = 0.0
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.50), hp2=torch.tensor(0.47))
    me, them = banner_ko_masks(s, torch.zeros(B, dtype=torch.long), cfg)
    assert not me.any() and not them.any()


def test_a_rollout_starting_inside_a_banner_is_not_paid():
    cfg = RewardConfig(ko_source="banner", ko_persist=1, ko_margin=0.1)
    B, T = 1, 4
    ban = torch.ones(B, T + 1)             # lit from the very first state
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.9), hp2=torch.tensor(0.1))
    me, them = banner_ko_masks(s, torch.zeros(B, dtype=torch.long), cfg)
    assert not me.any() and not them.any()


def test_reward_and_termination_agree_about_when_the_match_ended():
    """Split opinions would pay an outcome on a step the trajectory outlives."""
    cfg = RewardConfig(ko_source="banner", ko_persist=2, ko_margin=0.1)
    B, T = 4, 8
    ban = torch.zeros(B, T + 1)
    ban[:, 4:] = 1.0
    s = _states(B, T, banner=ban, hp1=torch.tensor(0.9), hp2=torch.tensor(0.2))
    side = torch.zeros(B, dtype=torch.long)
    actions = torch.zeros(B, T, 4, 20)
    _, alive, _ = compute_rewards(s, actions, side, cfg)
    term = terminal_mask(s, side, cfg)
    # `alive` is 1 up to and including the KO step; `terminal` marks that step.
    assert torch.equal(term.argmax(1), alive.sum(1).long() - 1)


def test_banner_source_needs_the_channel():
    cfg = RewardConfig(ko_source="banner")
    s = torch.rand(2, 5, 6)                # a six-channel probe, no ko_banner
    with pytest.raises(ValueError, match="at least"):
        ko_masks(s, torch.zeros(2, dtype=torch.long), cfg)


def test_unknown_source_is_rejected():
    cfg = RewardConfig(ko_source="vibes")
    with pytest.raises(ValueError, match="unknown ko_source"):
        ko_masks(torch.rand(2, 5, 9), torch.zeros(2, dtype=torch.long), cfg)
