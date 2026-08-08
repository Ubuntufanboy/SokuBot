"""Actor-critic advantages, and the parts shared with GRPO that must not drift.

    python -m pytest tests/test_ac.py -q
"""

from __future__ import annotations

import pytest
import torch

from sokubot.rl.ac import ACConfig, advantages_from_returns
from sokubot.rl.grpo import GRPOConfig


def test_acconfig_supplies_every_field_grpo_loss_reads():
    """`grpo_loss` is shared, so a missing field would be an AttributeError at
    step one of a GPU run. Subclassing is what guarantees this; the test is here
    so that replacing the subclass with a parallel dataclass fails loudly."""
    c = ACConfig()
    for f in ("clip_eps", "kl_coef", "kl_ref_coef", "entropy_coef",
              "max_log_ratio", "grad_clip", "epochs", "target_kl",
              "advantage_scale", "reward", "jitter_sigma"):
        assert hasattr(c, f), f
    assert isinstance(c, GRPOConfig)


def test_defaults_encode_the_design():
    c = ACConfig()
    assert c.group_size == 1, "a critic replaces the group; a group of 1 is the point"
    assert c.horizon == 16, "the critic bootstraps, so the rollout can outrun 0.27 s"


def test_advantage_is_return_minus_value():
    lam = torch.tensor([[3.0, 1.0]])
    val = torch.tensor([[1.0, 1.0]])
    a = advantages_from_returns(lam, val, torch.ones(1, 2), scale="none")
    assert torch.allclose(a, torch.tensor([[2.0, 0.0]]))


def test_dead_steps_are_zeroed_and_excluded_from_the_statistics():
    """A dead step is identically zero. Letting it into the standard deviation
    would shrink the divisor in proportion to how often trajectories terminate,
    so every surviving trajectory's advantage would inflate as the policy got
    *better* at scoring KOs -- a feedback loop with no error message."""
    lam = torch.tensor([[1.0, 5.0], [3.0, 0.0], [5.0, 0.0]])
    val = torch.zeros(3, 2)
    alive = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    a = advantages_from_returns(lam, val, alive)
    assert float(a[1, 1]) == 0.0 and float(a[2, 1]) == 0.0
    # Column 1 has one live entry, so it is its own mean -> advantage 0.
    assert float(a[0, 1]) == pytest.approx(0.0, abs=1e-5)
    # Column 0 is standardised over its three live entries.
    live = a[:, 0]
    assert float(live.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(live.std(unbiased=False)) == pytest.approx(1.0, rel=1e-3)


def test_normalisation_is_per_timestep_not_global():
    """Return-to-go shrinks as the horizon runs out, so one scalar for the whole
    trajectory would systematically over-weight early steps."""
    lam = torch.tensor([[10.0, 0.1], [-10.0, -0.1], [0.0, 0.0]])
    val = torch.zeros(3, 2)
    a = advantages_from_returns(lam, val, torch.ones(3, 2))
    # Both columns end up on the same scale despite differing by 100x.
    assert float(a[:, 0].std(unbiased=False)) == pytest.approx(
        float(a[:, 1].std(unbiased=False)), rel=1e-3)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must match"):
        advantages_from_returns(torch.zeros(2, 3), torch.zeros(2, 4),
                                torch.ones(2, 3))


def test_unknown_scale_is_rejected():
    with pytest.raises(ValueError, match="unknown scale"):
        advantages_from_returns(torch.zeros(2, 3), torch.zeros(2, 3),
                                torch.ones(2, 3), scale="group")


def test_group_centred_advantages_removes_state_variance_exactly():
    """The property the whole design rests on: state luck cancels, actions don't.

    Built so the two sources are separable by construction -- a large per-start
    offset and a small per-rollout term. A group mean must delete the first
    completely and keep the second, which is what makes it an *exact* conditional
    baseline rather than a good one.
    """
    from sokubot.rl.ac import group_centred_advantages

    torch.manual_seed(0)
    S, G, T = 32, 8, 4
    state = torch.randn(S, 1, T) * 10.0          # the 99.88%
    action = torch.randn(S, G, T) * 0.1          # the 0.12%
    lam_ret = (state + action).reshape(S * G, T)
    alive = torch.ones(S * G, T)

    adv = group_centred_advantages(lam_ret, alive, G, scale="none")
    centred_action = (action - action.mean(dim=1, keepdim=True)).reshape(S * G, T)
    assert torch.allclose(adv, centred_action, atol=1e-5)


def test_group_centring_rejects_a_batch_that_does_not_divide():
    from sokubot.rl.ac import group_centred_advantages

    with pytest.raises(ValueError, match="do not divide"):
        group_centred_advantages(torch.zeros(10, 4), torch.ones(10, 4), 4)


def test_two_sided_concatenation_keeps_groups_intact():
    """`cat` stacks the agent's rollouts then the opponent's, both group-major.

    If either half were interleaved instead, the view(-1, G, T) inside the
    centring would average across starts and the baseline would silently stop
    being conditional -- with no error and a plausible-looking number.
    """
    from sokubot.rl.ac import group_centred_advantages

    S, G, T = 4, 8, 2
    mine = torch.arange(S).repeat_interleave(G).float()[:, None].expand(-1, T)
    opp = mine + 100.0
    both = torch.cat([mine, opp], dim=0).contiguous()
    adv = group_centred_advantages(both, torch.ones_like(both), G, scale="none")
    # Every rollout in a group has the identical return here, so a correct
    # grouping centres all of them to exactly zero.
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)
