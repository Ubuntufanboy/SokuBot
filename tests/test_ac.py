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
