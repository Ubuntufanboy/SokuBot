"""The supervised state head, and the two ways it could quietly do nothing.

The failure modes worth testing are not "does it run". They are:

  * training on padded frames as if they were measured, which `label_valid`
    exists to prevent and which nothing downstream could detect; and
  * a binary channel at a 0.04% base rate being optimised to "never", which
    scores 99.96% accuracy and has learned nothing at all.
"""

from __future__ import annotations

import torch

from sokubot.config import Config
from sokubot.data.state import STATE_CHANNELS
from sokubot.model.state_head import (BINARY, CONTINUOUS, StateHead,
                                      default_pos_weight, state_loss)


def _cfg() -> Config:
    c = Config()
    c.latent_dim = 32
    return c


def test_the_head_shapes_per_player_and_channel():
    head = StateHead(_cfg())
    out = head(torch.randn(4, 7, 32))
    assert out.shape == (4, 7, 2, len(STATE_CHANNELS))


def test_channels_are_split_and_nothing_is_dropped():
    assert set(CONTINUOUS) | set(BINARY) == set(range(len(STATE_CHANNELS)))
    assert not set(CONTINUOUS) & set(BINARY)
    assert STATE_CHANNELS.index("dx") in CONTINUOUS
    assert STATE_CHANNELS.index("guarding") in BINARY


def test_padded_frames_do_not_contribute():
    """A frame marked invalid must not move the loss, however wrong it is.

    Built as: one batch where the invalid frames hold correct values, one where
    they hold garbage. If the mask works the two losses are identical.
    """
    torch.manual_seed(0)
    target = torch.zeros(2, 6, 2, len(STATE_CHANNELS))
    target[..., STATE_CHANNELS.index("dx")] = 0.4
    pred = torch.zeros_like(target)
    pred[..., STATE_CHANNELS.index("dx")] = 0.4

    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[:, -2:] = False

    clean, _ = state_loss(pred, target, valid)
    dirty_pred = pred.clone()
    dirty_pred[:, -2:] = 99.0            # nonsense, but on invalid frames
    dirty, _ = state_loss(dirty_pred, target, valid)
    assert torch.allclose(clean, dirty), "padded frames leaked into the loss"

    # And the same nonsense on a VALID frame must move it, or the test above
    # would pass for a loss that ignores everything.
    moved_pred = pred.clone()
    moved_pred[:, 0] = 99.0
    moved, _ = state_loss(moved_pred, target, valid)
    assert moved > clean * 10


def test_frame_count_reports_only_real_labels():
    valid = torch.ones(3, 10, dtype=torch.bool)
    valid[:, :4] = False
    x = torch.zeros(3, 10, 2, len(STATE_CHANNELS))
    _, m = state_loss(x, x, valid)
    assert m["state_frames"] == 18.0        # 3 * 6


def test_always_negative_is_not_rewarded_on_a_rare_channel():
    """`crushed` is 0.04% of frames. Predicting "never" scores 99.96% accuracy
    and must not also score a low loss, or the channel is decorative."""
    ci = STATE_CHANNELS.index("crushed")
    n = 2000
    target = torch.zeros(1, n, 2, len(STATE_CHANNELS))
    target[0, :2, :, ci] = 1.0             # 0.1% positive

    never = torch.full((1, n, 2, len(STATE_CHANNELS)), -10.0)
    honest = never.clone()
    honest[0, :2, :, ci] = 10.0            # gets the rare ones right

    pw = default_pos_weight()
    l_never, m_never = state_loss(never, target, None, pw)
    l_honest, _ = state_loss(honest, target, None, pw)
    assert l_honest < l_never, "weighting does not reward finding the rare class"
    # The accuracy metric alone would have called the degenerate head excellent.
    assert m_never["state_crushed_acc"] > 0.99


def test_pos_weight_is_clamped_so_the_rarest_channel_cannot_dominate():
    w = default_pos_weight()
    assert w.shape == (len(BINARY),)
    assert float(w.max()) <= 50.0
    # airborne is common, so its weight should be near 1 rather than large.
    ai = BINARY.index(STATE_CHANNELS.index("airborne"))
    assert 0.5 < float(w[ai]) < 2.0


def test_dx_r2_is_one_for_a_perfect_fit_and_low_for_a_constant():
    torch.manual_seed(1)
    dx = STATE_CHANNELS.index("dx")
    target = torch.zeros(1, 200, 2, len(STATE_CHANNELS))
    target[..., dx] = torch.randn(1, 200, 2)

    _, perfect = state_loss(target.clone(), target, None)
    assert perfect["state_dx_r2"] > 0.99

    flat = torch.zeros_like(target)
    _, useless = state_loss(flat, target, None)
    assert useless["state_dx_r2"] < 0.1


def test_gradients_reach_the_encoder_input():
    """The point is to shape the latent, so the gradient must flow back into z
    rather than stopping at the head."""
    head = StateHead(_cfg())
    z = torch.randn(2, 5, 32, requires_grad=True)
    target = torch.zeros(2, 5, 2, len(STATE_CHANNELS))
    loss, _ = state_loss(head(z), target, None)
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
