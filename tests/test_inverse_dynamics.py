"""The term that decides what the representation keeps."""

from __future__ import annotations

import pytest
import torch

from sokubot.config import Config
from sokubot.model.inverse_dynamics import (InverseDynamicsHead,
                                            inverse_dynamics_loss)
from sokubot.model.world_model import LeWorldModel


def test_gradient_reaches_the_encoder():
    """The whole point, and the one thing `counterfactual_loss` deliberately
    does not do. If this is ever detached the term becomes decorative."""
    cfg = Config.tiny()
    model = LeWorldModel(cfg)
    obs = torch.rand(2, 3, 3, cfg.image_size, cfg.image_size)
    actions = (torch.rand(2, 3, cfg.action_ticks, cfg.action_dim) > 0.9).float()
    out = model(obs, actions)
    loss, _ = inverse_dynamics_loss(model.idm_head, out.z, actions)
    loss.backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads), "encoder gradients are all zero"


def test_balanced_accuracy_is_not_fooled_by_never_pressing():
    """Humans hold ~9.85% of buttons, so plain accuracy reads ~0.90 for a head
    that always says 'released'. The reported figure must not."""
    cfg = Config.tiny()
    head = InverseDynamicsHead(cfg, width=32)
    with torch.no_grad():
        for p in head.net[-1].parameters():
            p.fill_(0.0)
        head.net[-1].bias.fill_(-50.0)          # always predicts "released"
    z = torch.randn(8, 3, cfg.latent_dim)
    actions = (torch.rand(8, 3, cfg.action_ticks, cfg.action_dim) < 0.10).float()
    _, m = inverse_dynamics_loss(head, z, actions)
    assert m["idm_recall_pressed"] == pytest.approx(0.0, abs=1e-6)
    assert m["idm_acc"] == pytest.approx(0.5, abs=1e-6)


def test_it_predicts_the_action_that_caused_the_transition():
    """Index alignment: the chunk at t carries state t to t+1, so the head that
    reads (z_t, z_{t+1}) must be scored against actions[t], not actions[t+1].
    Off by one here trains the model to predict the future, silently."""
    cfg = Config.tiny()
    head = InverseDynamicsHead(cfg, width=64)
    B, T = 16, 3
    z = torch.zeros(B, T, cfg.latent_dim)
    actions = torch.zeros(B, T, cfg.action_ticks, cfg.action_dim)
    # Make the transition a pure function of the action applied at that step.
    mark = torch.rand(B, T - 1) > 0.5
    actions[:, :-1, :, 0] = mark[..., None].float()
    for t in range(T - 1):
        z[:, t + 1] = z[:, t] + mark[:, t : t + 1].float()
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(300):
        loss, _ = inverse_dynamics_loss(head, z, actions)
        opt.zero_grad(); loss.backward(); opt.step()
    _, m = inverse_dynamics_loss(head, z, actions)
    assert m["idm_acc"] > 0.95, f"learnable signal not learned: {m}"


def test_head_absent_when_switched_off():
    cfg = Config.tiny()
    cfg.idm_coef = 0.0
    assert LeWorldModel(cfg).idm_head is None


def test_needs_a_transition():
    cfg = Config.tiny()
    head = InverseDynamicsHead(cfg, width=16)
    with pytest.raises(ValueError, match="two timesteps"):
        inverse_dynamics_loss(head, torch.randn(2, 1, cfg.latent_dim),
                              torch.zeros(2, 1, cfg.action_ticks, cfg.action_dim))
