"""The multi-step rollout loss, and the BatchNorm trap it is shaped to fall into.

Every check here guards a failure that produces plausible numbers rather than an
exception. The BatchNorm one in particular is `docs/BUGS.md` section 1 replayed:
a fine-tune that pushes off-distribution activations through a non-affine
BatchNorm corrupts its running statistics, and the saved checkpoint then scores
one-step skill -6.15 while its own blob records +0.82. Nothing threw then either.

    python -m pytest tests/test_rollout.py -q
"""

from __future__ import annotations

import copy

import pytest
import torch

from sokubot.config import Config
from sokubot.losses.prediction import rollout_loss
from sokubot.model.world_model import LeWorldModel
from scripts.finetune_predictor import freeze_batchnorm, valid_starts

import numpy as np


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.tiny(base=Config.soku())


@pytest.fixture(scope="module")
def model(cfg) -> LeWorldModel:
    torch.manual_seed(0)
    return LeWorldModel(cfg)


def _inputs(cfg, B=32, P=8):
    """Deliberately off-distribution: mean 5, std 3, where the latent is ~N(0,1)."""
    z = torch.randn(B, cfg.history, cfg.latent_dim) * 3 + 5
    a_hist = torch.rand(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim).round()
    a_plan = torch.rand(B, P, cfg.action_ticks, cfg.action_dim).round()
    return z, a_hist, a_plan


# --------------------------------------------------------------------------
# rollout_loss calibration
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_zero(cfg):
    zt = torch.randn(16, 4, cfg.latent_dim)
    z0 = torch.randn(16, cfg.latent_dim)
    loss, rep = rollout_loss(zt, zt, z0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in rep.values())


def test_copy_forward_scores_exactly_one(cfg):
    """The normaliser is copy-forward, so copying forward must score 1.0.

    This is what makes the per-horizon numbers readable without a reference:
    below 1 is better than doing nothing, above 1 is worse.
    """
    zt = torch.randn(16, 4, cfg.latent_dim)
    z0 = torch.randn(16, cfg.latent_dim)
    zhat = z0[:, None].expand_as(zt).contiguous()
    loss, rep = rollout_loss(zhat, zt, z0)
    assert float(loss) == pytest.approx(1.0, rel=1e-5)
    assert all(v == pytest.approx(1.0, rel=1e-5) for v in rep.values())


def test_normaliser_is_detached(cfg):
    """z0 is the model's own input, so an attached denominator is optimisable.

    Without the detach the loss could be lowered by making copy-forward *worse*
    rather than the prediction better -- an objective that rewards degrading the
    yardstick.
    """
    zt = torch.randn(8, 4, cfg.latent_dim)
    z0 = torch.randn(8, cfg.latent_dim, requires_grad=True)
    zhat = torch.randn(8, 4, cfg.latent_dim, requires_grad=True)
    rollout_loss(zhat, zt, z0)[0].backward()
    assert zhat.grad is not None and float(zhat.grad.abs().sum()) > 0
    assert z0.grad is None or float(z0.grad.abs().sum()) == pytest.approx(0.0)


def test_horizons_outside_the_rollout_are_rejected(cfg):
    zt = torch.randn(4, 3, cfg.latent_dim)
    z0 = torch.randn(4, cfg.latent_dim)
    with pytest.raises(ValueError, match="outside the rollout length"):
        rollout_loss(zt, zt, z0, horizons=[1, 4])


# --------------------------------------------------------------------------
# the BatchNorm trap
# --------------------------------------------------------------------------

def _bn_state(m):
    return [(b.running_mean.clone(), b.running_var.clone(),
             int(b.num_batches_tracked))
            for b in m.modules()
            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]


def test_unpinned_train_mode_does_corrupt_running_stats(cfg, model):
    """The positive control. Without this, the test below proves nothing.

    If an unpinned rollout did *not* move the statistics, `freeze_batchnorm`
    would be guarding a hazard that does not exist, and the test asserting the
    stats are unchanged would pass for the wrong reason forever.
    """
    m = copy.deepcopy(model).train()
    before = _bn_state(m)
    z, a_hist, a_plan = _inputs(cfg)
    m.rollout(z, a_plan, a_hist)
    after = _bn_state(m)
    assert any(not torch.allclose(a[0], b[0]) for a, b in zip(after, before))


def test_freeze_batchnorm_pins_stats_through_off_distribution_rollouts(cfg, model):
    m = copy.deepcopy(model)
    m.train()
    n = freeze_batchnorm(m)
    assert n > 0, "the projector is supposed to end in a BatchNorm"
    before = _bn_state(m)
    z, a_hist, a_plan = _inputs(cfg)
    for _ in range(5):
        out = m.rollout(z, a_plan, a_hist)
        rollout_loss(out, torch.randn_like(out), z[:, -1])[0].backward()
    after = _bn_state(m)
    for a, b in zip(after, before):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


def test_train_call_re_enables_batchnorm_so_pinning_must_be_repeated(cfg, model):
    """`model.train()` recurses, so it undoes the pin. The loop must re-apply it.

    This is the one way the guard silently stops working: pin once outside the
    loop, call `train()` inside it, and the statistics drift from step two on.
    """
    m = copy.deepcopy(model)
    m.train()
    freeze_batchnorm(m)
    bns = [b for b in m.modules()
           if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
    assert all(not b.training for b in bns)
    m.train()
    assert any(b.training for b in bns), "train() no longer re-enables BatchNorm"


# --------------------------------------------------------------------------
# gradient routing: the encoder defines the latent space and must not move
# --------------------------------------------------------------------------

def test_gradient_reaches_predictor_but_not_encoder(cfg, model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.predictor.parameters():
        p.requires_grad_(True)
    m.zero_grad(set_to_none=True)
    z, a_hist, a_plan = _inputs(cfg)
    out = m.rollout(z, a_plan, a_hist)
    rollout_loss(out, torch.randn_like(out), z[:, -1])[0].backward()
    assert sum(float(p.grad.abs().sum()) for p in m.predictor.parameters()
               if p.grad is not None) > 0
    assert all(p.grad is None for p in m.encoder.parameters())


# --------------------------------------------------------------------------
# start-index arithmetic
# --------------------------------------------------------------------------

def test_valid_starts_never_straddle_a_replay_boundary():
    """A window crossing a boundary sees health jump back to full.

    That is not a subtle error -- it teaches the predictor a transition that
    exists nowhere in the game -- but it is invisible in a loss curve.
    """
    ep = np.r_[np.zeros(50, int), np.ones(50, int), np.full(50, 2)]
    s = valid_starts(ep, history=3, plan=16)
    assert len(s) > 0
    assert (ep[s] == ep[s + 16]).all()
    assert (s >= 2).all()                       # room for history-1 context behind


def test_valid_starts_raises_when_replays_are_too_short():
    ep = np.repeat(np.arange(5), 8)
    with pytest.raises(SystemExit, match="no start states survive"):
        valid_starts(ep, history=3, plan=16)


def test_a_near_static_sample_cannot_dominate(cfg):
    """Copy-forward is nearly perfect during a freeze, so its error is nearly
    zero -- and a per-sample ratio then explodes and sets the whole batch mean.

    This is measured, not hypothetical: the first run of this loss reported
    training error 0.43-1.30 while validation on held-out replays read h4 = 364
    and h8 = 550, non-monotonic in the horizon, entirely from a few near-zero
    denominators. A ratio of sums is immune, because such a sample contributes
    nearly nothing to numerator and denominator alike.
    """
    torch.manual_seed(0)
    B, D = 64, cfg.latent_dim
    z0 = torch.randn(B, D)
    z_true = (z0[:, None] + torch.randn(B, 1, D) * 0.5).repeat(1, 3, 1)
    zhat = z_true + torch.randn(B, 3, D) * 0.1
    clean, _ = rollout_loss(zhat, z_true, z0)

    # One frozen sample: the latent does not move, so copy-forward is exact.
    z_true[0] = z0[0]
    zhat[0] = z0[0] + 1e-3
    frozen, rep = rollout_loss(zhat, z_true, z0)
    assert float(frozen) < float(clean) * 1.5, (
        f"one static sample moved the loss from {float(clean):.3f} to "
        f"{float(frozen):.3f}; the denominator is not robust")
    assert all(v < 10 for v in rep.values()), rep


def test_ratio_of_sums_matches_the_skill_definition(cfg):
    """`1 - rollout_loss` at h=1 must be `eval_ckpt.predictor_skill`'s quantity,
    so the two numbers quoted around this project mean the same thing."""
    torch.manual_seed(0)
    B, D = 128, cfg.latent_dim
    z0 = torch.randn(B, D)
    z1 = z0 + torch.randn(B, D) * 0.4
    zhat = z1 + torch.randn(B, D) * 0.15
    loss, rep = rollout_loss(zhat[:, None], z1[:, None], z0, horizons=[1])
    se_m = ((zhat - z1) ** 2).mean(1).sum()
    se_i = ((z0 - z1) ** 2).mean(1).sum()
    assert rep[1] == pytest.approx(float(se_m / se_i), rel=1e-5)


# --------------------------------------------------------------------------
# the HUD-augmented predictor (arm B)
# --------------------------------------------------------------------------

def test_augmented_starts_as_copy_forward_on_hud_and_identical_on_latent(cfg, model):
    """Both ends must be zeroed, and the second one was learned the hard way.

    The first version emitted `sigmoid(Linear(h))` with no path from the input
    HUD to the output, so the model had to reconstruct health from scratch each
    step. Health barely moves between two 15 Hz frames, so copy-forward is a very
    strong baseline, and that arm scored 1.53 at h=1 after 15000 steps -- half
    again *worse* than assuming nothing happens. A residual head starts at 1.0.
    """
    from sokubot.model.augmented import (N_HUD, AugmentedPredictor,
                                         AugmentedWorldModel)
    # Both sides in eval mode. The projector's BatchNorm normalises by *batch*
    # statistics in train mode and by *running* statistics in eval, so comparing
    # a train-mode base against an eval-mode copy differs for that reason alone
    # and says nothing about the weights. `nn.Module` defaults to train, so this
    # is easy to get wrong -- it is docs/BUGS.md section 1 seen from the
    # measurement side, and it failed exactly this way when first written.
    base = copy.deepcopy(model).eval()
    aug = AugmentedPredictor.from_pretrained(base.predictor, cfg).eval()
    B, T = 12, cfg.history
    z = torch.randn(B, T, cfg.latent_dim)
    hud = torch.rand(B, T, N_HUD)
    cond = base.action_encoder(
        torch.rand(B, T, cfg.action_ticks, cfg.action_dim).round())
    with torch.no_grad():
        z_base = base.predictor(z, cond)
        z_aug, h_aug = aug(z, hud, cond)
    from sokubot.model.augmented import HUD_EPS
    dz = float((z_base - z_aug).abs().max())
    assert dz < 1e-6, f"latent path must match the base predictor; max diff {dz:.3e}"

    # Copy-forward is exact only away from the gauge extremes: the residual is
    # taken in logit space, so the input is clamped into [HUD_EPS, 1-HUD_EPS]
    # first and a reading of exactly 0 comes back as HUD_EPS. That bound is the
    # honest tolerance -- a tighter one fails on whichever random draw happens to
    # land near zero, which is a property of the fixture rather than the model.
    dh = float((h_aug - hud).abs().max())
    assert dh <= HUD_EPS + 1e-6, f"hud path must be copy-forward; max diff {dh:.3e}"
    mid = (hud > 0.05) & (hud < 0.95)
    dmid = float((h_aug[mid] - hud[mid]).abs().max())
    assert dmid < 1e-5, f"hud must be exact away from the clamp; max diff {dmid:.3e}"

    awm = AugmentedWorldModel(base, aug).eval()
    with torch.no_grad():
        _, hr = awm.rollout(
            z, hud, torch.rand(B, 6, cfg.action_ticks, cfg.action_dim).round(),
            torch.rand(B, T - 1, cfg.action_ticks, cfg.action_dim).round())
        _, rep = rollout_loss(hr, torch.rand(B, 6, N_HUD), hud[:, -1])
    for h, v in rep.items():
        assert v == pytest.approx(1.0, rel=1e-4), f"h{h} scored {v}, not copy-forward"


def test_augmented_hud_stays_finite_at_the_gauge_extremes(cfg, model):
    """A spent spirit gauge reads exactly 0 and a full health bar exactly 1, and
    the residual is taken in logit space, so both need clamping or they are inf."""
    from sokubot.model.augmented import N_HUD, AugmentedPredictor
    aug = AugmentedPredictor.from_pretrained(model.predictor, cfg).eval()
    B, T = 4, cfg.history
    z = torch.randn(B, T, cfg.latent_dim)
    cond = model.action_encoder(
        torch.rand(B, T, cfg.action_ticks, cfg.action_dim).round())
    for val in (0.0, 1.0):
        with torch.no_grad():
            _, h = aug(z, torch.full((B, T, N_HUD), val), cond)
        assert torch.isfinite(h).all(), f"hud={val} produced non-finite output"
        assert ((h >= 0) & (h <= 1)).all()
