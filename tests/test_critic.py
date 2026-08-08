"""The twohot critic and the lambda-return recursion.

Most of these have exact analytic answers, which is the point: a value head that
is subtly miscalibrated does not fail, it just biases every advantage in the same
direction, and the training curve looks fine while the policy learns the wrong
ranking.

    python -m pytest tests/test_critic.py -q
"""

from __future__ import annotations

import pytest
import torch

from sokubot.rl.critic import (CriticConfig, SokuCritic, TargetCritic,
                               continuation, lambda_returns, symexp, symlog,
                               twohot)
from sokubot.rl.reward import RewardConfig, compute_rewards, terminal_mask

LATENT, HIST = 192, 3


# --------------------------------------------------------------------------
# symlog / twohot
# --------------------------------------------------------------------------

def test_symlog_roundtrips_across_the_reward_range():
    """Both ends of the range matter: chip damage ~0.1, win/lose +-5."""
    x = torch.tensor([-5.0, -1.0, -0.1, 0.0, 0.05, 0.1, 1.0, 5.0, 50.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)


def test_symlog_is_monotone():
    x = torch.linspace(-20, 20, 401)
    assert (symlog(x).diff() > 0).all()


def test_twohot_is_a_distribution_whose_mean_is_the_input():
    """This is the property the whole scheme rests on.

    If the encoded mean were not the input, the critic would be regressing
    towards a systematically shifted target and every advantage would inherit
    the shift.
    """
    bins = torch.linspace(-6.0, 6.0, 41)
    x = torch.tensor([-5.9, -2.3, -0.04, 0.0, 0.37, 1.5, 5.9])
    w = twohot(x, bins)
    assert torch.allclose(w.sum(-1), torch.ones_like(x), atol=1e-6)
    assert (w >= 0).all()
    assert torch.allclose((w * bins).sum(-1), x, atol=1e-5)


def test_twohot_puts_mass_on_at_most_two_adjacent_bins():
    bins = torch.linspace(-6.0, 6.0, 41)
    w = twohot(torch.tensor([0.37]), bins)[0]
    nz = torch.nonzero(w > 1e-9).flatten()
    assert len(nz) <= 2
    if len(nz) == 2:
        assert nz[1] - nz[0] == 1


def test_twohot_on_an_exact_bin_is_one_hot():
    bins = torch.linspace(-6.0, 6.0, 41)
    for i in (0, 20, 40):
        w = twohot(bins[i : i + 1], bins)[0]
        assert w[i] == pytest.approx(1.0, abs=1e-6)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_twohot_clamps_out_of_range_values_onto_the_end_bin():
    """Clamping is the designed behaviour, and `CriticConfig.clipped` reports it.

    The danger is not the clamp, it is a clamp nobody notices -- so the loss's
    diagnostics carry the clipped fraction.
    """
    bins = torch.linspace(-6.0, 6.0, 41)
    w = twohot(torch.tensor([100.0, -100.0]), bins)
    assert w[0, -1] == pytest.approx(1.0, abs=1e-6)
    assert w[1, 0] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# the critic head
# --------------------------------------------------------------------------

def test_critic_starts_at_zero_value():
    """Zero-init head -> uniform bins -> mean 0 by symmetry of the grid.

    A critic that started at some arbitrary large value would inject that error
    into every advantage in exactly the batches where the policy moves most.
    """
    c = SokuCritic(LATENT, HIST)
    v = c(torch.randn(16, HIST, LATENT), torch.zeros(16, dtype=torch.long))
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)


def test_critic_can_learn_a_constant_across_the_reward_scales():
    """Both a chip-damage-sized and a KO-sized target, to the same tolerance.

    This is the concrete claim made against a scalar MSE head: it would be
    dominated by the +-5 tail and read as noise at 0.1.
    """
    for target_value in (0.1, 5.0):
        torch.manual_seed(0)
        c = SokuCritic(LATENT, HIST)
        opt = torch.optim.Adam(c.parameters(), lr=3e-3)
        z = torch.randn(256, HIST, LATENT)
        side = torch.zeros(256, dtype=torch.long)
        tgt = torch.full((256,), target_value)
        for _ in range(300):
            loss, _ = c.loss(z, side, tgt)
            opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            opt.step()
        with torch.no_grad():
            got = float(c(z, side).mean())
        assert got == pytest.approx(target_value, rel=0.05), \
            f"target {target_value} -> {got}"


def test_loss_reports_the_clipped_fraction():
    c = SokuCritic(LATENT, HIST, CriticConfig(limit=1.0))
    z = torch.randn(8, HIST, LATENT)
    side = torch.zeros(8, dtype=torch.long)
    _, stats = c.loss(z, side, torch.full((8,), 100.0))
    assert stats["clipped"] == pytest.approx(1.0)
    _, stats = c.loss(z, side, torch.zeros(8))
    assert stats["clipped"] == pytest.approx(0.0)


def test_target_critic_lags_and_holds_no_gradients():
    c = SokuCritic(LATENT, HIST)
    tgt = TargetCritic(c, tau=0.98)
    assert all(not p.requires_grad for p in tgt.net.parameters())
    with torch.no_grad():
        for p in c.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = [p.clone() for p in tgt.net.parameters()]
    tgt.update(c)
    moved = [(a - b).abs().sum() for a, b in zip(tgt.net.parameters(), before)]
    assert sum(float(m) for m in moved) > 0, "target never moves"
    # It must move only 2% of the way, not all of it.
    for t, b, s in zip(tgt.net.parameters(), before, c.parameters()):
        assert torch.allclose(t, b.lerp(s, 0.02), atol=1e-6)


def test_target_critic_is_not_in_the_critics_parameters():
    """It must never reach the optimiser or the checkpoint as a trainable thing."""
    c = SokuCritic(LATENT, HIST)
    n_before = len(list(c.parameters()))
    TargetCritic(c, tau=0.98)
    assert len(list(c.parameters())) == n_before
    assert "net" not in c.state_dict()


# --------------------------------------------------------------------------
# lambda returns
# --------------------------------------------------------------------------

def test_lambda_one_is_the_discounted_monte_carlo_return():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.0, 0.0, 0.0, 7.0]])
    cont = torch.ones(1, 3)
    g = 0.9
    got = lambda_returns(r, v, cont, gamma=g, lam=1.0)
    want = 1 + g * (2 + g * (3 + g * 7))
    assert float(got[0, 0]) == pytest.approx(want, rel=1e-6)


def test_lambda_zero_is_the_one_step_td_target():
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.0, 5.0, 6.0, 7.0]])
    cont = torch.ones(1, 3)
    g = 0.9
    got = lambda_returns(r, v, cont, gamma=g, lam=0.0)
    assert float(got[0, 0]) == pytest.approx(1 + g * 5, rel=1e-6)
    assert float(got[0, 1]) == pytest.approx(2 + g * 6, rel=1e-6)
    assert float(got[0, 2]) == pytest.approx(3 + g * 7, rel=1e-6)


def test_terminal_stops_value_crossing_the_boundary():
    """cont = 0 must truncate the recursion, not merely zero the reward.

    A terminal step that still bootstraps credits the trajectory with whatever
    the critic predicts for a state that does not exist.
    """
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.0, 99.0, 99.0, 99.0]])
    cont = torch.tensor([[1.0, 0.0, 1.0]])          # step 1 is terminal
    got = lambda_returns(r, v, cont, gamma=0.9, lam=0.95)
    assert float(got[0, 1]) == pytest.approx(2.0, rel=1e-6)
    assert float(got[0, 0]) == pytest.approx(1 + 0.9 * (0.05 * 99 + 0.95 * 2), rel=1e-6)


def test_lambda_returns_rejects_mismatched_shapes():
    r = torch.zeros(2, 3)
    with pytest.raises(ValueError, match=r"value must be"):
        lambda_returns(r, torch.zeros(2, 3), torch.ones(2, 3), 0.99, 0.95)
    with pytest.raises(ValueError, match=r"cont must be"):
        lambda_returns(r, torch.zeros(2, 4), torch.ones(2, 4), 0.99, 0.95)


# --------------------------------------------------------------------------
# terminal / continuation, against the real reward
# --------------------------------------------------------------------------

def _states(hp1, hp2, n_extra=4):
    """[1, T+1, 6] with the two health traces given and everything else flat."""
    T = len(hp1)
    s = torch.zeros(1, T, 6)
    s[0, :, 0] = torch.tensor(hp1)
    s[0, :, 1] = torch.tensor(hp2)
    s[0, :, 2:4] = 1.0
    return s


def test_terminal_marks_the_ko_step_and_continuation_is_its_complement():
    cfg = RewardConfig()
    # P2 is knocked out and stays down long enough to satisfy ko_persist.
    hp2 = [1.0, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0]
    hp1 = [1.0] * 7
    s = _states(hp1, hp2)
    side = torch.zeros(1, dtype=torch.long)
    term = terminal_mask(s, side, cfg)
    assert term.shape == (1, len(hp1) - 1)
    assert float(term.sum()) == pytest.approx(1.0), "exactly one terminal step"
    cont = continuation(term)
    assert float(cont[0, int(term[0].argmax())]) == 0.0
    assert torch.allclose(cont + term, torch.ones_like(cont))


def test_no_ko_means_no_terminal_step_so_the_rollout_bootstraps():
    """A truncated rollout must keep cont = 1 everywhere, including the last step.

    This is the case `alive` alone cannot distinguish from a KO on the horizon,
    and getting it wrong throws away the critic's estimate of everything after
    the rollout -- which is the entire reason the critic exists.
    """
    s = _states([1.0] * 7, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    term = terminal_mask(s, torch.zeros(1, dtype=torch.long))
    assert float(term.sum()) == 0.0
    assert torch.allclose(continuation(term), torch.ones_like(term))


def test_terminal_agrees_with_the_alive_mask_away_from_the_horizon():
    """Two independent readings of the same event must agree where both are defined.

    `docs/BUGS.md` names measuring one quantity two ways as the defence that
    actually worked, so the two constructions are checked against each other
    rather than each being checked only against itself.
    """
    cfg = RewardConfig()
    s = _states([1.0] * 9, [1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    side = torch.zeros(1, dtype=torch.long)
    a = torch.zeros(1, s.shape[1] - 1, 4, 20)
    _, alive, _ = compute_rewards(s, a, side, cfg)
    term = terminal_mask(s, side, cfg)
    k = int(term[0].argmax())
    assert k < term.shape[1] - 1, "this fixture must KO before the horizon"
    assert float(alive[0, k]) == 1.0          # alive through the KO step
    assert float(alive[0, k + 1]) == 0.0      # and not after


# --------------------------------------------------------------------------
# per-step lambda: truncation is not termination
# --------------------------------------------------------------------------

def test_scalar_and_constant_tensor_lambda_agree():
    r = torch.randn(4, 6)
    v = torch.randn(4, 7)
    c = torch.ones(4, 6)
    a = lambda_returns(r, v, c, 0.99, 0.95)
    b = lambda_returns(r, v, c, 0.99, torch.full((4, 6), 0.95))
    assert torch.allclose(a, b, atol=1e-6)


def test_per_step_lambda_zero_is_a_pure_one_step_bootstrap():
    """What the OOD guard needs: stop trusting the rollout without ending it."""
    r = torch.tensor([[1.0, 2.0, 3.0]])
    v = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
    c = torch.ones(1, 3)
    lam = torch.tensor([[0.95, 0.0, 0.95]])
    got = lambda_returns(r, v, c, gamma=0.9, lam=lam)
    # Step 1 truncates: r_1 + gamma * v(s_2), ignoring everything after.
    assert float(got[0, 1]) == pytest.approx(2 + 0.9 * 20, rel=1e-6)


def test_truncation_and_termination_differ_at_the_same_step():
    """cont=0 pays nothing beyond; lam=0 pays the critic's estimate beyond.

    Collapsing the two would teach the policy that drifting off the manifold is
    worth whatever a KO is worth.
    """
    r = torch.tensor([[5.0]])
    v = torch.tensor([[0.0, 100.0]])
    terminated = lambda_returns(r, v, torch.zeros(1, 1), 0.9, 0.95)
    truncated = lambda_returns(r, v, torch.ones(1, 1), 0.9,
                               torch.zeros(1, 1))
    assert float(terminated[0, 0]) == pytest.approx(5.0)
    assert float(truncated[0, 0]) == pytest.approx(5 + 0.9 * 100)


def test_per_step_lambda_shape_is_checked():
    with pytest.raises(ValueError, match="per-step lam"):
        lambda_returns(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2, 3),
                       0.99, torch.zeros(2, 5))
