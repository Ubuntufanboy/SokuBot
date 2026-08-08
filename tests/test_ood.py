"""The out-of-distribution guard on imagined latents.

    python -m pytest tests/test_ood.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sokubot.rl.ood import LatentOOD, OODConfig, truncate_after

D = 32


def _corpus(n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g)


def test_fit_recovers_an_identity_covariance():
    """SIGReg is supposed to push the latent toward N(0, I), so on data that
    really is N(0, I) the fitted covariance must come back as identity. If this
    drifts, the score is measuring the fit rather than the data."""
    o = LatentOOD(D)
    rep = o.fit(_corpus())
    assert rep["cov_dev_from_identity"] < 0.02
    assert o.mu.abs().max() < 0.1


def test_thresholds_are_the_stated_corpus_quantiles_at_both_tails():
    """Their meaning is 'this fraction of *real* frames would be flagged at each
    end'. That is checkable; a bare number is not."""
    o = LatentOOD(D, OODConfig(quantile=0.99))
    z = _corpus()
    o.fit(z)
    s = o(z)
    assert float((s > o.hi).float().mean()) == pytest.approx(0.01, abs=0.003)
    assert float((s < o.lo).float().mean()) == pytest.approx(0.01, abs=0.003)


def test_median_score_sits_at_the_latent_dimension():
    """For genuine N(0, I) the squared Mahalanobis distance concentrates on D.
    This is what makes a *two-sided* test the right shape: real latents are
    neither unusually far from the mean nor unusually close to it."""
    o = LatentOOD(D)
    rep = o.fit(_corpus())
    assert rep["median"] == pytest.approx(D, rel=0.1)
    assert rep["expected_median"] == D


def test_extrapolation_away_from_the_mean_is_flagged():
    o = LatentOOD(D)
    o.fit(_corpus())
    excess, hard = o.flags(_corpus(2000, seed=2) * 3 + 4)
    assert float(excess.median()) > 0
    assert float(hard.float().mean()) > 0.9


def test_collapse_toward_the_mean_is_also_flagged():
    """The case that motivated making this two-sided, and the one a plain
    distance test gets exactly backwards.

    This model's documented long-horizon failure is blur -- rollout cosine 0.417
    and relative L2 0.925 by h=48. A blurred latent is *closer* to the mean than
    a real one, so a one-sided 'too far away' test would score a maximally
    degraded rollout as maximally in-distribution.
    """
    o = LatentOOD(D)
    o.fit(_corpus())
    blurred = _corpus(2000, seed=3) * 0.1          # collapsed toward the mean
    assert float(o(blurred).median()) < float(o.lo), "fixture is not collapsed"
    excess, hard = o.flags(blurred)
    assert float(excess.median()) > 0, "collapse must produce positive excess"
    assert float(hard.float().mean()) > 0.9


def test_in_distribution_latents_are_not_flagged():
    o = LatentOOD(D)
    o.fit(_corpus())
    excess, hard = o.flags(_corpus(4000, seed=1))
    assert float(excess.mean()) < 0.05
    assert float(hard.float().mean()) < 0.01


def test_flags_before_fit_is_an_error_not_a_default():
    o = LatentOOD(D)
    with pytest.raises(RuntimeError, match="fit has not been called"):
        o.flags(torch.randn(4, D))


def test_fit_refuses_a_bank_too_small_for_a_covariance():
    o = LatentOOD(D)
    with pytest.raises(ValueError, match="too small"):
        o.fit(torch.randn(D - 1, D))


def test_truncate_masks_from_the_first_violation_inclusive():
    """The violating step is masked too. Its reward was read off a latent the
    probe has no business reading, so keeping it would pay for the drift."""
    hard = torch.tensor([[False, False, True, False],
                         [False, False, False, False],
                         [True, False, False, False]])
    m = truncate_after(hard)
    assert m[0].tolist() == [1, 1, 0, 0]
    assert m[1].tolist() == [1, 1, 1, 1]
    assert m[2].tolist() == [0, 0, 0, 0]


def test_truncate_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        truncate_after(torch.zeros(3, dtype=torch.bool))


def test_score_is_invariant_to_device_moves_and_survives_state_dict():
    o = LatentOOD(D)
    o.fit(_corpus())
    z = _corpus(64, seed=7)
    want = o(z)
    o2 = LatentOOD(D)
    o2.load_state_dict(o.state_dict())
    assert torch.allclose(o2(z), want, atol=1e-5)
    assert bool(o2.fitted)
    assert float(o2.hi) == float(o.hi) and float(o2.lo) == float(o.lo)


def test_no_trainable_parameters():
    """It must never reach an optimiser: these are fitted statistics, not weights."""
    o = LatentOOD(D)
    assert list(o.parameters()) == []


# --------------------------------------------------------------------------
# state flags -> step masks
# --------------------------------------------------------------------------

def test_step_is_contaminated_if_either_endpoint_drifted():
    """Step t is paid for the transition state t -> state t+1, so a bad state at
    either end taints it. Taking only the arrival state would let the first bad
    state through with a full reward attached."""
    from sokubot.rl.ood import step_flags
    # states:            0      1      2      3
    sh = torch.tensor([[False, False, True, False]])
    ok, _ = step_flags(sh)
    # step 1 spans states 1->2, so it is the first contaminated step.
    assert ok[0].tolist() == [1, 0, 0]


def test_lam_scale_is_zero_exactly_at_the_last_live_step():
    from sokubot.rl.ood import step_flags
    sh = torch.tensor([[False, False, False, True, False]])
    ok, lam_scale = step_flags(sh)
    assert ok[0].tolist() == [1, 1, 0, 0]
    # Last live step is index 1; it must bootstrap rather than recurse.
    assert lam_scale[0].tolist() == [1, 0, 0, 0]


def test_a_clean_rollout_keeps_lambda_everywhere():
    """No violation must mean no behaviour change at all -- otherwise the guard
    silently shortens every rollout it was supposed to leave alone."""
    from sokubot.rl.ood import step_flags
    sh = torch.zeros(3, 6, dtype=torch.bool)
    ok, lam_scale = step_flags(sh)
    assert (ok == 1).all()
    assert (lam_scale == 1).all()


def test_step_flags_rejects_a_single_state():
    from sokubot.rl.ood import step_flags
    with pytest.raises(ValueError, match="at least two states"):
        step_flags(torch.zeros(2, 1, dtype=torch.bool))
