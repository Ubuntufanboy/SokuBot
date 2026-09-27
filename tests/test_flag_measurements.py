"""The simulator's flag channels come out of `rollout` as PROBABILITIES, and every instrument must
read them as such.

Found 2026-09-27: `sim_channel_table` (sigma table and BCE skill) and `train_state_dynamics`
(`guard_sigma_h*`, `block_gain`) all applied a second sigmoid to `rollout`'s output. That maps every
"not guarding" to 0.5, so the full-corpus simulator read as predicting guarding at ~0.67 against a
5% base rate (BCE skill -2.2, 2.3 sigma of error at one step) while its AUC was 0.997, and it
squashed the BLOCK gate's gain by the sigmoid's slope (~4x).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.train_state_dynamics as tsd
from scripts.make_smoke_assets import write_sim
from scripts.sim_channel_table import flag_skill, undo_pos_weight
from sokubot.data.state import CH, PROJ_FEATURES, STATE_CHANNELS
from sokubot.model.state_dynamics import load_sim, rollout
from sokubot.model.state_head import BINARY

C, SLOTS, H, TICKS, F = len(STATE_CHANNELS), 2, 4, 5, len(PROJ_FEATURES)


def test_rollout_returns_the_flag_it_fed_back_not_a_logit(tmp_path):
    model, _ = load_sim(write_sim(tmp_path / "sim.pt", slots=SLOTS, history=H, ticks=TICKS,
                                  width=32, depth=1), "cpu")
    g = torch.Generator().manual_seed(0)
    s = torch.rand(3, H, 2, C, generator=g)
    p = torch.rand(3, H, 2, SLOTS, F, generator=g)
    a = (torch.rand(3, H + 2, TICKS, 20, generator=g) > 0.8).float()
    out = rollout(model, s, p, a, 2)
    logits, _ = model(s, p, a[:, :H])
    binr = torch.tensor(BINARY)
    step1 = out[:, 0].index_select(-1, binr)
    assert torch.allclose(step1, torch.sigmoid(logits[:, -1].index_select(-1, binr)), atol=1e-6)
    assert float(step1.min()) >= 0.0 and float(step1.max()) <= 1.0


def corpus(n=400, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.random((n, 2, C)).astype(np.float32)
    S[:, :, CH["guarding"]] = (rng.random((n, 2)) < 0.05).astype(np.float32)
    S[:, :, CH["dx"]] = 0.0                     # "near", so block_gain has a pool
    P = rng.random((n, 2, SLOTS, F)).astype(np.float32)
    A = (rng.random((n, TICKS, 20)) < 0.1).astype(np.float32)
    E = np.zeros(n, np.int32)
    return S, P, A, E


def test_a_perfect_simulator_scores_zero_guard_error(monkeypatch):
    S, P, A, E = corpus()

    def truth(model, s, p, a, steps, mv=None):
        return torch.as_tensor(TRUE[0])[:, H:H + steps]
    TRUE = [None]
    real_as_tensor = torch.as_tensor

    def spy(x, *args, **kw):                      # the window eval_rollout builds, before .to()
        t = real_as_tensor(x, *args, **kw)
        if t.dim() == 4 and t.shape[-1] == C and TRUE[0] is None:
            TRUE[0] = t
        return t
    monkeypatch.setattr(tsd.torch, "as_tensor", spy)
    monkeypatch.setattr(tsd, "rollout", truth)
    out = tsd.eval_rollout(None, S, P, A, E, H, (1, 2), "cpu", n=64)
    for h in (1, 2):
        assert out[f"guard_sigma_h{h}"] == pytest.approx(0.0, abs=1e-7)
        assert out[f"guard_mean_h{h}"] == pytest.approx(out[f"guard_true_h{h}"])


def test_block_gain_is_the_difference_in_fed_back_probability(monkeypatch):
    S, P, A, E = corpus()
    calls = iter((0.2, 0.05))

    def const(model, s, p, a, steps, mv=None):
        out = torch.zeros(s.shape[0], steps, 2, C)
        out[..., CH["guarding"]] = next(calls)
        return out
    monkeypatch.setattr(tsd, "rollout", const)
    res = tsd.block_gain(None, S, P, A, E, H, "cpu", n=64, horizon=4)
    assert res["block_away"] == pytest.approx(0.2)
    assert res["block_toward"] == pytest.approx(0.05)
    assert res["block_gain"] == pytest.approx(0.15)


def test_flag_skill_reads_a_probability():
    g = torch.Generator().manual_seed(0)
    y = (torch.rand(20000, generator=g) < 0.05).float()
    base = float(y.mean())
    perfect, _ = flag_skill(y.clone(), y, base)
    constant, auc = flag_skill(torch.full_like(y, base), y, base)
    assert perfect > 0.99                        # a second sigmoid made this about -2.4
    assert constant == pytest.approx(0.0, abs=1e-3)
    assert auc == pytest.approx(0.5, abs=1e-9)   # all ties: exactly chance


def test_undoing_pos_weight_inverts_the_losss_minimiser():
    p = torch.linspace(0.001, 0.999, 101)
    for w in (1.0, 1.33, 5.0, 50.0):
        q = w * p / (w * p + 1 - p)
        assert torch.allclose(undo_pos_weight(q, w), p, atol=1e-5)
