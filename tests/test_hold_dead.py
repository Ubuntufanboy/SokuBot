"""Dead channels are HELD in every rollout, not fed back (state_dynamics.hold_dead).

The trainer leaves a channel whose per-step delta never moves out of the loss, so its residual head
is untrained. Feeding that output back told the full-corpus simulator it was reading its own
output, and its rare-flag heads fired from step 2 (guarding 0.85 against 5%).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.make_smoke_assets import make_bank, write_sim
from sokubot.data.state import CH, PROJ_FEATURES, STATE_CHANNELS
from sokubot.model.state_dynamics import load_sim, rollout
from sokubot.rl.ppo import PPOConfig
from sokubot.rl.state_arena import StateArena, StateObs, corpus_stats

C, SLOTS, H, TICKS, F = len(STATE_CHANNELS), 2, 4, 5, len(PROJ_FEATURES)
AX = CH["ax"]


@pytest.fixture
def sim_path(tmp_path):
    return write_sim(tmp_path / "sim.pt", slots=SLOTS, history=H, ticks=TICKS, width=32, depth=1)


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, H, 2, C, generator=g), torch.rand(3, H, 2, SLOTS, F, generator=g),
            (torch.rand(3, H + 3, TICKS, 20, generator=g) > 0.8).float())


def test_a_held_channel_keeps_its_last_value_and_nothing_else_changes(sim_path):
    model, _ = load_sim(sim_path, "cpu")
    s, p, a = inputs()
    free = rollout(model, s, p, a, 3)
    assert not torch.allclose(free[..., AX], s[:, -1:, :, AX].expand(-1, 3, -1))  # it does drift
    model.hold = (AX,)
    held = rollout(model, s, p, a, 3)
    assert torch.equal(held[..., AX], s[:, -1:, :, AX].expand(-1, 3, -1))
    others = [i for i in range(C) if i != AX]
    # Step 1 reads only real rows, so holding changes nothing else there.
    assert torch.equal(held[:, 0][..., others], free[:, 0][..., others])


def test_the_arena_holds_exactly_as_the_rollout_does(sim_path):
    model, _ = load_sim(sim_path, "cpu")
    model.hold = (AX,)
    S, P, _, _, _, _ = make_bank(replays=4, steps=60, slots=SLOTS, ticks=TICKS)
    arena = StateArena(model, StateObs(*corpus_stats(S, P), SLOTS), PPOConfig(horizon=2), H, TICKS)
    s, p, a = inputs(1)
    nxt_s, _ = arena._advance(s, p, a[:, :H])
    ref = rollout(model, s, p, a, 1)
    assert torch.equal(nxt_s[:, 0], ref[:, 0])
    assert torch.equal(nxt_s[:, 0, :, AX], s[:, -1, :, AX])


def test_load_sim_reads_hold_and_refuses_a_name_that_is_not_continuous(sim_path, tmp_path):
    assert load_sim(sim_path, "cpu")[0].hold == ()          # old checkpoints: as trained
    d = torch.load(sim_path, map_location="cpu", weights_only=False)
    d["hold"] = ["ax"]
    torch.save(d, tmp_path / "held.pt")
    assert load_sim(tmp_path / "held.pt", "cpu")[0].hold == (AX,)
    d["hold"] = ["guarding"]
    torch.save(d, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="not continuous"):
        load_sim(tmp_path / "bad.pt", "cpu")


def test_stamping_uses_the_trainers_dead_rule_and_writes_a_copy(sim_path, tmp_path):
    from scripts.stamp_sim_hold import main
    S, P, A, E, _, _ = make_bank(replays=4, steps=60, slots=SLOTS, ticks=TICKS)
    S = S.copy()
    S[:, :, AX] = -0.0                                   # as in the real corpus
    S[:, :, CH["x"]] = np.random.default_rng(0).random(S.shape[:2])   # certainly live
    np.savez(tmp_path / "cache.npz", S=S, P=P, A=A, E=E)
    before = sim_path.read_bytes()
    assert main(["--sim", str(sim_path), "--cache", str(tmp_path / "cache.npz"),
                 "--out", str(tmp_path / "held.pt")]) == 0
    assert sim_path.read_bytes() == before
    hold = torch.load(tmp_path / "held.pt", map_location="cpu", weights_only=False)["hold"]
    assert "ax" in hold and "x" not in hold
    with pytest.raises(SystemExit, match="new file"):
        main(["--sim", str(sim_path), "--cache", str(tmp_path / "cache.npz"),
              "--out", str(sim_path)])
