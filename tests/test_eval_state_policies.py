"""The policy evaluator measures exactly what the trainer's `net` means, and runs end to end."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from scripts.eval_state_policies import boot, ci, main, net_of, per_start
from scripts.make_smoke_assets import write_bank, write_sim
from sokubot.data.state import CH, STATE_CHANNELS
from sokubot.rl.ppo import evaluate_vs

C = len(STATE_CHANNELS)


class FixedArena:
    """Returns the same trajectory whatever the RNG, so two instruments can be compared exactly."""

    def __init__(self, B=24, T=8):
        g = torch.Generator().manual_seed(0)
        self.tr = []
        # One trajectory per chair: with the same one in both, the chair average is exactly 0.
        for _ in range(2):
            s = torch.rand(B, T + 1, 2, C, generator=g) * 0.1 + 0.5
            # Health that falls unevenly on both sides, so dealt and taken are both non-zero.
            s[..., CH["hp"]] = 0.9 - torch.cumsum(torch.rand(B, T + 1, 2, generator=g) * 0.02, 1)
            self.tr.append({"states": s,
                            "joint": (torch.rand(B, T, 5, 20, generator=g) > 0.8).float()})

    def rollout(self, s_ctx, p_ctx, a_hist, side, policy, opponent, two_sided=False):
        return self.tr[int(side[0])]


def test_net_is_evaluate_vs_net():
    arena = FixedArena()
    ctx = (torch.zeros(24, 4, 2, C), None, None)
    dummy = nn.Linear(1, 1)
    ref = evaluate_vs(arena, dummy, dummy, ctx)["net"]
    num, den, _ = per_start(arena, dummy, dummy, ctx, None, draws=3, seed=0)
    assert abs(ref) > 1e-4
    assert abs(net_of(num, den) - ref) < 1e-6


def test_the_bootstrap_of_a_constant_is_that_constant():
    vals = boot(lambda i: 0.25, 100, 50)
    assert ci(vals) == (0.25, 0.25)


def test_it_runs_end_to_end_and_writes_every_number(tmp_path):
    from scripts.train_state_ppo import main as train
    sim = write_sim(tmp_path / "sim.pt", slots=2, history=4, ticks=5, width=32, depth=1)
    bank = write_bank(tmp_path / "bank.npz", replays=6, steps=80, slots=2, ticks=5)
    for s in (0, 1):
        assert train(["--sim", str(sim), "--bank", str(bank), "--out", str(tmp_path / f"r{s}"),
                      "--device", "cpu", "--starts", "16", "--horizon", "2", "--epochs", "1",
                      "--minibatches", "1", "--steps", "2", "--eval-starts", "8",
                      "--seed", str(s)]) == 0
    out = tmp_path / "eval.json"
    assert main(["--sim", str(sim), "--bank", str(bank), "--runs", str(tmp_path / "r0"),
                 str(tmp_path / "r1"), "--starts", "16", "--draws", "1", "--horizon", "2",
                 "--boot", "50", "--device", "cpu", "--out", str(out)]) == 0
    table = json.loads(out.read_text())["worlds"][str(sim)]["table"]
    for tag in ("r0 final vs own reference", "r1 final vs replay", "r0 vs r1",
                "r0: final - prior, vs replay", "r0 vs r1 + reverse [must be 0]"):
        assert tag in table, tag
