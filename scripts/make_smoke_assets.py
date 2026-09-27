"""Synthetic stand-ins for a simulator checkpoint and a state bank, for smoke-testing the RL harness.

    python -m scripts.make_smoke_assets --out /tmp/smoke
    python -m scripts.train_state_ppo --sim /tmp/smoke/sim.pt --bank /tmp/smoke/bank.npz \
        --out /tmp/smoke/ppo --steps 50

Neither is a model of the game. The simulator is a real, tiny `StateDynamics` with randomly
perturbed heads, saved in exactly the format `load_sim` reads, so the load path, the arena, the
update, checkpoints and resume all run for real; the bank has the state-bank cache's arrays and
dtypes with made-up contents. They exist so the harness can be exercised on a cluster node BEFORE
the real sidecars and simulator exist -- a number trained against them means nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.state import CH, PROJ_FEATURES, STATE_CHANNELS
from sokubot.model.state_dynamics import StateDynamics
from sokubot.model.state_head import BINARY

N_STATE, N_PROJF = len(STATE_CHANNELS), len(PROJ_FEATURES)


def write_sim(path: Path, *, slots: int = 2, history: int = 4, ticks: int = 5, width: int = 64,
              depth: int = 2, heads: int = 4, noise: float = 0.02, seed: int = 0) -> Path:
    """A tiny StateDynamics whose zero-initialised heads are given random weights, so its rollouts
    actually move (an untouched one is an exact identity predictor and every reward is 0)."""
    torch.manual_seed(seed)
    cfg = Config.soku()
    m = StateDynamics(cfg, slots=slots, width=width, depth=depth, heads=heads, history=history,
                      ticks=ticks)
    with torch.no_grad():
        for head in (m.head_state, m.head_proj):
            head.weight.normal_(0.0, noise)
            head.bias.zero_()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "cfg": cfg, "step": 0, "slots": slots,
                "history": history, "width": width, "depth": depth, "ticks": ticks,
                "proj_feedback": "sigmoid", "n_moves": 0, "move_dim": 0, "move_vocab": None,
                "idm": False, "state_history": history, "act_skip": False,
                "synthetic": True}, path)
    return path


def make_bank(*, replays: int = 20, steps: int = 300, slots: int = 2, ticks: int = 5,
              seed: int = 0):
    """S, P, A, E, V, names with the state bank's shapes and dtypes and invented contents."""
    rng = np.random.default_rng(seed)
    n = replays * steps
    S = rng.normal(0.0, 0.3, size=(n, 2, N_STATE)).astype(np.float32)
    S[..., list(BINARY)] = (rng.random((n, 2, len(BINARY))) < 0.05).astype(np.float32)
    # Health: starts full each replay and drifts down, never to zero (no KO at the start).
    t = np.tile(np.arange(steps), replays) / steps
    S[..., CH["hp"]] = np.clip(1.0 - 0.5 * t[:, None] + rng.normal(0, 0.02, (n, 2)), 0.2, 1.0)
    P = np.zeros((n, 2, slots, N_PROJF), np.float32)
    # Buttons: the axes are exclusive (a policy can never press left and right at once).
    A = np.zeros((n, ticks, 20), np.uint8)
    for base in (0, 10):
        lr = rng.choice(3, size=(n, ticks), p=[0.7, 0.15, 0.15])
        ud = rng.choice(3, size=(n, ticks), p=[0.85, 0.1, 0.05])
        A[..., base + 2] = lr == 1
        A[..., base + 3] = lr == 2
        A[..., base + 0] = ud == 1
        A[..., base + 1] = ud == 2
        A[..., base + 4:base + 10] = rng.random((n, ticks, 6)) < 0.05
    E = np.repeat(np.arange(replays, dtype=np.int32), steps)
    V = np.ones(n, bool)
    names = [f"synthetic-{i:04d}" for i in range(replays)]
    return S, P, A, E, V, names


def write_bank(path: Path, **kw) -> Path:
    S, P, A, E, V, names = make_bank(**kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, S=S, P=P, A=A, E=E, V=V, names=np.array(names))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    a = ap.parse_args()
    sim = write_sim(a.out / "sim.pt", width=a.width, depth=a.depth)
    bank = write_bank(a.out / "bank.npz", replays=a.replays, steps=a.steps)
    print(f"wrote {sim} and {bank} -- SYNTHETIC, for exercising the harness only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
