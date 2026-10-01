"""PPO on the real game against its own AI: what the learner and the actors share.

The user's experiment (2026-10-01): train with the SAME reward the simulator runs used, but on the
real game against the Lunatic COM, with no world model anywhere. If that fails, the reward was
flawed to begin with; if it works, what is left is the world model.

LAYOUT
------
One learner (scripts/train_vscom_ppo.py, a GPU job) and any number of actor processes
(scripts/vscom_actor.py, CPU jobs on other nodes), over TCP; cross-job connections were measured
at 0.085 ms and ~500 MB/s on Amarel. Actors play with a local copy of the policy, cut each game's
play into fixed-length segments, and ship them; the learner updates and publishes new weights,
which actors fetch in the background. A segment can therefore be a few versions behind; it carries
the BEHAVIOUR log-probability of every action, so PPO's ratio is taken against the policy that
actually acted, and segments older than a cap are dropped.

THE REWARD IS THE SIMULATOR'S
-----------------------------
Every term but the outcome is `state_reward.compute_rewards` over one transition: its damage
("step" mode), crush and combo terms are local to a step, which the tests check against the
whole-round computation. The outcome is not: `ko_persist` asks a KO to hold for two states, which
guards against a simulator's noisy health but never fires on a real round, whose KO row is its
last. The real game reports the outcome exactly, in the round score, so it is paid from that,
once, on the round's last step, with the same win/lose magnitudes.
"""
from __future__ import annotations

import dataclasses
import pickle
import socket
import struct

import numpy as np
import torch

from sokubot.env.vscom import Tick, words_to_buttons
from sokubot.rl.state_reward import StateRewardConfig, compute_rewards

PROTOCOL = 1


# ---------------------------------------------------------------------------------------------
# wire: length-prefixed pickles. Trusted peers only: both ends are this repository on one cluster.
# ---------------------------------------------------------------------------------------------
def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def recv_msg(sock: socket.socket):
    def exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(1 << 22, n - len(buf)))
            if not chunk:
                raise ConnectionError("peer closed")
            buf += chunk
        return bytes(buf)
    (n,) = struct.unpack("!Q", exact(8))
    return pickle.loads(exact(n))


# ---------------------------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------------------------
def joint_from_words(words: np.ndarray) -> np.ndarray:
    """[2, ticks] uint16 (P1 row, P2 row) -> [ticks, 20] float32, P1's ten buttons first: the
    corpus's action layout (data/soku.ACTION_COLUMNS)."""
    b = words_to_buttons(words)                        # [2, ticks, 10]
    return np.concatenate([b[0], b[1]], axis=-1)


def round_outcome(before: tuple[int, int], after: tuple[int, int],
                  cfg: StateRewardConfig) -> float:
    """The outcome the score records for P1: a KO, a timeout and a double KO all follow it."""
    p1 = after[0] > before[0]
    p2 = after[1] > before[1]
    if p1 and not p2:
        return cfg.win
    if p2 and not p1:
        return cfg.lose
    return 0.0


def transition_rewards(s0: np.ndarray, s1: np.ndarray, joint: np.ndarray,
                       cfg: StateRewardConfig) -> np.ndarray:
    """`step_reward` for T transitions at once: s0, s1 [T, 2, C], joint [T, ticks, 20] -> [T].
    Each transition is its own two-state window, so the batch is exactly T `step_reward` calls;
    one `compute_rewards` call instead of T is what keeps it off the actor's per-decision path."""
    no_outcome = dataclasses.replace(cfg, win=0.0, lose=0.0)
    states = torch.from_numpy(np.stack([s0, s1], axis=1).astype(np.float32))      # [T, 2, 2, C]
    j = torch.from_numpy(joint.astype(np.float32))[:, None]                       # [T, 1, ticks, 20]
    total, _, _ = compute_rewards(states, j, torch.zeros(len(s0), dtype=torch.long), no_outcome)
    return total[:, 0].numpy()


def step_reward(s0: np.ndarray, s1: np.ndarray, joint: np.ndarray,
                cfg: StateRewardConfig) -> tuple[float, dict[str, float]]:
    """Every term of `compute_rewards` but the outcome, for the transition s0 -> s1 under `joint`
    ([ticks, 20]); P1 is the agent (side 0). The outcome is added by the caller, from the score."""
    no_outcome = dataclasses.replace(cfg, win=0.0, lose=0.0)
    states = torch.from_numpy(np.stack([s0, s1]).astype(np.float32))[None]
    j = torch.from_numpy(joint.astype(np.float32))[None, None]
    total, _, terms = compute_rewards(states, j, torch.zeros(1, dtype=torch.long), no_outcome)
    return float(total[0, 0]), {k: float(v[0, 0]) for k, v in terms.items()}


# ---------------------------------------------------------------------------------------------
# segments
# ---------------------------------------------------------------------------------------------
class SegmentBuilder:
    """One game's play, cut into fixed-length segments of `length` decisions.

    A segment may contain a round boundary: `terminal` marks the last step of a round, which stops
    both the bootstrap and the lambda-chain in `state_critic.lambda_returns`, and the next step is
    the new round's first decision. `obs_last` is the observation AFTER the segment's last step,
    the critic's bootstrap; it is ignored when that step was terminal.
    """

    def __init__(self, length: int, cfg: StateRewardConfig | None = None):
        self.length = length
        self.cfg = cfg or StateRewardConfig()
        self._clear()

    def _clear(self) -> None:
        self.obs, self.act, self.logp, self.terminal, self.version = [], [], [], [], []
        self.s0, self.s1, self.joint, self.outcome = [], [], [], []
        self.rounds: list[dict] = []

    def add(self, obs: np.ndarray, act: np.ndarray, logp: float, s0: np.ndarray,
            s1: np.ndarray, joint: np.ndarray, outcome: float, terminal: bool,
            version: int) -> None:
        """One decision. The reward is computed at `pop`, for the whole segment in one batch:
        every term of compute_rewards over s0 -> s1 under `joint`, plus `outcome` (from the
        score, nonzero only on a round's last step)."""
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.s0.append(s0)
        self.s1.append(s1)
        self.joint.append(joint)
        self.outcome.append(outcome)
        self.terminal.append(terminal)
        self.version.append(version)

    def full(self) -> bool:
        return len(self.obs) >= self.length

    def pop(self, obs_last: np.ndarray) -> dict:
        reward = transition_rewards(np.stack(self.s0), np.stack(self.s1), np.stack(self.joint),
                                    self.cfg) + np.asarray(self.outcome, np.float32)
        seg = {"obs": np.stack(self.obs).astype(np.float32),          # [T, H, dim]
               "act": np.stack(self.act).astype(np.uint8),            # [T, ticks, 10]
               "logp": np.asarray(self.logp, np.float32),             # [T] behaviour log-probs
               "reward": reward.astype(np.float32),
               "terminal": np.asarray(self.terminal, np.float32),
               "version": np.asarray(self.version, np.int64),
               "obs_last": obs_last.astype(np.float32),               # [H, dim]
               "rounds": self.rounds}
        self._clear()
        return seg


class Window:
    """The policy's observation history for one round: the last H (state, proj) pairs, padded at a
    round's start by repeating the oldest -- exactly `env.vscom.PolicyAgent`'s rule."""

    def __init__(self, history: int, slots: int):
        self.H, self.slots = history, slots
        self.s: list[np.ndarray] = []
        self.p: list[np.ndarray] = []

    def reset(self) -> None:
        self.s, self.p = [], []

    def push(self, t: Tick) -> tuple[np.ndarray, np.ndarray]:
        self.s.append(t.state)
        self.p.append(t.proj[:, :self.slots])
        self.s, self.p = self.s[-self.H:], self.p[-self.H:]
        pad = self.H - len(self.s)
        return (np.stack([self.s[0]] * pad + self.s), np.stack([self.p[0]] * pad + self.p))
