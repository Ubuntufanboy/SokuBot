"""The vs-COM environment against a fake game speaking the DLL's protocol over a real socket.

What the fake checks is the contract the DLL relies on (SokuFrameExtractor
dll/include/sfe/agent_link.hpp): every S is answered by exactly one A of `ticks` words -- the
end-of-round S included -- and the rows are read by the corpus's own parser.
"""
from __future__ import annotations

import socket
import threading

import numpy as np
import torch

from sokubot.data.state import parse_state
from sokubot.env.vscom import (AgentLink, NeutralAgent, PolicyAgent, VsComEnv, buttons_to_words,
                               words_to_buttons)
from test_state_loader import HEADER, SLOTS, _row

TICKS = 5


def row_text(**over) -> str:
    r = _row(**over)
    return ",".join(r[c] for c in HEADER)


class FakeGame(threading.Thread):
    """Plays the DLL: one round of `n` fight decisions, the end-of-round S, then a second round."""

    def __init__(self, port: int, n: int = 3):
        super().__init__(daemon=True)
        self.port, self.n = port, n
        self.replies: list[list[int]] = []
        self.sent_rows: list[str] = []

    def _send(self, s: socket.socket, line: str) -> None:
        s.sendall((line + "\n").encode())

    def _reply(self, f) -> bool:
        line = f.readline().strip()
        if not line:                      # the agent hung up: a test that stops early
            return False
        assert line.startswith("A "), line
        self.replies.append([int(x) for x in line.split()[1:]])
        return True

    def state(self, s, f, ms: int, score: tuple[int, int], p1_hp: int, x: float) -> None:
        words = " ".join(str(w) for w in [1, 2, 4, 8, 16] + [0x10, 0, 0, 0x200, 0])
        row = row_text(p1_hp=str(p1_hp), p1_x=f"{x:.3f}")
        self.sent_rows.append(row)
        self._send(s, f"S {ms} 0 {score[0]} {score[1]} {words} {row}")
        if not self._reply(f):
            raise ConnectionAbortedError

    def run(self) -> None:
        try:
            self._play()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _play(self) -> None:
        s = socket.create_connection(("127.0.0.1", self.port))
        f = s.makefile("r")
        self._send(s, f"H 1 16 16 3 {TICKS}")
        self._send(s, "C " + ",".join(HEADER))
        for i in range(self.n):
            self.state(s, f, 2, (0, 0), 10000 - 1000 * i, 100.0 + i)
        self.state(s, f, 3, (0, 1), 0, 150.0)          # P1 KO'd: the round is over
        self.state(s, f, 2, (0, 1), 10000, 120.0)      # the next round's first decision
        s.close()


def make_link():
    link = AgentLink()
    game = FakeGame(link.port)
    game.start()
    link.accept(10.0)
    link.handshake(10.0)
    return link, game


def test_a_round_ends_on_the_ko_and_every_state_is_answered():
    link, game = make_link()
    assert link.hello == (1, 16, 16, 3, TICKS) and link.ticks == TICKS
    env, agent = VsComEnv(link, timeout=10.0), NeutralAgent(TICKS)
    t = env.reset()
    assert t.fight and t.hp == (10000, 10000)
    n = 0
    while t.fight:
        t = env.step(agent.act(t))
        n += 1
    assert n == 3 and t.match_state == 3 and t.score == (0, 1) and t.hp[0] == 0
    t2 = env.reset()                                   # straight to the next round
    assert t2.fight and t2.score == (0, 1)
    env.link.send(np.zeros(TICKS, np.uint16))
    game.join(5.0)
    assert len(game.replies) == 5 and all(len(r) == TICKS for r in game.replies)
    link.close()


def test_observations_are_the_corpus_parser_on_the_same_rows():
    link, game = make_link()
    env = VsComEnv(link, timeout=10.0)
    t = env.reset()
    s, p, _, _ = parse_state([",".join(HEADER) + "\n", game.sent_rows[0] + "\n"])
    assert np.array_equal(t.state, s[0]) and np.array_equal(t.proj, p[0])
    assert t.words.tolist() == [[1, 2, 4, 8, 16], [0x10, 0, 0, 0x200, 0]]
    for _ in range(4):
        env.link.send(np.zeros(TICKS, np.uint16))
        if not game.is_alive():
            break
        try:
            env.link.recv(2.0)
        except Exception:
            break
    link.close()


def test_words_round_trip_through_buttons():
    w = np.arange(1024, dtype=np.uint16)
    assert np.array_equal(buttons_to_words(words_to_buttons(w)), w)
    assert words_to_buttons(np.array([0x200]))[0].tolist() == [0] * 9 + [1]   # spell is bit 9


def test_a_trained_policy_plays_from_its_own_checkpoint(tmp_path):
    from sokubot.rl.policy import SokuPolicy
    from sokubot.rl.state_arena import StateObs
    H = 4
    obs = StateObs(np.zeros(33, np.float32), np.ones(33, np.float32),
                   np.zeros(7, np.float32), np.ones(7, np.float32), SLOTS)
    pol = SokuPolicy(obs.dim, H, TICKS)
    torch.save({"policy": pol.state_dict(), "obs": obs.state_dict(), "history": H,
                "ticks": TICKS, "slots": SLOTS}, tmp_path / "policy.pt")
    link, game = make_link()
    env, agent = VsComEnv(link, timeout=10.0), PolicyAgent(tmp_path / "policy.pt")
    t = env.reset()
    words = agent.act(t)                     # one state in a window of 4: padded, not refused
    assert words.shape == (TICKS,) and words.dtype == np.uint16
    t = env.step(words)
    assert len(game.replies) == 1 and game.replies[0] == words.tolist()
    link.close()
