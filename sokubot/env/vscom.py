"""The real game against its own AI (vs COM), as an environment.

The game runs SokuFrameExtractor with SFE_VSCOM=1 and SFE_AGENT=host:port. The module starts a
vs-COM match with no menus (Cirno vs the COM at a chosen difficulty by default), restarts after
every match, and every `ticks` fight ticks sends P1's view of the game and waits for P1's next
`ticks` input words. Protocol: SokuFrameExtractor dll/include/sfe/agent_link.hpp.

THE OBSERVATION IS THE CORPUS'S OWN
-----------------------------------
Each state arrives as a row of the capture CSV, written by the same C++ function that writes the
sidecars (sfe/state_row.hpp), and is parsed here by the corpus's own parser
(`data.state.parse_state`). A policy trained on the corpus or in the simulator therefore sees
exactly the observation it trained on, not a re-derivation of it.

Measured on Amarel (SFE probe job 62073600): about 80-100 game ticks/s per game on 4 cpus, so
16-20 decisions/s at 5 ticks a decision; one CSV row is one tick, as in the corpus.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from sokubot.data.soku import BUTTONS
from sokubot.data.state import FULL_HP, RowParser

PROTOCOL = 2
# BattleManager matchState, read off the game in the probe: 0 init, 1 round intro, 2 fight,
# 3 round over, 5 match over, 6 results. Decisions are only ever asked for during 2.
FIGHT, ROUND_OVER, MATCH_OVER = 2, 3, 5


def words_to_buttons(words: np.ndarray) -> np.ndarray:
    """uint16 input words [...] -> float32 [..., 10] in BUTTONS order (bit i is BUTTONS[i])."""
    w = np.asarray(words, dtype=np.int64)[..., None]
    return ((w >> np.arange(len(BUTTONS))) & 1).astype(np.float32)


def buttons_to_words(buttons: np.ndarray) -> np.ndarray:
    """[..., 10] 0/1 -> uint16 words. The inverse of `words_to_buttons`."""
    b = (np.asarray(buttons) > 0.5).astype(np.int64)
    return (b << np.arange(len(BUTTONS))).sum(-1).astype(np.uint16)


@dataclass
class Tick:
    """One message from the game: P1's view at a decision, or at the tick a round stopped."""
    match_state: int
    round: int
    score: tuple[int, int]          # rounds won, P1 then P2
    words: np.ndarray               # [2, ticks] uint16: what P1 and P2 played, previous ticks
    state: np.ndarray               # [2, C], the corpus's state channels
    proj: np.ndarray                # [2, K, F]
    hp: tuple[int, int]             # raw HP, FULL_HP = full
    chars: tuple[int, int] = (-1, -1)   # this match's characters, P1 then the COM

    @property
    def fight(self) -> bool:
        return self.match_state == FIGHT

    @property
    def hp_frac(self) -> tuple[float, float]:
        return self.hp[0] / FULL_HP, self.hp[1] / FULL_HP


class LinkClosed(ConnectionError):
    pass


class AgentLink:
    """The agent's end of the DLL's TCP link. It listens; the game connects to it."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.host, self.port = self._srv.getsockname()
        self._conn: socket.socket | None = None
        self._buf = b""
        self.ticks = 0
        self.hello: tuple[int, ...] = ()
        self.header = ""
        self._hp_cols: tuple[int, int] = (-1, -1)
        # The current match, from the game's M line: p1 char, COM char, level, p1 cards, COM cards.
        self.match: tuple[int, int, int, int, int] = (-1, -1, -1, -1, -1)
        self.matches_seen = 0

    def accept(self, timeout: float, alive=lambda: True) -> None:
        """Wait for the game. `alive()` returning False (the game process died) ends the wait."""
        self._srv.settimeout(1.0)
        t0 = time.monotonic()
        while True:
            try:
                conn, _ = self._srv.accept()
                break
            except socket.timeout:
                if not alive():
                    raise LinkClosed("the game exited before it connected")
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"no connection within {timeout:.0f}s")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if self._conn is not None:
            self._conn.close()
        self._conn, self._buf = conn, b""

    def _line(self, timeout: float) -> str:
        assert self._conn is not None, "not connected"
        self._conn.settimeout(timeout)
        while b"\n" not in self._buf:
            try:
                chunk = self._conn.recv(65536)
            except socket.timeout as e:
                raise TimeoutError(f"no message from the game within {timeout:.0f}s") from e
            if not chunk:
                raise LinkClosed("the game closed the link")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode()

    def handshake(self, timeout: float = 60.0) -> None:
        h = self._line(timeout).split()
        if h[0] != "H" or int(h[1]) != PROTOCOL:
            raise ConnectionError(f"expected 'H {PROTOCOL} ...', got {' '.join(h[:3])!r}")
        self.hello = tuple(int(x) for x in h[1:])          # protocol, p1, p2, level, ticks
        self.ticks = self.hello[4]
        c = self._line(timeout)
        if not c.startswith("C "):
            raise ConnectionError(f"expected the CSV header, got {c[:40]!r}")
        self.header = c[2:]
        self.parser = RowParser(self.header, "vscom")
        cols = self.header.split(",")
        self._hp_cols = (cols.index("p1_hp"), cols.index("p2_hp"))

    def recv(self, timeout: float = 60.0) -> Tick:
        line = self._line(timeout)
        while line.startswith("M "):           # a new match: who is playing whom. No reply.
            self.match = tuple(int(x) for x in line.split()[1:6])
            self.matches_seen += 1
            line = self._line(timeout)
        if not line.startswith("S "):
            raise ConnectionError(f"expected a state line, got {line[:40]!r}")
        parts = line.split(" ", 5 + 2 * self.ticks)
        ms, rnd, s1, s2 = (int(x) for x in parts[1:5])
        words = np.array(parts[5:5 + 2 * self.ticks], dtype=np.uint16).reshape(2, self.ticks)
        row = parts[5 + 2 * self.ticks]
        state, proj = self.parser.parse(row)
        cells = row.split(",")
        hp = (int(cells[self._hp_cols[0]]), int(cells[self._hp_cols[1]]))
        return Tick(ms, rnd, (s1, s2), words, state, proj, hp, self.match[:2])

    def send(self, words) -> None:
        assert self._conn is not None, "not connected"
        w = np.asarray(words, dtype=np.int64).reshape(-1)
        if len(w) != self.ticks:
            raise ValueError(f"{len(w)} words for {self.ticks} ticks")
        self._conn.sendall(("A " + " ".join(str(int(x)) for x in w) + "\n").encode())

    def close(self) -> None:
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._conn = None


@dataclass
class VsComGame:
    """One game process playing vs COM, with its link.

    `argv` launches the game (on Amarel: the bwrap sandbox around the SFE runner); `env` adds the
    game's SFE_* settings. SFE_VSCOM and SFE_AGENT are set here. The game may be relaunched once by
    the runner (first launches in a fresh prefix sometimes exit before the module runs), so the
    connection wait covers both launches.
    """
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None
    connect_timeout: float = 300.0
    link: AgentLink | None = None
    proc: subprocess.Popen | None = None

    def start(self) -> AgentLink:
        self.link = AgentLink()
        env = {**os.environ, **self.env, "SFE_VSCOM": "1",
               "SFE_AGENT": f"{self.link.host}:{self.link.port}"}
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.link.accept(self.connect_timeout, alive=lambda: self.proc.poll() is None)
        self.link.handshake()
        return self.link

    def close(self, grace: float = 10.0) -> None:
        """Hang up and take the game down. Hanging up alone is not enough: the module exits only
        when it next reaches the title, which can be a whole match away."""
        if self.link is not None:
            self.link.close()
        if self.proc is not None and self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue


class VsComEnv:
    """P1 (the agent) against the COM, one ROUND per episode.

    reset() -> the first decision of the next round. step(words) -> the next Tick; when its
    `fight` is False the round is over (match_state 3, or 5 if that round ended the match) and the
    Tick holds the final state, the KO included. Matches restart by themselves in the game.
    """

    def __init__(self, link: AgentLink, timeout: float = 120.0):
        self.link = link
        self.timeout = timeout
        self.ticks = link.ticks

    def _recv(self) -> Tick:
        return self.link.recv(self.timeout)

    def reset(self) -> Tick:
        while True:
            t = self._recv()
            if t.fight:
                return t
            self.link.send(np.zeros(self.ticks, np.uint16))   # ack a stray end-of-round tick

    def step(self, words) -> Tick:
        self.link.send(words)
        t = self._recv()
        if not t.fight:
            # The game waits for an answer to every state, the end-of-round one included; its
            # action is ignored. Answer now so the next reset() reads the next round.
            self.link.send(np.zeros(self.ticks, np.uint16))
        return t


# ---------------------------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------------------------
class PolicyAgent:
    """A trained state policy (train_state_grpo / train_state_ppo checkpoint) as P1.

    It keeps its own history window per round. Before a round has `history` decisions the window
    is padded by repeating its oldest state, the nearest thing to a real past the round has; the
    simulator-trained policy only ever saw full windows of real rows.
    """

    def __init__(self, path: Path, device: str = "cpu", sample: bool = True, side: int = 0):
        from sokubot.rl.policy_io import load_agent
        self.pol, self.obs, self.H, self.ticks, self.slots, self.meta = load_agent(path, device)
        self.device, self.sample, self.side = device, sample, side
        self.reset()

    def reset(self) -> None:
        self._s: deque = deque(maxlen=self.H)
        self._p: deque = deque(maxlen=self.H)

    @torch.no_grad()
    def act(self, t: Tick) -> np.ndarray:
        self._s.append(t.state)
        self._p.append(t.proj[:, :self.slots])
        pad = self.H - len(self._s)
        s = np.stack([self._s[0]] * pad + list(self._s))
        p = np.stack([self._p[0]] * pad + list(self._p))
        sd = torch.tensor([self.side], dtype=torch.long, device=self.device)
        o = self.obs(torch.from_numpy(s)[None].to(self.device),
                     torch.from_numpy(p)[None].to(self.device), sd)
        act = self.pol(o, sd, sample=self.sample).actions[0].cpu().numpy()   # [ticks, 10]
        return buttons_to_words(act)


class NeutralAgent:
    """Presses nothing: the floor any trained policy has to clear."""

    def __init__(self, ticks: int = 5):
        self.ticks = ticks

    def reset(self) -> None:
        pass

    def act(self, t: Tick) -> np.ndarray:
        return np.zeros(self.ticks, np.uint16)


class RandomAgent:
    """Each button pressed independently at a fixed rate, held for a whole decision."""

    def __init__(self, ticks: int = 5, rate: float = 0.1, seed: int = 0):
        self.ticks, self.rate = ticks, rate
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, t: Tick) -> np.ndarray:
        btn = (self.rng.random(len(BUTTONS)) < self.rate).astype(np.float32)
        return np.repeat(buttons_to_words(btn)[None], self.ticks)
