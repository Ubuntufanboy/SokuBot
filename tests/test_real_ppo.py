"""PPO on the real game: the reward, the segments, and the learner-actor loop end to end.

The end-to-end test runs the real learner and the real actor code against fake games that speak
the DLL's protocol over sockets, so everything but the game itself is exercised.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest
import torch

from sokubot.data.state import CH, STATE_CHANNELS
from sokubot.env.vscom import AgentLink
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.ppo import PPOConfig, ppo_update
from sokubot.rl.real_ppo import (SegmentBuilder, Window, joint_from_words, round_outcome,
                                 step_reward)
from sokubot.rl.state_arena import StateObs
from sokubot.rl.state_critic import StateCritic
from sokubot.rl.state_reward import StateRewardConfig, compute_rewards
from test_state_loader import HEADER, SLOTS, _row

C = len(STATE_CHANNELS)
TICKS = 5


def round_states(n: int, seed: int = 0) -> np.ndarray:
    """[n+1, 2, C]: health falling unevenly on both sides, the odd crush and combo."""
    g = np.random.default_rng(seed)
    s = g.random((n + 1, 2, C)).astype(np.float32) * 0.1 + 0.4
    s[:, :, CH["hp"]] = 1.0 - np.cumsum(g.random((n + 1, 2)) * 0.05, 0)
    s[:, :, CH["crushed"]] = (g.random((n + 1, 2)) < 0.2).astype(np.float32)
    return s


def test_per_step_rewards_are_the_whole_round_computation_minus_the_outcome():
    cfg = StateRewardConfig(combo=0.3)
    s = round_states(12)
    joint = (np.random.default_rng(1).random((12, TICKS, 20)) < 0.2).astype(np.float32)
    whole, _, terms = compute_rewards(torch.from_numpy(s)[None], torch.from_numpy(joint)[None],
                                      torch.zeros(1, dtype=torch.long),
                                      StateRewardConfig(combo=0.3, win=0.0, lose=0.0))
    steps = [step_reward(s[t], s[t + 1], joint[t], cfg)[0] for t in range(12)]
    assert np.allclose(steps, whole[0].numpy(), atol=1e-6)
    assert abs(sum(steps)) > 1e-3                        # it is not all zeros


def test_the_outcome_follows_the_score():
    cfg = StateRewardConfig()
    assert round_outcome((0, 0), (1, 0), cfg) == cfg.win
    assert round_outcome((1, 1), (1, 2), cfg) == cfg.lose
    assert round_outcome((0, 0), (1, 1), cfg) == 0.0      # double KO: a draw


def test_joint_puts_p1s_ten_buttons_first():
    j = joint_from_words(np.array([[1, 0, 0, 0, 0], [0x200, 0, 0, 0, 0]], np.uint16))
    assert j.shape == (TICKS, 20)
    assert j[0, 0] == 1 and j[0, 19] == 1 and j[0].sum() == 2     # p1 up, p2 spell


def test_segments_cut_at_their_length_and_keep_round_ends():
    b = SegmentBuilder(3)
    for i in range(3):
        b.add(np.full((2, 4), i, np.float32), np.zeros((TICKS, 10)), -1.0, 0.5, i == 1, 7)
    assert b.full()
    seg = b.pop(np.zeros((2, 4), np.float32))
    assert seg["obs"].shape == (3, 2, 4) and seg["terminal"].tolist() == [0, 1, 0]
    assert not b.full() and b.obs == []


def test_a_window_pads_a_rounds_start_with_its_first_state():
    class T:
        def __init__(self, v):
            self.state = np.full((2, C), v, np.float32)
            self.proj = np.full((2, 8, 7), v, np.float32)
    w = Window(4, 3)
    s, p = w.push(T(1.0))
    assert s.shape == (4, 2, C) and p.shape == (4, 2, 3, 7) and (s == 1.0).all()
    w.push(T(2.0)); w.push(T(3.0)); w.push(T(4.0))
    s, _ = w.push(T(5.0))
    assert s[:, 0, 0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_ppo_update_takes_the_behaviour_log_probs_when_given():
    torch.manual_seed(0)
    H, dim = 2, 16
    pol = SokuPolicy(dim, H, TICKS)
    obs = torch.randn(64, H, dim)
    side = torch.zeros(64, dtype=torch.long)
    with torch.no_grad():
        act = pol(obs, side, sample=True).actions
        lp, _ = pol.log_prob_of(obs, side, act)
    base = {"obs": obs, "act": act, "side": side, "adv": torch.randn(64), "ret": torch.randn(64),
            "value": torch.zeros(64)}
    cfg = PPOConfig(epochs=1, minibatches=1)
    stats = {}
    for name, extra in (("on", {}), ("stale", {"logp_old": lp - 1.0})):
        p = SokuPolicy(dim, H, TICKS); p.load_state_dict(pol.state_dict())
        ref = SokuPolicy(dim, H, TICKS); ref.load_state_dict(pol.state_dict())
        c = StateCritic(dim, H)
        stats[name] = ppo_update(p, c, torch.optim.SGD(p.parameters(), lr=0.0),
                                 torch.optim.SGD(c.parameters(), lr=0.0), {**base, **extra},
                                 cfg, ref, np.random.default_rng(0))
    # Behaviour log-probs one nat below the current policy's: every ratio is e and clipped.
    assert stats["on"]["clip_frac"] == pytest.approx(0.0, abs=1e-6)
    assert stats["stale"]["clip_frac"] > 0.9


# ---- end to end ---------------------------------------------------------------------------------
def row(hp1: int, hp2: int, x: float) -> str:
    r = _row(p1_hp=str(hp1), p2_hp=str(hp2), p1_x=f"{x:.3f}")
    return ",".join(r[c] for c in HEADER)


class LoopingFakeGame:
    """A game that plays forever: matches of two rounds, each round 7 decisions then a KO."""

    def __init__(self, k: int):
        self.k = k
        self.stop = threading.Event()

    def start(self) -> AgentLink:
        link = AgentLink()
        self.thread = threading.Thread(target=self._play, args=(link.port,), daemon=True)
        self.thread.start()
        link.accept(10.0)
        link.handshake(10.0)
        self.link = link
        return link

    def _play(self, port: int) -> None:
        try:
            s = socket.create_connection(("127.0.0.1", port))
            f = s.makefile("r")
            send = lambda line: s.sendall((line + "\n").encode())
            send(f"H 2 16 -1 3 {TICKS}")
            send("C " + ",".join(HEADER))
            words = " ".join(["1"] * TICKS + ["16"] * TICKS)
            match = 0
            while not self.stop.is_set():
                match += 1
                send(f"M 16 {match % 20} 3 20 20")
                score = [0, 0]
                for rnd in range(2):
                    for i in range(7):
                        send(f"S 2 {rnd} {score[0]} {score[1]} {words} "
                             + row(10000 - 1000 * i, 10000 - 500 * i, 100.0 + i))
                        if not f.readline():
                            return
                    winner = (match + rnd) % 2
                    score[winner] += 1
                    ms = 5 if rnd == 1 else 3
                    hp = (0, 4000) if winner == 1 else (4000, 0)
                    send(f"S {ms} {rnd} {score[0]} {score[1]} {words} " + row(*hp, 150.0))
                    if not f.readline():
                        return
        except OSError:
            return

    def close(self) -> None:
        self.stop.set()
        self.link.close()


def make_init(tmp_path) -> tuple:
    H = 4
    obs = StateObs(np.zeros(33, np.float32), np.ones(33, np.float32),
                   np.zeros(7, np.float32), np.ones(7, np.float32), SLOTS)
    pol = SokuPolicy(obs.dim, H, TICKS)
    init = tmp_path / "init.pt"
    torch.save({"policy": pol.state_dict(), "obs": obs.state_dict(), "history": H,
                "ticks": TICKS, "slots": SLOTS}, init)
    rates = tmp_path / "rates.json"
    rates.write_text(json.dumps([0.05] * 10))
    return init, rates


def run_actor(run_dir, games=2):
    from scripts.vscom_actor import LearnerClient, play_game
    client = LearnerClient(run_dir, "test-actor", refresh_s=0.0)
    client.start()
    stats = {"decisions": 0, "rounds": 0, "failures": 0}
    ths = [threading.Thread(target=play_game, args=(k, client, LoopingFakeGame, 0, stats),
                            daemon=True) for k in range(games)]
    for t in ths:
        t.start()
    return client, stats, ths


def test_learner_and_actor_train_to_the_budget_and_resume(tmp_path):
    from scripts.train_vscom_ppo import main as learner
    init, rates = make_init(tmp_path)
    run = tmp_path / "run"
    args = ["--run-dir", str(run), "--init", str(init), "--button-rates", str(rates),
            "--batch", "32", "--segment", "8", "--device", "cpu", "--ckpt-every", "1",
            "--snapshot-minutes", "0", "--done-linger", "7"]
    client, stats, ths = run_actor(run)
    assert learner(args + ["--budget", "64"]) == 0
    assert (run / "DONE").exists() and (run / "latest.pt").exists()
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["version"] >= 2 and ck["decisions"] >= 64
    assert any(r["won"] for r in ck["rounds"]) and any(r["lost"] for r in ck["rounds"])
    assert any(r["match_over"] for r in ck["rounds"])
    lines = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    assert lines[-1]["version"] == ck["version"]
    client.stop.wait(60)                                   # told `done` on the next ack
    assert client.stop.is_set()
    snaps = list(run.glob("policy_v*.pt"))
    assert snaps                                           # load_agent-format snapshots
    from sokubot.rl.policy_io import load_agent
    load_agent(snaps[0])

    # Resume: a bigger budget continues from the checkpoint, not from scratch. The new actor waits
    # for the new learner's address -- the first learner still listens inside this process, and the
    # nonce check is what would refuse it if the actor got there first.
    (run / "DONE").unlink()
    old = (run / "learner.addr").read_text()
    started = {}

    def actor_when_address_changes():
        while (run / "learner.addr").read_text() == old:
            time.sleep(0.2)
        started["client"] = run_actor(run)[0]
    threading.Thread(target=actor_when_address_changes, daemon=True).start()
    assert learner(args + ["--budget", "128"]) == 0
    ck2 = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ck2["version"] > ck["version"] and ck2["decisions"] >= 128
    started["client"].stop.wait(60)
