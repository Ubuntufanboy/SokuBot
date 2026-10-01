"""PPO on the real game against its own AI (vs COM): an actor.

    python -m scripts.vscom_actor --run-dir ~/sokubot-runs/vscom-ppo-s0 --games 8 \\
        --cpus-per-game 2 --game-offset 0 --seed 11

Plays `--games` games at once, each in its own bwrap sandbox, with a shared local copy of the
learner's policy, and ships fixed-length segments to the learner named in `<run-dir>/learner.addr`
(see sokubot/rl/real_ppo.py). New weights are fetched by a background thread and swapped in whole,
so a slow download never holds a game past the DLL's reply timeout. If the learner goes away the
games keep playing on the last weights, segments are dropped, and the client reconnects to whatever
the address file names next. A dead game is relaunched. The actor exits when the learner says the
run is done, or at --stop-after-hours.
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.env.vscom import MATCH_OVER, VsComEnv, buttons_to_words
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.real_ppo import (PROTOCOL, SegmentBuilder, Window, joint_from_words, recv_msg,
                                 round_outcome, send_msg)
from sokubot.rl.state_arena import StateObs


class LearnerClient(threading.Thread):
    """The actor's one connection to the learner: segments out, weights in."""

    def __init__(self, run_dir: Path, actor_id: str, refresh_s: float = 5.0):
        super().__init__(daemon=True)
        self.run_dir, self.actor_id, self.refresh_s = run_dir, actor_id, refresh_s
        self.out: queue.Queue = queue.Queue(maxsize=512)
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.meta: dict | None = None
        self.obs: StateObs | None = None
        self.current: tuple[int, SokuPolicy] | None = None   # (version, policy), swapped whole
        self.sent = self.dropped = 0

    def _policy(self, sd: dict) -> SokuPolicy:
        pol = SokuPolicy(self.meta["obs_dim"], self.meta["history"], self.meta["ticks"])
        pol.load_state_dict(sd)
        pol.eval()
        return pol

    def _connect(self) -> socket.socket:
        while not self.stop.is_set():
            try:
                host, port, nonce = (self.run_dir / "learner.addr").read_text().split()
                s = socket.create_connection((host, int(port)), timeout=30)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(120)
                send_msg(s, {"type": "hello", "protocol": PROTOCOL, "actor": self.actor_id})
                w = recv_msg(s)
                if w.get("type") != "welcome":
                    raise ConnectionError(str(w))
                if w["meta"].get("nonce") != nonce:
                    # A learner that is not the one the address file names: an old process that
                    # still holds its port. Feeding it would train a run nobody is watching.
                    s.close()
                    raise ConnectionError("the learner's nonce is not the address file's")
                if self.meta is None:
                    self.meta = w["meta"]
                    obs = StateObs(np.zeros(33, np.float32), np.ones(33, np.float32),
                                   np.zeros(7, np.float32), np.ones(7, np.float32),
                                   self.meta["slots"])
                    obs.load_state_dict(self.meta["obs"])
                    self.obs = obs.eval()
                wt = recv_msg(s)
                self.current = (wt["version"], self._policy(wt["weights"]))
                if w.get("done"):
                    self.stop.set()
                self.ready.set()
                print(f"[{self.actor_id}] connected to {host}:{port}, weights v{wt['version']}",
                      flush=True)
                return s
            except (OSError, ConnectionError, ValueError, EOFError) as e:
                print(f"[{self.actor_id}] learner not reachable ({e}); retrying", flush=True)
                time.sleep(10)
        raise ConnectionError("stopped")

    def run(self) -> None:
        last_fetch = 0.0
        while not self.stop.is_set():
            try:
                s = self._connect()
            except ConnectionError:
                return
            try:
                while not self.stop.is_set():
                    try:
                        seg = self.out.get(timeout=5.0)
                        send_msg(s, {"type": "segment", "segment": seg})
                        self.sent += 1
                    except queue.Empty:
                        send_msg(s, {"type": "ping"})
                    ack = recv_msg(s)
                    if ack.get("done"):
                        print(f"[{self.actor_id}] the learner says the run is done", flush=True)
                        self.stop.set()
                        break
                    if (ack["version"] > self.current[0]
                            and time.time() - last_fetch >= self.refresh_s):
                        send_msg(s, {"type": "weights"})
                        wt = recv_msg(s)
                        self.current = (wt["version"], self._policy(wt["weights"]))
                        last_fetch = time.time()
            except (OSError, ConnectionError, EOFError) as e:
                print(f"[{self.actor_id}] lost the learner ({e}); reconnecting", flush=True)
                try:
                    s.close()
                except OSError:
                    pass
                time.sleep(5)

    def put(self, seg: dict) -> None:
        """Queue a segment; if the learner is away for long, drop the oldest (they go stale)."""
        while True:
            try:
                self.out.put_nowait(seg)
                return
            except queue.Full:
                try:
                    self.out.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass


def observe(obs: StateObs, s: np.ndarray, p: np.ndarray) -> np.ndarray:
    side = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        return obs(torch.from_numpy(s)[None], torch.from_numpy(p)[None], side)[0].numpy()


def sample(pol: SokuPolicy, o: np.ndarray, sigma: float, gen: torch.Generator):
    """-> (executed action [ticks, 10] uint8, its log-probability under `pol`). The EXECUTED,
    jittered chunk is what is scored, exactly as the simulator arena scores it."""
    side = torch.zeros(1, dtype=torch.long)
    x = torch.from_numpy(o)[None]
    with torch.no_grad():
        act, lp = pol.act_and_score(x, side, sigma, generator=gen)
    return act[0].numpy().astype(np.uint8), float(lp[0])


def play_game(k: int, client: LearnerClient, make_game, seed: int, stats: dict) -> None:
    """One game slot: launch, play rounds into segments until told to stop, relaunch on failure."""
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(seed * 1000 + k)
    client.ready.wait()
    meta = client.meta
    cfg = meta["reward"]
    failures = 0
    while not client.stop.is_set():
        game = None
        try:
            game = make_game(k)
            env = VsComEnv(game.start(), timeout=120.0)
            builder = SegmentBuilder(meta["segment"], cfg)
            win = Window(meta["history"], meta["slots"])
            failures = 0
            while not client.stop.is_set():
                win.reset()
                t = env.reset()
                start, n = t.score, 0
                o = observe(client.obs, *win.push(t))
                while True:
                    version, pol = client.current
                    act, lp = sample(pol, o, meta["jitter_sigma"], gen)
                    t2 = env.step(buttons_to_words(act))
                    terminal = not t2.fight
                    n += 1
                    out = 0.0
                    if terminal:
                        out = round_outcome(start, t2.score, cfg)
                        builder.rounds.append({
                            "opponent": int(t2.chars[1]), "won": out > 0, "lost": out < 0,
                            "p1_hp": t2.hp_frac[0], "p2_hp": t2.hp_frac[1], "decisions": n,
                            "match_over": t2.match_state == MATCH_OVER,
                            "match_won": t2.match_state == MATCH_OVER and t2.score[0] > t2.score[1],
                            "score": list(t2.score), "version": version})
                        o_next = o                     # unused: the step is terminal
                    else:
                        o_next = observe(client.obs, *win.push(t2))
                    builder.add(o, act, lp, t.state, t2.state, joint_from_words(t2.words),
                                out, terminal, version)
                    if builder.full():
                        client.put(builder.pop(o_next))
                    stats["decisions"] += 1
                    if terminal:
                        stats["rounds"] += 1
                        break
                    t, o = t2, o_next
        except Exception as e:                         # one dead game must not end the actor
            failures += 1
            stats["failures"] += 1
            print(f"[game {k}] {type(e).__name__}: {e} (failure {failures})", flush=True)
            if failures <= 2:
                traceback.print_exc()
        finally:
            if game is not None:
                game.close()
        if not client.stop.is_set():
            time.sleep(min(300, 30 * failures))


def main(argv: list[str] | None = None) -> int:
    from scripts import vscom_play
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--cpus-per-game", type=int, default=2)
    ap.add_argument("--game-offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-after-hours", type=float, default=0.0)
    ap.add_argument("--sfe", type=Path, default=Path("~/sfe").expanduser())
    ap.add_argument("--game-base", type=Path, default=Path("~/sfe-game").expanduser())
    ap.add_argument("--dll", type=Path,
                    default=Path("~/sfe-vscom-dll/SokuFrameExtractor.dll").expanduser())
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--local", type=Path, default=None)
    a = ap.parse_args(argv)
    host = socket.gethostname().split(".")[0]
    actor_id = f"{host}-{os.getpid()}"
    a.work = a.work or Path(f"~/sfe-vscom-work/actors/{actor_id}").expanduser()
    a.local = a.local or Path(os.environ.get("TMPDIR", "/tmp")) / f"vscom-actor-{os.getpid()}"
    for d in (a.work, a.local):
        d.mkdir(parents=True, exist_ok=True)
    replays = a.local / "replays"
    replays.mkdir(exist_ok=True)
    rep = next(Path("~/sfe-replays").expanduser().glob("*.rep"))
    shutil.copy2(rep, replays / rep.name)

    client = LearnerClient(a.run_dir, actor_id)
    client.start()
    if not client.ready.wait(timeout=3600):
        print("no learner within an hour; giving up", flush=True)
        return 1
    meta = client.meta
    # vscom_play's launcher, configured from the learner: the opponent schedule, the level.
    a.level, a.p2_char, a.p1_char, a.p1_deck = meta["level"], meta["p2_char"], 16, 0

    def make_game(k: int):
        return vscom_play.game_for(k, a, replays)

    stats = {"decisions": 0, "rounds": 0, "failures": 0}
    threads = []
    for k in range(a.games):
        th = threading.Thread(target=play_game, args=(k, client, make_game, a.seed, stats),
                              daemon=True)
        th.start()
        threads.append(th)
        time.sleep(2.0)
    t0, last = time.time(), 0
    while not client.stop.is_set():
        time.sleep(60)
        d = stats["decisions"]
        print(f"[{actor_id}] {d:,} decisions ({(d - last) / 60:.0f}/s), {stats['rounds']} rounds, "
              f"{client.sent} segments sent, {client.dropped} dropped, "
              f"{stats['failures']} game failures, weights v{client.current[0]}", flush=True)
        last = d
        if a.stop_after_hours and time.time() - t0 > a.stop_after_hours * 3600:
            client.stop.set()
    for th in threads:
        th.join(timeout=120)
    shutil.rmtree(a.local, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
