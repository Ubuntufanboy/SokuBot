"""Play matches against the game's own AI (vs COM) and score them the way the milestones are set.

    python -m scripts.vscom_play --agent policy --policy ~/sokubot-runs/ppo-hold-s0/policy.pt \\
        --games 8 --matches 200 --level 3 --out ~/sokubot-runs/vscom/ppo-hold-s0.json

Milestone 2 as the user set it: a 95% match win rate against the Lunatic COM (level 3), with at
least 20% of P1's health left at the end of at least 80% of those wins. Both numbers are reported
with Wilson 95% intervals. Telling 95% from 90% takes a few hundred matches; the interval says how
far a run of N actually gets.

Each game runs in its own bwrap sandbox (SokuFrameExtractor ops/bwrap/sfe-bwrap) from its own clone
of the game, with the vs-COM DLL copied in. Every game is one thread here; a small state policy
decides in about a millisecond on one CPU thread, so the games, not the policy, set the pace.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.env.vscom import (MATCH_OVER, NeutralAgent, PolicyAgent, RandomAgent, VsComEnv,
                               VsComGame)

CIRNO = 16


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def prepare_game(k: int, a) -> Path:
    """A clone of the base game with the vs-COM DLL, private to game k (clones copy everything but
    the .dat archives, so replacing the DLL cannot reach the base)."""
    g = a.work / f"game{k}"
    if g.exists():
        shutil.rmtree(g)
    subprocess.run([str(a.sfe / "ops/bwrap/clone-game.sh"), str(a.game_base), str(g)],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.copy2(a.dll, g / "modules/SokuFrameExtractor/SokuFrameExtractor.dll")
    # clone-game.sh hard-links every *.dat, the game's save files included, so all clones and the
    # base share one inode -- and the game saves with CREATE_ALWAYS, which rewrites that inode in
    # place. Unchanged after 30+ vs-COM matches (checked 2026-10-01), but config123.dat names the
    # profiles that decide P1's deck, so each game gets copies of its own.
    for name in ("config123.dat", "score123.dat", "score123Backup.dat"):
        f = g / name
        if f.exists():
            data = f.read_bytes()
            f.unlink()
            f.write_bytes(data)
    return g


def game_for(k: int, a, replays: Path) -> VsComGame:
    g = prepare_game(k, a)
    slot = a.game_offset + k          # node-wide: CPU block and X display must not collide
    prefix = a.local / f"prefix{k}"
    out = a.work / f"out{k}"
    for d in (prefix, out):
        d.mkdir(parents=True, exist_ok=True)
    env = {
        "SFE_GAME": str(g), "SFE_PREFIX": str(prefix), "SFE_OUT": str(out),
        "SFE_REPLAYS": str(replays), "SFE_CPU_BLOCK": str(slot),
        # The link is TCP to this process, so the sandbox shares the network namespace; each game
        # then needs its own X display block (sfe-bwrap explains the collision this avoids).
        "SFE_UNSHARE_NET": "0", "SFE_DISPLAY_BASE": str(100 + 20 * slot),
        "SFE_WIRE_KEYMGR": "1",
        "SFE_COM_LEVEL": str(a.level), "SFE_P1_CHAR": str(a.p1_char),
        "SFE_P2_CHAR": str(a.p2_char), "SFE_P1_DECK": str(a.p1_deck),
        # Each game its own opponent schedule when the COM's character is random (-1).
        "SFE_SEED": str(a.seed * 1000 + slot + 1),
    }
    argv = [str(a.sfe / "ops/bwrap/sfe-bwrap"), "/app/docker/entrypoint.sh",
            "--replay-dir", "/replays", "--out", "/out", "--shard", "0/1", "--no-video",
            "--cpus", str(a.cpus_per_game), "--timeout", "864000", "--min-free-gb", "1",
            "--limit", "1"]
    return VsComGame(argv, env, log_path=a.work / f"game{k}.log")


def make_agent(a, k: int):
    if a.agent == "policy":
        return PolicyAgent(a.policy, sample=not a.greedy)
    if a.agent == "random":
        return RandomAgent(rate=a.random_rate, seed=a.seed * 1000 + k)
    return NeutralAgent()


class Tally:
    def __init__(self, matches: int, out: Path | None, seconds: float = 0.0):
        self.lock = threading.Lock()
        self.deadline = time.monotonic() + seconds if seconds else 0.0
        self.target = matches
        self.matches: list[dict] = []
        self.rounds: list[dict] = []
        self.started = 0
        self.decisions = 0
        self.t0 = time.monotonic()
        self.fh = open(out.with_suffix(".rounds.jsonl"), "w") if out else None

    def claim_match(self) -> bool:
        with self.lock:
            if self.started >= self.target or (self.deadline and time.monotonic() > self.deadline):
                return False
            self.started += 1
            return True

    def round_done(self, rec: dict) -> None:
        with self.lock:
            self.rounds.append(rec)
            self.decisions += rec["decisions"]
            if self.fh:
                self.fh.write(json.dumps(rec) + "\n")
                self.fh.flush()

    def match_done(self, rec: dict) -> None:
        with self.lock:
            self.matches.append(rec)
            n = len(self.matches)
            w = sum(m["won"] for m in self.matches)
            dt = time.monotonic() - self.t0
            print(f"  match {n}/{self.target} vs char {rec['opponent']:>2}: "
                  f"{'WIN ' if rec['won'] else 'loss'} "
                  f"{rec['score'][0]}-{rec['score'][1]} p1 hp {rec['p1_hp']:.2f} | "
                  f"won {w}/{n} | {self.decisions / max(dt, 1e-9):.0f} decisions/s", flush=True)

    def summary(self) -> dict:
        m = self.matches
        n = len(m)
        wins = [x for x in m if x["won"]]
        healthy = sum(x["p1_hp"] >= 0.2 for x in wins)
        rw = sum(r["won"] for r in self.rounds)
        per_char: dict[int, list[int]] = {}
        for x in m:
            c = per_char.setdefault(x["opponent"], [0, 0, 0, 0])
            c[0] += x["won"]
            c[1] += 1
        for r in self.rounds:
            c = per_char.setdefault(r["opponent"], [0, 0, 0, 0])
            c[2] += r["won"]
            c[3] += 1
        return {
            "per_opponent": {str(k): {"match_wins": v[0], "matches": v[1], "round_wins": v[2],
                                      "rounds": v[3]} for k, v in sorted(per_char.items())},
            "matches": n, "match_wins": len(wins), "match_win_rate": len(wins) / n if n else None,
            "match_win_ci95": wilson(len(wins), n),
            "wins_with_20pct_hp": healthy,
            "healthy_win_frac": healthy / len(wins) if wins else None,
            "healthy_win_ci95": wilson(healthy, len(wins)),
            "rounds": len(self.rounds), "round_wins": rw,
            "round_win_ci95": wilson(rw, len(self.rounds)),
            "decisions": self.decisions,
            "decisions_per_s": self.decisions / max(time.monotonic() - self.t0, 1e-9),
            "milestone2_met": bool(n and len(wins) / n >= 0.95 and wins
                                   and healthy / len(wins) >= 0.80),
        }


def play(k: int, a, replays: Path, tally: Tally, errors: list) -> None:
    game = None
    try:
        torch.set_num_threads(1)
        game = game_for(k, a, replays)
        link = game.start()
        if link.ticks != 5 and a.agent == "policy":
            raise RuntimeError(f"game decides every {link.ticks} ticks; the policy was trained on 5")
        env = VsComEnv(link)
        agent = make_agent(a, k)
        while tally.claim_match():
            score, match_rounds = (0, 0), []
            while True:
                agent.reset()
                t = env.reset()
                start = t.score
                n, t0 = 0, time.monotonic()
                while t.fight:
                    t = env.step(agent.act(t))
                    n += 1
                won = t.score[0] > start[0]
                rec = {"game": k, "opponent": t.chars[1], "cards": list(link.match[3:5]),
                       "round": t.round, "won": bool(won), "p1_hp": t.hp_frac[0],
                       "p2_hp": t.hp_frac[1], "decisions": n, "seconds": time.monotonic() - t0,
                       "match_state": t.match_state, "score": list(t.score)}
                tally.round_done(rec)
                match_rounds.append(rec)
                score = t.score
                if t.match_state == MATCH_OVER:
                    break
            tally.match_done({"game": k, "opponent": match_rounds[-1]["opponent"],
                              "won": score[0] > score[1], "score": list(score),
                              "p1_hp": match_rounds[-1]["p1_hp"],
                              "rounds": len(match_rounds)})
    except Exception as e:                       # one dead game must not end the run
        errors.append(f"game {k}: {type(e).__name__}: {e}")
        print(f"  game {k} failed: {type(e).__name__}: {e}", flush=True)
    finally:
        if game is not None:
            game.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", choices=("policy", "neutral", "random"), default="policy")
    ap.add_argument("--policy", type=Path, help="a train_state_ppo / train_state_grpo policy.pt")
    ap.add_argument("--greedy", action="store_true", help="most likely action, not a sample")
    ap.add_argument("--random-rate", type=float, default=0.1)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--level", type=int, default=3, help="COM difficulty 0-3; 3 is Lunatic")
    ap.add_argument("--p1-char", type=int, default=CIRNO)
    ap.add_argument("--p2-char", type=int, default=CIRNO,
                    help="the COM's character; -1 draws one of the 20 per match")
    ap.add_argument("--p1-deck", type=int, default=0, help="profile deck slot for P1")
    ap.add_argument("--cpus-per-game", type=int, default=4)
    ap.add_argument("--game-offset", type=int, default=0,
                    help="index of this run's first game among all runs sharing the node; each "
                         "game takes CPU block and X display block (offset + k)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="start no new match after this long (0: only --matches limits)")
    ap.add_argument("--sfe", type=Path, default=Path("~/sfe").expanduser())
    ap.add_argument("--game-base", type=Path, default=Path("~/sfe-game").expanduser())
    ap.add_argument("--dll", type=Path,
                    default=Path("~/sfe-vscom-dll/SokuFrameExtractor.dll").expanduser())
    ap.add_argument("--replay", type=Path, default=None,
                    help="any .rep: the runner insists on one; vs COM never reads it")
    ap.add_argument("--work", type=Path, default=Path("~/sfe-vscom-work/play").expanduser())
    ap.add_argument("--local", type=Path,
                    default=Path(os.environ.get("TMPDIR", "/tmp")) / f"vscom-{os.getpid()}",
                    help="node-local scratch for the Wine prefixes")
    ap.add_argument("--out", type=Path, default=None, help="summary JSON (+ .rounds.jsonl)")
    a = ap.parse_args(argv)
    if a.agent == "policy" and a.policy is None:
        ap.error("--agent policy needs --policy")
    a.work.mkdir(parents=True, exist_ok=True)
    a.local.mkdir(parents=True, exist_ok=True)
    replays = a.local / "replays"
    replays.mkdir(exist_ok=True)
    rep = a.replay or next(Path("~/sfe-replays").expanduser().glob("*.rep"))
    shutil.copy2(rep, replays / rep.name)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"vs COM: {a.agent}{' ' + str(a.policy) if a.policy else ''} | level {a.level} | "
          f"P1 char {a.p1_char} vs COM char {a.p2_char} | {a.games} games, {a.matches} matches",
          flush=True)
    tally, errors = Tally(a.matches, a.out, a.seconds), []
    threads = [threading.Thread(target=play, args=(k, a, replays, tally, errors), daemon=True)
               for k in range(a.games)]
    for t in threads:
        t.start()
        time.sleep(2.0)          # staggered: simultaneous first launches exit early more often
    for t in threads:
        t.join()
    s = tally.summary()
    s.update({"agent": a.agent, "policy": str(a.policy) if a.policy else None, "level": a.level,
              "p1_char": a.p1_char, "p2_char": a.p2_char, "errors": errors})
    lo, hi = s["match_win_ci95"]
    print(f"\nmatches won {s['match_wins']}/{s['matches']} = "
          f"{(s['match_win_rate'] or 0):.3f} [95% CI {lo:.3f}, {hi:.3f}] | rounds won "
          f"{s['round_wins']}/{s['rounds']} | wins with >= 20% HP {s['wins_with_20pct_hp']}"
          f"/{s['match_wins']} | milestone 2 met: {s['milestone2_met']}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(s, indent=1))
    shutil.rmtree(a.local, ignore_errors=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
