"""PPO on the real game against its own AI (vs COM): the learner.

    python -m scripts.train_vscom_ppo --run-dir ~/sokubot-runs/vscom-ppo-s0 \\
        --init ~/sokubot-runs/vscom-arms/init-s0.pt --button-rates ~/sokubot-runs/button_rates.json

The game-playing actors (scripts/vscom_actor.py) run in other jobs and find this process through
`<run-dir>/learner.addr` ("host port nonce"). See sokubot/rl/real_ppo.py for the layout and the
reward.

SAME START, SAME OBJECTIVE AS THE SIMULATOR RUNS
-------------------------------------------------
`--init` is the corpus prior those runs started from (their frozen `reference`), and it stays the
KL reference here. The PPO hyperparameters, the entropy floor and the button-rate floor are
train_state_ppo's defaults. So a difference between this run and the simulator's is the
environment, not the recipe.

RESUMABLE
---------
SIGUSR1/SIGTERM (Slurm's warning before a time limit or a preemption) -> latest.pt, exit 99, and
the Slurm script requeues. A new learner process writes a new nonce; actors reconnect to whatever
the address file names. When the decision budget is spent it writes DONE and every actor's next
acknowledgement tells it to stop.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sokubot.rl.grpo import ButtonRateFloor, EntropyFloor
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.policy_io import load_agent
from sokubot.rl.ppo import PPOConfig, ppo_update
from sokubot.rl.real_ppo import PROTOCOL, recv_msg, send_msg
from sokubot.rl.state_critic import StateCritic, lambda_returns
from sokubot.rl.state_reward import StateRewardConfig
from scripts.train_state_ppo import atomic_save, rng_state


class Hub:
    """What the actor handlers and the update loop share."""

    def __init__(self, meta: dict):
        self.lock = threading.Condition()
        self.segments: deque = deque()
        self.steps = 0
        self.version = 0
        self.weights_blob = b""
        self.meta = meta
        self.done = False
        self.actors: dict[str, float] = {}
        self.rounds: list[dict] = []

    def publish(self, version: int, policy: SokuPolicy) -> None:
        sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        blob = pickle.dumps({"type": "weights", "version": version, "weights": sd},
                            protocol=pickle.HIGHEST_PROTOCOL)
        with self.lock:
            self.version, self.weights_blob = version, blob

    def add(self, seg: dict) -> None:
        with self.lock:
            self.segments.append(seg)
            self.steps += len(seg["reward"])
            self.rounds.extend(seg["rounds"])
            self.lock.notify_all()

    def take(self, steps: int, stop: dict) -> list[dict]:
        with self.lock:
            while self.steps < steps and not stop["why"]:
                self.lock.wait(timeout=1.0)
            out = []
            while self.segments and sum(len(s["reward"]) for s in out) < steps:
                s = self.segments.popleft()
                self.steps -= len(s["reward"])
                out.append(s)
            return out

    def drain_rounds(self) -> list[dict]:
        with self.lock:
            r, self.rounds = self.rounds, []
            return r


def send_blob(sock: socket.socket, blob: bytes) -> None:
    sock.sendall(struct.pack("!Q", len(blob)) + blob)


def serve(conn: socket.socket, peer, hub: Hub) -> None:
    name = f"{peer[0]}:{peer[1]}"
    try:
        hello = recv_msg(conn)
        if hello.get("protocol") != PROTOCOL:
            send_msg(conn, {"type": "error", "why": f"protocol {hello.get('protocol')} != {PROTOCOL}"})
            return
        name = hello.get("actor", name)
        with hub.lock:
            blob, done = hub.weights_blob, hub.done
            meta = dict(hub.meta)
        send_msg(conn, {"type": "welcome", "meta": meta, "done": done})
        send_blob(conn, blob)
        while True:
            msg = recv_msg(conn)
            hub.actors[name] = time.time()
            if msg["type"] == "segment":
                hub.add(msg["segment"])
            if msg["type"] in ("segment", "ping"):
                send_msg(conn, {"type": "ack", "version": hub.version, "done": hub.done})
            elif msg["type"] == "weights":
                with hub.lock:
                    blob = hub.weights_blob
                send_blob(conn, blob)
    except (ConnectionError, OSError, EOFError, pickle.UnpicklingError):
        pass
    finally:
        hub.actors.pop(name, None)
        conn.close()


def listen(hub: Hub, run_dir: Path) -> str:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 0))
    srv.listen(256)
    nonce = os.urandom(8).hex()
    hub.meta["nonce"] = nonce            # actors check it against the address file they read
    addr = f"{socket.gethostname()} {srv.getsockname()[1]} {nonce}"
    tmp = run_dir / "learner.addr.tmp"
    tmp.write_text(addr + "\n")
    os.replace(tmp, run_dir / "learner.addr")

    def accept_loop():
        while True:
            conn, peer = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=serve, args=(conn, peer, hub), daemon=True).start()
    threading.Thread(target=accept_loop, daemon=True).start()
    return addr


def batch_from(segs: list[dict], critic: StateCritic, cfg: PPOConfig, dev: str) -> dict:
    """Segments [B][T] -> flat PPO samples with GAE(lambda) and a true bootstrap."""
    obs = torch.from_numpy(np.stack([s["obs"] for s in segs])).to(dev)          # [B,T,H,dim]
    obs_last = torch.from_numpy(np.stack([s["obs_last"] for s in segs])).to(dev)
    act = torch.from_numpy(np.stack([s["act"] for s in segs])).to(dev).float()
    logp = torch.from_numpy(np.stack([s["logp"] for s in segs])).to(dev)
    reward = torch.from_numpy(np.stack([s["reward"] for s in segs])).to(dev)
    terminal = torch.from_numpy(np.stack([s["terminal"] for s in segs])).to(dev)
    B, T = reward.shape
    with torch.no_grad():
        v = critic.value(obs)
        vb = critic.value(obs_last)
        ret = lambda_returns(reward, torch.cat([v, vb[:, None]], 1), torch.ones_like(reward),
                             terminal, cfg.gamma, cfg.lam)
    adv = (ret - v).reshape(-1)
    adv = (adv - adv.mean()) / adv.std().clamp(min=1e-6)
    return {"obs": obs.reshape(B * T, *obs.shape[2:]), "act": act.reshape(B * T, *act.shape[2:]),
            "side": torch.zeros(B * T, dtype=torch.long, device=dev),
            "ret": ret.reshape(-1), "adv": adv, "value": v.reshape(-1),
            "logp_old": logp.reshape(-1)}


def summarize(rounds: list[dict]) -> dict:
    if not rounds:
        return {}
    won = sum(r["won"] for r in rounds)
    matches = [r for r in rounds if r["match_over"]]
    mw = sum(r["match_won"] for r in matches)
    return {"rounds": len(rounds), "round_win": won / len(rounds), "matches": len(matches),
            "match_win": mw / len(matches) if matches else None,
            "match_win_hp20": (sum(r["match_won"] and r["p1_hp"] >= 0.2 for r in matches)
                               / max(1, mw)) if mw else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--init", type=Path, help="policy.pt-format start (needed on a fresh run)")
    ap.add_argument("--button-rates", type=Path, help="JSON list of the corpus's 10 press rates")
    ap.add_argument("--budget", type=float, default=82e6, help="decisions to train on")
    ap.add_argument("--batch", type=int, default=16384, help="decisions per update")
    ap.add_argument("--segment", type=int, default=64, help="decisions per actor segment")
    ap.add_argument("--max-stale", type=int, default=4, help="drop segments this many versions old")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--kl-ref-coef", type=float, default=0.05)
    ap.add_argument("--entropy-floor-frac", type=float, default=0.8)
    ap.add_argument("--button-tolerance", type=float, default=2.0)
    ap.add_argument("--jitter-sigma", type=float, default=1.0,
                    help="frames of timing noise on executed actions, as in the simulator arena")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--p2-char", type=int, default=-1, help="-1: a random COM character per match")
    ap.add_argument("--ckpt-every", type=int, default=20, help="updates")
    ap.add_argument("--snapshot-minutes", type=float, default=30.0)
    ap.add_argument("--stop-after-hours", type=float, default=0.0)
    ap.add_argument("--done-linger", type=float, default=30.0,
                    help="seconds to keep serving after the budget, so every actor hears `done`")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = a.device
    a.run_dir.mkdir(parents=True, exist_ok=True)
    stop = {"why": None}
    for s in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(s, lambda signum, _f: stop.update(why=signal.Signals(signum).name))

    # ---- model, from the checkpoint if there is one ------------------------------------------
    latest = a.run_dir / "latest.pt"
    ck = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    init_path = Path(ck["init"]) if ck else a.init
    if init_path is None:
        ap.error("--init is needed on a fresh run")
    policy, obs, H, ticks, slots, init_meta = load_agent(init_path, "cpu")
    reference = SokuPolicy(obs.dim, H, ticks)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad_(False)
    critic = StateCritic(obs.dim, H)
    rates = json.loads(Path(ck["button_rates_path"] if ck else a.button_rates).read_text())
    reward_cfg = StateRewardConfig()
    cfg = PPOConfig(epochs=a.epochs, minibatches=a.minibatches, lr=a.lr, critic_lr=a.critic_lr,
                    gamma=a.gamma, lam=a.lam, clip_eps=a.clip_eps, target_kl=a.target_kl,
                    kl_ref_coef=a.kl_ref_coef, entropy_floor_frac=a.entropy_floor_frac,
                    jitter_sigma=a.jitter_sigma, reward=reward_cfg)
    policy.to(dev).train()
    reference.to(dev).eval()
    critic.to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    total_updates = max(1, int(a.budget // a.batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_updates * cfg.epochs * cfg.minibatches, eta_min=cfg.lr * 0.05)
    ent_floor = EntropyFloor(cfg, dev) if cfg.entropy_floor_frac > 0 else None
    btn_floor = (ButtonRateFloor(np.asarray(rates, np.float32), dev, tolerance=a.button_tolerance)
                 if a.button_tolerance > 0 else None)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    version, decisions, dropped = 0, 0, 0
    all_rounds: list[dict] = []
    if ck:
        policy.load_state_dict(ck["policy"])
        reference.load_state_dict(ck["reference"])
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        critic_opt.load_state_dict(ck["critic_opt"])
        sched.load_state_dict(ck["sched"])
        if ent_floor is not None and ck.get("ent_floor"):
            ent_floor.log_alpha.data.copy_(ck["ent_floor"]["log_alpha"])
            ent_floor.opt.load_state_dict(ck["ent_floor"]["opt"])
            ent_floor.floor = ck["ent_floor"]["floor"]
        if btn_floor is not None and ck.get("btn_floor"):
            btn_floor.log_alpha.data.copy_(ck["btn_floor"]["log_alpha"])
            btn_floor.opt.load_state_dict(ck["btn_floor"]["opt"])
        rng.bit_generator.state = ck["rng"]["numpy"]
        version, decisions, dropped = ck["version"], ck["decisions"], ck["dropped"]
        all_rounds = ck["rounds"]
        print(f"resumed from {latest} at version {version}, {decisions:,} decisions", flush=True)
    if (a.run_dir / "DONE").exists():
        print("DONE exists: this run is finished", flush=True)
        return 0

    meta = {"history": H, "ticks": ticks, "slots": slots, "obs": obs.state_dict(),
            "obs_dim": obs.dim, "segment": a.segment, "jitter_sigma": cfg.jitter_sigma,
            "reward": reward_cfg, "level": a.level, "p2_char": a.p2_char}
    hub = Hub(meta)
    hub.publish(version, policy)
    addr = listen(hub, a.run_dir)
    cfg_rec = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    (a.run_dir / "config.json").write_text(json.dumps(
        {"args": cfg_rec, "init": str(init_path), "address": addr}, indent=1))
    print(f"learner at {addr} | init {init_path} | budget {a.budget:,.0f} decisions in batches of "
          f"{a.batch} | device {dev}", flush=True)

    log = open(a.run_dir / "log.jsonl", "a")
    t_start = time.time()
    last_snap = time.time()
    last_t, last_dec = time.time(), decisions

    def save(tag: str = "latest.pt") -> None:
        atomic_save({"policy": policy.state_dict(), "reference": reference.state_dict(),
                     "critic": critic.state_dict(), "opt": opt.state_dict(),
                     "critic_opt": critic_opt.state_dict(), "sched": sched.state_dict(),
                     "ent_floor": None if ent_floor is None else
                     {"log_alpha": ent_floor.log_alpha.detach().clone(),
                      "opt": ent_floor.opt.state_dict(), "floor": ent_floor.floor},
                     "btn_floor": None if btn_floor is None else
                     {"log_alpha": btn_floor.log_alpha.detach().clone(),
                      "opt": btn_floor.opt.state_dict()},
                     "rng": rng_state(rng), "version": version, "decisions": decisions,
                     "dropped": dropped, "rounds": all_rounds, "init": str(init_path),
                     "button_rates_path": str(a.button_rates or ck["button_rates_path"])},
                    a.run_dir / tag)

    def snapshot() -> None:
        atomic_save({"policy": {k: v.cpu() for k, v in policy.state_dict().items()},
                     "obs": obs.state_dict(), "history": H, "ticks": ticks, "slots": slots,
                     "version": version, "decisions": decisions, "init": str(init_path)},
                    a.run_dir / f"policy_v{version:06d}.pt")

    rc = 0
    while decisions < a.budget:
        if a.stop_after_hours and time.time() - t_start > a.stop_after_hours * 3600:
            stop["why"] = "stop-after-hours"
        segs = hub.take(a.batch, stop)
        if stop["why"]:
            for s in segs:
                hub.add(s)
            print(f"stopping ({stop['why']}) at version {version}: checkpoint, exit 99", flush=True)
            save()
            rc = 99
            break
        fresh = [s for s in segs if int(s["version"].min()) >= version - a.max_stale]
        lag = [version - int(s["version"].min()) for s in segs]
        dropped += len(segs) - len(fresh)
        if not fresh:
            continue
        t0 = time.time()
        batch = batch_from(fresh, critic, cfg, dev)
        stats = ppo_update(policy, critic, opt, critic_opt, batch, cfg, reference, rng, sched,
                           ent_alpha=float(ent_floor.alpha) if ent_floor else 0.0,
                           button_floor=btn_floor)
        if ent_floor is not None and "entropy" in stats:
            if ent_floor.floor is None:
                ent_floor.set_floor_from(stats["entropy"])
            stats.update({f"ent_{k}": v for k, v in ent_floor.update(stats["entropy"]).items()})
        if btn_floor is not None and stats.get("samples"):
            with torch.no_grad():
                _, _, rates_now = policy.log_prob_of(batch["obs"], batch["side"], batch["act"],
                                                     return_rates=True)
            stats.update(btn_floor.update(rates_now))
        version += 1
        hub.publish(version, policy)
        n = int(sum(len(s["reward"]) for s in fresh))
        decisions += n
        new_rounds = hub.drain_rounds()
        all_rounds.extend(new_rounds)
        recent = summarize(all_rounds[-2000:])
        now = time.time()
        rec = {"version": version, "decisions": decisions, "samples": n,
               "decisions_per_s": (decisions - last_dec) / max(now - last_t, 1e-9),
               "update_s": now - t0, "dropped_segments": dropped,
               "lag_mean": float(np.mean(lag)), "lag_max": int(max(lag)),
               "actors": len(hub.actors), "reward_mean": float(batch["ret"].mean()),
               "recent": recent, "elapsed_h": (now - t_start) / 3600,
               **{k: v for k, v in stats.items() if isinstance(v, (int, float))}}
        last_t, last_dec = now, decisions
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if version % 5 == 0 or version == 1:
            r = recent or {}
            print(f"v{version:5d} | {decisions:>11,} decisions | {rec['decisions_per_s']:6.0f}/s | "
                  f"{len(hub.actors):3d} actors | lag {rec['lag_mean']:.1f} (max {rec['lag_max']}) "
                  f"| kl {stats.get('kl', 0):.4f} clip {stats.get('clip_frac', 0):.2f} "
                  f"ent {stats.get('entropy', 0):.2f} v_r2 {stats.get('v_r2', 0):+.2f} | "
                  f"recent rounds won {r.get('round_win', 0) or 0:.3f} of {r.get('rounds', 0)}, "
                  f"matches {r.get('match_win') if r.get('match_win') is not None else '-'}",
                  flush=True)
        if version % a.ckpt_every == 0:
            save()
        if now - last_snap > a.snapshot_minutes * 60:
            snapshot()
            last_snap = now
    else:
        save()
        snapshot()
        (a.run_dir / "DONE").write_text(f"{decisions} decisions, version {version}\n")
        with hub.lock:
            hub.done = True
        print(f"budget spent: {decisions:,} decisions, version {version}; DONE", flush=True)
        time.sleep(a.done_linger)        # let actors' next acknowledgements carry `done`
    log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
