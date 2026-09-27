"""Headless PPO self-play inside the frozen state simulator. Resumable; built for a Slurm GPU node.

    python -m scripts.train_state_ppo --sim sim.pt --bank bank.npz --out runs/ppo --steps 20000
    python -m scripts.train_state_ppo --sim sim.pt --corpus ~/corpus --out runs/ppo   # builds the bank

No display, no game, no pixels: the policy plays both chairs of imagined fights in state space, the
simulator is frozen, and only the policy and its critic learn. See `sokubot/rl/ppo.py` for what is
trained against what, and why.

MADE TO BE INTERRUPTED
----------------------
A Slurm job ends at its time limit, and a requeued one must pick up where it stopped, not restart:
    * `latest.pt` is written atomically every `--ckpt-every` steps and holds EVERYTHING -- policy,
      critic, both optimisers, the LR schedule, the league of past selves, the entropy-floor state
      and every RNG -- so a resumed run continues the same trajectory of decisions.
    * SIGUSR1 or SIGTERM (Slurm sends one before the time limit with `--signal=USR1@300`) makes the
      loop finish the current step, write `latest.pt` and exit with code 99, which the job script
      treats as "requeue me". `--stop-after-hours` does the same on a clock, in case no signal comes.
    * On start, an existing `latest.pt` in `--out` is resumed unless `--fresh`. It is REFUSED if the
      simulator or the bank has a different fingerprint than the run it came from: two simulators
      saved under one name are different models, and continuing a policy against a different one
      produces numbers that belong to neither.

OUTPUT
------
    config.json      every setting plus the simulator and bank fingerprints
    log.jsonl        one line per logged step; on resume, lines past the checkpoint are dropped
    latest.pt        the full resumable state
    policy_best.pt   the best policy by `net` vs the frozen reference, in the same format as
                     train_state_grpo writes (eval_state_policy / head_to_head read it)
    policy.pt        the final policy
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.data import state_bank
from sokubot.data.soku import BUTTONS
from sokubot.data.state import CH
from sokubot.model.state_dynamics import load_sim
from sokubot.rl.grpo import ButtonRateFloor, EntropyFloor, ReplayOpponent
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.ppo import (OPPONENTS, League, PPOConfig, assemble, bank_fingerprint,
                            evaluate_vs, file_fingerprint, init_policy_from_corpus, ppo_update)
from sokubot.rl.state_arena import StateArena, StateObs, StatePolicyOpponent, corpus_stats
from sokubot.rl.state_critic import StateCritic
from sokubot.rl.state_reward import StateRewardConfig
from scripts.train_state_grpo import valid_starts

REQUEUE = 99            # exit code: "stopped cleanly to be resumed"


def load_bank(a, ticks: int, slots: int):
    if a.bank is not None:
        d = np.load(a.bank, allow_pickle=True)
        S, P, A, E = d["S"], d["P"], d["A"], d["E"]
        if "V" in d.files:
            V = d["V"]
        else:
            # train_state_dynamics' corpus cache (S, P, A, E, M): the same arrays at the same stride,
            # so the 8.6 h parse of the full Amarel corpus is not done twice. It keeps every row and
            # has no validity mask, which is right for FRESH captures -- only a sidecar aligned to an
            # old video is ever padded -- and wrong for an aligned corpus, so say so.
            V = np.ones(len(S), bool)
            print(f"bank {a.bank}: no validity mask (a simulator corpus cache); every row taken as "
                  f"valid, which holds for fresh captures only", flush=True)
        names = [str(x) for x in d["names"]] if "names" in d.files else []
        A = as_buttons(A)
    else:
        S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache, a.replays)
    # A bank is only valid against the simulator it was built for. Checked here so a mismatch is a
    # one-line refusal and not a shape error deep inside the first rollout.
    if P.shape[2] != slots or A.shape[1] != ticks:
        raise SystemExit(f"bank has {P.shape[2]} projectile slots and {A.shape[1]} ticks per step; "
                         f"the simulator wants {slots} and {ticks}. Rebuild the bank for it.")
    return S, P, A, E, V, names


def as_buttons(A: np.ndarray) -> np.ndarray:
    """The button chunks as uint8, the bank's own dtype. The simulator's cache keeps them float32:
    10.4 GB for the full corpus instead of 2.6. Checked chunk by chunk, so a value that is not 0 or
    1 stops the run instead of being truncated, and no full-size temporary is made."""
    if A.dtype == np.uint8:
        return A
    out = np.empty(A.shape, np.uint8)
    for i in range(0, len(A), 1 << 20):
        a = A[i:i + (1 << 20)]
        out[i:i + len(a)] = a
        if not np.array_equal(out[i:i + len(a)], a):
            raise SystemExit(f"bank buttons are not 0/1 near row {i}: not a button chunk array")
    return out


# Summary statistics (normalisation, the button prior, button rates) come from at most this many
# evenly strided rows. The full Amarel bank is 25.9M decision steps: a std over its projectile block
# allocates a 12 GB temporary, and means over 4M rows are already exact to ~1e-3. Smaller banks are
# used whole, so every earlier run and test is unchanged.
STAT_ROWS = 4_000_000


def stat_view(x: np.ndarray) -> np.ndarray:
    return x[::max(1, -(-len(x) // STAT_ROWS))]


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state(rng: np.random.Generator) -> dict:
    out = {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng_state(rng: np.random.Generator, st: dict) -> None:
    rng.bit_generator.state = st["numpy"]
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, required=True, help="from scripts.train_state_dynamics")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bank", type=Path, default=None,
                     help="a prebuilt bank .npz (S, P, A, E, V, names) -- the state_bank cache format")
    src.add_argument("--corpus", type=Path, nargs="+", default=None,
                     help="sidecar roots; the bank is built (and cached at --cache) from these")
    ap.add_argument("--cache", type=Path, default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--replays", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing latest.pt in --out")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=8,
                    help="imagined decision steps per rollout (8 = 667 ms: where the simulator "
                         "is still clearly better than a no-op; the critic carries the rest)")
    ap.add_argument("--starts", type=int, default=512, help="distinct start states per update")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--kl-ref-coef", type=float, default=0.05,
                    help="anchor to the prior-initialised policy: keeps play near behaviour the "
                         "simulator can be trusted to simulate")
    ap.add_argument("--entropy-floor-frac", type=float, default=0.8)
    ap.add_argument("--button-tolerance", type=float, default=2.0,
                    help="cap each button's press rate at this multiple of the corpus rate (0 = "
                         "off). On by default here: GRPO runs pressed `spell` 10.8x the human rate, "
                         "where the simulator has seen least and is freest to be wrong")
    ap.add_argument("--opponents", default="self=0.5,league=0.35,reference=0.05,replay=0.1",
                    help="per-update opponent mixture, name=weight over " + ",".join(OPPONENTS))
    ap.add_argument("--no-two-sided", action="store_true",
                    help="do not train on the opponent's chair in self vs self updates")
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--max-snapshots", type=int, default=16)
    ap.add_argument("--damage-dealt", type=float, default=1.0)
    ap.add_argument("--win-magnitude", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-starts", type=int, default=512)
    ap.add_argument("--eval-league", type=int, default=4, help="league entries scored per eval")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--stop-after-hours", type=float, default=0.0,
                    help="checkpoint and exit 99 after this long (0 = never)")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help="behave as if signalled after this step: checkpoint and exit 99 (0 = off). "
                         "Runs a long job in fixed chunks, and is how the resume is tested")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.bank is None and a.corpus is None:
        ap.error("give --bank or --corpus")
    a.out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    t_start = time.time()
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    stop = {"why": None}

    def on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name
    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        return _run(a, ap, dev, t_start, stop)
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def _run(a, ap, dev: str, t_start: float, stop: dict) -> int:

    # ---- the frozen world ------------------------------------------------------------------
    sim, meta = load_sim(a.sim, dev)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    S, P, A, E, V, names = load_bank(a, ticks, slots)
    fp = {"sim": file_fingerprint(a.sim), "bank": bank_fingerprint(S, P, A, E, V)}
    s_mu, s_sd, p_mu, p_sd = corpus_stats(stat_view(S), stat_view(P))
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(dev)
    print(f"simulator {a.sim} [{fp['sim']}] step {meta.get('step')} | history {H} slots {slots} "
          f"ticks {ticks} | bank [{fp['bank']}] {len(S)} decision steps, {len(names) or int(E.max()) + 1} replays | "
          f"obs {obs.dim} | device {dev}", flush=True)

    weights = dict(kv.split("=") for kv in a.opponents.split(","))
    unknown = set(weights) - set(OPPONENTS)
    if unknown:
        ap.error(f"unknown opponents {sorted(unknown)}; have {OPPONENTS}")
    cfg = PPOConfig(horizon=a.horizon, starts_per_batch=a.starts, epochs=a.epochs,
                    minibatches=a.minibatches, lr=a.lr, critic_lr=a.critic_lr, gamma=a.gamma,
                    lam=a.lam, clip_eps=a.clip_eps, target_kl=a.target_kl,
                    kl_ref_coef=a.kl_ref_coef, entropy_floor_frac=a.entropy_floor_frac,
                    two_sided=not a.no_two_sided, snapshot_every=a.snapshot_every,
                    max_snapshots=a.max_snapshots,
                    **{f"p_{k}": float(weights.get(k, 0.0)) for k in OPPONENTS},
                    reward=StateRewardConfig(damage_dealt=a.damage_dealt, win=a.win_magnitude,
                                             lose=-a.win_magnitude))
    probs = cfg.opponent_probs()
    arena = StateArena(sim, obs, cfg, H, ticks)

    starts = valid_starts(E, V, H, a.horizon)
    if len(starts) == 0:
        raise SystemExit("the bank has no start with a full history and horizon inside one replay")

    # ---- the learners ----------------------------------------------------------------------
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    policy = SokuPolicy(obs.dim, H, ticks).to(dev)
    init_policy_from_corpus(policy, stat_view(A), BUTTONS[4:10])
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    critic = StateCritic(obs.dim, H).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.steps * cfg.epochs * cfg.minibatches), eta_min=cfg.lr * 0.05)
    league = League(cfg, policy)
    ent_floor = EntropyFloor(cfg, dev) if cfg.entropy_floor_frac > 0 else None
    btn_floor = None
    if a.button_tolerance > 0:
        cr = stat_view(A).reshape(-1, 20).astype(np.float32)
        btn_floor = ButtonRateFloor((cr[:, :10].mean(0) + cr[:, 10:].mean(0)) / 2, dev,
                                    tolerance=a.button_tolerance)

    # The bank stays in HOST memory; each update gathers its windows there and ships only those.
    # Moving the whole bank to the GPU was fine for a few hundred replays and is ~20 GB for the full
    # corpus -- more than most of Amarel's cards. A batch is 512 starts x `history` rows: tiny.
    St, Pt, At = (torch.from_numpy(x) for x in (S, P, A))
    off = torch.arange(H) - (H - 1)

    def context(idx: torch.Tensor):
        """idx: CPU long tensor of window-end rows -> the three context tensors, on the device."""
        w = idx[:, None] + off[None, :]
        return (St[w].to(dev, non_blocking=True), Pt[w].to(dev, non_blocking=True),
                At[w[:, :-1]].to(dev).float())

    # The yardstick: FIXED starts, drawn from their own seed, identical across runs and resumes.
    eval_idx = torch.from_numpy(np.random.default_rng(12345).choice(
        starts, size=min(a.eval_starts, len(starts)), replace=False))
    eval_ctx = context(eval_idx)

    # ---- resume ------------------------------------------------------------------------------
    step0, best = 0, {"net": float("-inf"), "step": -1}
    latest = a.out / "latest.pt"
    if latest.exists() and not a.fresh:
        # To CPU, always. Every module and optimiser copies its state onto its own device when
        # loaded, and the RNG states MUST stay CPU ByteTensors: map_location="cuda" moved them to the
        # GPU and torch.set_rng_state refused them, which the CPU-only tests could not see and the
        # first GPU smoke run did. It also keeps the league's snapshots on the CPU, as designed.
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["fingerprints"] != fp:
            raise SystemExit(f"refusing to resume {latest}: it was trained against "
                             f"{ck['fingerprints']} and this run has {fp}. Use --fresh or a new "
                             f"--out.")
        policy.load_state_dict(ck["policy"])
        reference.load_state_dict(ck["reference"])
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        critic_opt.load_state_dict(ck["critic_opt"])
        sched.load_state_dict(ck["sched"])
        league.load_state_dict(ck["league"])
        if ent_floor is not None and ck.get("ent_floor"):
            ent_floor.log_alpha.data.copy_(ck["ent_floor"]["log_alpha"])
            ent_floor.opt.load_state_dict(ck["ent_floor"]["opt"])
            ent_floor.floor = ck["ent_floor"]["floor"]
        if btn_floor is not None and ck.get("btn_floor") is not None:
            btn_floor.log_alpha.data.copy_(ck["btn_floor"]["log_alpha"])
            btn_floor.opt.load_state_dict(ck["btn_floor"]["opt"])
        set_rng_state(rng, ck["rng"])
        step0, best = int(ck["step"]), ck["best"]
        log_path = a.out / "log.jsonl"
        if log_path.exists():
            kept = [l for l in log_path.read_text().splitlines()
                    if l.strip() and json.loads(l).get("step", 0) <= step0]
            log_path.write_text("".join(l + "\n" for l in kept))
        print(f"resumed from {latest} at step {step0} (best net {best['net']:+.5f} at "
              f"{best['step']}, league {len(league)})", flush=True)
    else:
        (a.out / "log.jsonl").write_text("")

    (a.out / "config.json").write_text(json.dumps(
        {"argv": sys.argv, "args": {k: str(v) for k, v in vars(a).items()},
         "ppo": {k: (v if not dataclasses.is_dataclass(v) else dataclasses.asdict(v))
                 for k, v in dataclasses.asdict(cfg).items()},
         "fingerprints": fp, "history": H, "slots": slots, "ticks": ticks, "obs_dim": obs.dim,
         "bank_steps": int(len(S)), "bank_replays": len(names), "valid_starts": int(len(starts)),
         "sim_meta": {k: str(v) for k, v in meta.items() if k not in ("cfg", "move_vocab")}},
        indent=1))

    def checkpoint(step: int) -> None:
        atomic_save({"step": step, "policy": policy.state_dict(),
                     "reference": reference.state_dict(), "critic": critic.state_dict(),
                     "opt": opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                     "sched": sched.state_dict(), "league": league.state_dict(),
                     "ent_floor": (None if ent_floor is None else
                                   {"log_alpha": ent_floor.log_alpha.detach().clone(),
                                    "opt": ent_floor.opt.state_dict(),
                                    "floor": ent_floor.floor}),
                     "btn_floor": (None if btn_floor is None else
                                   {"log_alpha": btn_floor.log_alpha.detach().clone(),
                                    "opt": btn_floor.opt.state_dict()}),
                     "rng": rng_state(rng), "best": best, "fingerprints": fp,
                     "obs": obs.state_dict(), "obs_dim": obs.dim, "history": H, "ticks": ticks,
                     "slots": slots, "sim": str(a.sim)}, latest)

    def policy_record(step: int, net: float | None) -> dict:
        return {"policy": policy.state_dict(), "obs": obs.state_dict(), "obs_dim": obs.dim,
                "history": H, "ticks": ticks, "slots": slots, "sim": str(a.sim),
                "sim_fingerprint": fp["sim"], "step": step, "net": net,
                "reference": reference.state_dict()}

    # ---- the loop ----------------------------------------------------------------------------
    log = open(a.out / "log.jsonl", "a", buffering=1)
    t0 = time.time()
    step = step0
    rc = 0
    for step in range(step0 + 1, a.steps + 1):
        kind = OPPONENTS[int(rng.choice(len(OPPONENTS), p=probs))]
        li = None
        if kind == "league":
            li = league.sample(rng)
            if li is None:
                kind = "self"               # nothing banked yet: play the current self
        idx = torch.from_numpy(rng.choice(starts, size=cfg.starts_per_batch))
        side = torch.from_numpy(rng.integers(0, 2, cfg.starts_per_batch)).to(dev).long()
        ctx = context(idx)
        if kind == "replay":
            fut = At[idx[:, None] + torch.arange(cfg.horizon)[None, :]].to(dev)
            opponent = ReplayOpponent(fut.float())
        elif kind == "reference":
            opponent = StatePolicyOpponent(reference)
        elif kind == "league":
            opponent = StatePolicyOpponent(league.policy(li))
        else:
            opponent = StatePolicyOpponent(policy)
        two = cfg.two_sided and kind == "self"

        traj = arena.rollout(*ctx, side, policy, opponent, two_sided=two)
        batch = assemble(traj, critic, cfg, both_chairs=two)
        stats = ppo_update(policy, critic, opt, critic_opt, batch, cfg, reference, rng, sched,
                           ent_alpha=float(ent_floor.alpha) if ent_floor else 0.0,
                           button_floor=btn_floor)
        if ent_floor is not None and "entropy" in stats:
            if ent_floor.floor is None:
                ent_floor.set_floor_from(stats["entropy"])
            stats.update({f"ent_{k}": v for k, v in ent_floor.update(stats["entropy"]).items()})
        if btn_floor is not None and stats.get("samples"):
            with torch.no_grad():
                _, _, rates = policy.log_prob_of(batch["obs"], batch["side"], batch["act"],
                                                 return_rates=True)
            stats.update(btn_floor.update(rates))
        alive = traj["alive"]
        n = alive.sum().clamp(min=1)
        train_net = float(((traj["terms"]["dealt"] + traj["terms"]["taken"]) * alive).sum() / n)
        if li is not None:
            league.record(li, train_net)
        league.maybe_add(policy, step)

        if step % a.log_every == 0 or step == step0 + 1:
            m = traj["mine"]
            rec = {"step": step, "opponent": kind, "two_sided": two, "train_net": train_net,
                   "ret": float(traj["reward"].sum(1).mean()), "alive_frac": float(alive.mean()),
                   "press_rate": float(m.mean()), "attack_rate": float(m[..., 4:8].mean()),
                   "lr": float(sched.get_last_lr()[0]), "league": len(league),
                   "guard_rate": float(traj["states"][..., CH["guarding"]].gather(
                       -1, side.view(-1, 1, 1).expand(-1, traj["states"].shape[1], 1)).mean()),
                   "steps_per_s": (step - step0) / max(time.time() - t0, 1e-9),
                   "elapsed_h": (time.time() - t_start) / 3600, **stats}
            if step % a.eval_every == 0:
                ev = evaluate_vs(arena, policy, reference, eval_ctx)
                rec.update({f"eval_{k}": v for k, v in ev.items()})
                # Self vs self should be ~0: anything else is the simulator favouring a seat.
                rec["eval_seat_bias"] = evaluate_vs(arena, policy, policy, eval_ctx)["net"]
                for j in range(max(0, len(league) - a.eval_league), len(league)):
                    net_j = evaluate_vs(arena, policy, league.policy(j), eval_ctx)["net"]
                    league.record(j, net_j)
                    rec[f"eval_vs_step{league.entries[j]['step']}"] = net_j
                if ev["net"] > best["net"]:
                    best = {"net": ev["net"], "step": step}
                    atomic_save(policy_record(step, ev["net"]), a.out / "policy_best.pt")
                print(f"  [eval] step {step} | net vs reference {ev['net']:+.5f} | seat bias "
                      f"{rec['eval_seat_bias']:+.5f} | best {best['net']:+.5f} @ {best['step']}",
                      flush=True)
            log.write(json.dumps(rec) + "\n")
            print(f"step {step:6d} vs {kind:<9} | net {train_net:+.5f} | kl {stats.get('kl', 0):.4f} "
                  f"clip {stats.get('clip_frac', 0):.2f} ent {stats.get('entropy', 0):.2f} | "
                  f"v_r2 {stats.get('v_r2', 0):+.2f} | press {rec['press_rate']:.3f} | "
                  f"{rec['steps_per_s']:.2f} it/s", flush=True)

        if step % a.ckpt_every == 0:
            checkpoint(step)
        over_time = a.stop_after_hours and (time.time() - t_start) / 3600 >= a.stop_after_hours
        # EXACTLY that step: a requeued job re-runs with the same arguments, and `>=` would stop it
        # again one step after every resume, forever.
        at_step = a.stop_at_step and step == a.stop_at_step
        if stop["why"] or over_time or at_step:
            checkpoint(step)
            why = stop["why"] or (f"--stop-at-step {a.stop_at_step}" if at_step
                                  else f"--stop-after-hours {a.stop_after_hours}")
            print(f"stopping at step {step} ({why}); latest.pt written, exit {REQUEUE} to requeue",
                  flush=True)
            rc = REQUEUE
            break
    else:
        checkpoint(step)
        atomic_save(policy_record(step, None), a.out / "policy.pt")
        print(f"done: {step} steps | best net {best['net']:+.5f} at {best['step']} -> {a.out}",
              flush=True)
    log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
