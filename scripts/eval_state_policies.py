"""Re-measure trained PPO policies on common ground, with confidence intervals.

The trainer's `[eval]` line is ONE sampled draw on 512 starts against the run's OWN reference (its
own random initialisation), and its `best` is the maximum of ~200 such draws. That is fine for
choosing a checkpoint and wrong for a claim: the maximum of noisy draws is biased upward, and two
runs' numbers come from different references -- and, across simulators, different worlds.

Here every run is scored in the SAME simulator(s), on the same fresh starts, over several sampling
draws and both chairs, against:

  own reference   the run's frozen initial policy (what the trainer reports, measured properly)
  replay          the human's recorded buttons from each start's own replay: one opponent shared
                  by every run, so `final - prior` against it is comparable across runs
  each other      head to head between the final policies

`net` is exactly `ppo.evaluate_vs`'s: damage dealt plus damage taken per living step (EVAL_RCFG:
no crush, combo, win or idle terms), averaged over the two chairs. 95% intervals come from a
bootstrap over START STATES (all draws and chairs of a start move together). Two checks run on the
instrument itself: a policy against itself must read 0, and A-vs-B must be minus B-vs-A.

    python -m scripts.eval_state_policies --sim sim_hold.pt [--sim other.pt] \\
        --bank sim_corpus_full.npz --runs ppo-hold-s0 ppo-hold-s1 ppo-full-s0 ppo-full-s1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_state_grpo import valid_starts
from scripts.train_state_ppo import as_buttons, stat_view
from sokubot.data.state import STATE_CHANNELS
from sokubot.model.state_dynamics import load_sim
from sokubot.rl.grpo import ReplayOpponent
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.ppo import EVAL_RCFG, PPOConfig
from sokubot.rl.state_arena import StateArena, StateObs, StatePolicyOpponent, corpus_stats
from sokubot.rl.state_reward import compute_rewards


def per_start(arena, policy, opponent, ctx, fut, draws: int, seed: int):
    """-> num, den, dealt: [2 chairs, B] sums over draws of (dealt+taken)*alive, alive, dealt*alive."""
    B = ctx[0].shape[0]
    num, den, dealt = (np.zeros((2, B)) for _ in range(3))
    for d in range(draws):
        for chair in (0, 1):
            # Same seed for the same (draw, chair) in every matchup: common random numbers.
            torch.manual_seed(seed * 1000 + d * 2 + chair)
            side = torch.full((B,), chair, device=ctx[0].device, dtype=torch.long)
            opp = ReplayOpponent(fut) if opponent == "replay" else StatePolicyOpponent(opponent)
            tr = arena.rollout(*ctx, side, policy, opp)
            _, al, terms = compute_rewards(tr["states"], tr["joint"], side, EVAL_RCFG)
            al = al.float()
            num[chair] += ((terms["dealt"] + terms["taken"]) * al).sum(1).cpu().numpy()
            den[chair] += al.sum(1).cpu().numpy()
            dealt[chair] += (terms["dealt"] * al).sum(1).cpu().numpy()
    return num, den, dealt


def net_of(num, den, idx=None):
    """evaluate_vs's definition: a pooled ratio per chair, then the mean of the two chairs."""
    if idx is not None:
        num, den = num[:, idx], den[:, idx]
    return float(np.mean(num.sum(1) / np.maximum(den.sum(1), 1.0)))


def boot(stats, B: int, n: int, seed: int = 0) -> np.ndarray:
    """Bootstrap over starts. `stats(idx) -> float`; returns the n resampled values."""
    rng = np.random.default_rng(seed)
    return np.array([stats(rng.integers(0, B, B)) for _ in range(n)])


def ci(vals: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_run(run: Path, obs_dim: int, H: int, ticks: int, dev: str):
    ck = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    pols = {}
    for key in ("policy", "reference"):
        p = SokuPolicy(obs_dim, H, ticks).to(dev).eval()
        p.load_state_dict(ck[key])
        pols[key] = p
    cfg = json.loads((run / "config.json").read_text())
    return pols, int(ck["step"]), cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, action="append", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--starts", type=int, default=4096)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=777,
                    help="picks the starts; NOT the trainer's eval seed (12345), so these are fresh")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="write every number as JSON here")
    a = ap.parse_args(argv)
    dev = a.device

    d = np.load(a.bank)
    S, P, A, E = d["S"], d["P"], as_buttons(d["A"]), d["E"]
    V = d["V"] if "V" in d.files else np.ones(len(S), bool)
    slots = P.shape[2]
    # Exactly the trainer's observation normalisation (train_state_ppo._run).
    obs = StateObs(*corpus_stats(stat_view(S), stat_view(P)), slots).to(dev)
    results = {"starts": a.starts, "draws": a.draws, "horizon": a.horizon, "worlds": {}}

    for sim_path in a.sim:
        sim, meta = load_sim(sim_path, dev)
        H, ticks = int(meta["history"]), int(meta["ticks"])
        hold = [STATE_CHANNELS[i] for i in sim.hold]
        print(f"\n######## world {sim_path} | hold {hold} | history {H} ticks {ticks}", flush=True)
        starts = valid_starts(E, V, H, a.horizon)
        idx = np.random.default_rng(a.seed).choice(starts, min(a.starts, len(starts)),
                                                   replace=False)
        off = np.arange(H) - (H - 1)
        w = idx[:, None] + off[None, :]
        ctx = (torch.from_numpy(S[w]).to(dev), torch.from_numpy(P[w]).to(dev),
               torch.from_numpy(A[w[:, :-1]]).to(dev).float())
        fut = torch.from_numpy(A[idx[:, None] + np.arange(a.horizon)[None, :]]).to(dev).float()
        arena = StateArena(sim, obs, PPOConfig(horizon=a.horizon), H, ticks)
        B = len(idx)

        runs = {}
        for r in a.runs:
            pols, step, cfg = load_run(r, obs.dim, H, ticks, dev)
            runs[r.name] = pols
            print(f"  run {r.name}: step {step}, trained in {cfg['args']['sim']}", flush=True)

        raw = {}

        def match(tag, pol, opp):
            raw[tag] = per_start(arena, pol, opp, ctx, fut, a.draws, a.seed)
            return raw[tag]

        W = {}
        print(f"\n  {'matchup':<44}{'net':>10}{'95% CI':>24}{'dealt/step':>12}", flush=True)

        def show(tag):
            num, den, dealt = raw[tag]
            est = net_of(num, den)
            lo, hi = ci(boot(lambda i: net_of(num, den, i), B, a.boot))
            dps = float(dealt.sum() / max(den.sum(), 1.0))
            W[tag] = {"net": est, "lo": lo, "hi": hi, "dealt_per_step": dps}
            print(f"  {tag:<44}{est:+10.5f}   [{lo:+.5f}, {hi:+.5f}]{dps:12.5f}", flush=True)

        for name, pols in runs.items():
            for tag, pol, opp in ((f"{name} final vs own reference", pols["policy"], pols["reference"]),
                                  (f"{name} final vs replay", pols["policy"], "replay"),
                                  (f"{name} prior vs replay", pols["reference"], "replay"),
                                  (f"{name} final vs itself [must be 0]", pols["policy"],
                                   pols["policy"])):
                match(tag, pol, opp)
                show(tag)
        names = list(runs)
        for x, y in itertools.permutations(names, 2):
            match(f"{x} vs {y}", runs[x]["policy"], runs[y]["policy"])
            show(f"{x} vs {y}")

        # Paired differences on the same starts: improvement over the prior against the human,
        # and the head-to-head antisymmetry check.
        print(f"\n  {'paired difference':<44}{'value':>10}{'95% CI':>24}", flush=True)

        def diff(tag, t1, t2, sign=-1.0):
            n1, d1, _ = raw[t1]
            n2, d2, _ = raw[t2]
            f = lambda i=None: net_of(n1, d1, i) + sign * net_of(n2, d2, i)
            est = f()
            lo, hi = ci(boot(f, B, a.boot))
            W[tag] = {"value": est, "lo": lo, "hi": hi}
            print(f"  {tag:<44}{est:+10.5f}   [{lo:+.5f}, {hi:+.5f}]", flush=True)

        for name in names:
            diff(f"{name}: final - prior, vs replay", f"{name} final vs replay",
                 f"{name} prior vs replay")
        for x, y in itertools.combinations(names, 2):
            diff(f"{x} vs {y} + reverse [must be 0]", f"{x} vs {y}", f"{y} vs {x}", sign=+1.0)
        for x, y in itertools.combinations(names, 2):
            diff(f"{x} - {y}, vs replay", f"{x} final vs replay", f"{y} final vs replay")

        bad = [t for t, v in W.items()
               if "[must be 0]" in t and not v["lo"] <= 0.0 <= v["hi"]]
        print(f"\n  instrument checks: {'ALL PASS' if not bad else 'FAIL: ' + ', '.join(bad)}",
              flush=True)
        results["worlds"][str(sim_path)] = {"hold": hold, "table": W, "checks_failed": bad}

    if a.out:
        a.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
