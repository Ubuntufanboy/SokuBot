"""Play two trained policies against each other, instead of scoring both
against a third.

    python -m scripts.head_to_head --a ~/rl/grpo_sym --b ~/rl/grpo_def

WHY THIS EXISTS
---------------
`eval_state_policy` scores every policy against the same frozen corpus-prior
reference, re-scored with one fixed symmetric reward so the number means the
same thing whatever the arm trained on. That makes the METRIC comparable. It
does not make the CRITERION neutral: net damage exchange is essentially the
symmetric arm's own objective, so an arm that deliberately traded net for not
being hit is guaranteed to score lower on it. Ranking them that way assumes the
answer.

Two policies playing each other has no such assumption. Whoever comes out ahead
on the exchange is ahead, and the reward each was trained with does not enter.

BOTH CHAIRS, SAME STARTS
------------------------
Every start is played twice with the sides swapped and the results averaged, so
neither policy can score by exploiting whichever seat the simulator happens to
favour -- the same construction the single-policy evaluation uses, and for the
same reason.

ONE SIMULATOR AT A TIME
-----------------------
Two policies trained against DIFFERENT world models cannot be played off
against each other honestly: the match has to happen inside one of them, and
that one is home turf. This refuses the cross-model case rather than reporting
a number that looks fair and is not.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.data import state_bank
from sokubot.data.state import CH, FULL_HP
from sokubot.model.state_dynamics import load_sim
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.state_arena import (StateArena, StateGRPOConfig, StateObs,
                                    StatePolicyOpponent, corpus_stats)
from sokubot.rl.state_reward import StateRewardConfig, compute_rewards
from scripts.build_gyms import build as build_gyms
from scripts.train_state_grpo import valid_starts

SCORE = StateRewardConfig(damage_mode="step", combo=0.0, idle=0.0, crush=0.0,
                          win=0.0, lose=0.0)


def load_policy(run: Path, name: str, obs_dim, H, ticks, device):
    ck = torch.load(run / name, map_location=device, weights_only=False)
    pol = SokuPolicy(obs_dim, H, ticks).to(device)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, ck


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--policy", default="policy_best.pt")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--starts", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234,
                    help="COMMON RANDOM NUMBERS. Every rollout in the run is "
                         "drawn from this same seed, so the sampling noise is "
                         "shared between the A-vs-B match and the A-vs-A "
                         "control and cancels in their difference. Without it "
                         "the control read +0.61 HP/step on a quantity that is "
                         "exactly zero by construction, which is larger than "
                         "most of the differences being measured.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    ca = json.loads((a.a / "config.json").read_text())
    cb = json.loads((a.b / "config.json").read_text())
    if ca["sim"] != cb["sim"]:
        raise SystemExit(
            f"these were trained against different world models:\n"
            f"  {a.a.name}: {ca['sim']}\n  {a.b.name}: {cb['sim']}\n"
            f"A head-to-head has to be played inside one of them, and that one "
            f"is home turf. Compare them via eval_state_policy against their "
            f"own references instead.")
    sim, meta = load_sim(ca["sim"], a.device)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache)
    s_mu, s_sd, p_mu, p_sd = corpus_stats(S, P)
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(a.device)
    pa, cka = load_policy(a.a, a.policy, obs.dim, H, ticks, a.device)
    pb, ckb = load_policy(a.b, a.policy, obs.dim, H, ticks, a.device)
    if "obs" in cka:
        obs.load_state_dict(cka["obs"])

    hor = int(ca["horizon"])
    arena = StateArena(sim, obs, StateGRPOConfig(horizon=hor), H, ticks)
    gyms = build_gyms(S, P, V, E, hor, H)
    starts = valid_starts(E, V, H, hor)
    rng = np.random.default_rng(2024)
    sets = {"(corpus)": rng.choice(starts, min(a.starts, len(starts)))}
    for k in sorted(gyms):
        st, _ = gyms[k]
        if len(st):
            sets[k] = st[rng.choice(len(st), min(a.starts, len(st)))]

    off = torch.arange(H, device=a.device) - (H - 1)
    St = torch.from_numpy(S).to(a.device)
    Pt = torch.from_numpy(P).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    @torch.no_grad()
    def match(idx, actor, opponent):
        w = torch.from_numpy(idx).to(a.device)[:, None] + off[None, :]
        s_ctx, p_ctx, a_hist = St[w], Pt[w], At[w[:, :-1]].float()
        out = {}
        for tag, s0 in (("p1", 0), ("p2", 1)):
            side = torch.full((len(idx),), s0, device=a.device, dtype=torch.long)
            # Reset before EVERY rollout, so the action sampling and the
            # jitter draw the same stream in the match and in the control.
            # The rollout is otherwise deterministic given (start, actions),
            # so this removes essentially all of the comparison's variance.
            torch.manual_seed(a.seed + s0)
            tr = arena.rollout(s_ctx, p_ctx, a_hist, side, actor,
                               StatePolicyOpponent(opponent))
            _, al, terms = compute_rewards(tr["states"], tr["joint"], side, SCORE)
            n = al.sum().clamp(min=1)
            out[f"{tag}_dealt"] = float((terms["dealt"] * al).sum() / n)
            out[f"{tag}_taken"] = float((terms["taken"] * al).sum() / n)
        # A's edge over B, averaged over both seats.
        return {"net": ((out["p1_dealt"] + out["p1_taken"])
                        + (out["p2_dealt"] + out["p2_taken"])) / 2, **out}

    print(f"\n{a.a.name} (A) vs {a.b.name} (B), inside {Path(ca['sim']).parent.name}")
    print(f"steps {cka.get('step')} vs {ckb.get('step')} | horizon {hor} | "
          f"{a.starts} starts/gym, both chairs")
    print(f"trained with damage_dealt {ca['damage_dealt']}/"
          f"{ca.get('defence_dealt')} and {cb['damage_dealt']}/"
          f"{cb.get('defence_dealt')}\n")
    print(f"  {'gym':<20} {'A net edge':>12} {'HP/step':>9}")
    res = {}
    for name, ix in sets.items():
        m = match(ix, pa, pb)
        res[name] = m
        print(f"  {name:<20} {m['net']:+12.6f} {m['net']*FULL_HP:+9.2f}")
    # A cannot beat itself: the same weights on both chairs must net to zero,
    # and if they do not the construction is wrong rather than the policy good.
    ctrl = match(sets["(corpus)"], pa, pa)
    print(f"\n  {'(A vs itself)':<20} {ctrl['net']:+12.6f} "
          f"{ctrl['net']*FULL_HP:+9.2f}   <- must be ~0")
    if a.out:
        a.out.write_text(json.dumps({"a": str(a.a), "b": str(a.b),
                                     "self_control": ctrl["net"],
                                     "gyms": res}, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
