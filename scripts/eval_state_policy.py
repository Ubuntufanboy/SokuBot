"""Score a trained state-space policy against the frozen prior, per gym.

    python -m scripts.eval_state_policy --run ~/rl/grpo_state \
        --policy policy_best.pt

The training log's `net` is one number per evaluation and answers "is it better".
This answers "at what", which is the question a set of gyms was built to ask,
and it reports the ABSOLUTE behaviour on both sides -- guard rate, press rate,
damage taken -- rather than only the difference, because a difference cannot
distinguish "learned to block" from "the reference got worse".

THE REFERENCE IS RECONSTRUCTED, AND CHECKED
-------------------------------------------
The yardstick is the corpus-prior-initialised policy the run started from. It
is rebuilt here from the same seed and the same bank statistics rather than
loaded, because runs before this script existed did not save it. That is a real
risk -- a reference off by a random trunk is a different opponent and every
number would shift -- so it is verified rather than assumed: the reference is
played against itself, and its net must come out at the noise floor. If it does
not, the reconstruction is wrong and the script says so instead of reporting
numbers.
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

# Pure damage exchange, so the number means the same thing whatever reward the
# run was trained under. See train_state_grpo.EVAL_RCFG.
EVAL_RCFG = StateRewardConfig(damage_mode="step", combo=0.0, idle=0.0,
                              crush=0.0, win=0.0, lose=0.0)


@torch.no_grad()
def play(arena, ctx, side_val, actor, opp_policy, n):
    s_ctx, p_ctx, a_hist = ctx
    side = torch.full((n,), side_val, device=s_ctx.device, dtype=torch.long)
    tr = arena.rollout(s_ctx, p_ctx, a_hist, side, actor,
                       StatePolicyOpponent(opp_policy))
    _, al, terms = compute_rewards(tr["states"], tr["joint"], side, EVAL_RCFG)
    d = al.sum().clamp(min=1)
    mine = tr["states"].gather(
        2, side.view(-1, 1, 1, 1).expand(-1, tr["states"].shape[1], 1,
                                         tr["states"].shape[-1])).squeeze(2)
    m = tr["mine"]
    return {"dealt": float((terms["dealt"] * al).sum() / d),
            "taken": float((terms["taken"] * al).sum() / d),
            "guard": float((mine[..., CH["guarding"]]
                            + mine[..., CH["wrongblock"]]).clamp(max=1).mean()),
            "airborne": float(mine[..., CH["airborne"]].mean()),
            "press": float(m.mean()),
            "attack": float(m[..., 4:8].mean()),
            # Per button, because an aggregate press rate cannot tell a policy
            # that moves more from one that mashes. The failure mode on record
            # is a 0.53 press rate against a human's 0.097, and it was ONE
            # button that got there first.
            "buttons": [float(m[..., i].mean()) for i in range(10)],
            "idle": float((m.amax(dim=2).amax(dim=-1) < 0.5).float().mean())}


BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")


def both_chairs(arena, ctx, actor, opp, n):
    """Average the two seats, cancelling any bias the simulator has toward one."""
    a, b = play(arena, ctx, 0, actor, opp, n), play(arena, ctx, 1, actor, opp, n)
    return {k: ([(x + y) / 2 for x, y in zip(a[k], b[k])]
                if isinstance(a[k], list) else (a[k] + b[k]) / 2) for k in a}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--policy", default="policy_best.pt")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--starts", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-buttons", nargs="*", default=None, metavar="NAME",
                    help="force these buttons to 0 for BOTH the agent and the "
                         "reference, and report how much of the advantage "
                         "survives. `--mask-buttons spell change` is the "
                         "measurement this was built for: the policy presses "
                         "those 12.3x and 7.4x more than any human in the "
                         "corpus, and they are the two rarest buttons in it, "
                         "so they are where the simulator is freest to be "
                         "wrong. An advantage that vanishes under the mask was "
                         "an advantage over the model, not over the game.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = json.loads((a.run.expanduser() / "config.json").read_text())
    ck = torch.load(a.run.expanduser() / a.policy, map_location=a.device,
                    weights_only=False)
    sim, meta = load_sim(cfg["sim"], a.device)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache,
                                           int(cfg.get("replays", 0) or 0))
    # The normalisation must be the one TRAINING used, so it is loaded from the
    # checkpoint rather than recomputed. `StateObs` is a module precisely so it
    # can travel this way, and recomputing it here would silently move the
    # policy's input space whenever the corpus or the statistic changed --
    # the same class of error as reading a probe fit on other weights.
    s_mu, s_sd, p_mu, p_sd = corpus_stats(S, P)
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(a.device)
    if "obs" in ck:
        obs.load_state_dict(ck["obs"])
        drift = max(float((obs.s_sd.cpu() - torch.as_tensor(s_sd)).abs().max()),
                    float((obs.s_mu.cpu() - torch.as_tensor(s_mu)).abs().max()))
        print(f"observation normalisation loaded from the checkpoint "
              f"(max drift vs recomputed: {drift:.2e})")
    else:
        print("WARNING: checkpoint carries no observation normalisation; "
              "recomputed from this corpus, which may not be the one it was "
              "trained on")

    torch.manual_seed(a.seed)
    reference = SokuPolicy(obs.dim, H, ticks).to(a.device)
    if "reference" in ck:
        reference.load_state_dict(ck["reference"])
        ref_src = "loaded from the checkpoint"
    else:
        # Rebuild it as the run did: same seed, same first torch RNG consumer,
        # same corpus prior. Verified below rather than trusted.
        p1 = A.reshape(-1, 20)[:, :10].astype(np.float32)
        lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                         float(p1[:, 2].mean()), float(p1[:, 3].mean())])
        ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                         float(p1[:, 0].mean()), float(p1[:, 1].mean())])
        reference.set_action_prior(lr_p / lr_p.sum(), ud_p / ud_p.sum(),
                                   p1[:, 4:10].mean(0))
        ref_src = f"reconstructed from seed {a.seed}"
    reference.eval()
    policy = copy.deepcopy(reference)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    print(f"{a.run}/{a.policy}: step {ck.get('step')} net "
          f"{ck.get('net', float('nan')):+.5f} | reference {ref_src}")
    print(f"simulator {cfg['sim']} | horizon {cfg['horizon']} steps "
          f"({cfg['horizon']*ticks*1000/60:.0f} ms) | {a.starts} starts/gym, "
          f"both chairs", flush=True)

    gcfg = StateGRPOConfig(horizon=int(cfg["horizon"]), reward=EVAL_RCFG)
    mask = tuple(BUTTONS.index(b) for b in (a.mask_buttons or []))
    if a.mask_buttons:
        bad = [b for b in a.mask_buttons if b not in BUTTONS]
        if bad:
            raise SystemExit(f"unknown button(s) {bad}; want {list(BUTTONS)}")
        print(f"MASKED for both chairs: {a.mask_buttons}")
    arena = StateArena(sim, obs, gcfg, H, ticks, button_mask=mask)
    gyms = build_gyms(S, P, V, E, int(cfg["horizon"]), H)
    starts = valid_starts(E, V, H, int(cfg["horizon"]))
    rng = np.random.default_rng(12345)
    off = torch.arange(H, device=a.device) - (H - 1)
    St = torch.from_numpy(S).to(a.device)
    Pt = torch.from_numpy(P).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    def ctx_for(idx):
        w = torch.from_numpy(idx).to(a.device)[:, None] + off[None, :]
        return St[w], Pt[w], At[w[:, :-1]].float()

    sets = {"(corpus)": rng.choice(starts, min(a.starts, len(starts)))}
    for k in sorted(gyms):
        st, _ = gyms[k]
        if len(st):
            sets[k] = st[rng.choice(len(st), min(a.starts, len(st)))]

    # The reconstruction check. Same weights on both chairs must net to zero.
    idx = sets["(corpus)"]
    floor = both_chairs(arena, ctx_for(idx), reference, reference, len(idx))
    fnet = floor["dealt"] + floor["taken"]
    print(f"\nnoise floor (reference vs itself): net {fnet:+.6f} "
          f"({fnet*FULL_HP:+.2f} HP/step)")
    if abs(fnet) > 5e-5:
        print("  WARNING: the reference plays itself to a non-zero net. The "
              "reconstruction is not the policy this run started from, and "
              "every difference below is measured against the wrong yardstick.")

    # `dealt` and `taken` are split out because their SUM is the thing that can
    # hide the answer: a policy that deals 10 HP/step more and takes 10 more is
    # not the same agent as one that deals nothing more and takes 10 less, and
    # a net column reports them identically. On a set of gyms half of which
    # drill not being hit, that distinction is the whole result.
    print(f"\n  {'gym':<20} {'net HP/step':>12} {'dealt d':>9} {'taken d':>9}"
          f" {'guard  (ref)':>16} {'press':>8}")
    out = {"run": str(a.run), "policy": a.policy, "step": int(ck.get("step", -1)),
           "noise_floor_net": fnet, "gyms": {}}
    for name, ix in sets.items():
        c = ctx_for(ix)
        ag = both_chairs(arena, c, policy, reference, len(ix))
        rf = both_chairs(arena, c, reference, reference, len(ix))
        net = (ag["dealt"] + ag["taken"]) - (rf["dealt"] + rf["taken"])
        out["gyms"][name] = {"net": net, "agent": ag, "reference": rf,
                             "n": int(len(ix))}
        print(f"  {name:<20} {net*FULL_HP:+12.1f} "
              f"{(ag['dealt']-rf['dealt'])*FULL_HP:+9.1f} "
              f"{(ag['taken']-rf['taken'])*FULL_HP:+9.1f} "
              f"{ag['guard']:7.4f} ({rf['guard']:.4f}) {ag['press']:8.4f}")
    print("  columns are the agent MINUS the reference, in game HP per "
          "decision step;\n  a positive `taken d` means it is hit LESS than "
          "the reference.")

    g = out["gyms"]
    blocks = [k for k in ("block_enter", "block_gap", "escape_pressure",
                          "cornered") if k in g]
    if blocks:
        dg = np.mean([g[k]["agent"]["guard"] - g[k]["reference"]["guard"]
                      for k in blocks])
        dt = np.mean([g[k]["agent"]["taken"] - g[k]["reference"]["taken"]
                      for k in blocks])
        print(f"\nOn the four defensive gyms {blocks}:")
        print(f"  guarding {dg:+.4f} vs the reference, damage taken "
              f"{dt:+.5f} ({dt*FULL_HP:+.1f} HP/step; less negative is better)")
    # Is it still playing like a person? The corpus rate is the third column,
    # because "drifted from the reference" and "drifted from human play" are
    # different questions and only the second one bounds whether any of this
    # transfers.
    corpus = A.reshape(-1, 20).astype(np.float32)
    corpus = (corpus[:, :10].mean(0) + corpus[:, 10:].mean(0)) / 2
    cg = g["(corpus)"]
    print(f"\n  {'button':<8} {'agent':>8} {'ref':>8} {'humans':>8} {'x human':>8}")
    for i, b in enumerate(BUTTONS):
        ag_i, rf_i = cg["agent"]["buttons"][i], cg["reference"]["buttons"][i]
        r = ag_i / corpus[i] if corpus[i] > 1e-6 else float("nan")
        print(f"  {b:<8} {ag_i:8.4f} {rf_i:8.4f} {corpus[i]:8.4f} {r:7.2f}x")
    out["buttons_corpus"] = corpus.tolist()

    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
