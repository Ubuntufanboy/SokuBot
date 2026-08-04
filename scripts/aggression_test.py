"""Does the imagined world reward attacking, in the regime GRPO plays in?

    python -m scripts.aggression_test --ckpt /root/ckpt_cf/best.pt

scripts/reward_sensitivity compared eight constant action sequences -- hold left
for 1.6 s, hold A for 1.6 s -- and found doing nothing dealt the most damage.
That is suggestive and not conclusive, because nobody plays that way and neither
does GRPO. Constant holds are as far out of distribution as uniform random
sampling is.

This asks the same question inside the distribution the policy actually samples
from. Many action sequences are drawn from the corpus prior, rolled from
identical starts, and each is scored on two axes: how much it attacks, and how
much damage the model says it dealt. The correlation between them is the whole
answer.

  positive   attacking earns damage; there is a gradient for GRPO to climb and
             the remaining problems are ones of optimisation
  zero       the model is indifferent to what the policy does, and no reward
             shaping or learning rate reaches that
  negative   the model believes attacking is counterproductive, and GRPO will
             faithfully learn to stop doing it

The opponent is drawn from the same prior rather than left idle, because an
opponent holding no buttons for 1.6 s is itself a state the corpus almost never
contains, and it was the confound in the previous measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import ProbeHead
from sokubot.rl.policy import FREE_BUTTONS, SokuPolicy, to_joint
from sokubot.rl.reward import RewardConfig, compute_rewards

ATTACK = (4, 5, 6, 7)          # a b c d within a player's ten-button block


def corpus_prior(A: np.ndarray, action_dim: int):
    p1 = A.astype(np.float32).reshape(-1, action_dim)[:, :10]
    lr = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                   float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                   float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    return lr / lr.sum(), ud / ud.sum(), p1[:, 4:10].mean(0)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_best/reward_probe.npz"))
    ap.add_argument("--bank", type=Path, default=Path("/root/bank_best.npz"))
    ap.add_argument("--starts", type=int, default=256)
    ap.add_argument("--samples", type=int, default=32, help="rollouts per start")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="*", default=None,
                    help="report the correlation at each of these rollout "
                         "lengths as well as the full one. The full-horizon "
                         "number alone cannot tell an action effect that never "
                         "existed from one that existed and washed out.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/aggression.json"))
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    d = np.load(a.probe, allow_pickle=True)
    head = ProbeHead(LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"],
                                 ysd=d["ysd"], W=d["W"],
                                 names=[str(x) for x in d["names"]])).to(a.device)

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(a.device)
    ep = bank["ep"]
    lo = cfg.history - 1
    ok = np.flatnonzero(np.diff(ep) == 0)
    ok = ok[(ok >= lo) & (ok < len(ep) - a.horizon - 1)]
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(ok, size=a.starts, replace=False)).to(a.device)

    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
    policy.set_action_prior(*corpus_prior(bank["a"], cfg.action_dim))
    policy.eval()

    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    G, P = a.samples, a.horizon
    rep = idx.repeat_interleave(G)
    z_ctx = Z[rep[:, None] + off[None, :]].float()
    B = len(rep)
    side = torch.zeros(B, dtype=torch.long, device=a.device)
    z_win = z_ctx
    a_win = torch.zeros(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                        device=a.device)

    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020)
    # No encoder latent in the sequence: see ImaginedArena.rollout. The probe is
    # calibrated for predictor outputs and reads encoder outputs 2.6x
    # over-dispersed, which fabricates damage at the first step.
    zs, joints = [], []
    for _ in range(P + 1):
        mine = policy(z_win, side, sample=True).actions
        theirs = policy(z_win, 1 - side, sample=True).actions
        joint = to_joint(mine, theirs, side)
        zhat = wm.predictor(z_win, wm.action_encoder(
            torch.cat([a_win, joint[:, None]], dim=1)))[:, -1]
        zs.append(zhat)
        joints.append(joint)
        z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
        if cfg.history > 1:
            a_win = torch.cat([a_win[:, 1:], joint[:, None]], dim=1)

    states = head(torch.stack(zs, dim=1))               # [B,P+1,K] all predicted
    J = torch.stack(joints[1:], dim=1)                  # [B,P,ticks,20]
    _, alive, terms = compute_rewards(states, J, side, rcfg)

    mine_blk = J[..., :10]
    attack = mine_blk[..., list(ATTACK)].mean(dim=(1, 2, 3))     # [B]
    press = mine_blk.mean(dim=(1, 2, 3))
    dealt = terms["dealt"].sum(dim=1)
    taken = terms["taken"].sum(dim=1)
    net = dealt + taken

    def within_h(x, y):
        xs = x.view(a.starts, G) - x.view(a.starts, G).mean(1, keepdim=True)
        ys = y.view(a.starts, G) - y.view(a.starts, G).mean(1, keepdim=True)
        return float(((xs * ys).sum(1) /
                      (xs.norm(dim=1) * ys.norm(dim=1) + 1e-12)).mean())

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))

    # Correlate *within* each start, then average. Across starts the situation
    # dominates and would swamp the effect of the actions, which is the same
    # reason GRPO centres advantages within a group.
    def within(x, y):
        xs = x.view(a.starts, G)
        ys = y.view(a.starts, G)
        xs = xs - xs.mean(1, keepdim=True)
        ys = ys - ys.mean(1, keepdim=True)
        num = (xs * ys).sum(1)
        den = xs.norm(dim=1) * ys.norm(dim=1) + 1e-12
        return float((num / den).mean())

    res = {
        "attack_vs_dealt_within": within(attack, dealt),
        "attack_vs_net_within": within(attack, net),
        "press_vs_dealt_within": within(press, dealt),
        "attack_vs_dealt_pooled": corr(attack, dealt),
        "mean_attack": float(attack.mean()), "mean_dealt": float(dealt.mean()),
        "mean_taken": float(taken.mean()),
        "dealt_spread_within": float(dealt.view(a.starts, G).std(1).mean()),
        "dealt_spread_across": float(dealt.view(a.starts, G).mean(1).std()),
    }

    print(f"{a.starts} starts x {G} sampled rollouts, horizon {P}\n")
    print(f"mean attack rate      {res['mean_attack']:.4f}")
    print(f"mean damage dealt     {res['mean_dealt']:.4f}")
    print(f"mean damage taken     {res['mean_taken']:.4f}")
    print(f"damage spread within a start  {res['dealt_spread_within']:.4f}")
    print(f"damage spread across starts   {res['dealt_spread_across']:.4f}")
    print()
    if a.horizons:
        # Recompute rewards over truncated prefixes of the same rollouts, so the
        # only thing changing is how far the rollout is allowed to run.
        print(f"{'steps':>6} {'seconds':>8} {'mean dealt':>11} {'spread':>9} "
              f"{'corr(attack,dealt)':>19}")
        print("-" * 58)
        sweep = {}
        for h in a.horizons:
            if h > P:
                continue
            st = states[:, : h + 1]
            _, _, tm = compute_rewards(st, J[:, :h], side, rcfg)
            dh = tm["dealt"].sum(dim=1)
            c = within_h(attack, dh)
            sweep[h] = {"corr": c, "mean_dealt": float(dh.mean()),
                        "spread_within": float(dh.view(a.starts, G).std(1).mean())}
            print(f"{h:6d} {h*4/60:8.3f} {dh.mean():11.4f} "
                  f"{sweep[h]['spread_within']:9.4f} {c:+19.4f}")
        print("-" * 58)
        res["by_horizon"] = sweep
        print()
    print(f"corr(attack, dealt) within start : {res['attack_vs_dealt_within']:+.4f}")
    print(f"corr(attack, net)   within start : {res['attack_vs_net_within']:+.4f}")
    print(f"corr(press,  dealt) within start : {res['press_vs_dealt_within']:+.4f}")
    print()
    c = res["attack_vs_dealt_within"]
    if c > 0.05:
        v = "attacking earns damage; the gradient exists and this is an optimisation problem"
    elif c < -0.05:
        v = "the model believes attacking is counterproductive; GRPO will learn not to"
    else:
        v = "the model is indifferent to what the policy does; no tuning reaches this"
    print(f"VERDICT: {v}")
    a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
