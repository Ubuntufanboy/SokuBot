"""Can a policy actually move the reward, as opposed to the imagined state?

    python -m scripts.reward_sensitivity --ckpt /root/ckpt_cf/best.pt

scripts/action_sensitivity showed actions move imagined *health* by 0.235 of a
bar at the 1.6 s horizon, well above the probe's ~0.14 residual. GRPO still did
not learn: the policy collapsed to near-deterministic and tied a random
opponent, and the evaluation numbers barely moved across every configuration
tried -- as P1 about +0.0030 and as P2 about -0.0031, whatever the policy did.

Those two facts are only compatible if the reward reads something other than
what action_sensitivity measured. It does. action_sensitivity compares health
*levels* at the end of a rollout; the reward sums health *decreases* along it:

    dealt_t = clamp(hp_opp[t+1] - hp_opp[t], max=0).abs()

That clamp rectifies. The probe's per-step residual is around 0.14 of a bar and
roughly independent across steps, so even on a perfectly flat health trajectory
each step contributes about 0.4 * sigma of fake damage, and sixteen of them
accumulate close to a bar of it. Real damage over 1.6 s is a fraction of that.
Rectified noise does not depend on the actions, so most of the reward is a
constant the policy cannot influence, and the part it can influence is buried.

This measures the thing that matters directly: roll fixed, deliberately
different action sequences from identical starts, compute the *actual reward*,
and compare how much it varies across actions against how much it varies across
start states -- the same ratio action_sensitivity reports, but on the reward
rather than on the state.

Both damage modes are measured side by side:

  step   sum of per-step clamped decreases -- what shipped
  net    one clamped decrease over the whole rollout, hp[0] - hp[T]

`net` cannot rectify per-step noise because there are no per-step differences to
rectify; the endpoint noise enters once instead of sixteen times.
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
from sokubot.rl.reward import RewardConfig, compute_rewards
from scripts.action_sensitivity import action_set


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_best/reward_probe.npz"))
    ap.add_argument("--bank", type=Path, default=Path("/root/bank_best.npz"))
    ap.add_argument("--starts", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/reward_sensitivity.json"))
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
    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Z[idx[:, None] + off[None, :]].float()

    acts = action_set(cfg.action_ticks, cfg.action_dim, a.device)
    names = list(acts)
    S, P = a.starts, a.horizon
    side = torch.zeros(S, dtype=torch.long, device=a.device)
    a_hist = torch.zeros(S, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                         device=a.device)

    results = {}
    for mode in ("step", "net"):
        rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                            flying=0.0015, idle=-0.020, damage_mode=mode)
        rets, dealt = [], []
        for name in names:
            plan = acts[name][None, None].expand(S, P, cfg.action_ticks,
                                                 cfg.action_dim).contiguous()
            zi = wm.rollout(z_ctx, plan, a_hist)
            states = head(torch.cat([z_ctx[:, -1:], zi], dim=1))
            r, alive, terms = compute_rewards(states, plan, side, rcfg)
            rets.append(r.sum(dim=1))                       # [S] return per start
            dealt.append(terms["dealt"].sum(dim=1))
        R = torch.stack(rets)                              # [n_actions, S]
        D = torch.stack(dealt)

        across_action = float(R.std(dim=0).mean())
        across_start = float(R.mean(dim=0).std())
        results[mode] = {
            "across_action": across_action,
            "across_start": across_start,
            "ratio": across_action / (across_start + 1e-12),
            "mean_return": float(R.mean()),
            "mean_dealt": float(D.mean()),
            "dealt_across_action": float(D.std(dim=0).mean()),
            "per_action_dealt": {n: float(D[i].mean()) for i, n in enumerate(names)},
        }

    print(f"world model {a.ckpt} step {blob.get('step','?')} | {S} starts | "
          f"{len(names)} action sequences | horizon {P}\n")
    print(f"{'mode':>6} {'mean dealt':>11} {'dealt spread':>13} "
          f"{'return spread':>14} {'across start':>13} {'ratio':>7}")
    print("-" * 68)
    for mode, r in results.items():
        print(f"{mode:>6} {r['mean_dealt']:11.4f} {r['dealt_across_action']:13.4f} "
              f"{r['across_action']:14.4f} {r['across_start']:13.4f} "
              f"{r['ratio']:7.4f}")
    print("-" * 68)
    print("per-action damage dealt to the opponent:")
    for n in names:
        print(f"   {n:10s} step {results['step']['per_action_dealt'][n]:7.4f}   "
              f"net {results['net']['per_action_dealt'][n]:7.4f}")

    s, nt = results["step"], results["net"]
    print()
    print(f"step mode: {s['dealt_across_action']/max(s['mean_dealt'],1e-9)*100:5.1f}% "
          f"of the damage signal varies with the action")
    print(f"net  mode: {nt['dealt_across_action']/max(nt['mean_dealt'],1e-9)*100:5.1f}% "
          f"of the damage signal varies with the action")
    a.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
