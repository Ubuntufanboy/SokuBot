"""Train the FF-JEPA latent planner and keep it as a terminal value for GRPO.

    python -m scripts.train_planner --ckpt /root/ckpt_cf/best_bnfix.pt \
        --horizon 4 --ctx 1 --out /root/planner_h4.pt

`scripts/subgoal_test.py` measured what this is for: the action-free planner
predicts health change over the next H steps at skill +0.72 (H=4) and +0.59
(H=16) against a "no damage happens" baseline. It does not know which button
caused the damage -- it cannot, having never seen an action -- so it is a value
function, not a controller.

That is exactly the missing piece for GRPO here. The transferable
action->outcome signal is strongest at short horizons (+0.32 at four steps,
+0.09 at sixteen), so the rollout wants to be short; but a four-step rollout
sees 0.27 s and knows nothing about what its choices set up. Bootstrapping the
tail with V(s_T) lets a short, trustworthy rollout carry information about the
next couple of seconds, which is the standard fix and the one the measurements
support.

WHY ctx=1
---------
The deterministic planner in the paper uses a sliding window of three past
subgoals, which at stride H needs 2H steps of history at stride H -- history a
four-step rollout does not have. The diffusion variant uses W_G = 1 and does
fine, so a single-latent context is both faithful to the paper and the only
thing that composes with a short rollout. `--ctx 3` is kept for comparison.

The value head is not learned separately: V is read straight off the predicted
subgoal with the frozen reward probe, as the damage exchange the planner expects
over the interval,

    V(s) = [hp_them(s) - hp_them(G(s))] - [hp_me(s) - hp_me(G(s))],

which is in the same units as the `dealt + taken` the reward already pays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from scripts.subgoal_test import LatentPlanner, subgoal_index
from scripts.train_grpo import model_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best_bnfix.pt"))
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo_h4c/bank.npz"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_bnfix/reward_probe.npz"))
    ap.add_argument("--horizon", type=int, default=4, help="stride H")
    ap.add_argument("--ctx", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = a.device

    blob = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = dev
    wm = LeWorldModel(cfg).to(dev)
    wm.load_state_dict(blob["model"])
    wm.eval()
    fp = model_fingerprint(wm)

    from sokubot.probe import LinearProbe
    from sokubot.rl.grpo import ProbeHead
    d = np.load(a.probe, allow_pickle=True)
    head = ProbeHead(LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"],
                                 ysd=d["ysd"], W=d["W"],
                                 names=[str(x) for x in d["names"]])).to(dev)

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(dev)
    ep = bank["ep"]
    n_rep = int(ep.max()) + 1
    rng = np.random.default_rng(0)
    val_reps = set(rng.choice(n_rep, size=max(1, int(n_rep * a.val_frac)),
                              replace=False).tolist())
    is_val = np.array([e in val_reps for e in ep])

    idx = subgoal_index(ep, a.horizon, a.ctx)
    vm = is_val[idx[:, -1]]
    tr = torch.from_numpy(idx[~vm]).to(dev)
    va = torch.from_numpy(idx[vm]).to(dev)
    print(f"H={a.horizon} ctx={a.ctx}  {len(tr):,} train / {len(va):,} val chains",
          flush=True)

    G = LatentPlanner(cfg.latent_dim, ctx=a.ctx).to(dev)
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    @torch.no_grad()
    def score():
        G.eval()
        se = zero = lat = ident = 0.0
        n = 0
        for s in range(0, min(len(va), 16384), 2048):
            rows = va[s : s + 2048]
            cur, tg = Z[rows[:, -2]].float(), Z[rows[:, -1]].float()
            zp = G(Z[rows[:, :-1]].float())
            lat += float(((zp - tg) ** 2).mean(1).sum())
            ident += float(((cur - tg) ** 2).mean(1).sum())
            hc, ht, hp_ = head(cur), head(tg), head(zp)
            se += float((((hp_ - hc)[:, :2] - (ht - hc)[:, :2]) ** 2).sum())
            zero += float(((ht - hc)[:, :2] ** 2).sum())
            n += len(rows)
        G.train()
        return 1 - lat / ident, 1 - se / zero

    for s in range(a.steps):
        j = torch.randint(0, len(tr), (a.batch,), device=dev)
        rows = tr[j]
        loss = nn.functional.mse_loss(G(Z[rows[:, :-1]].float()),
                                      Z[rows[:, -1]].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if (s + 1) % max(1, a.steps // 8) == 0:
            lsk, hsk = score()
            print(f"  step {s+1:6d} | mse {loss.item():.5f} | latent skill "
                  f"{lsk:+.4f} | health-delta skill {hsk:+.4f}", flush=True)

    lsk, hsk = score()
    G.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"planner": G.state_dict(), "horizon": a.horizon, "ctx": a.ctx,
                "latent_dim": cfg.latent_dim, "wm_fingerprint": fp,
                "ckpt": str(a.ckpt), "latent_skill": lsk,
                "health_delta_skill": hsk}, a.out)
    print(f"\nwrote {a.out}  (latent {lsk:+.4f}, health-delta {hsk:+.4f}, "
          f"world model {fp})")
    if hsk < 0.2:
        print("WARNING: this planner barely predicts damage; bootstrapping a "
              "rollout with it will add noise rather than horizon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
