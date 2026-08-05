"""Is a sparse subgoal easier to predict than an autoregressive rollout?

    python -m scripts.subgoal_test --ckpt /root/ckpt_cf/best_bnfix.pt

FF-JEPA (Masip et al., arXiv:2606.09311) sits a second, *action-free* model on
top of a frozen LeWM world model. Where the world model steps one frame at a
time, the latent planner G jumps H steps in one shot, trained on demonstrations
subsampled with stride H:

    z_sg,m+1_hat = G(z_sg, m-W_G+1 : m)

Their claim is not that the far future is predictable. It is that predicting one
sparse subgoal beats autoregressing H steps to the same place, because a single
jump cannot compound its own error. On PushT that takes flat LeWM from 3.52%
success at t=75 to 91.80%.

That maps onto a decay we have already measured here: the transferable
action->outcome signal is +0.55 at one step and +0.09 at sixteen, and the drift
curve shows the same shape. If the compounding is what kills it, a one-shot jump
should hold up where sixteen chained steps do not.

WHAT IS DIFFERENT ABOUT SOKU, AND WHY IT MIGHT NOT TRANSFER
-----------------------------------------------------------
PushT has one agent and a goal state. Soku has two agents and no goal. An
action-free model asked where the game will be in half a second is averaging
over *the opponent's* choices as well as its own, and the mean of an adversarial
distribution is a state that never occurs -- the same conditional-mean trap that
makes long rollouts useless. So this is a real question and not a formality.

THE COMPARISON
--------------
Everything is scored against the same held-out replays and the same denominator,
`1 - MSE / MSE_identity`, where identity is copying the current subgoal forward
and changing nothing. Three ways of getting from z_t to z_{t+H}:

  planner          G, one shot, no actions. The FF-JEPA arm.
  flat + true      the world model autoregressed H steps on the buttons that
                   were really pressed. Privileged -- it is told the future --
                   so it is an upper bound rather than a fair rival.
  flat + prior     the same rollout on actions sampled from the corpus prior,
                   which is what planning actually has available.

The planner beating `flat + prior` is the result that would justify building on
this. Beating `flat + true` as well would mean the compounding is so severe that
knowing the real actions does not rescue it. Failing to beat *identity* kills it
outright, and is a live possibility in an adversarial game.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel


class LatentPlanner(nn.Module):
    """G_Det: LeWM's predictor without the action conditioning.

    Sized to the paper's 9.5M deterministic planner rather than to what is
    convenient, so a null result cannot be blamed on starving the arm under
    test.
    """

    def __init__(self, latent: int, width: int = 384, depth: int = 6,
                 heads: int = 6, ctx: int = 3, action_dim: int = 0):
        super().__init__()
        self.inp = nn.Linear(latent, width)
        # Optional action conditioning. FF-JEPA's planner is deliberately
        # action-free, which is right for a single agent moving toward a goal.
        # Here it is the whole question: if telling the planner what both
        # players did over the interval does not improve it, then the H-step
        # future is action-independent, the subgoal is an accurate number that
        # no policy can steer, and there is nothing for GRPO to optimise toward.
        self.act = nn.Linear(action_dim, width) if action_dim else None
        self.pos = nn.Parameter(torch.zeros(1, ctx, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.body = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, latent)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, a=None):               # [B, ctx, latent], [B, action]
        h = self.inp(z) + self.pos[:, -z.shape[1]:]
        if self.act is not None and a is not None:
            h = h + self.act(a)[:, None]
        mask = nn.Transformer.generate_square_subsequent_mask(
            z.shape[1], device=z.device)
        h = self.norm(self.body(h, mask=mask, is_causal=True))
        # Residual on the last observed subgoal: the planner predicts a *change*,
        # so an untrained G starts exactly at the identity baseline it is being
        # scored against and any skill above zero is something it learned.
        return z[:, -1] + self.out(h[:, -1])


def subgoal_index(ep: np.ndarray, H: int, ctx: int):
    """Start indices m such that m-ctx+1 .. m+1 are all in one replay at stride H."""
    out = []
    edges = np.flatnonzero(np.diff(ep)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ep)]):
        chain = np.arange(lo, hi, H)
        if len(chain) < ctx + 1:
            continue
        out.append(np.stack([chain[i : len(chain) - (ctx - i)]
                             for i in range(ctx + 1)], axis=1))
    return np.concatenate(out) if out else np.zeros((0, ctx + 1), dtype=np.int64)


@torch.no_grad()
def flat_rollout(wm, Z, A, base, H, cfg, dev, prior_actions=None):
    """Autoregress the world model H steps from `base`, returning the latent."""
    off = torch.arange(cfg.history, device=dev) - (cfg.history - 1)
    z_win = Z[base[:, None] + off[None, :]].float()
    a_win = A[base[:, None] + off[None, :]][:, :-1].float()
    for t in range(H):
        act = (A[base + t].float() if prior_actions is None
               else prior_actions[:, t])
        zhat = wm.predictor(z_win, wm.action_encoder(
            torch.cat([a_win, act[:, None]], dim=1)))[:, -1]
        z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
        if cfg.history > 1:
            a_win = torch.cat([a_win[:, 1:], act[:, None]], dim=1)
    return z_win[:, -1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best_bnfix.pt"))
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo_h4c/bank.npz"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--ctx", type=int, default=3, help="W_G, the planner's context")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2, help="held-out replays")
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe", type=Path, default=None,
                    help="reward probe. With it, the subgoal is also scored on "
                         "the quantity GRPO's reward is actually made of -- the "
                         "health change over the interval -- instead of on total "
                         "latent MSE, which is dominated by structure nobody "
                         "controls and cannot resolve an effect this small.")
    ap.add_argument("--out", type=Path, default=Path("/root/subgoal_test.json"))
    a = ap.parse_args()
    dev = a.device

    blob = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = dev
    wm = LeWorldModel(cfg).to(dev)
    wm.load_state_dict(blob["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(dev)
    A = torch.from_numpy(bank["a"]).to(dev)
    ep = bank["ep"]
    n_rep = int(ep.max()) + 1
    rng = np.random.default_rng(0)
    val_reps = set(rng.choice(n_rep, size=max(1, int(n_rep * a.val_frac)),
                              replace=False).tolist())
    is_val = np.array([e in val_reps for e in ep])
    print(f"{len(Z)} latents, {n_rep} replays, {len(val_reps)} held out "
          f"({(is_val).sum()} latents)\n", flush=True)

    # Corpus action prior, for the honest "planning does not know the future" arm.
    flat_a = A.reshape(-1, cfg.action_dim).float()
    prior_p = flat_a.mean(0)

    head = None
    if a.probe is not None:
        from sokubot.probe import LinearProbe
        from sokubot.rl.grpo import ProbeHead
        d = np.load(a.probe, allow_pickle=True)
        head = ProbeHead(LinearProbe(
            zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"], W=d["W"],
            names=[str(x) for x in d["names"]])).to(dev)
        print(f"probe: {[str(x) for x in d['names']]}\n")

    results = {}
    print(f"{'H':>3} {'sec':>6} {'n_train':>9} | {'planner':>9} "
          f"{'plan+act':>9} {'act gain':>9} | {'flat+true':>10} "
          f"{'flat+prior':>11}")
    print("-" * 82)

    for H in a.horizons:
        idx = subgoal_index(ep, H, a.ctx)
        if len(idx) < 1000:
            print(f"{H:>3} -- too few subgoal chains ({len(idx)})")
            continue
        val_mask = is_val[idx[:, -1]]
        tr_idx = torch.from_numpy(idx[~val_mask]).to(dev)
        va_idx = torch.from_numpy(idx[val_mask]).to(dev)

        def interval_actions(base):
            """Mean of each button over the H steps between two subgoals.

            A summary rather than the full sequence, which makes this arm a
            *lower* bound on how much the interval's actions explain -- if even
            the aggregate helps, actions matter."""
            steps = torch.arange(H, device=dev)
            chunk = A[base[:, None] + steps[None, :]].float()   # [B,H,ticks,dim]
            return chunk.mean(dim=(1, 2))                       # [B, action_dim]

        planners = {}
        for arm, adim in (("planner", 0), ("planner+act", cfg.action_dim)):
            G = LatentPlanner(cfg.latent_dim, ctx=a.ctx, action_dim=adim).to(dev)
            opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=1e-2)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            for s in range(a.steps):
                j = torch.randint(0, len(tr_idx), (a.batch,), device=dev)
                rows = tr_idx[j]
                zin = Z[rows[:, :-1]].float()
                ztg = Z[rows[:, -1]].float()
                av = interval_actions(rows[:, -2]) if adim else None
                loss = nn.functional.mse_loss(G(zin, av), ztg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
            G.eval()
            planners[arm] = G

        # ---- held-out evaluation, one denominator for everything -----------
        sel = va_idx[torch.randperm(len(va_idx), device=dev)[: a.eval_n]]
        base = sel[:, -2]                       # the current subgoal's index
        target = Z[sel[:, -1]].float()
        se = {k: 0.0 for k in ("planner", "planner+act", "flat_true",
                               "flat_prior", "identity")}
        dse = {"pred": 0.0, "zero": 0.0, "absdelta": 0.0}
        n = 0
        with torch.no_grad():
            for s0 in range(0, len(sel), 1024):
                rows = sel[s0 : s0 + 1024]
                b = rows[:, -2]
                tg = Z[rows[:, -1]].float()
                cur = Z[b].float()
                zp = planners["planner"](Z[rows[:, :-1]].float())
                zpa = planners["planner+act"](Z[rows[:, :-1]].float(),
                                              interval_actions(b))
                pri = (torch.rand(len(b), H, cfg.action_ticks, cfg.action_dim,
                                  device=dev) < prior_p).float()
                zt = flat_rollout(wm, Z, A, b, H, cfg, dev)
                zr = flat_rollout(wm, Z, A, b, H, cfg, dev, prior_actions=pri)
                se["planner"] += float(((zp - tg) ** 2).mean(1).sum())
                se["planner+act"] += float(((zpa - tg) ** 2).mean(1).sum())
                se["flat_true"] += float(((zt - tg) ** 2).mean(1).sum())
                se["flat_prior"] += float(((zr - tg) ** 2).mean(1).sum())
                se["identity"] += float(((cur - tg) ** 2).mean(1).sum())
                if head is not None:
                    # Health change over the interval, in bar units. The
                    # identity baseline here is "no damage happens", which is
                    # what a subgoal that knows nothing about the fight predicts.
                    h_cur, h_tg, h_p = head(cur), head(tg), head(zp)
                    d_true = (h_tg - h_cur)[:, :2]
                    d_pred = (h_p - h_cur)[:, :2]
                    dse["pred"] += float(((d_pred - d_true) ** 2).sum())
                    dse["zero"] += float((d_true ** 2).sum())
                    dse["absdelta"] += float(d_true.abs().sum())
                n += len(b)
        sk = {k: 1.0 - se[k] / se["identity"] for k in
              ("planner", "planner+act", "flat_true", "flat_prior")}
        best = max(sk, key=sk.get)
        results[H] = {"seconds": H * cfg.frame_skip / 60, "n_train": len(tr_idx),
                      "n_eval": n, "skill": sk,
                      "mse": {k: se[k] / n for k in se}}
        gain = sk["planner+act"] - sk["planner"]
        results[H]["action_gain"] = gain
        if head is not None:
            hp_skill = 1.0 - dse["pred"] / max(dse["zero"], 1e-12)
            results[H]["health_delta_skill"] = hp_skill
            results[H]["mean_abs_health_delta"] = dse["absdelta"] / (2 * n)
            print(f"      health delta over the interval: skill {hp_skill:+.4f} "
                  f"vs 'no damage', mean |delta| "
                  f"{dse['absdelta']/(2*n):.4f} bar", flush=True)
        print(f"{H:>3} {H*cfg.frame_skip/60:6.2f} {len(tr_idx):>9,} | "
              f"{sk['planner']:+9.4f} {sk['planner+act']:+9.4f} {gain:+9.4f} | "
              f"{sk['flat_true']:+10.4f} {sk['flat_prior']:+11.4f}", flush=True)

    print("-" * 82)
    print("skill = 1 - MSE/MSE_identity; identity copies the current subgoal.")
    print("positive beats 'nothing changed'; negative is worse than doing nothing.\n")

    # Beating the flat rollout on total latent MSE is necessary and nowhere near
    # sufficient. Most of that variance is structure nobody controls -- health
    # bars that barely move, a weather timer counting down, positions reverting
    # to the mean -- and a planner can win it while knowing nothing about the
    # fight. What decides whether this is worth building on is whether the
    # subgoal carries the quantity the reward is made of, scored only with
    # --probe.
    wins = [H for H, r in results.items()
            if r["skill"]["planner"] > max(r["skill"]["flat_prior"], 0.0)]
    hp = {H: r["health_delta_skill"] for H, r in results.items()
          if "health_delta_skill" in r}
    if not results:
        v = "no horizon had enough data to score"
    elif not wins:
        v = ("the planner never beats both the flat rollout and the identity "
             "baseline, so FF-JEPA's premise does not carry to this game: "
             "spend the time on the entropy floor and a longer GRPO run")
    elif not hp:
        v = (f"the planner beats the flat rollout at {sorted(wins)}, but this "
             f"was run without --probe, so whether the subgoal knows anything "
             f"about damage -- the only part that could help a policy -- is "
             f"unmeasured. Re-run with --probe before concluding anything")
    elif max(hp.values()) < 0.2:
        v = (f"the planner predicts the latent well and the health change badly "
             f"(best {max(hp.values()):+.3f}), so the subgoal is an accurate "
             f"account of the parts nobody controls. Nothing for a policy to "
             f"steer toward; prefer the entropy floor")
    else:
        bh = max(hp, key=hp.get)
        v = (f"the subgoal predicts health change over the interval at "
             f"{hp[bh]:+.3f} (H={bh}) against 'no damage happens', so it "
             f"carries reward-relevant state action-free. That is a value "
             f"estimate, not a controller: the use for it is bootstrapping the "
             f"tail of a short GRPO rollout, not CEM subgoal-steering")
    print(f"VERDICT: {v}")
    if hp:
        print("\nNote: an action-free predictor of damage is a value function. "
              "It says how the fight is about to go, never which button caused "
              "it, so it extends the horizon rather than replacing the policy.")
    a.out.write_text(json.dumps({"ckpt": str(a.ckpt), "ctx": a.ctx,
                                 "steps": a.steps, "results": results},
                                indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
