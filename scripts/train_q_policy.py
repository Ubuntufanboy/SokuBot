"""Act on the one-step signal directly, instead of policy-gradient through it.

    python -m scripts.train_q_policy --ckpt /root/ckpt_cf/best_bnfix.pt

WHY NOT MORE GRPO
-----------------
Four GRPO runs peaked at +0.001 to +0.0016 and none held it. Every one died the
same way: the policy's logits ran away, entropy hit zero, press rate went to
0.5 against a human's 0.097, and `klref` reached 1e14. At that point the policy
is far outside anything the world model was trained on, so it is optimising
noise.

The human baseline says where the real gap is. The corpus-prior policy already
presses buttons at a human rate -- 0.10 against 0.097 -- and deals 0.0064
against a human's 0.0255. So a four-fold gap comes from *timing*, not from how
often it presses.

And the world model has that information: `action_effect_test` measured the
action->damage rule transferring to unseen start states at r = +0.55 one step
ahead, decaying to +0.09 by sixteen. GRPO consumes the weak end of that. It
gets one scalar return per rollout and has to spread the credit across four
steps by four ticks by twenty buttons, while an unconstrained optimiser walks
out of distribution.

That regressor from `action_effect_test` was a Q-function all along. Fit it
properly and act on it.

WHAT THIS IS
------------
Q(z, a) predicts the damage exchange over one 67 ms step, trained on rollouts
of the frozen world model under actions sampled from the corpus prior. The
policy is then argmax over `--candidates` sampled actions, which is one-step
model-predictive control.

Nothing here can collapse. There is no policy network to saturate, no entropy
to defend, no trust region and no reference to drift from -- the action
distribution is the corpus prior by construction, so the controller cannot
leave the region the world model understands. The cost is that it is greedy
over one step and cannot plan, which is the honest trade: the measurement says
one step is where the signal actually is.

Scored with the same eval GRPO used -- damage dealt and taken against the frozen
prior policy, both chairs, fixed starts -- so the numbers are directly
comparable to GRPO's 0.0085 and the human 0.0255.
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
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import ProbeHead
from sokubot.rl.policy import SokuPolicy, to_joint
from sokubot.rl.reward import RewardConfig, compute_rewards
from scripts.eval_ckpt import assert_predictor_sane


class QNet(nn.Module):
    """Q(z_ctx, my_action) -> expected damage exchange over one step."""

    def __init__(self, latent: int, history: int, ticks: int, width: int = 1024):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent * history + ticks * 10, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1))

    def forward(self, z, a):                     # [B,H,latent], [B,ticks,10]
        return self.body(torch.cat([z.flatten(1), a.flatten(1)], 1)).squeeze(-1)


def corpus_prior(A: np.ndarray, action_dim: int):
    p1 = A.astype(np.float32).reshape(-1, action_dim)[:, :10]
    lr = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                   float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                   float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    return lr / lr.sum(), ud / ud.sum(), p1[:, 4:10].mean(0)


@torch.no_grad()
def step_damage(wm, head, z_win, a_win, mine, theirs, side, cfg, rcfg):
    """One predictor step; returns (dealt + taken) for `side` and the new latent."""
    joint = to_joint(mine, theirs, side)
    zhat = wm.predictor(z_win, wm.action_encoder(
        torch.cat([a_win, joint[:, None]], dim=1)))[:, -1]
    # Two probed states around one action, which is exactly the contract
    # compute_rewards wants: T+1 states, T actions.
    z2 = wm.predictor(torch.cat([z_win[:, 1:], zhat[:, None]], dim=1),
                      wm.action_encoder(torch.cat([a_win[:, 1:], joint[:, None],
                                                   joint[:, None]], dim=1)))[:, -1]
    states = head(torch.stack([zhat, z2], dim=1))
    _, _, terms = compute_rewards(states, joint[:, None], side, rcfg)
    return terms["dealt"][:, 0] + terms["taken"][:, 0], zhat, joint


@torch.no_grad()
def rollout_q(wm, head, q, prior, z_ctx, side, cfg, rcfg, T, cand, dev,
              greedy=True):
    """Roll `T` steps choosing each action as argmax Q over `cand` samples."""
    B = len(side)
    z_win = z_ctx
    a_win = torch.zeros(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                        device=dev)
    zs, mines = [], []
    for _ in range(T + 1):
        theirs = prior(z_win, 1 - side, sample=True).actions
        if q is None or not greedy:
            mine = prior(z_win, side, sample=True).actions
        else:
            # Score `cand` prior samples and keep the best. Sampling the
            # candidates from the corpus prior is what keeps the controller
            # inside the distribution the world model was trained on -- it can
            # only ever choose among actions a person might have pressed.
            flat_z = z_win.repeat_interleave(cand, 0)
            flat_s = side.repeat_interleave(cand, 0)
            cands = prior(flat_z, flat_s, sample=True).actions
            scores = q(flat_z, cands).view(B, cand)
            best = scores.argmax(1)
            mine = cands.view(B, cand, cfg.action_ticks, 10)[torch.arange(B, device=dev), best]
        joint = to_joint(mine, theirs, side)
        zhat = wm.predictor(z_win, wm.action_encoder(
            torch.cat([a_win, joint[:, None]], dim=1)))[:, -1]
        zs.append(zhat)
        mines.append(joint)
        z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
        if cfg.history > 1:
            a_win = torch.cat([a_win[:, 1:], joint[:, None]], dim=1)
    states = head(torch.stack(zs, dim=1))
    J = torch.stack(mines[1:], dim=1)
    _, _, terms = compute_rewards(states, J, side, rcfg)
    return terms["dealt"].sum(1), terms["taken"].sum(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best_bnfix.pt"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_bnfix/reward_probe.npz"))
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo_h4c/bank.npz"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--eval-starts", type=int, default=2048)
    ap.add_argument("--eval-horizon", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/qpolicy"))
    a = ap.parse_args()
    dev = a.device
    a.out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = dev
    wm = LeWorldModel(cfg).to(dev)
    wm.load_state_dict(blob["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    d = np.load(a.probe, allow_pickle=True)
    head = ProbeHead(LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"],
                                 ysd=d["ysd"], W=d["W"],
                                 names=[str(x) for x in d["names"]])).to(dev)

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(dev)
    A = torch.from_numpy(bank["a"]).to(dev)
    ep = bank["ep"]
    assert_predictor_sane(wm, Z, A, ep, cfg, what=str(a.ckpt))

    lo = cfg.history - 1
    ok = np.flatnonzero(np.diff(ep) == 0)
    ok = ok[(ok >= lo) & (ok < len(ep) - a.eval_horizon - 3)]
    rng = np.random.default_rng(0)
    eval_idx = torch.from_numpy(rng.choice(ok, a.eval_starts, replace=False)).to(dev)
    train_pool = torch.from_numpy(np.setdiff1d(ok, eval_idx.cpu().numpy())).to(dev)
    print(f"{len(train_pool)} train starts, {len(eval_idx)} held-out eval starts",
          flush=True)

    prior = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(dev)
    prior.set_action_prior(*corpus_prior(bank["a"], cfg.action_dim))
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    q = QNet(cfg.latent_dim, cfg.history, cfg.action_ticks).to(dev)
    opt = torch.optim.AdamW(q.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020)
    off = torch.arange(cfg.history, device=dev) - (cfg.history - 1)

    @torch.no_grad()
    def evaluate():
        out = {}
        for tag, use_q in (("prior", False), ("qpolicy", True)):
            ds, ts = [], []
            for side_val in (0, 1):
                for s in range(0, len(eval_idx), 1024):
                    base = eval_idx[s : s + 1024]
                    zc = Z[base[:, None] + off[None, :]].float()
                    sd = torch.full((len(base),), side_val, dtype=torch.long,
                                    device=dev)
                    dd, tt = rollout_q(wm, head, q if use_q else None, prior, zc,
                                       sd, cfg, rcfg, a.eval_horizon,
                                       a.candidates, dev, greedy=use_q)
                    ds.append(dd.cpu()); ts.append(tt.cpu())
            dd, tt = torch.cat(ds), torch.cat(ts)
            out[tag] = {"dealt": float(dd.mean()), "taken": float(tt.mean()),
                        "net": float((dd + tt).mean())}
        out["gain_net"] = out["qpolicy"]["net"] - out["prior"]["net"]
        out["gain_dealt"] = out["qpolicy"]["dealt"] - out["prior"]["dealt"]
        return out

    print(f"{'step':>7} {'loss':>9} {'prior dealt':>12} {'q dealt':>9} "
          f"{'q net':>9} {'gain net':>9}", flush=True)
    hist = []
    for step in range(1, a.steps + 1):
        j = torch.randint(0, len(train_pool), (a.batch,), device=dev)
        base = train_pool[j]
        z_win = Z[base[:, None] + off[None, :]].float()
        a_win = A[base[:, None] + off[None, :]][:, :-1].float()
        side = torch.randint(0, 2, (a.batch,), device=dev)
        with torch.no_grad():
            mine = prior(z_win, side, sample=True).actions
            theirs = prior(z_win, 1 - side, sample=True).actions
            y, _, _ = step_damage(wm, head, z_win, a_win, mine, theirs, side,
                                  cfg, rcfg)
        loss = nn.functional.mse_loss(q(z_win, mine), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()

        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            ev.update({"step": step, "loss": float(loss)})
            hist.append(ev)
            (a.out / "log.json").write_text(json.dumps(hist, indent=2))
            torch.save({"q": q.state_dict(), "cfg": cfg, "step": step,
                        "eval": ev}, a.out / "q.pt")
            print(f"{step:>7} {float(loss):9.6f} {ev['prior']['dealt']:12.4f} "
                  f"{ev['qpolicy']['dealt']:9.4f} {ev['qpolicy']['net']:+9.5f} "
                  f"{ev['gain_net']:+9.5f}", flush=True)

    best = max(hist, key=lambda r: r["gain_net"])
    print(f"\nbest gain_net {best['gain_net']:+.5f} at step {best['step']}")
    print(f"  q policy deals {best['qpolicy']['dealt']:.4f} vs prior "
          f"{best['prior']['dealt']:.4f}")
    print(f"  human, same units, through the world model: 0.0255")
    print(f"  best GRPO policy reached: 0.0085")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
