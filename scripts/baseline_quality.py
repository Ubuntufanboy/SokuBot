"""How much return variance does a critic remove, against a group baseline?

    python -m scripts.baseline_quality --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --probe ~/gate_base/reward_probe.npz --bank ~/bank_hud.npz \
        --critic ~/ac_A/policy_best.pt

WHY THIS QUESTION DECIDES PHASE 2
---------------------------------
A policy gradient consumes `advantage = return - baseline`. Everything the
baseline fails to remove is noise the gradient has to average away, and here the
ratio of signal to noise is known to be brutal: `docs/HANDOFF.md` §3 measures
action-driven variance at **3.2% of across-state variance**. Which start state a
rollout happens to draw matters roughly thirty times more than what the policy
does in it.

The two baselines remove that differently, and the difference is not a detail:

  **Group baseline (GRPO).** Roll the same start `G` times and subtract the group
  mean. Across-state variance is removed *exactly*, by construction, because
  every rollout in the group shares the state. What remains is exactly the
  action-driven part -- the signal.

  **Learned critic (this plan's Phase 2).** Subtract `v(s)`. Across-state
  variance is removed only as well as `v` predicts it. Whatever `v` misses stays
  in the advantage as state luck, mislabelled as credit for the action.

The plan justified the swap on throughput: a critic frees the factor of `G` that
GRPO spent on baseline estimation, buying `G` times more distinct starts. That
argument is only sound if the critic's residual noise is small compared with the
3.2% signal. If `v` explains, say, 60% of the state variance, the leftover 40% is
still more than ten times the signal, and the extra starts are bought at a far
worse exchange rate than they cost.

So this measures, on one batch of real starts, three numbers:

  var_total    variance of the lambda-return across all rollouts
  var_within   variance within groups sharing a start  <- what GRPO leaves
  var_resid    variance of (lambda_return - v(s))      <- what the critic leaves

and reports the implied signal-to-noise of each advantage. `var_within` is both
the group baseline's residual *and* an estimate of the action-driven signal
itself, which is what makes the comparison interpretable rather than relative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.model.loading import load_world_model
from sokubot.probe import LinearProbe
from sokubot.rl.ac import ACConfig
from sokubot.rl.critic import SokuCritic, continuation, lambda_returns
from sokubot.rl.grpo import ImaginedArena, PolicyOpponent, ProbeHead
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.reward import RewardConfig
from scripts.eval_policy import RCFG, build_reference
from scripts.train_grpo import model_fingerprint, valid_starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--critic", type=Path, default=None,
                    help="a checkpoint from train_ac holding a trained critic. "
                         "Without it only the group numbers are reported.")
    ap.add_argument("--policy", type=Path, default=None,
                    help="policy to roll under; defaults to the corpus prior")
    ap.add_argument("--starts", type=int, default=256)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("baseline_quality.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    wm, cfg, _ = load_world_model(a.wm, a.device)
    fp = model_fingerprint(wm)
    d = np.load(a.probe.expanduser(), allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    b = np.load(a.bank.expanduser())
    Z, A, E = b["z"], b["a"], b["ep"]
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    starts = valid_starts(E, cfg.history, a.horizon)
    rng = np.random.default_rng(a.seed)
    idx0 = torch.from_numpy(rng.choice(starts, size=a.starts)).to(a.device)
    # Every rollout in a group shares one start AND one side, which is what makes
    # the group mean an exact conditional baseline rather than an approximate one.
    idx = idx0.repeat_interleave(a.group)
    side = torch.randint(0, 2, (a.starts,), device=a.device).repeat_interleave(a.group)

    acfg = ACConfig(horizon=a.horizon, reward=RCFG)
    arena = ImaginedArena(wm, ProbeHead(probe).to(a.device), acfg, cfg.history,
                          cfg.action_ticks)
    policy = build_reference(A, cfg, a.device)
    if a.policy is not None:
        blob = torch.load(a.policy.expanduser(), map_location=a.device,
                          weights_only=False)
        policy = SokuPolicy(cfg.latent_dim, cfg.history,
                            cfg.action_ticks).to(a.device)
        policy.load_state_dict(blob["policy"])
        policy.eval()
    reference = build_reference(A, cfg, a.device)

    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Zt[idx[:, None] + off[None, :]].float()
    a_hist = At[idx[:, None] + off[None, :-1]].float()
    with torch.no_grad():
        tr = arena.rollout(z_ctx, a_hist, side, policy, PolicyOpponent(reference))

    B, T = tr["reward"].shape
    G, S = a.group, a.starts

    critic = None
    if a.critic is not None:
        blob = torch.load(a.critic.expanduser(), map_location=a.device,
                          weights_only=False)
        ccfg = blob["acfg"].critic
        critic = SokuCritic(cfg.latent_dim, cfg.history, ccfg).to(a.device)
        critic.load_state_dict(blob["critic"])
        critic.eval()

    with torch.no_grad():
        if critic is not None:
            flat = tr["obs"].reshape(B * T, cfg.history, cfg.latent_dim)
            fside = side[:, None].expand(B, T).reshape(-1)
            v = critic(flat, fside).view(B, T)
            v_boot = critic(tr["obs_boot"], side)
        else:
            v = torch.zeros(B, T, device=a.device)
            v_boot = torch.zeros(B, device=a.device)
        values = torch.cat([v, v_boot[:, None]], dim=1)
        cont = continuation(tr["terminal"])
        lam_ret = lambda_returns(tr["reward"], values, cont, a.gamma, a.lam)

    # All variances are taken over the same objects: one number per rollout,
    # summed over the horizon, so "state" and "action" mean the same thing in
    # each. Taking them per step would let a policy that terminates early look
    # like it had lower variance.
    R = lam_ret.sum(dim=1)                       # [B]
    Rg = R.view(S, G)
    var_total = float(R.var(unbiased=False))
    var_within = float(Rg.var(dim=1, unbiased=False).mean())
    var_between = float(Rg.mean(dim=1).var(unbiased=False))
    out = {"wm": str(a.wm), "fingerprint": fp, "starts": S, "group": G,
           "horizon": a.horizon, "var_total": var_total,
           "var_within_group": var_within, "var_between_states": var_between,
           "action_share": var_within / max(var_total, 1e-12)}

    print(f"{S} starts x {G} rollouts, horizon {a.horizon}\n")
    print(f"var(lambda-return) total          {var_total:.6e}")
    print(f"  within a group (same start)     {var_within:.6e}   "
          f"<- action-driven; this IS the signal")
    print(f"  between starts                  {var_between:.6e}")
    print(f"  action share of total           {out['action_share']:.2%}"
          f"   (HANDOFF measured 3.2%)")

    # ---- is the critic's bootstrap signal or noise? ----
    # At horizon 4 the accumulated reward is ~0.009 while v(s_H) is ~0.07, so the
    # lambda-return is roughly 86% bootstrap. Once the group mean is subtracted
    # what survives is the *within-group* differences, and those now come from
    # two places: four steps of real reward, and the critic's opinion of four
    # different end states. If the second is approximation error rather than
    # judgement, the bootstrap is diluting exactly the signal it was added to
    # extend. The correlation against the plain discounted return says which.
    with torch.no_grad():
        plain = torch.zeros_like(tr["reward"])
        run = torch.zeros(B, device=a.device)
        for t in range(T - 1, -1, -1):
            run = tr["reward"][:, t] + a.gamma * run * tr["alive"][:, t]
            plain[:, t] = run
    Pg = plain.sum(dim=1).view(S, G)
    Pc = (Pg - Pg.mean(dim=1, keepdim=True)).flatten()
    Lc = (Rg - Rg.mean(dim=1, keepdim=True)).flatten()
    # With no critic v is identically 0, so the lambda-return *is* the plain
    # return and the correlation would be a tautological 1.0.
    corr = (float(torch.corrcoef(torch.stack([Pc, Lc]))[0, 1])
            if critic is not None else float("nan"))
    out["within_group_var_plain_return"] = float(Pg.var(dim=1, unbiased=False).mean())
    out["bootstrap_alignment"] = corr
    print(f"\n--- what the critic's bootstrap does inside a group ---")
    print(f"within-group var, plain discounted return  "
          f"{out['within_group_var_plain_return']:.6e}")
    print(f"within-group var, lambda-return            {var_within:.6e}")
    if critic is not None:
        print(f"correlation between the two                {corr:+.3f}"
              f"   (1.0 = the bootstrap adds nothing but scale)")
        if corr < 0.5:
            print(f"  LOW. Most of what the group baseline now sees is the "
                  f"critic's opinion of\n  the end state, not four steps of "
                  f"reward -- and at R^2 0.385 that opinion is\n  mostly "
                  f"approximation error. The bootstrap is diluting the signal.")

    print(f"\n--- what each baseline leaves in the advantage ---")
    print(f"group mean   residual var {var_within:.6e}  SNR 1.00  (exact by "
          f"construction)")
    if critic is not None:
        # Advantage as train_ac forms it, summed to match R's units.
        adv = (lam_ret - v).sum(dim=1)
        var_resid = float(adv.var(unbiased=False))
        # How much of the *state* variation the critic captured.
        vg = v[:, 0].view(S, G).mean(dim=1)
        state_mean = Rg.mean(dim=1)
        ss_res = float(((state_mean - state_mean.mean())
                        - (vg - vg.mean())).pow(2).mean())
        r2 = 1.0 - ss_res / max(float(state_mean.var(unbiased=False)), 1e-12)
        snr = var_within / max(var_resid, 1e-12)
        out.update({"var_after_critic": var_resid, "critic_state_r2": r2,
                    "critic_snr_vs_group": snr})
        print(f"critic v(s)  residual var {var_resid:.6e}  SNR {snr:.3f}")
        print(f"             critic R^2 on the across-state mean return {r2:+.3f}")
        print("\n" + "=" * 68)
        if snr < 0.5:
            print(f"The critic's advantage carries {1/max(snr,1e-9):.1f}x more "
                  f"noise per sample than the group baseline's.\n"
                  f"Phase 2 traded an exact conditional baseline for an "
                  f"approximate one in\nthe one setting where that is worst: the "
                  f"signal being isolated is {out['action_share']:.1%} of\nthe "
                  f"variance being removed. The extra starts a critic buys do not "
                  f"pay for\nthat. Keep a group AND use the critic for the "
                  f"bootstrap past the horizon.")
        else:
            print(f"The critic is a competitive baseline (SNR {snr:.2f} vs the "
                  f"group's 1.00).")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
