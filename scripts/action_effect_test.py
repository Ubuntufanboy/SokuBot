"""Is the imagined world's action-dependence mechanical, or state-specific noise?

    python -m scripts.action_effect_test --ckpt /root/ckpt_cf/best.pt

THE PUZZLE
----------
GRPO does not move. Across every configuration tried -- learning rate, horizon,
advantage scaling, KL anchor, and after the encoder/predictor probe bug was
fixed -- `net vs frozen init` stays at +-0.00004. At step 600 the policy's
*behaviour* had changed materially (press rate 0.099 -> 0.116, attack 0.063 ->
0.074, KL from the reference 0.31) while the damage outcome was identical to
four decimal places.

That is strange, because the model is not indifferent. Within a group of
rollouts from one start, returns spread by 0.0140 against a mean of 0.0969 --
14%. Something about the actions is changing the outcome. It just yields no
direction the optimiser can climb.

Two hypotheses fit those facts, and they call for opposite work:

  MECHANICAL   The action->outcome map is a real, transferable rule ("this
               button here, at this range, deals damage"). Then the signal is
               learnable and GRPO's failure is an optimisation or credit
               assignment problem: fix the optimiser.

  CHAOTIC      Which action is better varies arbitrarily from state to state
               and encodes no rule. Then the within-group spread is
               state-specific noise, every gradient step points somewhere
               unrelated to the last, and they cancel -- exactly the flat
               curve observed. No optimiser reaches this. It is what
               MSE-trained deterministic prediction produces on a stochastic
               game, and the fix is a stochastic latent.

THE TEST
--------
The rollout is deterministic: one start plus one joint action sequence gives
exactly one return. So `f(start, actions) -> outcome` is a *function*, and the
only question is whether it is a smooth one that generalises or a hash. Fit a
regressor to it and check where it stops working:

  fit on the same starts, same samples   capacity check. If this fails, the
                                         regressor is too weak and nothing
                                         below means anything.
  held-out samples, same starts          can it interpolate in action space at
                                         a state it has seen?
  held-out starts                        the money number. Does the rule
                                         transfer to a state it has never
                                         seen? This is what GRPO needs.

Scores are reported *within* each start, because across starts the situation
dominates and would swamp the action effect -- the same reason GRPO centres
advantages within a group. Within-start correlation on held-out starts is
precisely the quantity a policy gradient consumes.

WHAT ELSE IS MEASURED, AND WHY
------------------------------
Predicting damage runs the signal through the probe, a threshold and a
difference, any of which could destroy it on its own. So the same features are
also asked to predict random projections of the final *latent*. That separates
three different worlds:

  latent predictable, damage not    the dynamics are fine and the reward
                                    readout is throwing the signal away
  latent predictable, health not    actions move the latent, but not along the
                                    direction health is read from
  latent not predictable            the dynamics themselves are chaotic

Alongside every target is its *leverage*: the within-start spread over the
across-start spread. A target the actions barely move at all is a different
finding from one they move unpredictably, and the correlation alone cannot
tell them apart.

Both players' actions are features. The rollout is driven by the joint action,
so with only one side's buttons the opponent's sampled behaviour is
unexplainable variance and every score would be capped for a reason that has
nothing to do with the question. The one-sided version is reported too, since
that is the marginal effect a policy can actually exploit.

THE CONTROL THAT MAKES A NULL RESULT MEAN ANYTHING
--------------------------------------------------
"The regressor could not predict it" is only evidence if the regressor could
have. Two synthetic targets ride along, computed from the same features by
construction:

  ctl_lin     how much this rollout attacked. A pure function of the actions,
              and the features contain them, so anything below ~1.0 means the
              features or the plumbing are broken.
  ctl_inter   attacking, scaled by the opponent's health at the start, plus
              moving, scaled by own spirit. This has the exact shape a real
              mechanic would -- *the value of an action depends on the state* --
              and it is the one that matters. If it transfers to unseen starts
              and the return does not, the null on the return is a fact about
              the world model. If it does not transfer either, the fitting
              budget is too small and nothing here is conclusive.

Starts are split three ways rather than two, and the regressor is selected on
the validation starts by the same within-start correlation the test reports.
That is deliberately the most generous setup available to the "mechanical"
hypothesis: if a transferable rule exists at all, this finds it.
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


def corpus_prior(A: np.ndarray, action_dim: int):
    p1 = A.astype(np.float32).reshape(-1, action_dim)[:, :10]
    lr = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                   float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                   float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    return lr / lr.sum(), ud / ud.sum(), p1[:, 4:10].mean(0)


@torch.no_grad()
def collect(wm, head, policy, Z, idx, cfg, T, G, side_val, chunk, dev):
    """Roll `G` sampled action sequences from each start and record everything.

    Mirrors `ImaginedArena.rollout` exactly -- T+1 predictor steps, no encoder
    latent in the state sequence -- except that actions are not jittered, so
    what is recorded as a feature is byte-for-byte what drove the dynamics.
    """
    off = torch.arange(cfg.history, device=dev) - (cfg.history - 1)
    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    proj = torch.randn(cfg.latent_dim, 16, generator=gen).to(dev)
    proj /= proj.norm(dim=0, keepdim=True)

    out = {k: [] for k in ("zctx", "joint", "mine", "ret", "net", "dealt",
                           "taken", "hp_me", "hp_them", "ctl_lin", "ctl_inter",
                           "zproj", "start")}
    rep_all = idx.repeat_interleave(G)
    for s in range(0, len(rep_all), chunk):
        rep = rep_all[s : s + chunk]
        z_win = Z[rep[:, None] + off[None, :]].float()
        zctx0 = z_win.clone()
        B = len(rep)
        side = torch.full((B,), side_val, dtype=torch.long, device=dev)
        a_win = torch.zeros(B, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                            device=dev)
        zs, joints = [], []
        for _ in range(T + 1):
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

        ZT = torch.stack(zs, dim=1)                       # [B,T+1,latent]
        states = head(ZT)                                 # [B,T+1,K]
        J = torch.stack(joints, dim=1)                    # [B,T+1,ticks,20]
        # T+1 states against the last T actions, as in ImaginedArena: joints[k]
        # carries zs[k-1] to zs[k], so joints[0] made the baseline state and is
        # a feature but not a reward-bearing action.
        reward, alive, terms = compute_rewards(states, J[:, 1:], side, rcfg)

        # Discounted return-to-go at t=0: the scalar GRPO's advantage is built
        # from, not a convenience sum.
        ret = torch.zeros(B, device=dev)
        for t in range(reward.shape[1] - 1, -1, -1):
            ret = reward[:, t] + 0.99 * ret * alive[:, t]

        dealt, taken = terms["dealt"].sum(1), terms["taken"].sum(1)
        me, them = (0, 1) if side_val == 0 else (1, 0)
        hp_me, hp_them = states[:, -1, me], states[:, -1, them]

        # Synthetic controls, from the same actions and the same start state.
        blk = J[..., :10] if side_val == 0 else J[..., 10:]
        atk = blk[..., 4:8].mean(dim=(1, 2, 3))
        mov = blk[..., 0:4].mean(dim=(1, 2, 3))
        out["ctl_lin"].append(atk)
        out["ctl_inter"].append(atk * states[:, 0, them] + mov * states[:, 0, 2 + me])

        out["zctx"].append(zctx0.reshape(B, -1))
        out["joint"].append(J.reshape(B, -1))
        out["mine"].append(J[..., :10].reshape(B, -1) if side_val == 0
                           else J[..., 10:].reshape(B, -1))
        out["ret"].append(ret)
        out["net"].append(dealt + taken)
        out["dealt"].append(dealt)
        out["taken"].append(taken)
        out["hp_me"].append(hp_me)
        out["hp_them"].append(hp_them)
        out["zproj"].append(ZT[:, -1] @ proj)
        out["start"].append(rep)
    return {k: torch.cat(v) for k, v in out.items()}


def within_stats(pred, true, sid):
    """Correlation, R^2 and pairwise ordering accuracy, computed within start."""
    uniq, inv = torch.unique(sid, return_inverse=True)
    n = len(uniq)
    cnt = torch.zeros(n, device=pred.device).index_add_(0, inv, torch.ones_like(pred))
    mp = torch.zeros(n, device=pred.device).index_add_(0, inv, pred) / cnt
    mt = torch.zeros(n, device=pred.device).index_add_(0, inv, true) / cnt
    dp, dt = pred - mp[inv], true - mt[inv]
    spp = torch.zeros(n, device=pred.device).index_add_(0, inv, dp * dp)
    stt = torch.zeros(n, device=pred.device).index_add_(0, inv, dt * dt)
    spt = torch.zeros(n, device=pred.device).index_add_(0, inv, dp * dt)
    ok = (spp > 1e-20) & (stt > 1e-20)
    corr = float((spt[ok] / (spp[ok] * stt[ok]).sqrt()).mean()) if ok.any() else float("nan")
    r2 = float(1.0 - ((dp - dt) ** 2).sum() / (dt ** 2).sum().clamp_min(1e-20))
    # Pairwise ordering within a start, on a subsample of pairs.
    g = torch.randperm(len(pred), device=pred.device)
    a, b = g[: len(g) // 2], g[len(g) // 2 :]
    m = inv[a] == inv[b]
    acc = float(((dp[a] - dp[b]).sign() == (dt[a] - dt[b]).sign())[m].float().mean()) \
        if m.any() else float("nan")
    return {"corr": corr, "r2": r2, "pair_acc": acc}


def leverage(true, sid):
    uniq, inv = torch.unique(sid, return_inverse=True)
    n = len(uniq)
    cnt = torch.zeros(n, device=true.device).index_add_(0, inv, torch.ones_like(true))
    mt = torch.zeros(n, device=true.device).index_add_(0, inv, true) / cnt
    within = float((true - mt[inv]).std())
    return {"within_std": within, "across_std": float(mt.std()),
            "ratio": within / (float(mt.std()) + 1e-12)}


def fit_mlp(Xtr, Ytr, Xva, Yva, sva, steps, dev, hidden=1024, lr=1e-3,
            bs=1024, drop=0.1, seed=0, key=0):
    """Fit, and keep the checkpoint that scores best on the *validation starts*.

    Selection uses within-start correlation on `key` -- the same statistic the
    test reports -- so the mechanical hypothesis gets every advantage. A model
    that merely memorises the training rows is discarded here rather than
    dragging the held-out number down.
    """
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(drop),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
                        nn.Linear(hidden, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_sd = -2.0, None
    every = max(1, steps // 20)
    for i in range(steps):
        net.train()
        j = torch.randint(0, len(Xtr), (bs,), device=dev)
        loss = nn.functional.mse_loss(net(Xtr[j]), Ytr[j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if (i + 1) % every == 0:
            net.eval()
            with torch.no_grad():
                c = within_stats(net(Xva)[:, key], Yva[:, key], sva)["corr"]
            if c > best:
                best = c
                best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if (i + 1) % (every * 5) == 0:
                print(f"      step {i+1}/{steps}  mse {loss.item():.5f}  "
                      f"val corr {c:+.4f}  best {best:+.4f}", flush=True)
    if best_sd is not None:
        net.load_state_dict(best_sd)
    return net.eval(), best


def ridge_select(Xtr, Ytr, Xva, Yva, sva, alphas, key=0):
    """Ridge over a sweep of alpha, selected on the validation starts.

    The Gram matrix does not depend on alpha, so it is formed once; only the
    small solve repeats. At 130k rows and 1936 features the design matrix alone
    is 2 GB in double, and rebuilding it seven times is the whole cost.
    """
    Xb = torch.cat([Xtr, torch.ones_like(Xtr[:, :1])], 1).double()
    G = Xb.T @ Xb
    RHS = Xb.T @ Ytr.double()
    del Xb
    Vb = torch.cat([Xva, torch.ones_like(Xva[:, :1])], 1).double()
    eye = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    best, best_W, best_a = -2.0, None, None
    for al in alphas:
        W = torch.linalg.solve(G + al * eye, RHS)
        c = within_stats((Vb @ W)[:, key].float(), Yva[:, key], sva)["corr"]
        print(f"      alpha {al:>9.4g}  val corr {c:+.4f}", flush=True)
        if c > best:
            best, best_W, best_a = c, W, al
    return (lambda Z: (torch.cat([Z, torch.ones_like(Z[:, :1])], 1).double()
                       @ best_W).float()), best_a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_best/reward_probe.npz"))
    ap.add_argument("--bank", type=Path, default=Path("/root/bank_best.npz"))
    ap.add_argument("--starts", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=32, help="rollouts per start")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--held-samples", type=int, default=8,
                    help="of --samples, how many are held out at seen starts")
    ap.add_argument("--held-starts", type=float, default=0.25)
    ap.add_argument("--val-starts", type=float, default=0.15,
                    help="starts reserved for model selection, disjoint from "
                         "both the fit set and the held-out test set")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/action_effect.json"))
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
    ok = ok[(ok >= lo) & (ok < len(ep) - a.horizon - 2)]
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(ok, size=a.starts, replace=False)).to(a.device)

    A_bank = torch.from_numpy(bank["a"]).to(a.device)
    sk = assert_predictor_sane(wm, Z, A_bank, ep, cfg, what=str(a.ckpt))
    print(f"one-step skill of {a.ckpt}: {sk:+.4f}")
    results_skill = sk

    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
    policy.set_action_prior(*corpus_prior(bank["a"], cfg.action_dim))
    policy.eval()

    print(f"rolling {a.starts} starts x {a.samples} samples, horizon {a.horizon}",
          flush=True)
    D = collect(wm, head, policy, Z, idx, cfg, a.horizon, a.samples, 0,
                a.chunk, a.device)
    N = len(D["ret"])
    print(f"{N} rollouts collected\n", flush=True)

    # ---- targets -----------------------------------------------------------
    scalar = ["ret", "net", "dealt", "taken", "hp_me", "hp_them",
              "ctl_lin", "ctl_inter"]
    Y = torch.cat([torch.stack([D[k] for k in scalar], 1), D["zproj"]], 1)
    names = scalar + [f"z{i}" for i in range(D["zproj"].shape[1])]
    sid = D["start"]

    print(f"{'target':10} {'within std':>11} {'across std':>11} {'leverage':>9}")
    print("-" * 45)
    lev = {}
    for c, nm in enumerate(names):
        lv = leverage(Y[:, c], sid)
        lev[nm] = lv
        if c < len(scalar):
            print(f"{nm:10} {lv['within_std']:11.5f} {lv['across_std']:11.5f} "
                  f"{lv['ratio']:9.4f}")
    zr = np.mean([lev[f"z{i}"]["ratio"] for i in range(D["zproj"].shape[1])])
    print(f"{'z1..z15':10} {'':11} {'':11} {zr:9.4f}  (mean of 16 projections)")
    print()

    # ---- splits: by start, and by sample within the training starts ---------
    uniq = torch.unique(sid)
    torch.manual_seed(0)
    perm = torch.randperm(len(uniq), device=a.device)
    n_te = int(len(uniq) * a.held_starts)
    n_va = max(1, int(len(uniq) * a.val_starts))
    te_start = torch.isin(sid, uniq[perm[:n_te]])
    va_start = torch.isin(sid, uniq[perm[n_te : n_te + n_va]])
    # Within each start the samples arrive in a fixed order, so the last
    # `held_samples` of every start form a clean held-out slice.
    rank = torch.arange(N, device=a.device) % a.samples
    is_hold_sample = rank >= (a.samples - a.held_samples)

    fit_m = ~te_start & ~va_start & ~is_hold_sample
    te_sample = ~te_start & ~va_start & is_hold_sample
    print(f"fit {int(fit_m.sum())}  val starts {int(va_start.sum())}  "
          f"held-out samples {int(te_sample.sum())}  "
          f"held-out starts {int(te_start.sum())}\n")

    ymu, ysd = Y[fit_m].mean(0), Y[fit_m].std(0).clamp_min(1e-8)
    Yn = (Y - ymu) / ysd

    results = {"leverage": lev, "n_rollouts": N, "horizon": a.horizon,
               "starts": a.starts, "samples": a.samples,
               "ckpt": str(a.ckpt), "one_step_skill": results_skill, "fits": {}}

    for tag, cols in (("joint", ["zctx", "joint"]), ("mine-only", ["zctx", "mine"])):
        X = torch.cat([D[c] for c in cols], 1)
        xmu, xsd = X[fit_m].mean(0), X[fit_m].std(0).clamp_min(1e-6)
        Xn = (X - xmu) / xsd
        print(f"=== features: {tag}  ({Xn.shape[1]} dims) ===", flush=True)
        net, vbest = fit_mlp(Xn[fit_m], Yn[fit_m], Xn[va_start], Yn[va_start],
                             sid[va_start], a.steps, a.device)
        print(f"    mlp selected at val corr {vbest:+.4f}")
        alphas = [10.0 ** k for k in range(0, 7)]
        lin, abest = ridge_select(Xn[fit_m], Yn[fit_m], Xn[va_start],
                                  Yn[va_start], sid[va_start], alphas)
        print(f"    ridge selected alpha {abest:g}")
        with torch.no_grad():
            Pm, Pl = net(Xn), lin(Xn)

        for mtag, model_pred in (("mlp", Pm), ("ridge", Pl)):
            print(f"\n  {mtag}")
            print(f"  {'target':10} {'fit corr':>10} {'held smpl':>11} "
                  f"{'HELD START':>11} {'start r2':>10} {'start pair':>11}")
            print("  " + "-" * 67)
            for c, nm in enumerate(names):
                if c >= len(scalar) and nm != "z0":
                    continue
                a1 = within_stats(model_pred[fit_m, c], Yn[fit_m, c], sid[fit_m])
                a2 = within_stats(model_pred[te_sample, c], Yn[te_sample, c],
                                  sid[te_sample])
                a3 = within_stats(model_pred[te_start, c], Yn[te_start, c],
                                  sid[te_start])
                results["fits"][f"{tag}/{mtag}/{nm}"] = {
                    "fit": a1, "held_sample": a2, "held_start": a3}
                print(f"  {nm:10} {a1['corr']:+10.4f} {a2['corr']:+11.4f} "
                      f"{a3['corr']:+11.4f} {a3['r2']:+10.4f} {a3['pair_acc']:11.4f}")
            # The latent projections summarised, since 16 rows of them is noise.
            zc = np.mean([[within_stats(model_pred[m, len(scalar) + i],
                                        Yn[m, len(scalar) + i], sid[m])["corr"]
                           for i in range(D["zproj"].shape[1])]
                          for m in (fit_m, te_sample, te_start)], axis=1)
            results["fits"][f"{tag}/{mtag}/zproj_mean"] = {
                "fit": float(zc[0]), "held_sample": float(zc[1]),
                "held_start": float(zc[2])}
            print(f"  {'zproj x16':10} {zc[0]:+10.4f} {zc[1]:+11.4f} "
                  f"{zc[2]:+11.4f}     (mean over 16 projections)")
        print()

    # ---- verdict -----------------------------------------------------------
    key = results["fits"]["joint/mlp/ret"]
    zk = results["fits"]["joint/mlp/zproj_mean"]
    ctl = results["fits"]["joint/mlp/ctl_inter"]
    lin_ctl = results["fits"]["joint/mlp/ctl_lin"]
    print("=" * 72)
    for label, tr_, hs, ht in (
            ("ctl_lin  ", lin_ctl["fit"]["corr"], lin_ctl["held_sample"]["corr"],
             lin_ctl["held_start"]["corr"]),
            ("ctl_inter", ctl["fit"]["corr"], ctl["held_sample"]["corr"],
             ctl["held_start"]["corr"]),
            ("return   ", key["fit"]["corr"], key["held_sample"]["corr"],
             key["held_start"]["corr"]),
            ("latent   ", zk["fit"], zk["held_sample"], zk["held_start"])):
        print(f"{label} : fit {tr_:+.4f}  held sample {hs:+.4f}  "
              f"HELD START {ht:+.4f}")
    print()
    if ctl["held_start"]["corr"] < 0.5:
        v = ("the synthetic state-modulated control does not transfer either, "
             f"at {ctl['held_start']['corr']:+.4f} -- the fitting budget is too "
             "small to detect a mechanic of this size, so the null on the "
             "return is not yet evidence of anything")
    elif key["held_start"]["corr"] > 0.25:
        v = ("the action->return rule transfers to unseen starts, so the signal "
             "is real and learnable; GRPO's failure is optimisation or credit "
             "assignment, not the world model")
    elif key["held_sample"]["corr"] > 0.25:
        v = ("the rule fits within a start but does not transfer to new ones: "
             "the action-dependence is state-specific and encodes no rule a "
             "policy could carry between situations -- chaos, not mechanics")
    elif zk["held_start"] > 0.25:
        v = ("actions move the latent predictably but the return is "
             "unpredictable from them: the dynamics are learnable and the "
             "reward readout is destroying the signal")
    else:
        v = ("neither the return nor the latent is predictable from the actions "
             "at an unseen start: the imagined dynamics are chaotic in the "
             "action direction, which is what MSE-trained deterministic "
             "prediction gives on a stochastic game")
    print(f"VERDICT: {v}")
    a.out.write_text(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
