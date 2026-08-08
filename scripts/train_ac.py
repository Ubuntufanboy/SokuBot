"""Train the Soku policy with a lambda-return actor-critic inside imagination.

    python -m scripts.train_ac --wm ~/art/best_bnfix.pt \
        --probe ~/art/reward_probe.npz --out ~/ac --steps 40000

This is `train_grpo.py` with the group baseline replaced by a critic. Everything
that was hard-won stays: bounded logits (the one change that stopped four
identical entropy collapses), the entropy floor with its anti-wind-up bounds, the
KL anchor to a corpus-prior reference, batch-scaled advantages, and the
side-swapped evaluation that cancels the probe's mirrored chair bias.

WHAT CHANGES, AND WHY EACH
--------------------------
**The baseline.** GRPO rolled each start `group_size` times and subtracted the
group mean, so fifteen sixteenths of the predictor budget bought variance
reduction rather than start-state coverage. A critic amortises the baseline over
the batch, and the freed budget goes into distinct starts -- which matters here
specifically, because low-health starts are rare in the corpus and their scarcity
is the leading suspect for the agent giving up when hurt.

**The horizon, from 4 to 16.** `--horizon 4` was set by `action_effect_test.py`:
beyond about 0.27 s the action->outcome correlation a policy gradient consumes
falls away. That bound applies to a *return that has to cover the whole rollout*.
A lambda-return does not -- imagination only has to be right over its own length,
and `v(s_H)` carries the rest. This is the mechanism by which the world model's
trustworthy horizon stops being a ceiling on credit assignment, and it is the
single change here most likely to move the number.

**Both chairs.** The predictor is conditioned on both players' twenty buttons, so
one imagined rollout already is a two-player game and the opponent's half was
being discarded. Collecting it doubles the experience per predictor step, at zero
extra predictor steps -- but only when the opponent *is* the current policy, so
it is skipped on snapshot and replay steps rather than silently used off-policy.

COMPARABILITY
-------------
`evaluate()` is the same construction `train_grpo.py` uses: the same frozen
prior-initialised reference, the same fixed eval starts (seed 12345), the same
side-swap, and the same per-step units. The number it prints is therefore
directly comparable to the +0.00215 on record in `docs/HANDOFF.md`.

That comparability rests on the two scripts agreeing, which is an assumption, so
the GRPO baseline is re-run from `train_grpo.py` itself rather than reproduced
here -- this script has no group baseline to reproduce it with. `tests/test_ac.py`
pins the shared parts against drift.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.probe import LinearProbe
from sokubot.model.loading import load_world_model
from sokubot.rl.ac import (ACConfig, advantages_from_returns,
                           group_centred_advantages)
from sokubot.rl.critic import (SokuCritic, TargetCritic, continuation,
                               lambda_returns)
from sokubot.rl.grpo import (EntropyFloor, GRPOConfig, ImaginedArena,
                             PolicyOpponent, ProbeHead, ReplayOpponent,
                             SnapshotPool, grpo_loss)
from sokubot.rl.ood import LatentOOD
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.reward import RewardConfig
from scripts.eval_policy import score as score_policy
from scripts.train_grpo import build_bank, model_fingerprint, valid_starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bank", type=Path, default=None,
                    help="reuse a bank built by train_grpo; defaults to <out>/bank.npz")
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--starts", type=int, default=256,
                    help="distinct start states per batch. Total rollouts are "
                         "--starts x --group.")
    ap.add_argument("--group", type=int, default=8,
                    help="rollouts sharing one start and side. The advantage is "
                         "then the lambda-return centred within the group rather "
                         "than against v(s), and the critic's job narrows to the "
                         "bootstrap inside the return. --group 1 restores the "
                         "critic-as-baseline form, which scripts/"
                         "baseline_quality.py measures at 21.8x the noise per "
                         "sample because only 0.12% of return variance is "
                         "action-driven.")
    ap.add_argument("--bank-replays", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.98)
    ap.add_argument("--kl-ref-coef", type=float, default=None)
    ap.add_argument("--entropy-floor-frac", type=float, default=0.8)
    ap.add_argument("--replay-share", type=float, default=0.0)
    ap.add_argument("--no-two-sided", action="store_true",
                    help="ignore the opponent's half of self-play rollouts")
    ap.add_argument("--no-ood", action="store_true",
                    help="disable the out-of-distribution guard entirely, which "
                         "makes the rollout byte-identical to the unguarded one")
    ap.add_argument("--ood-penalty", type=float, default=0.0,
                    help="reward penalty per nat of drift outside the band. 0 "
                         "leaves only truncation, which is the stronger and "
                         "less tunable of the two mechanisms")
    ap.add_argument("--ood-quantile", type=float, default=0.99)
    ap.add_argument("--ood-hard-mult", type=float, default=2.0,
                    help="truncate once the score is this many times outside the "
                         "band at either end")
    ap.add_argument("--critic-warmup", type=int, default=200,
                    help="steps during which only the critic learns. A critic at "
                         "initialisation predicts 0 everywhere, so before it has "
                         "seen anything the advantage is just the return and the "
                         "actor takes its largest steps against its worst "
                         "baseline.")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-starts", type=int, default=1024)
    ap.add_argument("--eval-horizon", type=int, default=4,
                    help="horizon the reported `net` is measured over, held "
                         "separate from --horizon on purpose. `net` is damage "
                         "per alive step averaged over a rollout of this length, "
                         "so it changes with the length: GRPO's +0.00215 was "
                         "taken at 4, and reading a horizon-16 number against it "
                         "would compare an instrument change and a policy change "
                         "at once. Training still runs at --horizon.")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--uniform-init", action="store_true")
    ap.add_argument("--ko-source", default="health", choices=("health", "banner"),
                    help="where 'the match ended' comes from. 'health' is the "
                         "probe threshold every recorded number was measured "
                         "with, at precision 0.003. 'banner' reads the KNOCK OUT "
                         "announcement from a probe channel at precision 0.844, "
                         "and needs a probe fitted by scripts.fit_banner_channel.")
    ap.add_argument("--win-magnitude", type=float, default=None,
                    help="override RewardConfig.win/lose to +-this. 5.0 was set "
                         "when the KO detector essentially never fired at "
                         "horizon 4 (alive_frac 0.9999), so the term's size "
                         "barely mattered. With --ko-source banner it fires at "
                         "roughly the true rate, and at 0.93%% of rollouts a "
                         "+-5 payout contributes a standard deviation of ~0.48 "
                         "against a damage signal of ~0.009 -- so the outcome "
                         "term, and any error in detecting it, now dominates "
                         "the gradient. A KO is worth about one health bar, "
                         "which is what 1.0 means in these units.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    wm, cfg, _ = load_world_model(a.wm, a.device)

    d = np.load(a.probe, allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    wm_fp = model_fingerprint(wm)
    probe_fp = str(d["fingerprint"]) if "fingerprint" in d.files else None
    if probe_fp is None:
        print(f"WARNING: {a.probe} predates the fingerprint stamp and cannot be "
              f"checked against this world model ({wm_fp}).", flush=True)
    elif probe_fp != wm_fp:
        raise SystemExit(
            f"reward probe was fit on weights {probe_fp} but --wm {a.wm} is "
            f"{wm_fp}. A probe does not transfer between latent spaces; re-run "
            f"scripts.horizon_ablation --ckpt {a.wm}.")
    print(f"reward probe: alpha {float(d['alpha']):g}, targets {probe.names}",
          flush=True)

    # Identical to train_grpo's, so the comparison is against the same landscape.
    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020, ko_source=a.ko_source)
    if a.win_magnitude is not None:
        rcfg.win, rcfg.lose = a.win_magnitude, -a.win_magnitude
    if a.ko_source == "banner" and "ko_banner" not in probe.names:
        raise SystemExit(
            f"--ko-source banner needs a probe with a ko_banner channel; "
            f"{a.probe} has {probe.names}. Build one with "
            f"scripts.fit_banner_channel.")
    print(f"KO detector: {rcfg.ko_source}"
          + (f" (threshold {rcfg.ko_banner_threshold}, margin {rcfg.ko_margin})"
             if rcfg.ko_source == "banner" else " (probe health threshold)"),
          flush=True)
    acfg = ACConfig(horizon=a.horizon, starts_per_batch=a.starts, lr=a.lr,
                    reward=rcfg, two_sided=not a.no_two_sided)
    acfg.critic.gamma, acfg.critic.lam = a.gamma, a.lam
    acfg.critic.tau, acfg.critic.lr = a.tau, a.critic_lr
    if a.kl_ref_coef is not None:
        acfg.kl_ref_coef = a.kl_ref_coef
    acfg.entropy_floor_frac = a.entropy_floor_frac

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    rng.shuffle(rows)
    bank_path = a.bank or (a.out / "bank.npz")
    Z, A, E = build_bank(rows, manifest, wm, cfg, a.device, a.bank_replays,
                         bank_path)
    # Sized at whichever horizon reaches further. A start valid for the longer
    # one is valid for the shorter, so this keeps training and evaluation drawing
    # from one pool -- otherwise a short eval horizon would silently widen the
    # eval pool and score the policy on start states it never trained near.
    starts = valid_starts(E, cfg.history, max(a.horizon, a.eval_horizon))
    print(f"{len(starts)} valid start states (horizon {a.horizon}, eval "
          f"{a.eval_horizon})", flush=True)
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
    if not a.uniform_init:
        p1 = A.astype(np.float32).reshape(-1, cfg.action_dim)[:, :10]
        lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                         float(p1[:, 2].mean()), float(p1[:, 3].mean())])
        ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                         float(p1[:, 0].mean()), float(p1[:, 1].mean())])
        policy.set_action_prior(lr_p / lr_p.sum(), ud_p / ud_p.sum(),
                                p1[:, 4:10].mean(0))
        with torch.no_grad():
            samp = policy(torch.zeros(512, cfg.history, cfg.latent_dim,
                                      device=a.device),
                          torch.zeros(512, dtype=torch.long, device=a.device))
        print(f"policy prior: press rate {float(samp.actions.mean()):.4f} "
              f"(corpus {float(p1.mean()):.4f}, uniform ~0.44)", flush=True)

    critic = SokuCritic(cfg.latent_dim, cfg.history, acfg.critic).to(a.device)
    target = TargetCritic(critic, acfg.critic.tau)

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.steps * acfg.epochs), eta_min=a.lr * 0.05)

    # Fit the OOD guard on the bank's *encoder* latents -- what real frames
    # produce, and therefore the manifold. Fitting it on predictor outputs would
    # bake the drift being measured into the reference.
    ood = None
    if not a.no_ood:
        acfg.ood.penalty = a.ood_penalty
        acfg.ood.quantile = a.ood_quantile
        acfg.ood.hard_mult = a.ood_hard_mult
        ood = LatentOOD(cfg.latent_dim, acfg.ood).to(a.device)
        rep = ood.fit(Zt.float())
        print(f"ood: n {rep['n']} dim {rep['dim']} | band [{rep['lo']:.1f}, "
              f"{rep['hi']:.1f}] | median {rep['median']:.1f} vs D "
              f"{rep['expected_median']} | cov dev from I "
              f"{rep['cov_dev_from_identity']:.4f} | latent var "
              f"{rep['latent_var']:.4f}", flush=True)
        # The median sitting far from D says the latent is much less Gaussian
        # than SIGReg is supposed to make it, which does not stop the guard
        # working -- the thresholds are empirical quantiles either way -- but it
        # does mean the "typicality" reading is looser than the theory suggests.
        if not 0.5 * rep["dim"] <= rep["median"] <= 2.0 * rep["dim"]:
            print(f"  NOTE: median {rep['median']:.1f} is far from the latent "
                  f"dimension {rep['dim']}; the latent is less Gaussian than "
                  f"SIGReg's target, so read the band as empirical only.",
                  flush=True)

    probe_head = ProbeHead(probe).to(a.device)
    arena = ImaginedArena(wm, probe_head, acfg, cfg.history, cfg.action_ticks,
                          ood=ood)
    pool = SnapshotPool(acfg)
    S, G, T = a.starts, a.group, a.horizon
    acfg.group_size = G
    print(f"policy {sum(p.numel() for p in policy.parameters())/1e6:.2f}M | "
          f"critic {sum(p.numel() for p in critic.parameters())/1e6:.2f}M | "
          f"{S} starts x {G} = {S * G} rollouts x {T} steps | "
          f"baseline {'group mean' if G > 1 else 'critic v(s)'} | "
          f"two_sided {acfg.two_sided}", flush=True)

    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    eval_rng = np.random.default_rng(12345)
    eval_idx = torch.from_numpy(eval_rng.choice(starts, size=a.eval_starts)).to(a.device)

    # A second arena purely for measurement, at its own horizon and **without the
    # OOD guard**. Both are deliberate. The horizon is separated because `net` is
    # a per-step average over a rollout of that length and therefore changes with
    # it. The guard is dropped because it truncates rollouts, which would make
    # the reported number depend on how far off-manifold the policy currently is
    # -- a policy could then improve its score by drifting sooner. Training keeps
    # the guard; the ruler must not have one.
    eval_cfg = GRPOConfig(horizon=a.eval_horizon, reward=rcfg)
    eval_arena = ImaginedArena(wm, probe_head, eval_cfg, cfg.history,
                               cfg.action_ticks)

    def evaluate() -> dict:
        """Net damage against the frozen initial policy, per step, both chairs.

        Shares `scripts.eval_policy.score`, so the number logged here and the
        number that script prints for a saved checkpoint are produced by the same
        code rather than by two implementations that agree until one is edited.
        """
        return score_policy(eval_arena, policy, reference, Zt, At, eval_idx,
                            cfg, a.device)

    def chair(traj: dict, opp: bool) -> dict:
        """One chair's view of a rollout as a flat trajectory dict.

        The OOD masks are folded into `alive` here rather than at the source.
        `alive` means "this step's reward is worth learning from", and a step
        read off a drifted latent is not -- but `compute_rewards`' own `alive`
        has to keep meaning "not yet KO'd" so that `train_grpo` reproduces its
        recorded numbers unchanged.
        """
        s = "_opp" if opp else ""
        alive = traj[f"alive{s}"]
        lam_scale = traj.get(f"ood_lam_scale{s}")
        if f"ood_ok{s}" in traj:
            alive = alive * traj[f"ood_ok{s}"]
        return {"obs": traj["obs"], "obs_boot": traj["obs_boot"],
                "mine": traj[f"mine{s}"], "side": traj[f"side{s}"],
                "reward": traj[f"reward{s}"], "alive": alive,
                "terminal": traj[f"terminal{s}"],
                "lam_scale": (lam_scale if lam_scale is not None
                              else torch.ones_like(alive)),
                "ood_excess": traj.get(f"ood_excess{s}",
                                       torch.zeros_like(alive))}

    def cat(a_: dict, b_: dict) -> dict:
        return {k: torch.cat([a_[k], b_[k]], dim=0) for k in a_}

    hist, t0 = [], time.time()
    best_net, best_step = float("-inf"), -1
    ent_floor = EntropyFloor(acfg, a.device) if acfg.entropy_floor_frac > 0 else None

    for step in range(1, a.steps + 1):
        idx = torch.from_numpy(rng.choice(starts, size=S)).to(a.device)
        side = torch.randint(0, 2, (S,), device=a.device)
        # Every rollout in a group must share the start *and* the side, or the
        # group mean stops being a conditional baseline and starts averaging over
        # the probe's mirrored chair bias instead.
        if G > 1:
            idx = idx.repeat_interleave(G)
            side = side.repeat_interleave(G)
        off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
        z_ctx = Zt[idx[:, None] + off[None, :]].float()
        a_hist = At[idx[:, None] + off[None, :-1]].float()

        if rng.random() < a.replay_share:
            fut = At[idx[:, None] + torch.arange(T, device=a.device)[None, :]]
            opponent, kind = ReplayOpponent(fut.float()), "replay"
        else:
            snap = pool.sample(rng)
            use_snap = snap is not None and rng.random() < 0.5
            opponent = PolicyOpponent(snap if use_snap else policy)
            kind = "snapshot" if use_snap else "self"

        # Only a self-play opponent's experience is on-policy; see the rollout's
        # docstring. Against a snapshot or a replayed human it would be plain
        # off-policy data with no importance correction.
        both = acfg.two_sided and kind == "self"
        traj = arena.rollout(z_ctx, a_hist, side, policy, opponent, two_sided=both)
        tr = cat(chair(traj, False), chair(traj, True)) if both else chair(traj, False)
        B = tr["obs"].shape[0]

        # ---- values, lambda-returns, advantages ----
        flat_obs = tr["obs"].reshape(B * T, cfg.history, cfg.latent_dim)
        flat_side = tr["side"][:, None].expand(B, T).reshape(-1)
        with torch.no_grad():
            v = target(flat_obs, flat_side).view(B, T)
            v_boot = target(tr["obs_boot"], tr["side"])
            values = torch.cat([v, v_boot[:, None]], dim=1)          # [B, T+1]
            cont = continuation(tr["terminal"])
            # A drifted step is a *truncation*, not a termination: the match is
            # still going, only the simulation's licence to describe it has run
            # out. So it lowers lambda (bootstrap here) instead of clearing cont
            # (nothing follows) -- pricing drift as though it were a KO would
            # teach the policy to fear unfamiliar states as if they were losses.
            reward = tr["reward"]
            if acfg.ood.penalty:
                reward = reward - acfg.ood.penalty * tr["ood_excess"]
            lam_ret = lambda_returns(reward, values, cont, a.gamma,
                                     a.lam * tr["lam_scale"])
            # With a group, the baseline is the group mean and the critic's only
            # remaining job is the bootstrap already folded into `lam_ret`. The
            # concatenated two-sided batch stays group-major in both halves, so
            # the reshape inside finds 2*S groups of G rather than mixing chairs.
            if G > 1:
                adv = group_centred_advantages(lam_ret, tr["alive"], G,
                                               scale=acfg.advantage_scale)
            else:
                adv = advantages_from_returns(lam_ret, v, tr["alive"],
                                              scale=acfg.advantage_scale)

        # ---- critic: regress the lambda-return, on live steps only ----
        alive_flat = tr["alive"].reshape(-1)
        c_loss_per, c_stats = critic.loss(flat_obs, flat_side,
                                          lam_ret.reshape(-1).detach())
        c_loss = (c_loss_per * alive_flat).sum() / alive_flat.sum().clamp(min=1)
        opt_c.zero_grad(set_to_none=True)
        c_loss.backward()
        cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), acfg.grad_clip)
        if torch.isfinite(c_loss) and torch.isfinite(cgn):
            opt_c.step()
            target.update(critic)

        # ---- actor ----
        with torch.no_grad():
            old_logp = policy.log_prob_of(
                flat_obs, flat_side,
                tr["mine"].reshape(-1, cfg.action_ticks, 10))[0].view(B, T)
            ref_logp = reference.log_prob_of(
                flat_obs, flat_side,
                tr["mine"].reshape(-1, cfg.action_ticks, 10))[0].view(B, T)

        stats, skipped, stopped_early = {}, 0, 0
        if step > a.critic_warmup:
            for _ep in range(acfg.epochs):
                loss, stats = grpo_loss(
                    policy, tr, adv, old_logp, acfg, ref_logp,
                    ent_alpha=float(ent_floor.alpha) if ent_floor else 0.0)
                if _ep > 0 and stats["kl"] > acfg.target_kl:
                    stopped_early = 1
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), acfg.grad_clip)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                sched.step()
        else:
            with torch.no_grad():
                _, stats = grpo_loss(policy, tr, adv, old_logp, acfg, ref_logp)
            gn = torch.tensor(0.0)
        pool.maybe_add(policy, step)

        ent_stats = {"alpha": 0.0, "floor": float("nan")}
        if ent_floor is not None:
            if ent_floor.floor is None:
                f = ent_floor.set_floor_from(stats["entropy"])
                print(f"entropy floor {f:.2f} nats "
                      f"({acfg.entropy_floor_frac:.0%} of the initial "
                      f"{stats['entropy']:.2f})", flush=True)
            ent_stats = ent_floor.update(stats["entropy"])

        if step % a.log_every == 0 or step == 1:
            alive = tr["alive"]
            n = alive.sum().clamp(min=1)
            rec = {"step": step, "opponent": kind, "two_sided": bool(both),
                   "critic_loss": float(c_loss.detach()),
                   "reward": float((tr["reward"] * alive).sum() / n),
                   "ret": float(tr["reward"].sum(1).mean()),
                   "lam_ret": float((lam_ret * alive).sum() / n),
                   "adv_abs": float((adv.abs() * alive).sum() / n),
                   "grad_norm": float(gn), "critic_grad_norm": float(cgn),
                   "alive_frac": float(alive.mean()),
                   "term_frac": float(tr["terminal"].mean()),
                   # How much of the rollout the guard is throwing away. If this
                   # climbs over training the policy is walking off-manifold and
                   # the effective horizon is shrinking back toward GRPO's --
                   # which would make the whole longer-horizon argument moot, so
                   # it is worth watching rather than merely recording.
                   "ood_kept": float(tr["lam_scale"].mean()),
                   "ood_excess": float(tr["ood_excess"].mean()),
                   "skipped": skipped, "kl_early_stop": stopped_early,
                   "lr": float(sched.get_last_lr()[0]),
                   "ent_alpha": ent_stats["alpha"], "ent_floor": ent_stats["floor"],
                   **stats, **{f"v_{k}": v_ for k, v_ in c_stats.items()}}
            with torch.no_grad():
                m = tr["mine"]
                rec["press_rate"] = float(m.mean())
                rec["attack_rate"] = float(m[..., 4:8].mean())
                rec["idle_rate"] = float(
                    (m.amax(dim=2).amax(dim=-1) < 0.5).float().mean())
            if step % a.eval_every == 0 or step == 1:
                ev = evaluate()
                rec.update({f"eval_{k}": v_ for k, v_ in ev.items()})
                if ev["net"] > best_net:
                    best_net, best_step = ev["net"], step
                    torch.save({"policy": policy.state_dict(),
                                "critic": critic.state_dict(), "cfg": cfg,
                                "acfg": acfg, "rcfg": rcfg, "step": step,
                                "net": ev["net"]}, a.out / "policy_best.pt")
                print(f"  [eval] step {step:6d} | net vs frozen init "
                      f"{ev['net']:+.5f} | as P1 {ev['p1_dealt']:+.4f}/"
                      f"{ev['p1_taken']:+.4f} | as P2 {ev['p2_dealt']:+.4f}/"
                      f"{ev['p2_taken']:+.4f} | press {rec['press_rate']:.3f} "
                      f"| best {best_net:+.5f} @ {best_step}", flush=True)
            rec["elapsed_h"] = (time.time() - t0) / 3600
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            print(f"step {step:6d} | ret {rec['ret']:+8.4f} lam {rec['lam_ret']:+.4f} "
                  f"| V {rec['v_value_mean']:+.4f}+-{rec['v_value_std']:.3f} "
                  f"clip {rec['v_clipped']:.3f} | closs {rec['critic_loss']:.4f} "
                  f"| KL {rec['kl']:.4f} ent {rec['entropy']:.2f} "
                  f"a {rec['ent_alpha']:.3f} | ood keep {rec['ood_kept']:.3f} "
                  f"| vs {kind} | "
                  f"{rec['elapsed_h']:.2f}h", flush=True)

        if step % a.ckpt_every == 0:
            torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                        "cfg": cfg, "acfg": acfg, "rcfg": rcfg, "step": step},
                       a.out / "policy.pt")

    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(),
                "cfg": cfg, "acfg": acfg, "rcfg": rcfg, "step": a.steps},
               a.out / "policy.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h | best net {best_net:+.5f} "
          f"at step {best_step} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
