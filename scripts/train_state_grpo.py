"""GRPO on the gyms, inside the state-space simulator.

    python -m scripts.train_state_grpo --sim ~/rl/fix5/sim.pt \
        --corpus ~/corpus --out ~/rl/grpo_state --steps 20000

The simulator is frozen; only the policy learns. Nothing here reads a pixel or
a probe: the rollout is in the game's own state, so the reward reads `hp`
exactly instead of through a linear map with a 0.116-of-a-bar residual.

WHY THE GYMS ARE BUILT HERE INSTEAD OF LOADED
----------------------------------------------
A gym is a filtered start-state distribution, and a start index only means
something against the array it indexes. `gyms_full.npz` indexes corpus sidecars
frame by frame; this run samples a bank that is strided to the decision rate,
carries pre-chunked actions and holds only the projectile slots the simulator
was built for. Loading one into the other would silently drill different
situations than the ones named. So `build_gyms.build` is called on the same
arrays the trainer samples, and the two cannot disagree.

The horizon passed to that call is the RL horizon, in decision steps. That
widens what a selector means -- `block_enter` becomes "blockstun begins within
667 ms" rather than "within 8 frames" -- and it is the right widening: the
window a gym looks ahead over should be the window the agent is being paid
over.

ONE-SIDED, AND THAT IS THE POINT OF A GYM
------------------------------------------
`rl/state_arena.py` can return both chairs of a rollout for free, and for
general self-play that is the cheapest available improvement. It is OFF here.
A gym pair is `(start, side)` precisely because the same frame is `okizeme`
for one player and `escape_pressure` for the other; harvesting the opponent's
half of a blocking rep would train the attacker's decision under the blocker's
name, which is the exact failure `build_gyms`' docstring exists to prevent.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.data import state_bank
from sokubot.data.soku import BUTTONS
from sokubot.data.state import CH, FULL_HP
from sokubot.model.state_dynamics import load_sim
from sokubot.rl.grpo import (ButtonRateFloor, EntropyFloor, ReplayOpponent,
                             SnapshotPool, group_advantages, grpo_loss)
from sokubot.rl.state_critic import StateCritic, lambda_returns
from sokubot.rl.policy import representable_prior, SokuPolicy
from sokubot.rl.state_arena import (StateArena, StateGRPOConfig, StateObs,
                                    StatePolicyOpponent, corpus_stats)
from sokubot.rl.state_reward import StateRewardConfig
from scripts.build_gyms import build as build_gyms


def valid_starts(E: np.ndarray, V: np.ndarray, history: int,
                 horizon: int) -> np.ndarray:
    """Decision frames whose history AND future are real and in one replay.

    `idx` is the LAST frame of the history window, matching `build_gyms`'
    convention and `train_grpo`'s `off = arange(history) - (history - 1)`. The
    two have to agree or a gym start would be read as a window that begins
    where the gym meant it to end.
    """
    n = len(E)
    ok = np.ones(n, bool)
    ok[:history - 1] = False
    ok[n - horizon:] = False
    for k in range(1, horizon + 1):
        ok[:n - k] &= (E[k:] == E[:n - k])
        ok[:n - k] &= V[k:]
    for k in range(1, history):
        ok[k:] &= (E[:n - k] == E[k:])
        ok[k:] &= V[:n - k]
    ok &= V
    return np.flatnonzero(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, required=True,
                    help="a state simulator from scripts.train_state_dynamics")
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replays", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=8,
                    help="decision steps per rollout. 8 is 667 ms, where "
                         "kinematic skill is still well above a no-op; by 16 "
                         "steps the edge over standing still has narrowed from "
                         "7.7-vs-14.3 game units to 124-vs-168.")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--critic", action="store_true",
                    help="replace the group baseline with a learned value "
                         "function and lambda-returns. Imagination then only "
                         "has to be right over the horizon; everything past it "
                         "comes from v(s_H), which is what lets a reward whose "
                         "payoff lands seconds away (approach, okizeme, winning "
                         "the match) reach the policy at all. Also frees the G "
                         "rollouts per start that existed only to build a "
                         "baseline -- with --critic, --group-size 1 and 8x the "
                         "starts is the same simulator cost for 8x the state "
                         "diversity.")
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="0.99 is a ~100-decision (8 s) effective horizon. A "
                         "match is ~720 decisions, so valuing the OUTCOME "
                         "needs 0.997+; that costs variance, hence a flag.")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--init", type=Path, default=None,
                    help="warm-start the policy from a checkpoint. The frozen "
                         "reference stays the PRIOR-initialised one, so `net` "
                         "remains comparable with every run on record.")
    ap.add_argument("--starts", type=int, default=64, help="groups per batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-ref-coef", type=float, default=None)
    ap.add_argument("--entropy-floor-frac", type=float, default=None)
    ap.add_argument("--gyms", nargs="*", default=None,
                    help="which gyms to drill (default: all that select "
                         "anything). Sampled uniformly, so a rare mechanic "
                         "gets the same number of reps as a common one -- "
                         "which is the entire reason for drilling.")
    ap.add_argument("--corpus-share", type=float, default=0.25,
                    help="fraction of batches drawn from unfiltered start "
                         "states. Gyms are situations someone chose; this is "
                         "the distribution the agent will actually meet, and "
                         "training only on drills optimises for the drills.")
    ap.add_argument("--replay-share", type=float, default=0.30,
                    help="fraction of batches facing the recorded human input "
                         "from the start's own replay. It is open-loop -- the "
                         "recorded human cannot react -- which is tolerable at "
                         "667 ms and is the only opponent in the pool that is "
                         "not a copy of ourselves.")
    ap.add_argument("--damage-mode", default="step", choices=("step", "net"),
                    help="set from scripts.state_preflight, not from taste")
    ap.add_argument("--damage-dealt", type=float, default=1.0,
                    help="weight on damage the agent deals")
    ap.add_argument("--defence-dealt", type=float, default=None,
                    help="`--damage-dealt` used ONLY on the defensive gyms. "
                         "A GYM CHOOSES WHERE THE AGENT STARTS, NOT WHAT IT IS "
                         "PAID FOR, and measuring that distinction is what "
                         "motivates this flag: at step 800 of the symmetric "
                         "run every gym was positive and every gain was "
                         "offence -- the agent guarded 0.021 LESS than the "
                         "reference and took the same damage, on the four "
                         "gyms built to drill not being hit. Set this below "
                         "--damage-dealt (0.25 is a reasonable first try) so "
                         "that inside a defensive drill the exchange is worth "
                         "less than the hit avoided.")
    ap.add_argument("--defensive-gyms", nargs="*",
                    default=["block_enter", "block_gap", "escape_pressure",
                             "cornered", "projectile_dodge", "spirit_starved"],
                    help="which gyms --defence-dealt applies to: the six whose "
                         "mechanic is surviving something rather than starting "
                         "it")
    ap.add_argument("--combo", type=float, default=0.0,
                    help="weight on growth in my combo's damage. Ownership was "
                         "verified 2026-08-16 (the DEALER owns it, 95.9%% vs "
                         "1.5%%), so this is safe to raise; the damage term "
                         "already pays for every point a combo does, so w is a "
                         "PREMIUM -- combo damage becomes ~(1+0.72w)x plain.")
    ap.add_argument("--proximity", type=float, default=0.0,
                    help="bars per step at zero separation, falling linearly to "
                         "0 at --proximity-range. The anchor it competes with "
                         "is 0.00089 bars/step of real damage, so 0.0002 is "
                         "~20%% of the damage signal. Being close is only a "
                         "1.33x lift on landing a hit -- this is a nudge, and "
                         "a large value is farmable by approaching and idling.")
    ap.add_argument("--proximity-range", type=float, default=300.0,
                    help="game units at which the proximity payment reaches 0. "
                         "Hits land at a median separation of 144u; 300u covers "
                         "78.8%% of them and still gives a gradient at the 208u "
                         "median of all steps.")
    ap.add_argument("--whiff", type=float, default=0.0,
                    help="bars charged per attack press (rising edge) that draws "
                         "no health within --whiff-window steps. SIZE THIS "
                         "CAREFULLY: humans whiff 92.5%% of presses and press on "
                         "5.5%% of steps, so the charge lands on ~5.1%% of steps "
                         "and -0.01 would cost 57%% of the whole damage signal, "
                         "i.e. train the agent never to attack. -0.002 is ~11%%.")
    ap.add_argument("--whiff-window", type=int, default=2,
                    help="decision steps to wait for damage before calling a "
                         "press a whiff. Corpus whiff rate: 96.8%% at 1, 92.5%% "
                         "at 2, 86.3%% at 3, 81.9%% at 4.")
    ap.add_argument("--idle", type=float, default=0.0)
    ap.add_argument("--crush", type=float, default=-0.5)
    ap.add_argument("--win-magnitude", type=float, default=1.0)
    ap.add_argument("--proj-feedback", default="sigmoid",
                    choices=("sigmoid", "raw"),
                    help="'sigmoid' reproduces the path the simulator was "
                         "trained through, including its defect (the whole "
                         "projectile tensor is squashed, destroying the sign "
                         "of dx/vx). 'raw' is the fix, and is only honest "
                         "against a simulator retrained with it.")
    ap.add_argument("--button-tolerance", type=float, default=0.0,
                    help="cap each button's press rate at this MULTIPLE of the "
                         "corpus rate, with one Lagrange multiplier per button. "
                         "0 disables it, which is the default because every "
                         "number recorded so far was measured without it. 2.0 "
                         "is the value the measurement argues for: the trained "
                         "policy presses `spell` 10.8x and `change` 7.2x the "
                         "human rate while the aggregate reads a benign 1.4x, "
                         "and those are the two rarest buttons in the corpus -- "
                         "so they are where the simulator has seen least and is "
                         "freest to be wrong. Masking both showed 84-86%% of the "
                         "gain survives, so this constrains about a sixth of "
                         "the behaviour and none of the learning.")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-starts", type=int, default=512)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device

    sim, meta = load_sim(a.sim, dev)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    print(f"simulator {a.sim} step {meta.get('step')} | history {H} slots "
          f"{slots} ticks {ticks} ({ticks*1000/60:.0f} ms/step) | "
          f"{sum(p.numel() for p in sim.parameters())/1e6:.2f}M params",
          flush=True)

    S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache,
                                           a.replays)
    s_mu, s_sd, p_mu, p_sd = corpus_stats(S, P)
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(dev)
    print(f"observation {obs.dim} = 2x{S.shape[-1]} state + 2x{slots}x"
          f"{P.shape[-1]} projectile, ego-ordered", flush=True)

    # ---- gyms, built from the very arrays this run samples ------------------
    print(f"\nbuilding gyms over {len(S)} decision steps "
          f"(horizon {a.horizon} steps = {a.horizon*ticks*1000/60:.0f} ms) ...",
          flush=True)
    gyms = build_gyms(S, P, V, E, a.horizon, H)
    starts = valid_starts(E, V, H, a.horizon)
    wanted = a.gyms if a.gyms else sorted(gyms)
    pool: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    print(f"  {'gym':<20} {'pairs':>9}  {'per-1k':>7}  mean hp(me)  mean |dx|")
    for name in wanted:
        if name not in gyms:
            raise SystemExit(f"unknown gym {name!r}; have {sorted(gyms)}")
        st, sd = gyms[name]
        if len(st) == 0:
            print(f"  {name:<20} {0:>9}  SELECTS NOTHING -- dropped")
            continue
        pool[name] = (st, sd)
        print(f"  {name:<20} {len(st):>9}  {1000*len(st)/len(S):7.1f}  "
              f"{float(S[st, sd, CH['hp']].mean()):11.3f}  "
              f"{float(np.abs(S[st, sd, CH['dx']]).mean()):9.3f}")
    if not pool:
        raise SystemExit("every requested gym selects nothing")
    print(f"  {'(corpus)':<20} {len(starts):>9}  {1000*len(starts)/len(S):7.1f}"
          f"   share {a.corpus_share:.2f}")

    # ---- reward, policy, arena ---------------------------------------------
    def make_rcfg(dealt: float) -> StateRewardConfig:
        return StateRewardConfig(damage_mode=a.damage_mode, damage_dealt=dealt,
                                 combo=a.combo, idle=a.idle, crush=a.crush,
                                 win=a.win_magnitude, lose=-a.win_magnitude,
                                 proximity=a.proximity,
                                 proximity_range=a.proximity_range,
                                 whiff=a.whiff, whiff_window=a.whiff_window)

    rcfg = make_rcfg(a.damage_dealt)
    # A second reward, swapped in for the gyms whose mechanic is defensive.
    # Same object type, same everything else -- only the price of a hit landed
    # changes, so the difference in what the agent learns is attributable to
    # that one number and not to a different drill.
    defensive = set(a.defensive_gyms) if a.defence_dealt is not None else set()
    rcfg_def = (make_rcfg(a.defence_dealt) if defensive else rcfg)
    gcfg = StateGRPOConfig(horizon=a.horizon, group_size=a.group_size,
                           starts_per_batch=a.starts, lr=a.lr, reward=rcfg)
    if a.kl_ref_coef is not None:
        gcfg.kl_ref_coef = a.kl_ref_coef
    if a.entropy_floor_frac is not None:
        gcfg.entropy_floor_frac = a.entropy_floor_frac
    arena = StateArena(sim, obs, gcfg, H, ticks, proj_feedback=a.proj_feedback)

    policy = SokuPolicy(obs.dim, H, ticks).to(dev)
    # Start at the corpus's own button statistics. A uniform policy holds 44%
    # of buttons per tick against a human's 9.85% and is vertically neutral 33%
    # of the time against 82%, so every rollout from a uniform start is a
    # regime the simulator never saw -- and optimising against extrapolation is
    # what four earlier GRPO runs were doing.
    p1 = A.reshape(-1, 20)[:, :10].astype(np.float32)
    lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                     float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                     float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    policy.set_action_prior(
        lr_p / lr_p.sum(), ud_p / ud_p.sum(),
        representable_prior(p1[:, 4:10].mean(0), policy.logit_bound,
                            BUTTONS[4:10]))
    with torch.no_grad():
        samp = policy(torch.zeros(512, H, obs.dim, device=dev),
                      torch.zeros(512, dtype=torch.long, device=dev))
    print(f"\npolicy {sum(p.numel() for p in policy.parameters())/1e6:.2f}M | "
          f"prior press rate {float(samp.actions.mean()):.4f} "
          f"(corpus {float(p1.mean()):.4f}, uniform ~0.44)", flush=True)

    if a.init is not None:
        # Load the policy only. The reference is the yardstick every previous
        # run was measured against; warm-starting it too would silently move
        # the zero point and make `net` incomparable.
        ick = torch.load(a.init, map_location=dev, weights_only=False)
        policy.load_state_dict(ick["policy"])
        print(f"warm start from {a.init} (step {ick.get('step')}, net "
              f"{ick.get('net', float('nan')):+.5f}); reference left at the "
              f"prior init so `net` stays comparable", flush=True)

    critic = critic_opt = None
    if a.critic:
        critic = StateCritic(obs.dim, H).to(dev)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr)
        print(f"critic: {sum(p.numel() for p in critic.parameters())/1e6:.2f}M "
              f"params, {critic.bins} symlog bins, gamma {a.gamma} "
              f"(~{1/(1-a.gamma):.0f} decisions = {1/(1-a.gamma)*5/60:.1f}s "
              f"effective horizon), lambda {a.lam}", flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.steps * gcfg.epochs), eta_min=a.lr * 0.05)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    snaps = SnapshotPool(gcfg)
    ent_floor = (EntropyFloor(gcfg, dev) if gcfg.entropy_floor_frac > 0 else None)
    btn_floor = None
    if a.button_tolerance > 0:
        # Per-button corpus rates, averaged over both chairs -- the same figure
        # `eval_state_policy` reports drift against, so the constraint and the
        # measurement use one definition.
        cr = A.reshape(-1, 20).astype(np.float32)
        cr = (cr[:, :10].mean(0) + cr[:, 10:].mean(0)) / 2
        btn_floor = ButtonRateFloor(cr, dev, tolerance=a.button_tolerance)
        print("button ceilings (%.1fx corpus): %s" % (a.button_tolerance,
              ", ".join(f"{n} {c:.4f}" for n, c in
                        zip(("up","down","left","right","a","b","c","d",
                             "change","spell"), cr * a.button_tolerance))),
              flush=True)

    G, NS = a.group_size, a.starts
    B = G * NS
    off = torch.arange(H, device=dev) - (H - 1)
    St = torch.from_numpy(S).to(dev)
    Pt = torch.from_numpy(P).to(dev)
    At = torch.from_numpy(A).to(dev)
    print(f"{NS} groups x {G} = {B} rollouts x {a.horizon} steps | bank "
          f"{(S.nbytes + P.nbytes + A.nbytes)/1e9:.2f} GB on {dev}", flush=True)

    def context(idx: torch.Tensor):
        w = idx[:, None] + off[None, :]
        return St[w], Pt[w], At[w[:, :-1]].float()

    # ---- the yardstick self-play cannot provide ----------------------------
    # In self-play one chair's gain is the other's loss, so the mean training
    # reward is zero however well the policy plays. Progress has to be measured
    # against a FIXED opponent on FIXED starts, with the sides swapped so a
    # policy cannot score by exploiting whichever chair the simulator favours.
    eval_rng = np.random.default_rng(12345)
    eval_sets = {"corpus": eval_rng.choice(starts, size=min(a.eval_starts,
                                                            len(starts)))}
    for name, (st, _sd) in pool.items():
        eval_sets[name] = st[eval_rng.choice(len(st),
                                             size=min(a.eval_starts, len(st)))]

    # THE YARDSTICK IS NOT THE TRAINING REWARD.
    #
    # `net` has to mean the same thing across runs, and the training reward
    # does not: under `damage_mode="net"` the whole exchange is paid at one
    # step, so a mean over live steps is the same damage divided by eight.
    # Two arms that differ only in `--damage-mode` would then report numbers an
    # order of magnitude apart while playing identically. So the evaluation
    # re-scores every rollout with a FIXED config -- per-step damage exchange,
    # nothing else -- and that number is comparable across every run.
    EVAL_RCFG = StateRewardConfig(damage_mode="step", combo=0.0, idle=0.0,
                                  crush=0.0, win=0.0, lose=0.0)

    @torch.no_grad()
    def evaluate(which: str) -> dict:
        from sokubot.rl.state_reward import compute_rewards as _score
        idx = torch.from_numpy(eval_sets[which]).to(dev)
        s_ctx, p_ctx, a_hist = context(idx)
        out = {}
        for tag, s0 in (("p1", 0), ("p2", 1)):
            side = torch.full((len(idx),), s0, device=dev, dtype=torch.long)
            tr = arena.rollout(s_ctx, p_ctx, a_hist, side, policy,
                               StatePolicyOpponent(reference))
            _, al, terms = _score(tr["states"], tr["joint"], side, EVAL_RCFG)
            n = al.sum().clamp(min=1)
            out[f"{tag}_dealt"] = float((terms["dealt"] * al).sum() / n)
            out[f"{tag}_taken"] = float((terms["taken"] * al).sum() / n)
            out[f"{tag}_guard"] = float(
                tr["states"][..., CH["guarding"]]
                .gather(-1, side.view(-1, 1, 1)
                        .expand(-1, tr["states"].shape[1], 1)).mean())
        # dealt is positive and taken negative, so their sum is the exchange;
        # averaging the two chairs cancels any bias the simulator has toward
        # one seat, which a policy could otherwise score against.
        out["net"] = ((out["p1_dealt"] + out["p1_taken"])
                      + (out["p2_dealt"] + out["p2_taken"])) / 2
        out["guard"] = (out["p1_guard"] + out["p2_guard"]) / 2
        return out

    hist, t0 = [], time.time()
    best_net, best_step = float("-inf"), -1
    gym_names = sorted(pool)
    (a.out / "config.json").write_text(json.dumps(
        {"sim": str(a.sim), "sim_step": int(meta.get("step", -1)),
         "history": H, "slots": slots, "ticks": ticks, "obs_dim": obs.dim,
         "horizon": a.horizon, "group_size": G, "starts": NS,
         "proj_feedback": a.proj_feedback, "damage_mode": a.damage_mode,
         "damage_dealt": a.damage_dealt, "defence_dealt": a.defence_dealt,
         "defensive_gyms": sorted(defensive),
         "combo": a.combo, "idle": a.idle, "crush": a.crush,
         "button_tolerance": a.button_tolerance,
         "win": a.win_magnitude, "corpus_share": a.corpus_share,
         "replay_share": a.replay_share, "lr": a.lr, "steps": a.steps,
         "gyms": {k: int(len(v[0])) for k, v in pool.items()},
         "bank_replays": len(names), "bank_steps": int(len(S))}, indent=1))

    for step in range(1, a.steps + 1):
        # A gym carries the side with each start. Re-rolling the side would
        # seat the agent in the attacker's chair on half the blocking reps and
        # drill the opposite mechanic in the same breath.
        if rng.random() < a.corpus_share:
            src = "corpus"
            idx_np = rng.choice(starts, size=NS)
            side_np = rng.integers(0, 2, NS)
        else:
            src = gym_names[rng.integers(len(gym_names))]
            st, sd = pool[src]
            pick = rng.choice(len(st), size=NS)
            idx_np, side_np = st[pick], sd[pick]
        idx = torch.from_numpy(np.repeat(idx_np, G)).to(dev)
        side = torch.from_numpy(np.repeat(side_np, G)).to(dev).long()
        s_ctx, p_ctx, a_hist = context(idx)
        # Swapped on the arena rather than passed down, because `rollout` reads
        # `cfg.reward` and both configs are otherwise the same object.
        gcfg.reward = rcfg_def if src in defensive else rcfg

        if rng.random() < a.replay_share:
            fut = At[idx[:, None] + torch.arange(a.horizon, device=dev)[None, :]]
            opponent, kind = ReplayOpponent(fut.float()), "replay"
        else:
            snap = snaps.sample(rng)
            use = snap is not None and rng.random() < 0.5
            opponent = StatePolicyOpponent(snap if use else policy)
            kind = "snapshot" if use else "self"

        traj = arena.rollout(s_ctx, p_ctx, a_hist, side, policy, opponent)
        if critic is not None:
            # v at every visited state plus the bootstrap at the edge. The
            # rollout stores T+1 states but only T observations, so the final
            # value is read from the last observation -- one step stale, which
            # is what a truncated rollout can honestly know.
            with torch.no_grad():
                v_seq = critic.value(traj["obs"])              # [B, T]
                v_all = torch.cat([v_seq, v_seq[:, -1:]], 1)   # [B, T+1]
                ret = lambda_returns(traj["reward"], v_all, traj["alive"],
                                     traj["terminal"], a.gamma, a.lam)
                adv = ret - v_seq
                m = traj["alive"] > 0
                if m.any():
                    adv = (adv - adv[m].mean()) / adv[m].std().clamp(min=1e-6)
                adv = adv * traj["alive"]
        else:
            adv = group_advantages(traj["reward"], traj["alive"], G, gcfg.gamma,
                                   scale=gcfg.advantage_scale)
        flat_side = side[:, None].expand(B, a.horizon).reshape(-1)
        flat_obs = traj["obs"].reshape(-1, H, obs.dim)
        flat_act = traj["mine"].reshape(-1, ticks, 10)
        with torch.no_grad():
            # Frozen before any update: this is the policy the actions were
            # actually sampled from, which is what makes the ratio mean
            # anything.
            old_logp = policy.log_prob_of(flat_obs, flat_side,
                                          flat_act)[0].view(B, a.horizon)
            ref_logp = reference.log_prob_of(flat_obs, flat_side,
                                             flat_act)[0].view(B, a.horizon)

        skipped = stopped_early = 0
        for ep in range(gcfg.epochs):
            loss, stats = grpo_loss(policy, traj, adv, old_logp, gcfg, ref_logp,
                                    ent_alpha=float(ent_floor.alpha)
                                    if ent_floor else 0.0,
                                    button_floor=btn_floor)
            # Measured BEFORE the step, so this refuses an update from a policy
            # that has already drifted rather than noticing afterwards.
            if ep > 0 and stats["kl"] > gcfg.target_kl:
                stopped_early = 1
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                gcfg.grad_clip)
            # One non-finite batch otherwise writes NaN into every weight and
            # nothing recovers from there. Skipping costs one update.
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()
        if critic is not None:
            # Fitted to the SAME lambda-returns the actor was scored against,
            # so the baseline it subtracts is the one it is learning to predict.
            for _ in range(2):
                closs = (critic.loss(traj["obs"], ret) * traj["alive"]).sum() \
                    / traj["alive"].sum().clamp(min=1)
                critic_opt.zero_grad(set_to_none=True)
                closs.backward()
                cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                if torch.isfinite(closs) and torch.isfinite(cgn):
                    critic_opt.step()
            with torch.no_grad():
                v_now = critic.value(traj["obs"])
                stats["critic_loss"] = float(closs)
                stats["v_mean"] = float((v_now * traj["alive"]).sum()
                                        / traj["alive"].sum().clamp(min=1))
                ss = ((ret - v_now) ** 2 * traj["alive"]).sum()
                st_ = ((ret - ret.mean()) ** 2 * traj["alive"]).sum()
                stats["v_r2"] = float(1 - ss / st_.clamp(min=1e-9))
        snaps.maybe_add(policy, step)
        if btn_floor is not None:
            with torch.no_grad():
                _, _, r = policy.log_prob_of(flat_obs, flat_side, flat_act,
                                             return_rates=True)
            stats.update(btn_floor.update(r))

        ent_stats = {"alpha": 0.0, "floor": float("nan")}
        if ent_floor is not None:
            if ent_floor.floor is None:
                f = ent_floor.set_floor_from(stats["entropy"])
                print(f"entropy floor {f:.2f} nats ({gcfg.entropy_floor_frac:.0%}"
                      f" of the initial {stats['entropy']:.2f})", flush=True)
            ent_stats = ent_floor.update(stats["entropy"])

        if step % a.log_every == 0 or step == 1:
            alive = traj["alive"]
            n = alive.sum().clamp(min=1)
            with torch.no_grad():
                m = traj["mine"]
                guard = float(traj["states"][..., CH["guarding"]]
                              .gather(-1, side.view(-1, 1, 1)
                                      .expand(-1, traj["states"].shape[1], 1))
                              .mean())
            rec = {"step": step, "loss": float(loss.detach()), "src": src,
                   "opponent": kind, "grad_norm": float(gn),
                   "ret": float(traj["reward"].sum(1).mean()),
                   "alive_frac": float(alive.mean()), "skipped": skipped,
                   "kl_early_stop": stopped_early,
                   "lr": float(sched.get_last_lr()[0]),
                   "ent_alpha": ent_stats["alpha"], "ent_floor": ent_stats["floor"],
                   "press_rate": float(m.mean()),
                   "attack_rate": float(m[..., 4:8].mean()),
                   "move_rate": float(m[..., :4].mean()),
                   "idle_rate": float((m.amax(dim=2).amax(dim=-1) < 0.5).float().mean()),
                   # The mechanic that no previous world model could represent.
                   # Tracked every log line because "did the agent start
                   # blocking" is the question this whole redesign was for.
                   "guard_rate": guard,
                   **stats,
                   **{f"r_{k}": float((v * alive).sum() / n)
                      for k, v in traj["terms"].items()}}
            if step % a.eval_every == 0 or step == 1:
                ev = evaluate("corpus")
                rec.update({f"eval_{k}": v for k, v in ev.items()})
                for g in gym_names:
                    rec[f"gym_{g}_net"] = evaluate(g)["net"]
                if ev["net"] > best_net:
                    best_net, best_step = ev["net"], step
                    torch.save({"policy": policy.state_dict(), "obs": obs.state_dict(),
                                "obs_dim": obs.dim, "history": H, "ticks": ticks,
                                "slots": slots, "sim": str(a.sim),
                                "step": step, "net": ev["net"],
                                # The yardstick travels with the checkpoint.
                                # Reconstructing it from a seed works and is
                                # checked, but it depends on the torch RNG
                                # being consumed in the same order by a script
                                # that does other things first -- a dependency
                                # nobody should have to know about.
                                "reference": reference.state_dict()},
                               a.out / "policy_best.pt")
                worst = min(gym_names, key=lambda g: rec[f"gym_{g}_net"])
                best_g = max(gym_names, key=lambda g: rec[f"gym_{g}_net"])
                print(f"  [eval] step {step:6d} | net vs frozen init "
                      f"{ev['net']:+.5f} ({ev['net']*FULL_HP:+.1f} HP/step) | "
                      f"P1 {ev['p1_dealt']:+.4f}/{ev['p1_taken']:+.4f} "
                      f"P2 {ev['p2_dealt']:+.4f}/{ev['p2_taken']:+.4f} | "
                      f"guard {ev['guard']:.3f} | best {best_g} "
                      f"{rec[f'gym_{best_g}_net']:+.5f} worst {worst} "
                      f"{rec[f'gym_{worst}_net']:+.5f}", flush=True)
            rec["elapsed_h"] = (time.time() - t0) / 3600
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            # `attack` is printed beside the shaping terms on purpose: the whiff
            # penalty's failure mode is that it stops the agent attacking at
            # all, and the attack rate is the only line where that is visible
            # before the eval catches it 200 steps later.
            shape = ""
            if a.proximity:
                shape += f" prox {rec['r_prox']:+.5f}"
            if a.whiff:
                shape += f" whiff {rec['r_whiff']:+.5f}"
            if a.combo:
                shape += f" combo {rec['r_combo']:+.5f}"
            print(f"step {step:6d} | {src:<17} vs {kind:<8} | ret "
                  f"{rec['ret']:+8.5f} | dealt {rec['r_dealt']:+.5f} taken "
                  f"{rec['r_taken']:+.5f}{shape} | guard {rec['guard_rate']:.3f} "
                  f"atk {rec['attack_rate']:.3f} idle {rec['idle_rate']:.3f} | "
                  f"KL {rec['kl']:.4f} ent {rec['entropy']:.2f} "
                  f"a {rec['ent_alpha']:.3f} | {rec['elapsed_h']:.2f}h",
                  flush=True)

        if step % a.ckpt_every == 0:
            torch.save({"policy": policy.state_dict(), "obs": obs.state_dict(),
                        "obs_dim": obs.dim, "history": H, "ticks": ticks,
                        "slots": slots, "sim": str(a.sim), "step": step},
                       a.out / "policy.pt")

    torch.save({"policy": policy.state_dict(), "obs": obs.state_dict(),
                "obs_dim": obs.dim, "history": H, "ticks": ticks,
                "slots": slots, "sim": str(a.sim), "step": a.steps},
               a.out / "policy.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h | best net {best_net:+.5f} "
          f"at step {best_step} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
