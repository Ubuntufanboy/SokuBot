"""Five measurements that decide how to train RL against the state simulator.

    python -m scripts.state_preflight --sim ~/rl/fix5/sim.pt --corpus ~/corpus

Every one of them settles a choice that would otherwise be made by taste, and
each is cheap enough to run before a training run rather than after it. The
history that motivates this: four wrong diagnoses in a row came from trusting a
pooled aggregate nobody had decomposed, and a GRPO run on the blocking gym
learned NOT to block because the reward penalised it -- neither was visible in
the training curves, and both were one measurement away.

  1 proj_feedback   `state_dynamics._unroll_impl` sigmoids the WHOLE projectile
                    tensor when feeding it back, so a bullet's dx/dy/vx/vy are
                    squashed into (0,1) and their sign is destroyed from step 1
                    on. Does fixing it help, hurt, or do nothing to kinematic
                    skill on THIS checkpoint? Decides whether the fix can be
                    switched on or has to be retrained into the model.

  2 block_damage    THE GATE. `block_gain` measures predicted `guarding`, but
                    GRPO learns from `hp`. So: from real start states where a
                    defender is in range and free to act, hold AWAY versus
                    TOWARD and compare predicted damage TAKEN. If this is flat
                    the agent cannot learn blocking from damage however long it
                    trains, and that has to be said before the run, not after.

  3 damage_mode     `rl/reward.py` pays net-over-rollout because per-step
                    clamping rectified the probe's 0.116 residual into fake
                    damage. That argument is about a probe that no longer
                    exists. Compare the simulator's per-step `hp` residual
                    against the corpus's real per-step |dhp| and choose.

  4 action_share    Fraction of return variance attributable to the agent's own
                    actions rather than to which start state it drew.
                    `docs/HANDOFF.md` §9 measured 3.2% on the pixel world model
                    and attributed the GRPO plateau to it. Same number, same
                    construction, new simulator.

  5 combo_owner     Do `combo_hits` and `combo_damage` belong to the player
                    DOING the combo or the one RECEIVING it? The reward's combo
                    term has the opposite sign depending on the answer, so it
                    stays at 0 until this says which.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.data.soku import BUTTONS
from sokubot.data.state import CH, FULL_HP, STAGE_SPAN, STATE_CHANNELS
from sokubot.data import state_bank
from sokubot.model.state_dynamics import load_sim
from sokubot.model.state_head import CONTINUOUS
from sokubot.rl.policy import representable_prior, SokuPolicy
from sokubot.rl.state_arena import (StateArena, StateGRPOConfig, StateObs,
                                    corpus_stats)
from sokubot.rl.state_reward import StateRewardConfig, compute_rewards

BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
L, R = BUTTONS.index("left"), BUTTONS.index("right")
NEAR = 250.0 / STAGE_SPAN


def windows(E, V, span, rng, n, require_valid=True):
    """`n` start indices whose whole `span`-long window is real and in-replay.

    Returns (sample, pool) so a caller that needs to intersect the pool with
    its own condition -- `block_damage` does -- can do so without re-deriving
    the boundary logic and drifting away from it.
    """
    ok = np.ones(len(E), bool)
    ok[len(E) - span:] = False
    for k in range(1, span):
        ok[:len(E) - k] &= (E[k:] == E[:len(E) - k])
    if require_valid:
        for k in range(span):
            ok[:len(E) - k] &= V[k:]
    pool = np.flatnonzero(ok)
    if len(pool) == 0:
        raise SystemExit(f"no valid windows of length {span} in the bank")
    return rng.choice(pool, min(n, len(pool)), replace=False), pool


def _fetch(S, P, A, idx, span, device):
    w = idx[:, None] + np.arange(span)[None, :]
    return (torch.as_tensor(S[w]).to(device),
            torch.as_tensor(P[w]).to(device),
            torch.as_tensor(A[w]).float().to(device))


# ---------------------------------------------------------------------------
# 1. projectile feedback
# ---------------------------------------------------------------------------
@torch.no_grad()
def proj_feedback(sim, obs, S, P, A, E, V, H, horizons, device, n=512, seed=0):
    """Kinematic skill per horizon under each feedback convention.

    Runs the arena rather than `state_dynamics.rollout` so the thing measured
    is the thing RL will use, and forces the recorded human buttons on BOTH
    chairs so the only difference between the two arms is the convention.
    """
    rng = np.random.default_rng(seed)
    span = H + max(horizons)
    idx, _ = windows(E, V, span, rng, n)
    s, p, a = _fetch(S, P, A, idx, span, device)
    cont = torch.tensor(CONTINUOUS, device=device)
    names = [STATE_CHANNELS[i] for i in CONTINUOUS]
    kin = torch.tensor([names.index(x) for x in
                        ("dx", "dy", "x", "y", "vx", "vy", "ay")], device=device)
    sc = torch.as_tensor(
        S[:, :, np.array(CONTINUOUS)].reshape(-1, len(CONTINUOUS)).std(0)
        .clip(1e-6)).to(device)

    out = {}
    for mode in ("sigmoid", "raw"):
        cfg = StateGRPOConfig(horizon=max(horizons))
        arena = StateArena(sim, obs, cfg, H, a.shape[2], proj_feedback=mode)
        # Replay both chairs: `_replay_rollout` feeds the recorded joint input
        # straight in, so no policy is involved and no sampling noise enters.
        pred = _replay_rollout(arena, s[:, :H], p[:, :H], a, max(horizons))
        for h in horizons:
            tgt = (s[:, H + h - 1].index_select(-1, cont) / sc).index_select(-1, kin)
            pr = (pred[:, h - 1].index_select(-1, cont) / sc).index_select(-1, kin)
            idn = (s[:, H - 1].index_select(-1, cont) / sc).index_select(-1, kin)
            out[f"{mode}_h{h}"] = float(
                1 - F.mse_loss(pr, tgt) / F.mse_loss(idn, tgt).clamp(min=1e-9))
    return out


@torch.no_grad()
def _replay_rollout(arena, s_ctx, p_ctx, a_all, steps):
    """Autoregressive rollout driven by the RECORDED joint buttons.

    The arena's own `rollout` needs a policy on one chair; this is the same
    stepping with both chairs supplied by the corpus, which is what an accuracy
    measurement wants -- any policy would add sampling noise to a number that
    is about the simulator.
    """
    H = s_ctx.shape[1]
    s, p = s_ctx, p_ctx
    out = []
    for k in range(steps):
        nxt_s, nxt_p = arena._advance(s, p, a_all[:, k:k + H])
        s = torch.cat([s[:, 1:], nxt_s], dim=1)
        p = torch.cat([p[:, 1:], nxt_p], dim=1)
        out.append(s[:, -1])
    return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# 2. THE GATE: does holding away cost the attacker damage?
# ---------------------------------------------------------------------------
@torch.no_grad()
def block_damage(sim, obs, S, P, A, E, V, H, device, horizon=8, n=1024,
                 seed=0, mode=None):
    """Predicted damage taken while holding AWAY versus TOWARD.

    Identical real start states, identical opponent input (the recorded human's
    buttons, replayed), and the defender's stick forced one way or the other.
    Guarding in Hisoutensoku IS holding away, so a simulator that cannot
    reproduce this in HEALTH cannot teach blocking through a damage reward --
    whatever it does to the `guarding` flag.

    Restricted to frames where the defender is in range and not already
    committed: at full screen the stick is not a blocking decision, and inside
    blockstun it is not a decision at all.
    """
    rng = np.random.default_rng(seed)
    span = H + horizon
    _, pool = windows(E, V, span, rng, 1)
    near = np.abs(S[:, 0, CH["dx"]]) < NEAR
    free = ~((S[:, 0, CH["hitstop"]] > 0) | (S[:, 0, CH["knockdown"]] > 0.5)
             | (S[:, 0, CH["crushed"]] > 0.5) | (S[:, 0, CH["guarding"]] > 0.5)
             | (S[:, 0, CH["wrongblock"]] > 0.5))
    # The defender must also be under threat, or "blocking" is a no-op that
    # cannot show up in damage: require the attacker's hitbox live within the
    # window. Without this the measurement is dominated by neutral frames where
    # neither arm takes any damage and the difference is exactly zero.
    threat = np.zeros(len(S), bool)
    for k in range(1, horizon + 1):
        threat[:len(S) - k] |= (S[k:, 1, CH["hitboxes"]] > 0)
    sel = np.zeros(len(S), bool)
    sel[pool] = True
    sel &= near & free & threat
    idx = np.flatnonzero(sel)
    if len(idx) < 64:
        return {"block_damage_n": int(len(idx))}
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    s, p, a = _fetch(S, P, A, idx, span, device)

    cfg = StateGRPOConfig(horizon=horizon)
    arena = StateArena(sim, obs, cfg, H, a.shape[2], proj_feedback=mode)
    dx = s[:, H - 1, 0, CH["dx"]]                 # P1 is the defender here
    res = {}
    for tag, away in (("away", True), ("toward", False)):
        af = a.clone()
        af[:, :, :, L] = 0.0; af[:, :, :, R] = 0.0
        left = (dx > 0) if away else (dx < 0)     # dx > 0: they are to my right
        af[left, :, :, L] = 1.0
        af[~left, :, :, R] = 1.0
        pred = _replay_rollout(arena, s[:, :H], p[:, :H], af, horizon)
        seq = torch.cat([s[:, H - 1:H], pred], dim=1)
        side = torch.zeros(len(idx), dtype=torch.long, device=device)
        _, _, terms = compute_rewards(seq, af[:, H - 1:H - 1 + horizon], side,
                                      StateRewardConfig(damage_mode="step"))
        res[f"{tag}_taken"] = float(terms["taken"].sum(1).mean())
        res[f"{tag}_guard"] = float(seq[:, :, 0, CH["guarding"]].mean())
    # Positive = holding away costs LESS health, i.e. the simulator pays for
    # blocking in the currency the reward is denominated in.
    res["block_damage_gain"] = res["away_taken"] - res["toward_taken"]
    res["block_damage_gain_hp"] = res["block_damage_gain"] * FULL_HP
    res["block_guard_gain"] = res["away_guard"] - res["toward_guard"]
    res["block_damage_n"] = int(len(idx))
    return res


# ---------------------------------------------------------------------------
# 3. per-step residual versus real per-step damage
# ---------------------------------------------------------------------------
@torch.no_grad()
def damage_scale(sim, obs, S, P, A, E, V, H, device, horizon=8, n=1024, seed=0,
                 mode=None):
    """Is the simulator's `hp` error small against the damage it must resolve?

    Two numbers decide `damage_mode`. If the per-step residual is comparable to
    or larger than a real per-step health change, clamping it rectifies noise
    into action-independent damage once per step and `net` is the safer
    denomination. If it is well under, `step` credits the hit where it landed
    and the short horizon keeps its advantage.
    """
    rng = np.random.default_rng(seed)
    span = H + horizon
    idx, _ = windows(E, V, span, rng, n)
    s, p, a = _fetch(S, P, A, idx, span, device)
    cfg = StateGRPOConfig(horizon=horizon)
    arena = StateArena(sim, obs, cfg, H, a.shape[2], proj_feedback=mode)
    pred = _replay_rollout(arena, s[:, :H], p[:, :H], a, horizon)
    tgt = s[:, H:H + horizon]

    hp_p = pred[..., CH["hp"]]
    hp_t = tgt[..., CH["hp"]]
    # Per STEP, so both are the quantity `damage_mode="step"` would clamp.
    d_p = hp_p[:, 1:] - hp_p[:, :-1]
    d_t = hp_t[:, 1:] - hp_t[:, :-1]
    real = S[:, :, CH["hp"]]
    same = np.zeros(len(S), bool); same[:-1] = E[1:] == E[:-1]
    d_real = (real[1:][same[:-1]] - real[:-1][same[:-1]]).ravel()
    drop = d_real[d_real < 0]
    return {"hp_resid_step": float((d_p - d_t).std()),
            "hp_resid_step_hp": float((d_p - d_t).std()) * FULL_HP,
            "hp_resid_net": float(((hp_p[:, -1] - hp_p[:, 0])
                                   - (hp_t[:, -1] - hp_t[:, 0])).std()) * FULL_HP,
            "hp_real_step_std": float(d_real.std()) * FULL_HP,
            "hp_real_drop_mean": float(-drop.mean()) * FULL_HP if len(drop) else 0.0,
            "hp_real_drop_rate": float((d_real < 0).mean())}


# ---------------------------------------------------------------------------
# 4. how much of the return the agent actually controls
# ---------------------------------------------------------------------------
@torch.no_grad()
def action_share(sim, obs, S, P, A, E, V, H, device, policy, horizon=8,
                 starts=128, group=16, seed=0, mode=None):
    """Var(return | actions) as a fraction of total return variance.

    GRPO's whole signal is the spread WITHIN a group -- rollouts that share a
    start and a side and differ only in the actions sampled. If that spread is
    a rounding error against the spread ACROSS start states, the advantage is
    mostly measurement and the optimiser has nothing to climb. `HANDOFF.md` §9
    put this at 3.2% on the pixel world model and blamed the GRPO plateau on
    it, so it is the first number worth having about a replacement.
    """
    from sokubot.rl.state_arena import StatePolicyOpponent
    rng = np.random.default_rng(seed)
    span = H + horizon + 1
    idx, _ = windows(E, V, span, rng, starts)
    idx = np.repeat(idx, group)
    s, p, a = _fetch(S, P, A, idx, H, device)
    side = torch.from_numpy(rng.integers(0, 2, len(idx))).to(device)
    cfg = StateGRPOConfig(horizon=horizon)
    arena = StateArena(sim, obs, cfg, H, a.shape[2], proj_feedback=mode)
    tr = arena.rollout(s, p, a[:, :-1], side, policy,
                       StatePolicyOpponent(policy))
    ret = tr["reward"].sum(1).view(-1, group)
    within = float(ret.var(dim=1, unbiased=False).mean())
    total = float(ret.reshape(-1).var(unbiased=False))
    return {"action_share": within / max(total, 1e-12),
            "return_std": float(ret.std()),
            "return_mean": float(ret.mean()),
            "within_std": float(np.sqrt(within))}


# ---------------------------------------------------------------------------
# 5. who owns the combo counters
# ---------------------------------------------------------------------------
def combo_owner(S, E):
    """Does `combo_damage[p]` track damage p DEALS or damage p TAKES?

    Asked of real corpus data, with no model involved. The reward's combo term
    has the opposite sign under the two readings, and `build_gyms` assumes the
    attacker owns it, so the assumption is worth one correlation.
    """
    same = np.zeros(len(S), bool); same[:-1] = E[1:] == E[:-1]
    m = same[:-1]
    d_cd = S[1:, 0, CH["combo_damage"]][m] - S[:-1, 0, CH["combo_damage"]][m]
    d_hp1 = S[1:, 0, CH["hp"]][m] - S[:-1, 0, CH["hp"]][m]
    d_hp2 = S[1:, 1, CH["hp"]][m] - S[:-1, 1, CH["hp"]][m]
    rise = d_cd > 1e-6
    if rise.sum() < 100:
        return {"combo_owner": "undetermined", "combo_rise_n": int(rise.sum())}
    # On the steps where P1's combo counter GROWS, whose health fell?
    p1_lost = float((d_hp1[rise] < -1e-6).mean())
    p2_lost = float((d_hp2[rise] < -1e-6).mean())
    return {"combo_owner": "attacker" if p2_lost > p1_lost else "victim",
            "combo_rise_n": int(rise.sum()),
            "combo_rise_p1_hp_fell": p1_lost,
            "combo_rise_p2_hp_fell": p2_lost}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path, default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--replays", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    sim, meta = load_sim(a.sim, a.device)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    print(f"simulator {a.sim} step {meta.get('step')} | history {H} slots "
          f"{slots} ticks {ticks} ({ticks*1000/60:.0f} ms/step)", flush=True)
    # Stated, because measuring a model through a feedback path it was not
    # trained through is a silent way to score the wrong function -- and
    # measurements 2-4 below all default to the model's own for that reason.
    # Only measurement 1 forces a convention, because comparing them is its job.
    print(f"  feedback path: {meta['proj_feedback']} (the model's own; "
          f"measurement 1 overrides it deliberately)", flush=True)

    S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache,
                                           a.replays)
    s_mu, s_sd, p_mu, p_sd = corpus_stats(S, P)
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(a.device)

    policy = SokuPolicy(obs.dim, H, ticks).to(a.device)
    p1 = A.reshape(-1, 20)[:, :10].astype(np.float32)
    lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                     float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                     float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    policy.set_action_prior(
        lr_p / lr_p.sum(), ud_p / ud_p.sum(),
        representable_prior(p1[:, 4:10].mean(0), policy.logit_bound,
                            BUTTONS[4:10]))

    out = {"sim": str(a.sim), "step": int(meta.get("step", -1)),
           "history": H, "slots": slots, "ticks": ticks,
           "frames": int(len(S)), "replays": len(names)}
    print("\n1. projectile feedback (kinematic skill; higher is better)")
    r = proj_feedback(sim, obs, S, P, A, E, V, H, (1, 4, 8), a.device, seed=a.seed)
    out.update(r)
    for h in (1, 4, 8):
        print(f"   h{h:<2} sigmoid {r[f'sigmoid_h{h}']:+.4f}   "
              f"raw {r[f'raw_h{h}']:+.4f}   "
              f"delta {r[f'raw_h{h}'] - r[f'sigmoid_h{h}']:+.4f}")

    print("\n2. THE GATE: block -> damage")
    r = block_damage(sim, obs, S, P, A, E, V, H, a.device, a.horizon, seed=a.seed)
    out.update(r)
    if r.get("block_damage_n", 0) >= 64:
        print(f"   damage taken: away {r['away_taken']:+.5f}  toward "
              f"{r['toward_taken']:+.5f}")
        print(f"   gain {r['block_damage_gain']:+.5f} of a bar "
              f"({r['block_damage_gain_hp']:+.1f} HP) over {a.horizon} steps, "
              f"n={r['block_damage_n']}")
        print(f"   guarding: away {r['away_guard']:.4f} toward "
              f"{r['toward_guard']:.4f} gain {r['block_guard_gain']:+.5f}")
    else:
        print(f"   too few qualifying starts ({r.get('block_damage_n')})")

    print("\n3. damage_mode")
    r = damage_scale(sim, obs, S, P, A, E, V, H, a.device, a.horizon, seed=a.seed)
    out.update(r)
    print(f"   simulator per-step hp residual  {r['hp_resid_step_hp']:8.1f} HP")
    print(f"   simulator net-over-rollout      {r['hp_resid_net']:8.1f} HP")
    print(f"   real per-step |dhp| std         {r['hp_real_step_std']:8.1f} HP")
    print(f"   real mean drop when it drops    {r['hp_real_drop_mean']:8.1f} HP "
          f"on {r['hp_real_drop_rate']:.1%} of steps")
    print(f"   -> damage_mode "
          f"{'step' if r['hp_resid_step_hp'] < r['hp_real_drop_mean'] else 'net'}")

    print("\n4. action-attributable return variance")
    r = action_share(sim, obs, S, P, A, E, V, H, a.device, policy, a.horizon,
                     seed=a.seed)
    out.update(r)
    print(f"   {r['action_share']:.1%} of return variance is the agent's own "
          f"actions (pixel model: 3.2%)")
    print(f"   return {r['return_mean']:+.5f} +- {r['return_std']:.5f}, "
          f"within-group sd {r['within_std']:.5f}")

    print("\n5. combo counter ownership")
    r = combo_owner(S, E)
    out.update(r)
    print(f"   on steps where P1's combo_damage rises: P1 hp fell "
          f"{r.get('combo_rise_p1_hp_fell', float('nan')):.1%}, P2 hp fell "
          f"{r.get('combo_rise_p2_hp_fell', float('nan')):.1%} "
          f"-> {r['combo_owner']}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
