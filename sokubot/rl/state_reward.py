"""The reward, read off the game's own state instead of a linear probe.

    states  [B, T+1, 2, C]   the simulator's own output, in data/state.py units
    actions [B, T, ticks, 20]  both players' buttons, P1-first
    side    [B]              0 if the agent is P1, 1 if P2

WHAT CHANGES BY LEAVING LATENT SPACE
------------------------------------
`rl/reward.py` reads six numbers off a linear probe with a 0.116 residual, and
almost every complication in it is a defence against that residual: `net`
damage exists because per-step clamping rectifies probe noise into ~0.4 sigma of
fake damage per step, `ko_persist` and `ko_alive_margin` exist because a bare
health threshold fires on noise twenty to forty-five times per real KO, and
`spell_cost_min` is set to 1e9 because spirit is not decodable from the latent
at all (R^2 0.036 / 0.015).

Here `hp` and `spirit` are channels the extractor read out of the game's memory
and `pipeline/verify_extended.py` checked against the running game. There is no
probe and no residual. What remains is the SIMULATOR's own prediction error,
which is a different quantity and has to be measured on its own terms --
`scripts/state_preflight.py` does that, and `damage_mode` is set from the
answer rather than from the argument that applied to the probe.

WHAT IS NOT HERE, DELIBERATELY
------------------------------
Every proxy term in `rl/reward.py` existed because the HUD could not see the
thing it stood for, and each one is now either exact or unnecessary:

  flying     was "holding up", a proxy for altitude. `airborne` and `y` are
             exact channels now, and paying an agent for being in the air is
             not a mechanic anyone asked for. Removed rather than converted.
  idle       kept, at 0 by default. It was set to -0.020 against a specific
             measured hazard -- the pixel world model credited "P1 presses
             nothing" with 0.126 of a bar of damage to P2, because a silent
             frame in the corpus is usually a player in hitstun who cannot
             input. That is a property of that model, not of this one, so the
             penalty is available and off until the same hazard is measured
             here.

THREE TERMS ADDED 2026-08-16, AND THE RATES THAT SIZE THEM
-----------------------------------------------------------
`combo`, `proximity` and `whiff`, all defaulting to 0 so that every existing
run and the fixed evaluation reward are bit-identical to before. What makes
them safe to turn on is that their weights are set against a measured anchor
rather than chosen for feel. Over 150 replays at the decision rate:

  DAMAGE, THE ANCHOR   averaged over all player-steps, 0.00089 bars/step
                       (8.9 HP). A hit lands on 3.54% of steps and is worth
                       0.0251 bars when it does.
  whiff                attacks fire on 5.5% of player-steps and 92.5% of them
                       draw no health within two steps -- that is the HUMAN
                       rate, in the corpus this simulator was trained on. So a
                       whiff penalty is charged on ~5.1% of steps, and at a
                       weight of 0.01 bars it would cost 0.00051 bars/step:
                       57% of the entire damage signal, spent teaching the
                       agent never to press a button. The weight has to be
                       roughly two orders of magnitude under the naive guess.
  proximity            being close is only weakly predictive. Hits land inside
                       200 units 64.5% of the time against a 48.5% base rate,
                       a lift of 1.33x. This is a nudge and cannot carry more
                       than a nudge's weight; a standing payment is farmable by
                       an agent that approaches and does nothing.
  combo                combo_damage growth totals 0.72x damage dealt, so weight
                       w makes combo damage worth about (1 + 0.72w)x plain
                       damage. Ownership verified: the channel belongs to the
                       DEALER (95.9% against 1.5%).

ONLY DOWNWARD HEALTH CHANGES COUNT, STILL
-----------------------------------------
Unchanged from the probe reward and for the unchanged reason: health rises on
calm-weather regen, heavy fog, and the end-of-match heal back to full. The last
is enormous and would teach the agent that losing a round is good. Positive
deltas are discarded and termination handles round boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.state import CH, FULL_HP, STAGE_SPAN

# Button offsets within one player's 10-wide block (data/soku.py order).
UP, DOWN, LEFT, RIGHT, A, B, C, D, CHANGE, SPELL = range(10)


@dataclass
class StateRewardConfig:
    """Weights in units of a full health bar, so 1.0 == 10000 game HP."""

    damage_dealt: float = 1.0
    damage_taken: float = -1.0

    # How health change becomes damage.
    #
    #   "step"  sum of per-step clamped decreases. This is the honest choice
    #           whenever the per-step measurement is cleaner than the signal:
    #           it gives the policy credit at the step the hit landed, which is
    #           what a short horizon needs.
    #   "net"   one clamped decrease across the whole rollout, paid at the last
    #           live step. Correct when the per-step reading is noisy enough
    #           that clamping rectifies it into a large action-independent
    #           constant -- which is what the linear probe did.
    #
    # Which one applies is a measurement, not a preference. See
    # scripts/state_preflight.py: it compares the simulator's per-step `hp`
    # residual against the corpus's real per-step |dhp|. Chosen by measurement,
    # recorded in the run's config.json.
    damage_mode: str = "step"

    # Extending a combo. The double-counting is the POINT: the damage term
    # already pays for every point a combo does, so this weight is the premium
    # on top -- at `combo=w`, damage landed inside a combo is worth roughly
    # (1 + w) times the same damage landed in isolation. That is what "reward
    # combos more" can mean once damage is already exact.
    #
    # It stayed at 0 until ownership was settled, because if `combo_damage[p]`
    # tracked damage p RECEIVED, paying for its growth would pay the agent for
    # being comboed and the run would look healthy the whole way. Measured
    # 2026-08-16 over 60 replays and 6237 rising edges: the opponent's health
    # fell on 95.9% of them against 1.5% for the agent's own. The channel is
    # the DEALER's, and weighting it is safe.
    combo: float = 0.0

    # ---- proximity: a small standing payment for being in range ----
    #
    # Paid on the resulting state, falling linearly from `proximity` at zero
    # separation to nothing at `proximity_range` game units. This is shaping,
    # and shaping is dangerous in proportion to its size against the thing it
    # shapes, so the weight is quoted against measured damage-per-step and the
    # range is set from where hits actually land in the corpus.
    #
    # The known failure mode, stated up front: an agent can farm a standing
    # payment by sitting at range and doing nothing, which is worth more than a
    # risky approach if the weight is too large. `dealt` per step is the number
    # to keep it under, and the gym breakdown is where it would show up.
    proximity: float = 0.0
    proximity_range: float = 300.0        # game units

    # ---- whiff: attacking into nothing ----
    #
    # An attack press -- rising edge, so holding a button is charged once, not
    # every step -- that draws no damage from the opponent within
    # `whiff_window` decision steps.
    #
    # The module docstring says whiff was unbuildable, and that was true of the
    # PROBE reward: it defined a whiff as a card press with the cost inferred
    # from a spirit drop, and the sidecar never logged card counts. This
    # definition needs only the opponent's exact health and the agent's own
    # buttons, both of which are present, so the objection does not carry over.
    #
    # Blocked attacks count as whiffs. That is a choice, not an oversight:
    # "I swung and no health moved" is one event in the channels being paid on,
    # and splitting it would need a guard-vs-miss distinction the reward cannot
    # see. It also happens to be what the user asked for -- attacks that do not
    # connect should cost something.
    whiff: float = 0.0
    whiff_window: int = 2
    whiff_buttons: tuple[int, ...] = (A, B, C)

    # Spirit reaching zero is a guard crush: the block broke and the agent is
    # open. Exact here -- `crushed` is one of the five verified flags, checked
    # against the running game rather than probed at R^2 0.015.
    crush: float = -0.5

    # Paid once, at the KO step.
    #
    # 1.0, not the 5.0 the probe reward used. A KO is worth about one health
    # bar, which is what 1.0 means in these units, and the +-5 was set when the
    # detector essentially never fired so its size did not matter. It fires on
    # a real health reading now.
    win: float = 1.0
    lose: float = -1.0

    # 50 of 10000 HP. The threshold sits above zero because the SIMULATOR
    # predicts health with a delta head and can undershoot; it does not sit at
    # 0.06 like the probe version, because there is no 0.116 residual to clear.
    ko_threshold: float = 0.005
    ko_persist: int = 2
    ko_alive_margin: float = 0.05

    idle: float = 0.0              # see the module docstring


def _mine_theirs(states: torch.Tensor, side: torch.Tensor, ch: int):
    """Channel `ch` for (the agent, the opponent), each [B, T+1].

    `states` is [B, T+1, 2, C] and player-major, so one gather with the side
    index picks the agent's own row whichever chair it is sitting in.
    """
    B, T = states.shape[0], states.shape[1]
    col = states[..., ch]                                   # [B, T+1, 2]
    me = side.view(B, 1, 1).expand(B, T, 1)
    return (col.gather(-1, me).squeeze(-1),
            col.gather(-1, 1 - me).squeeze(-1))


def ko_mask(hp: torch.Tensor, cfg: StateRewardConfig) -> torch.Tensor:
    """[B, T+1] health -> [B, T] "this side is KO'd from here".

    Down for `ko_persist` consecutive steps, having started alive. Both guards
    survive the move out of latent space, for a weaker but real reason: the
    reading is exact in the corpus but PREDICTED inside a rollout, and a delta
    head that undershoots once should not pay a match outcome.
    """
    B, dev = hp.shape[0], hp.device
    down = (hp[:, 1:] <= cfg.ko_threshold).float()
    if cfg.ko_persist > 1:
        k = cfg.ko_persist
        # Pad with "not down", so a KO in the last k steps is missed rather
        # than assumed. The alternative pays a match outcome for one noisy read
        # at the boundary.
        pad = torch.zeros(B, k - 1, device=dev)
        run = F.avg_pool1d(torch.cat([down, pad], dim=1)[:, None], k, 1)[:, 0]
        down = (run >= 1.0 - 1e-6).float()
    started_alive = (hp[:, :1] > cfg.ko_threshold + cfg.ko_alive_margin).float()
    return (down * started_alive).bool()


def ko_masks(states: torch.Tensor, side: torch.Tensor,
             cfg: StateRewardConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(I am KO'd, they are KO'd). One dispatch point, as in rl/reward.py, so
    `terminal_mask` and `compute_rewards` cannot disagree about whether the
    match ended -- a split that would pay an outcome on a step the trajectory
    was already dead past."""
    mine_hp, thr_hp = _mine_theirs(states, side, CH["hp"])
    return ko_mask(mine_hp, cfg), ko_mask(thr_hp, cfg)


def terminal_mask(states: torch.Tensor, side: torch.Tensor,
                  cfg: StateRewardConfig | None = None) -> torch.Tensor:
    """[B, T] "the episode *ends* here", as opposed to "stops paying here".

    Kept distinct from `alive` for the same reason `rl/reward.py` keeps it: a
    terminal state must not carry value across the boundary while a truncated
    one must, and `alive` alone cannot tell a KO on the final step from a
    rollout that merely ran out of horizon.
    """
    cfg = cfg or StateRewardConfig()
    ko_me, ko_them = ko_masks(states, side, cfg)
    ko_any = ko_me | ko_them
    T, dev = ko_any.shape[1], states.device
    first = torch.where(ko_any.any(1), ko_any.float().argmax(1),
                        torch.full((states.shape[0],), T, device=dev))
    steps = torch.arange(T, device=dev)[None, :]
    return (steps == first[:, None]).float()


def _my_buttons(actions: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """[B,T,ticks,20] -> [B,T,ticks,10] for the side the agent controls."""
    B = actions.shape[0]
    idx = (side * 10)[:, None, None, None] + torch.arange(10, device=actions.device)
    idx = idx.expand(B, actions.shape[1], actions.shape[2], 10)
    return actions.gather(-1, idx)


def compute_rewards(
    states: torch.Tensor,           # [B, T+1, 2, C]
    actions: torch.Tensor,          # [B, T, ticks, 20]
    side: torch.Tensor,             # [B] long, 0 = agent is P1
    cfg: StateRewardConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """-> (reward [B, T], alive [B, T], per-term breakdown).

    Step t is paid for the transition from state t to state t+1 under action t,
    so there is exactly one fewer reward than there are states.
    """
    cfg = cfg or StateRewardConfig()
    B, T1 = states.shape[0], states.shape[1]
    T = T1 - 1
    dev = states.device
    if actions.shape[1] != T:
        raise ValueError(
            f"got {T1} states and {actions.shape[1]} actions; action t drives "
            f"the transition from state t to state t+1, so there must be "
            f"exactly {T}")
    mine_hp, thr_hp = _mine_theirs(states, side, CH["hp"])
    mine_cr, _ = _mine_theirs(states, side, CH["crushed"])
    mine_cd, _ = _mine_theirs(states, side, CH["combo_damage"])
    btn = _my_buttons(actions, side)

    # ---- termination: everything after the first KO is masked out ----
    ko_me, ko_them = ko_masks(states, side, cfg)
    ko_any = ko_me | ko_them
    first_ko = torch.where(ko_any.any(1), ko_any.float().argmax(1),
                           torch.full((B,), T - 1, device=dev))
    steps = torch.arange(T, device=dev)[None, :]
    alive = (steps <= first_ko[:, None]).float()

    # ---- health deltas; increases discarded (see the module docstring) ----
    if cfg.damage_mode == "step":
        d_them = (thr_hp[:, 1:] - thr_hp[:, :-1]).clamp(max=0.0).abs()
        d_me = (mine_hp[:, 1:] - mine_hp[:, :-1]).clamp(max=0.0).abs()
    elif cfg.damage_mode == "net":
        last = first_ko.clamp(max=T - 1)[:, None]
        tot_them = (thr_hp[:, 1:].gather(1, last) - thr_hp[:, :1]).clamp(max=0.0).abs()
        tot_me = (mine_hp[:, 1:].gather(1, last) - mine_hp[:, :1]).clamp(max=0.0).abs()
        at_last = (steps == last).float()
        d_them, d_me = tot_them * at_last, tot_me * at_last
    else:
        raise ValueError(f"unknown damage_mode {cfg.damage_mode!r}; want step or net")

    r_dealt = cfg.damage_dealt * d_them
    r_taken = cfg.damage_taken * d_me
    # Growth in my combo's damage, not its level: the level persists after a
    # string ends (the counters hold their last values), so paying for the
    # level would pay every step of the following neutral for a combo that is
    # already over. `build_gyms.extending` documents the same trap on
    # `combo_hits`, where the level is true on 64% of the corpus.
    r_combo = cfg.combo * (mine_cd[:, 1:] - mine_cd[:, :-1]).clamp(min=0.0)

    # ---- guard crush: the flag turning on ----
    crushed = (mine_cr[:, 1:] > 0.5) & (mine_cr[:, :-1] <= 0.5)
    r_crush = cfg.crush * crushed.float()

    # ---- match outcome, paid once at the KO step ----
    # A simultaneous read is a double KO, which is a draw. Paying `lose`
    # whenever `ko_me` fires regardless of `ko_them` makes every ambiguous
    # reading negative.
    both = ko_me & ko_them
    at_ko = (steps == first_ko[:, None]).float()
    r_out = (cfg.win * (ko_them & ~both).float()
             + cfg.lose * (ko_me & ~both).float()) * at_ko

    r_idle = cfg.idle * (btn.amax(dim=(2, 3)) < 0.5).float()

    # ---- proximity: linear falloff on the RESULTING state ----
    # Paid on state t+1 because it is the state the action produced; paying on
    # state t would pay for where the agent already was.
    if cfg.proximity != 0.0:
        mine_dx, _ = _mine_theirs(states, side, CH["dx"])
        sep = mine_dx[:, 1:].abs() * STAGE_SPAN
        r_prox = cfg.proximity * (1.0 - sep / cfg.proximity_range).clamp(min=0.0)
    else:
        r_prox = torch.zeros_like(r_dealt)

    # ---- whiff: an attack press that drew no health ----
    if cfg.whiff != 0.0:
        # The hit test reads the RAW per-step decrease, not `d_them`. Under
        # damage_mode="net" `d_them` is a single lump at the last live step, so
        # using it here would call every attack in the rollout a whiff except
        # possibly the last -- a bug that would train the agent to stop
        # attacking and would look like a reward-shaping result.
        hit = (thr_hp[:, 1:] - thr_hp[:, :-1]) < -1e-6            # [B, T]
        atk = (btn[..., list(cfg.whiff_buttons)] > 0.5).any(-1).any(-1)
        # Rising edge, so holding a button is charged once rather than every
        # step. Step 0 has no predecessor inside the rollout and is treated as
        # a press, matching how the corpus rate was measured.
        prev = torch.cat([torch.zeros_like(atk[:, :1]), atk[:, :-1]], dim=1)
        edge = atk & ~prev
        land = torch.zeros_like(hit)
        for k in range(cfg.whiff_window):
            land[:, :T - k] |= hit[:, k:]
        # A press in the last `window-1` steps cannot be judged: its window
        # runs past the end of the rollout. Not charged, rather than charged as
        # a miss -- the alternative puts a penalty on the horizon boundary that
        # has nothing to do with the action.
        judgeable = (torch.arange(T, device=dev)[None, :] <= T - cfg.whiff_window)
        r_whiff = cfg.whiff * (edge & ~land & judgeable).float()
    else:
        r_whiff = torch.zeros_like(r_dealt)

    terms = {"dealt": r_dealt, "taken": r_taken, "combo": r_combo,
             "crush": r_crush, "outcome": r_out, "idle": r_idle,
             "prox": r_prox, "whiff": r_whiff}
    total = sum(terms.values()) * alive
    return total, alive, {k: v * alive for k, v in terms.items()}


def hp_units(x: float) -> float:
    """Bar-fraction -> the game's own HP integer, for reporting.

    Every weight here is in bars because that is the comparable unit across
    characters, and every number that reaches a human should also be quotable
    in the units the game shows.
    """
    return x * FULL_HP
