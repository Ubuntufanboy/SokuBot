"""The reward landscape, computed from probed latent state inside imagination.

GRPO never sees a pixel. Each imagined latent is turned into game state by the
linear probe (health, spirit, combo red), and this module turns a *trajectory*
of that state, plus the actions that produced it, into a scalar per step.

    states  [B, T+1, 6]        probe output: hp1 hp2 spirit1 spirit2 combo1 combo2
    actions [B, T, ticks, 20]  both players' buttons, agent's side included
    side    [B]                0 if the agent is P1, 1 if P2

There is one more state than action because action t is what carries state t to
state t+1, and it is scored by what that transition did.

Everything is written from the agent's point of view via `side`, so one policy
plays either side and the reward is always "what happened to me".

WHAT IS MEASURED, AND WHAT IS PROXIED
-------------------------------------
Four terms read real probed state: damage dealt, damage taken, combo size, and
guard crush. Three are proxies that read the *action* instead, because the HUD
exposes no signal for them:

  flying    holding up. The HUD has no altitude, and the latent was never given
            an altitude label to probe, so this rewards the input rather than
            the state. Kept small deliberately -- it is a nudge toward using the
            air, not a claim about where the character is.
  idle      no buttons pressed at all during a decision step.
  spellcard the `spell` button, which is ground truth (read from the game's own
            input struct, not from pixels) but means "a card was played", not
            "a spell card was played". Whether it *was* a spell card, and what
            it cost, is inferred from the spirit drop at the press.

ONLY DOWNWARD HEALTH CHANGES COUNT
----------------------------------
Health rises for three reasons -- calm-weather regen, heavy fog, and the
end-of-match heal back to full -- and the last one is enormous. Rewarding a
positive delta would hand the agent a colossal fake payout every time a round
boundary landed inside a rollout, and would teach it that losing a round is
good. So positive deltas are discarded outright and only decreases are scored.
The regen signal lost this way is worth far less than the failure mode avoided.

Round boundaries are handled by termination, not by reward: once a side's health
crosses `ko_threshold` the match outcome is paid once and every later step in
that trajectory is masked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Column order of the probe's output, fixed by scripts/horizon_ablation.py.
HP1, HP2, SPIRIT1, SPIRIT2, COMBO1, COMBO2 = range(6)
# Appended by scripts/fit_banner_channel.py, after cards1/cards2. Read by index
# rather than by name because `ProbeHead` hands `compute_rewards` a bare tensor;
# the fitter writes the probe's `names` array and refuses to append if the
# position it would land in is not this one.
KO_BANNER = 8

# Button offsets within one player's 10-wide block (see data/soku.py).
UP, DOWN, LEFT, RIGHT, A, B, C, D, CHANGE, SPELL = range(10)


@dataclass
class RewardConfig:
    """Weights are in units of "bars of health", so 1.0 == a full health bar."""

    damage_dealt: float = 1.0
    damage_taken: float = -1.0

    # How health change becomes damage.
    #
    #   "step"  sum of per-step clamped decreases. The obvious choice and the
    #           wrong one here: the clamp rectifies the probe's per-step
    #           residual, which is around 0.14 of a bar and roughly independent
    #           across steps, so a perfectly flat health trajectory still
    #           accumulates about 0.4*sigma of fake damage per step. Over
    #           sixteen steps that is close to a bar, against real damage of a
    #           fraction of one. Rectified noise does not depend on the actions,
    #           so it is a large constant the policy cannot move, and the part it
    #           can move sits underneath.
    #
    #   "net"   one clamped decrease across the whole rollout, hp[0] - hp[T],
    #           paid at the end. There are no per-step differences to rectify, so
    #           endpoint noise enters once instead of once per step.
    damage_mode: str = "net"

    # Red on the opponent's bar is how much the *current combo* has done. Paying
    # for its size rewards extending a combo rather than trading single hits.
    combo: float = 0.30

    # A hit that lands while the agent's cards are spent counts double.
    spell_multiplier: float = 2.0

    win: float = 5.0
    lose: float = -5.0

    # Pressing the card button and paying spirit without landing anything inside
    # `spell_window` steps. Scaled by the spirit actually spent, so throwing away
    # an expensive card hurts more than a cheap one.
    whiff: float = -1.0
    spell_window: int = 8

    # Spirit reaching zero is a guard crush: the block broke and the agent is
    # open. This is real probed state, not a proxy.
    crush: float = -0.5
    crush_threshold: float = 0.02

    flying: float = 0.005          # action proxy; see module docstring
    idle: float = -0.010           # action proxy

    # A KO is read off a *probed* health value, and the probe's residual noise is
    # around 0.13 of a bar (label std 0.32 at R^2 0.83). A bare "health <= 0.02"
    # test therefore fires on noise many times per rollout, and at +-5 it swamps
    # damage, which lives around 0.1. Three things make it mean something:
    # a threshold above the noise floor, a persistence requirement (noise is
    # roughly independent across steps, a real KO is not), and a requirement that
    # the side started the rollout alive, so a start state that merely *reads*
    # low does not pay out.
    ko_threshold: float = 0.06
    ko_persist: int = 3
    ko_alive_margin: float = 0.10

    # Where "the match ended" comes from.
    #
    #   "health"  the threshold test above. Measured at precision **0.003** by
    #             `scripts/probe_reliability.py` -- twenty to forty-five false
    #             fires per real KO, each paying +-5 against damage worth ~0.1
    #             and masking the rest of the trajectory. Kept as the default so
    #             every recorded number stays reproducible, not because it works.
    #
    #   "banner"  a probe channel trained on the KNOCK OUT banner the game draws
    #             across the screen. Precision **0.803** at recall 0.487 on
    #             held-out replays (`scripts/banner_latent_probe.py`), i.e. 268x
    #             the health detector.
    #
    # THE SPLIT THAT MAKES THIS WORK
    # ------------------------------
    # The banner says the match ended. It does *not* say who won -- the glyphs
    # are the same whoever lost. So the two questions go to different
    # instruments, each where it is strong:
    #
    #   *when*  from the banner, because detecting a drawn event is what it is
    #           good at;
    #   *who*   from sign(my health - their health), because that is a
    #           **difference** and the probe's error is largely common-mode. The
    #           absolute test needs |error| < 0.06 against a residual of 0.116
    #           and loses. The comparison only needs the sign of a gap that is
    #           half a bar wide at a KO, and wins comfortably.
    #
    # Which is the point: the failing test was never "can the latent see a KO",
    # it was "can it resolve an absolute health level to a twentieth of a bar".
    # It cannot, and for this purpose it does not have to.
    ko_source: str = "health"
    # Score above which the banner channel counts as lit.
    #
    # **Not 0.5.** The channel is a ridge fit to a 0/1 target whose positive
    # class is 1.4% of frames, so its output is compressed toward zero and never
    # reaches a half. Measured on held-out replays by
    # `scripts/fit_banner_channel.py`:
    #
    #     threshold   precision   recall   fires per 1000 frames
    #          0.20       0.844    0.447                     8.1
    #          0.30       0.851    0.333                     6.0
    #          0.40       0.827    0.148                     2.7
    #          0.50       0.000    0.000                     0.0
    #
    # 0.20 is chosen for the recall, since precision is flat across the usable
    # range. Left at the natural-looking 0.5 the detector would simply never
    # fire, `win`/`lose` would be identically zero, and the run would look like a
    # reward-shaping null rather than a threshold set from taste.
    ko_banner_threshold: float = 0.20
    # How far apart the two health readings must be before a winner is called.
    # Below this the banner is still believed for termination, but no outcome is
    # scored: a KO the probe cannot attribute is worth nothing, and a coin flip
    # at +-5 is worth rather less than nothing.
    ko_margin: float = 0.10

    # A spirit drop this large at a card press means a card actually went off.
    # One orb is 0.2 of the gauge; a real cast spends at least one. Set above 1.0
    # to disable card detection entirely, which is correct wherever spirit is not
    # decodable from the latent.
    spell_cost_min: float = 0.12


def _sides(states: torch.Tensor, side: torch.Tensor):
    """Split probed state into (mine, theirs) given which player the agent is.

    `side` is [B] with 0 = P1. Health, spirit and combo are all stored P1-first,
    so one gather with the same index pattern reorders every channel at once.
    """
    mine_hp = torch.where(side[:, None] == 0, states[..., HP1], states[..., HP2])
    thr_hp = torch.where(side[:, None] == 0, states[..., HP2], states[..., HP1])
    mine_sp = torch.where(side[:, None] == 0, states[..., SPIRIT1], states[..., SPIRIT2])
    # Red on *my* bar is damage being done to me; red on theirs is damage I am
    # doing. The combo reward reads theirs.
    thr_cb = torch.where(side[:, None] == 0, states[..., COMBO2], states[..., COMBO1])
    return mine_hp, thr_hp, mine_sp, thr_cb


def ko_mask(hp: torch.Tensor, cfg: RewardConfig) -> torch.Tensor:
    """[B, T+1] probed health -> [B, T] "this side is KO'd from here".

    Down for `ko_persist` consecutive steps, having started alive.

    Module-level rather than a closure inside `compute_rewards` so that anything
    auditing the detector -- `scripts/probe_reliability.py` measures its false
    positive and false negative rates against true HUD health -- tests the
    detector the reward actually uses instead of a reimplementation that can
    drift away from it.
    """
    B, dev = hp.shape[0], hp.device
    down = (hp[:, 1:] <= cfg.ko_threshold).float()
    if cfg.ko_persist > 1:
        k = cfg.ko_persist
        # Pad with "not down". A KO within k steps of the end is therefore
        # missed rather than assumed, which is the right way to be wrong:
        # the alternative pays +-5 for a single noisy read at the boundary.
        pad = torch.zeros(B, k - 1, device=dev)
        run = F.avg_pool1d(torch.cat([down, pad], dim=1)[:, None], k, 1)[:, 0]
        down = (run >= 1.0 - 1e-6).float()
    started_alive = (hp[:, :1] > cfg.ko_threshold + cfg.ko_alive_margin).float()
    return (down * started_alive).bool()


def banner_ko_masks(states: torch.Tensor, side: torch.Tensor,
                    cfg: RewardConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T+1] probed state -> (I am KO'd, they are KO'd), each [B, T].

    The banner answers *when*; the health comparison answers *who*. See
    `RewardConfig.ko_source` for why those are two different instruments.

    The persistence requirement carries over from `ko_mask` and does the same
    job for the same reason: the banner probe's false positives are roughly
    independent across steps while a real banner is drawn for about two seconds,
    so requiring `ko_persist` consecutive lit steps costs almost no true
    detections and removes most of the isolated ones.

    Both outputs are False where the margin is too small to attribute the loss.
    That is not a fudge: the alternative is paying +-5 on the sign of a
    difference smaller than the probe's own noise, which is a coin flip with a
    five-bar stake.
    """
    if states.shape[-1] <= KO_BANNER:
        raise ValueError(
            f"ko_source='banner' needs a probe with at least {KO_BANNER + 1} "
            f"channels, got {states.shape[-1]}. Fit one with "
            f"scripts/fit_banner_channel.py.")
    lit = (states[..., KO_BANNER][:, 1:] >= cfg.ko_banner_threshold).float()
    if cfg.ko_persist > 1:
        k = cfg.ko_persist
        pad = torch.zeros(states.shape[0], k - 1, device=states.device)
        run = F.avg_pool1d(torch.cat([lit, pad], dim=1)[:, None], k, 1)[:, 0]
        lit = (run >= 1.0 - 1e-6).float()

    mine_hp, thr_hp, _, _ = _sides(states, side)
    gap = (mine_hp - thr_hp)[:, 1:]
    decided = (gap.abs() >= cfg.ko_margin).float()
    # Same "started alive" guard as the health detector, for the same reason: a
    # rollout that begins at the tail of someone else's KO must not be paid for
    # it. Read from the banner rather than from health, so it does not reinherit
    # the absolute-level test this exists to replace.
    began_clear = (states[..., KO_BANNER][:, :1] < cfg.ko_banner_threshold).float()

    i_lost = lit * decided * began_clear * (gap < 0).float()
    they_lost = lit * decided * began_clear * (gap > 0).float()
    return i_lost.bool(), they_lost.bool()


def ko_masks(states: torch.Tensor, side: torch.Tensor,
             cfg: RewardConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(I am KO'd, they are KO'd), from whichever detector `cfg.ko_source` names.

    One dispatch point, used by both `terminal_mask` and `compute_rewards`, so
    the two can never disagree about whether the match ended -- a split that
    would pay a match outcome on one step and keep the trajectory alive past it.
    """
    if cfg.ko_source == "health":
        mine_hp, thr_hp, _, _ = _sides(states, side)
        return ko_mask(mine_hp, cfg), ko_mask(thr_hp, cfg)
    if cfg.ko_source == "banner":
        return banner_ko_masks(states, side, cfg)
    raise ValueError(f"unknown ko_source {cfg.ko_source!r}; want health or banner")


def terminal_mask(states: torch.Tensor, side: torch.Tensor,
                  cfg: RewardConfig | None = None) -> torch.Tensor:
    """[B, T+1] probed states -> [B, T] "the episode *ends* at this step".

    Distinct from `alive`, and the distinction is not cosmetic. `alive` is 1 up
    to and including the KO step, so it cannot tell a KO that lands on the final
    step of a rollout from a rollout that simply ran out of horizon. Those need
    opposite treatment when bootstrapping: a terminal state must not carry value
    across the boundary, while a truncated one must, because the episode really
    does continue and the critic's estimate of what follows is the whole point.

    Reading it from the KO masks rather than inferring it from `alive` makes the
    difference explicit instead of guessed.
    """
    cfg = cfg or RewardConfig()
    ko_me, ko_them = ko_masks(states, side, cfg)
    ko_any = ko_me | ko_them
    T = ko_any.shape[1]
    dev = states.device
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
    states: torch.Tensor,          # [B, T+1, 6]  probed, in bar units
    actions: torch.Tensor,         # [B, T, ticks, 20]
    side: torch.Tensor,            # [B] long, 0 = agent is P1
    cfg: RewardConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """-> (reward [B, T], alive_mask [B, T], per-term breakdown).

    Step t's reward is paid for the transition from state t to state t+1 under
    the action at t, so there is one fewer reward than there are states.
    """
    cfg = cfg or RewardConfig()
    B, T, _ = states.shape
    dev = states.device
    if actions.shape[1] != T - 1:
        raise ValueError(
            f"got {T} states and {actions.shape[1]} actions; action t drives the "
            f"transition from state t to state t+1, so there must be exactly "
            f"{T - 1}")
    mine_hp, thr_hp, mine_sp, thr_cb = _sides(states, side)
    btn = _my_buttons(actions, side)                       # [B,T,ticks,10]

    # ---- termination: everything after the first KO is masked out ----
    ko_me, ko_them = ko_masks(states, side, cfg)
    ko_any = ko_me | ko_them
    first_ko = torch.where(ko_any.any(1), ko_any.float().argmax(1),
                           torch.full((B,), T - 1, device=dev))
    steps = torch.arange(T - 1, device=dev)[None, :]
    alive = (steps <= first_ko[:, None]).float()

    # ---- health deltas; increases discarded (see module docstring) ----
    if cfg.damage_mode == "step":
        d_them = (thr_hp[:, 1:] - thr_hp[:, :-1]).clamp(max=0.0).abs()
        d_me = (mine_hp[:, 1:] - mine_hp[:, :-1]).clamp(max=0.0).abs()
    elif cfg.damage_mode == "net":
        # One decrease over the rollout, credited at the last live step, so the
        # probe's per-step residual is not rectified sixteen times over.
        last = first_ko.clamp(max=T - 2)
        gather = last[:, None]
        end_them = thr_hp[:, 1:].gather(1, gather)
        end_me = mine_hp[:, 1:].gather(1, gather)
        tot_them = (end_them - thr_hp[:, :1]).clamp(max=0.0).abs()
        tot_me = (end_me - mine_hp[:, :1]).clamp(max=0.0).abs()
        at_last = (torch.arange(T - 1, device=dev)[None, :] == last[:, None]).float()
        d_them = tot_them * at_last
        d_me = tot_me * at_last
    else:
        raise ValueError(f"unknown damage_mode {cfg.damage_mode!r}; want step or net")

    # ---- card use: ground-truth button, cost inferred from the spirit drop ----
    pressed = btn[..., SPELL].amax(dim=2)                  # [B,T] any tick in chunk
    sp_drop = (mine_sp[:, :-1] - mine_sp[:, 1:]).clamp(min=0.0)
    cast = (pressed > 0.5) & (sp_drop >= cfg.spell_cost_min)
    cost = torch.where(cast, sp_drop, torch.zeros_like(sp_drop))

    # A cast is "active" for the following `spell_window` steps. Damage landed
    # while it is active is what the 2x bonus pays for.
    active = torch.zeros(B, T - 1, device=dev)
    run = torch.zeros(B, device=dev)
    for t in range(T - 1):
        run = torch.where(cast[:, t], float(cfg.spell_window), (run - 1).clamp(min=0))
        active[:, t] = (run > 0).float()

    mult = 1.0 + (cfg.spell_multiplier - 1.0) * active
    r_dealt = cfg.damage_dealt * d_them * mult
    r_taken = cfg.damage_taken * d_me
    r_combo = cfg.combo * thr_cb[:, 1:] * mult

    # A whiff is a paid-for cast that landed nothing before its window expired.
    landed = torch.zeros(B, T - 1, device=dev)
    for t in range(T - 1):
        hi = min(t + cfg.spell_window, T - 1)
        landed[:, t] = d_them[:, t:hi].sum(1)
    r_whiff = cfg.whiff * cost * (landed <= 1e-4).float()

    # ---- guard crush: spirit crossing to zero ----
    crushed = (mine_sp[:, 1:] <= cfg.crush_threshold) & (mine_sp[:, :-1] > cfg.crush_threshold)
    r_crush = cfg.crush * crushed.float()

    # ---- match outcome, paid once at the KO step ----
    # A simultaneous read is a double KO, which is a draw, not a loss. Paying
    # `lose` whenever `ko_me` fires regardless of `ko_them` made every ambiguous
    # reading negative, and with a noisy probe ambiguous readings are common.
    both = ko_me & ko_them
    r_out = (cfg.win * (ko_them & ~both).float() + cfg.lose * (ko_me & ~both).float())
    at_ko = (steps == first_ko[:, None]).float()
    r_out = r_out * at_ko

    # ---- action proxies ----
    held = btn.mean(dim=2)                                  # [B,T,10] duty cycle
    r_fly = cfg.flying * held[..., UP]
    r_idle = cfg.idle * (btn.amax(dim=(2, 3)) < 0.5).float()

    terms = {"dealt": r_dealt, "taken": r_taken, "combo": r_combo, "whiff": r_whiff,
             "crush": r_crush, "outcome": r_out, "flying": r_fly, "idle": r_idle}
    total = sum(terms.values()) * alive
    return total, alive, {k: v * alive for k, v in terms.items()}
