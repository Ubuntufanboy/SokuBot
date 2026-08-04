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

# Column order of the probe's output, fixed by scripts/horizon_ablation.py.
HP1, HP2, SPIRIT1, SPIRIT2, COMBO1, COMBO2 = range(6)

# Button offsets within one player's 10-wide block (see data/soku.py).
UP, DOWN, LEFT, RIGHT, A, B, C, D, CHANGE, SPELL = range(10)


@dataclass
class RewardConfig:
    """Weights are in units of "bars of health", so 1.0 == a full health bar."""

    damage_dealt: float = 1.0
    damage_taken: float = -1.0

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

    ko_threshold: float = 0.02
    # A spirit drop this large at a card press means a card actually went off.
    # One orb is 0.2 of the gauge; a real cast spends at least one.
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
    ko_me = mine_hp[:, 1:] <= cfg.ko_threshold
    ko_them = thr_hp[:, 1:] <= cfg.ko_threshold
    ko_any = ko_me | ko_them
    first_ko = torch.where(ko_any.any(1), ko_any.float().argmax(1), torch.full((B,), T - 1, device=dev))
    steps = torch.arange(T - 1, device=dev)[None, :]
    alive = (steps <= first_ko[:, None]).float()

    # ---- health deltas; increases discarded (see module docstring) ----
    d_them = (thr_hp[:, 1:] - thr_hp[:, :-1]).clamp(max=0.0).abs()   # I dealt this
    d_me = (mine_hp[:, 1:] - mine_hp[:, :-1]).clamp(max=0.0).abs()   # I took this

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
    r_out = (cfg.win * (ko_them & ~ko_me).float() + cfg.lose * ko_me.float())
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
