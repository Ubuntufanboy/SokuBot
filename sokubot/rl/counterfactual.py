"""Damage avoided: what the agent's actions were worth, against doing nothing.

WHAT THIS IS FOR
----------------
The agent does not block, and the reward as written gives it no way to discover
why it should. `damage_taken` pays for the damage that *did* land, so a rollout
where the agent guarded perfectly and one where it stood still are separated only
by the difference in what got through -- a small number, buried in a return whose
across-state variance is 800x its action-driven part
(`scripts/baseline_quality.py`).

The counterfactual states the lesson directly. Roll the same start twice: once
with the agent's actions, once with the agent doing nothing at all. The
difference in damage taken is what the agent's choices were worth defensively,
and crediting it says "holding that direction is why that combo did 0.02 instead
of 0.09" in one number.

    avoided = damage_taken(null) - damage_taken(mine)

Positive means the agent defended. It is zero for a policy that does nothing,
by construction, so it cannot be farmed by idling.

THE OPPONENT MUST NOT REACT
---------------------------
Both arms replay the *same* opponent action sequence, taken from the agent's own
rollout. This is the whole reason the comparison means anything: an opponent
policy re-rolled against the counterfactual state would see a different situation
and act differently, and the damage difference would then contain its reaction
as well as the agent's choice. The world model is deterministic given a start and
an action sequence, so with the opponent pinned the two arms differ in exactly
one thing.

WHY IT IS ANNEALED
------------------
Because "never take damage" is not the game. A policy optimising avoided damage
alone learns to hold guard forever, which loses slowly instead of quickly --
in Soku a permanent blocker is opened up by throws and guard crush, and does no
damage in the meantime. The term is a teaching signal for the *causal* link
between guarding and not being hit, and it should decay once that link is
established, leaving the ordinary damage exchange to decide how much guarding is
worth. `anneal` returns the coefficient at a given step.
"""

from __future__ import annotations

import torch

from .reward import compute_rewards


def anneal(step: int, coef0: float, half_life: int) -> float:
    """Exponential decay. `half_life` steps to halve, 0 disables the decay.

    Exponential rather than linear-to-zero because the point is to hand over
    smoothly rather than to switch the lesson off on a particular step, and
    because a term that reaches exactly zero at a known step invites reading the
    curve either side of it as an effect.
    """
    if coef0 <= 0:
        return 0.0
    if half_life <= 0:
        return coef0
    return coef0 * (0.5 ** (step / half_life))


@torch.no_grad()
def damage_avoided(arena, z_ctx: torch.Tensor, a_hist: torch.Tensor,
                   side: torch.Tensor, traj: dict) -> torch.Tensor:
    """-> [B, T] damage the agent's actions prevented, against doing nothing.

    Costs one extra rollout of the predictor per batch. That is the same price
    the group baseline pays for one extra group member, and unlike a group member
    this one answers a question the group cannot: every member of a group is an
    *alternative action*, so their spread says which action is better, never
    whether acting at all was worth anything.
    """
    from .grpo import ReplayOpponent

    T = arena.cfg.horizon
    B = z_ctx.shape[0]
    dev = z_ctx.device
    # The null agent: every button released, for the whole rollout. `NullPolicy`
    # is deliberately not a policy object -- it is fed through ReplayOpponent's
    # path so that nothing about it can depend on the state it sees.
    null = torch.zeros(B, T, arena.ticks, 10, device=dev)

    # Same start, same opponent inputs, agent does nothing.
    #
    # jitter is switched off for this arm, and that is not a detail.
    # `traj["joint"]` is recorded *after* jitter_actions has shifted it, so
    # replaying it through a jittering rollout would shift it a second time --
    # the two arms would then face different opponent timing and the difference
    # would no longer isolate the agent. The null actions are all zeros, which a
    # tick-shift leaves unchanged, so nothing else is affected.
    sigma = arena.cfg.jitter_sigma
    arena.cfg.jitter_sigma = 0.0
    try:
        cf = arena.rollout(z_ctx, a_hist, side, _Frozen(null),
                           ReplayOpponent(traj["joint"]))
    finally:
        arena.cfg.jitter_sigma = sigma
    mine_taken = traj["terms"]["taken"]            # negative: damage to me
    null_taken = cf["terms"]["taken"]
    # `taken` is negative, so (mine - null) is positive when the agent took less.
    return mine_taken - null_taken


class _Frozen:
    """Replays a fixed action sequence as if it were the agent's policy.

    `ImaginedArena.rollout` asks the policy for an action at each step, so the
    null arm needs an object with that shape. It cannot be a `SokuPolicy` with
    zeroed weights -- that would still sample, and the counterfactual has to be
    exactly "nothing", not "nothing on average".
    """

    def __init__(self, actions: torch.Tensor):
        self.actions = actions
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def __call__(self, z_win, side, sample: bool = True):
        from .policy import PolicyOutput
        a = self.actions[:, min(self.t, self.actions.shape[1] - 1)]
        self.t += 1
        z = a.new_zeros(a.shape[0])
        return PolicyOutput(actions=a, log_prob=z, entropy=z)
