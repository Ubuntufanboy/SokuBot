"""Actor-critic in imagination: the pieces GRPO's update does not have.

This deliberately reuses `grpo_loss` for the actor rather than writing a second
clipped surrogate. The trust-region machinery there -- the clamped log-ratio, the
k3 KL to the sampling policy, the anchored KL to the corpus-prior reference, the
entropy floor's multiplier -- was arrived at by four runs that collapsed
identically, and none of it is specific to how the advantage was estimated.
What changes is only *where the advantage comes from*:

    GRPO   advantage = return-to-go, centred against G rollouts of the same start
    here   advantage = lambda-return - v(s),  with the critic supplying v

so the group disappears and, with it, the factor of `group_size` that GRPO spent
on baseline estimation rather than on covering start states.

WHAT THIS INHERITS ON PURPOSE
-----------------------------
`ACConfig` subclasses `GRPOConfig` instead of copying its fields. That is not
tidiness: `grpo_loss` reads `clip_eps`, `kl_coef`, `kl_ref_coef`, `entropy_coef`
and `max_log_ratio` off whatever config it is handed, and a parallel dataclass
would let those drift apart silently -- which is exactly the class of bug
`docs/BUGS.md` is a catalogue of. Subclassing makes a missing field a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .critic import CriticConfig
from .grpo import GRPOConfig
from .ood import OODConfig


@dataclass
class ACConfig(GRPOConfig):
    """GRPO's trust region, with a critic replacing the group baseline."""

    # One rollout per start. The group existed only to estimate a baseline.
    group_size: int = 1
    # 16 decision steps = 1.07 s. Longer than GRPO's 4, because the critic
    # bootstraps the tail instead of the return having to cover it: the rollout
    # now only has to be accurate over its own length, and `horizon_ablation`
    # puts rollout cosine at 0.83 out to sixteen steps.
    horizon: int = 16
    starts_per_batch: int = 1024
    critic: CriticConfig = field(default_factory=CriticConfig)
    # The OOD guard is part of this change rather than a later one: horizon 16 is
    # four times the measured trustworthy horizon, and the critic's bootstrap
    # justifies that for credit assignment but says nothing about fidelity.
    ood: OODConfig = field(default_factory=OODConfig)
    # Weight on the critic's own regression loss. The two heads are separate
    # networks with separate optimisers, so this is only here to make a single
    # combined number loggable.
    critic_coef: float = 1.0
    # Use both chairs of a self-play rollout. Off for a like-for-like comparison
    # against the GRPO baseline; see ImaginedArena.rollout's docstring for why it
    # is only valid when the opponent *is* the current policy.
    two_sided: bool = True


def advantages_from_returns(lam_ret: torch.Tensor, value: torch.Tensor,
                            alive: torch.Tensor, scale: str = "batch",
                            eps: float = 1e-6) -> torch.Tensor:
    """lambda-return minus the critic's estimate, normalised the way GRPO's was.

    `value` is v(s_t) for t = 0..T-1, i.e. `lam_ret`'s own states -- not the
    bootstrap value, which has already been consumed inside the return.

    The scaling is inherited from `group_advantages` and for the same reason
    given there: dividing by a *per-timestep* batch spread rather than one scalar
    keeps early and late steps comparable, since return-to-go shrinks as the
    horizon runs out. Dead steps are excluded from the statistics -- they are
    identically zero, and letting them into the standard deviation would shrink
    it in proportion to how often trajectories terminate, quietly inflating the
    advantage of every surviving trajectory as the policy got better at KOs.
    """
    if lam_ret.shape != value.shape:
        raise ValueError(
            f"lam_ret {tuple(lam_ret.shape)} and value {tuple(value.shape)} "
            f"must match; value must be v(s_t) over the same steps")
    adv = lam_ret - value
    if scale == "none":
        return adv * alive
    if scale != "batch":
        raise ValueError(f"unknown scale {scale!r}; want batch or none")
    m = alive > 0
    n = m.sum(dim=0, keepdim=True).clamp(min=1)
    mean = (adv * m).sum(dim=0, keepdim=True) / n
    var = (((adv - mean) ** 2) * m).sum(dim=0, keepdim=True) / n
    return ((adv - mean) / (var.sqrt() + eps)) * alive
