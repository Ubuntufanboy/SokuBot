"""A state-space value head, and lambda-returns, so credit can outlive the horizon.

This is the critic for `train_state_grpo.py`, which rolls the 33-channel state
simulator. It lives apart from `critic.py`, which is the latent-space critic
that `ac.py`, `train_ac.py` and `baseline_quality.py` still use: the two share a
name for `lambda_returns` and take DIFFERENT arguments (here `alive` and
`terminal` separately, there a single `cont`), so they cannot occupy one module
without one caller silently passing the wrong tensor.

WHY THIS EXISTS
---------------
GRPO scores a whole rollout with one number and compares it against the other
members of its group. That makes the quality of the estimate the quality of the
LONGEST rollout, and it caps what the agent can learn at what imagination can
still describe. Measured on this simulator, that is not far: rolling with the
real recorded actions, health error grows from 0.05 sigma at 8 steps to 0.15 at
24 and 0.19 at 32, and `dx` -- the separation an approach is trying to change --
reaches 0.57 sigma (210 game units) by step 24.

So the horizon cannot simply be raised. Anything whose payoff lands past a
couple of seconds is invisible to the objective:

  approach     the agent sits at 268 units and closes only 7 inside an 8-step
               window, while holding toward would close 56 -- retreat is
               locally optimal because backing off avoids damage NOW and the
               offence given up is paid after the window shuts.
  okizeme      +1.6 HP/step against neutral's +23.4, on the same policy.
  winning      a KO lands inside the rollout on 0.8-2.7% of starts even when
               the start is chosen with a player under 8% health. A sparse
               match-outcome reward is unlearnable by rollout alone: over 97%
               of episodes would carry no signal at all.

A critic fixes the shape of the problem rather than the size. Imagination only
has to be right over H short steps; everything beyond H comes from v(s_H),
which is fitted across millions of rollouts. A win 60 seconds away never has to
be imagined -- it only has to be *predicted*, and the 1-3% of rollouts that do
contain a KO are what teach the prediction. At 512 rollouts a step that is tens
of thousands of real outcomes over a run.

The second benefit is compute, and it is the one that answers "the models keep
getting stuck": GRPO spends G rollouts per start purely to build a baseline.
The critic is that baseline, so the same simulator budget covers G times as
many DISTINCT start states. Same cost, eight times the state diversity.

WHY TWOHOT AND SYMLOG
----------------------
The reward spans three orders of magnitude in this game: chip damage is ~0.001
of a health bar per step while a match outcome is 1.0, and combos land 0.025 in
a step. A scalar MSE critic fits the large values and treats the small ones as
noise, which is backwards -- the small ones are almost all of the signal. So
values are predicted as a distribution over symlog-spaced bins and trained with
cross-entropy against a two-hot target, which gives the same relative resolution
at every magnitude. This is DreamerV3's construction and it is used here for
its reason, not its pedigree.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .critic import symexp, symlog


class StateCritic(nn.Module):
    """v(observation) as a distribution over symlog-spaced bins.

    Shares nothing with the policy on purpose. Weight sharing would couple the
    value loss into the action distribution, and the entropy floor and bounded
    logits that stopped four straight policy collapses were tuned against a
    trunk that only ever saw the policy gradient.
    """

    def __init__(self, obs_dim: int, history: int, bins: int = 41,
                 vmax: float = 20.0, width: int = 512):
        super().__init__()
        self.bins = bins
        # Bin centres are symlog-spaced, so resolution is proportional rather
        # than absolute: as fine near 0.001 as it is near 1.0.
        edge = symlog(torch.tensor(vmax))
        self.register_buffer("centres", symexp(torch.linspace(-edge, edge, bins)))
        self.net = nn.Sequential(
            nn.Linear(obs_dim * history, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, bins),
        )
        # Start at "value zero everywhere" rather than at a random opinion: an
        # untrained critic that confidently disagrees with reality produces a
        # large early advantage that the policy chases before the critic has
        # learned anything.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [..., H, dim] -> [..., bins]."""
        return self.net(obs.flatten(-2))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Expected value under the predicted distribution. [..., ]"""
        p = torch.softmax(self.logits(obs), dim=-1)
        return (p * self.centres).sum(-1)

    def loss(self, obs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the two-hot encoding of `target`."""
        return -(self.twohot(target) * torch.log_softmax(self.logits(obs), -1)
                 ).sum(-1)

    def twohot(self, x: torch.Tensor) -> torch.Tensor:
        """Split each value between its two neighbouring bins, linearly."""
        c = self.centres
        x = x.clamp(c[0], c[-1])
        hi = torch.searchsorted(c.contiguous(), x.contiguous().unsqueeze(-1)
                                ).squeeze(-1).clamp(1, len(c) - 1)
        lo = hi - 1
        c_lo, c_hi = c[lo], c[hi]
        w_hi = ((x - c_lo) / (c_hi - c_lo).clamp(min=1e-8)).clamp(0, 1)
        out = torch.zeros(*x.shape, len(c), device=x.device, dtype=x.dtype)
        out.scatter_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        out.scatter_add_(-1, lo.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        return out


@torch.no_grad()
def lambda_returns(reward: torch.Tensor, value: torch.Tensor,
                   alive: torch.Tensor, terminal: torch.Tensor,
                   gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """Backward recursion for V^lambda. reward/alive/terminal [B,T], value [B,T+1].

        V_t = r_t + gamma * ((1 - lam) * v_{t+1} + lam * V_{t+1})
        V_T = v_T

    `terminal` zeroes the bootstrap where the episode genuinely ended, which is
    the distinction that makes a KO worth something: a truncated rollout must
    carry value across its edge, a finished one must not, and a single `alive`
    mask cannot tell them apart.
    """
    T = reward.shape[1]
    out = torch.zeros_like(reward)
    nxt = value[:, T]
    for t in range(T - 1, -1, -1):
        cont = (1.0 - terminal[:, t])
        nxt = reward[:, t] + gamma * cont * ((1 - lam) * value[:, t + 1]
                                             + lam * nxt)
        out[:, t] = nxt
    return out * alive
