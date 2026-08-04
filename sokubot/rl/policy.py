"""The policy GRPO optimises, and the actuation noise that makes it human.

Input is a short history of latents plus which player the agent is; output is
one decision step's worth of buttons for that player -- `frame_skip` ticks of
them, because the world model consumes actions in chunks of that size.

THE ACTION SPACE IS NOT TWENTY INDEPENDENT BITS
-----------------------------------------------
The CSV stores ten booleans per player, but the game does not: `SWRCHARINPUT`
holds `lr` and `ud` as signed integers, and the extractor derives left/right
from the sign of one number (dll/src/session.cpp). Left and right can therefore
never both be set, and neither can up and down -- no such frame exists anywhere
in 200 hours of training data.

A policy over ten independent Bernoullis can emit those frames, and would, early
in training when it is near-uniform. The world model has never seen one, so it
would be rolled forward into a latent that means nothing, the probe would read a
number off it, and GRPO would optimise against that number. Factoring the two
axes as 3-way categoricals makes impossible inputs unrepresentable rather than
merely unlikely, which keeps imagined rollouts on the manifold the predictor was
trained on.

So one tick is: lr in {none, left, right}, ud in {none, up, down}, and six
independent buttons (a, b, c, d, change, spell).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Positions within a player's ten-button block, matching data/soku.py's order:
# up down left right a b c d change spell
IDX_UP, IDX_DOWN, IDX_LEFT, IDX_RIGHT = 0, 1, 2, 3
FREE_BUTTONS = (4, 5, 6, 7, 8, 9)          # a b c d change spell
N_FREE = len(FREE_BUTTONS)


@dataclass
class PolicyOutput:
    actions: torch.Tensor      # [B, ticks, 10] float 0/1, the agent's own block
    log_prob: torch.Tensor     # [B] summed over ticks and axes
    entropy: torch.Tensor      # [B]


class SokuPolicy(nn.Module):
    """[B, H, latent] + side -> a distribution over one decision step."""

    def __init__(self, latent_dim: int = 192, history: int = 3, ticks: int = 4,
                 width: int = 512, depth: int = 3):
        super().__init__()
        self.ticks = ticks
        self.history = history
        # Which player the agent is. The user asked for the agent to know this,
        # and it is load-bearing: the probe's channels are P1-first, so "my
        # health" is a different column depending on the side.
        self.side_emb = nn.Embedding(2, width)
        layers: list[nn.Module] = [nn.Linear(latent_dim * history, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        # Per tick: 3 lr logits + 3 ud logits + 6 button logits.
        self.head = nn.Linear(width, ticks * (3 + 3 + N_FREE))
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)   # near-uniform at init

    def logits(self, z_hist: torch.Tensor, side: torch.Tensor):
        B = z_hist.shape[0]
        h = self.trunk(z_hist.reshape(B, -1)) + self.side_emb(side)
        o = self.head(h).view(B, self.ticks, 3 + 3 + N_FREE)
        return o[..., :3], o[..., 3:6], o[..., 6:]

    def distributions(self, z_hist: torch.Tensor, side: torch.Tensor):
        lr, ud, btn = self.logits(z_hist, side)
        return (torch.distributions.Categorical(logits=lr),
                torch.distributions.Categorical(logits=ud),
                torch.distributions.Bernoulli(logits=btn))

    @staticmethod
    def _assemble(lr_i: torch.Tensor, ud_i: torch.Tensor,
                  free: torch.Tensor) -> torch.Tensor:
        """Category indices -> the ten-wide button block the game would produce."""
        B, T = lr_i.shape
        a = torch.zeros(B, T, 10, device=lr_i.device, dtype=free.dtype)
        a[..., IDX_LEFT] = (lr_i == 1).to(free.dtype)
        a[..., IDX_RIGHT] = (lr_i == 2).to(free.dtype)
        a[..., IDX_UP] = (ud_i == 1).to(free.dtype)
        a[..., IDX_DOWN] = (ud_i == 2).to(free.dtype)
        a[..., list(FREE_BUTTONS)] = free
        return a

    def forward(self, z_hist: torch.Tensor, side: torch.Tensor,
                sample: bool = True) -> PolicyOutput:
        d_lr, d_ud, d_btn = self.distributions(z_hist, side)
        if sample:
            lr_i, ud_i = d_lr.sample(), d_ud.sample()
            free = d_btn.sample()
        else:
            lr_i, ud_i = d_lr.probs.argmax(-1), d_ud.probs.argmax(-1)
            free = (d_btn.probs > 0.5).float()
        lp = (d_lr.log_prob(lr_i) + d_ud.log_prob(ud_i)).sum(-1) \
            + d_btn.log_prob(free).sum((-1, -2))
        ent = (d_lr.entropy() + d_ud.entropy()).sum(-1) + d_btn.entropy().sum((-1, -2))
        return PolicyOutput(self._assemble(lr_i, ud_i, free), lp, ent)

    def log_prob_of(self, z_hist: torch.Tensor, side: torch.Tensor,
                    actions: torch.Tensor):
        """Re-score stored actions under current parameters (the GRPO ratio).

        Recovers the categorical index from the stored one-hot pair: this is
        lossless because `_assemble` can never set both bits of an axis.
        """
        d_lr, d_ud, d_btn = self.distributions(z_hist, side)
        lr_i = (actions[..., IDX_LEFT] > 0.5).long() + 2 * (actions[..., IDX_RIGHT] > 0.5).long()
        ud_i = (actions[..., IDX_UP] > 0.5).long() + 2 * (actions[..., IDX_DOWN] > 0.5).long()
        free = actions[..., list(FREE_BUTTONS)]
        lp = (d_lr.log_prob(lr_i) + d_ud.log_prob(ud_i)).sum(-1) \
            + d_btn.log_prob(free).sum((-1, -2))
        ent = (d_lr.entropy() + d_ud.entropy()).sum(-1) + d_btn.entropy().sum((-1, -2))
        return lp, ent


def jitter_actions(actions: torch.Tensor, sigma_frames: float = 1.0,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Shift each chunk along the tick axis by a rounded N(0, sigma) frames.

    The intent is the user's: the agent presses slightly off from where it meant
    to, so it plays like a person rather than a frame-perfect machine. Shifting
    the whole chunk in time -- rather than flipping individual bits -- is what
    makes it a *timing* error; a button flip would be a different button, which
    is a different mistake.

    Vacated ticks repeat the edge value, so a shift never invents a neutral
    frame in the middle of a held input.
    """
    B, T, _ = actions.shape
    dev = actions.device
    shift = torch.round(
        torch.randn(B, device=dev, generator=generator) * sigma_frames
    ).long().clamp(-(T - 1), T - 1)
    idx = (torch.arange(T, device=dev)[None, :] - shift[:, None]).clamp(0, T - 1)
    return actions.gather(1, idx[:, :, None].expand(B, T, actions.shape[2]))


def to_joint(mine: torch.Tensor, theirs: torch.Tensor,
             side: torch.Tensor) -> torch.Tensor:
    """Two ten-wide blocks -> the twenty-wide vector the world model consumes.

    `side` says which block is the agent's, so the halves land in the order the
    predictor was trained on: P1 first, always.
    """
    B, T, _ = mine.shape
    p1 = torch.where(side[:, None, None] == 0, mine, theirs)
    p2 = torch.where(side[:, None, None] == 0, theirs, mine)
    return torch.cat([p1, p2], dim=-1)
