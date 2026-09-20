"""Projectiles as a spatial map, not as regressed coordinates.

WHY THE COORDINATE VERSION FAILED
----------------------------------
Asking a pooled feature vector for "the nearest projectile's dx" is the same
mistake that made character position hard, in a worse form. Trained on 2003
replays -- thirteen times the coverage that fixed everything else -- the slot
regression peaked at step 4500 and then went backwards:

    present  +0.133 -> +0.122      proj_n   +0.062 -> +0.017
    hb       +0.179 -> +0.137      proj_hb  +0.019 -> -0.084
    dx, dy, vx, closing            ~0, then negative

Presence saturates around 0.15 and geometry never starts. The network finds a
weak global cue for "something is in the air" and cannot say where, because a
single coordinate output has to be produced by averaging over a feature map
that contains several projectiles in different places. Averaging over the
spatial axes is precisely the operation that discards where something is --
which is why `Net`'s soft-argmax head exists for the characters.

WHAT THIS DOES INSTEAD
-----------------------
Predict an OCCUPANCY MAP over projectile positions relative to the player they
are flying at, one map per owner. Dense supervision, one target per cell, and
the head is the thing the architecture is already good at.

Slot-0 `dx`/`dy` are then read off the map by soft-argmax rather than
regressed, so the POLICY'S INPUT DOES NOT CHANGE: it still receives
`[2, slots, 7]` with the danger-slot filled. Nothing downstream has to be
retrained to benefit from a better-conditioned target.

Relative rather than screen coordinates on purpose. The camera pans and zooms
-- measured, the encoder's positional error nearly triples as the players
separate and the view pulls back -- so a screen-space target would make the
network learn the camera as well as the projectile. Relative-to-target is also
the frame the policy reasons in: "something is coming at me from over there".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The map covers +-RANGE game units around the target player, which contains
# essentially every projectile that matters; anything further away is not a
# threat inside the policy's horizon.
RANGE = 600.0
GRID = 16


def render_targets(proj: torch.Tensor, present_idx: int, dx_idx: int,
                   dy_idx: int, stage_span: float, sigma: float = 1.2
                   ) -> torch.Tensor:
    """[B, 2, slots, F] -> [B, 2, GRID, GRID] occupancy, per owner.

    Every live projectile contributes a Gaussian blob at its relative position.
    Blobs rather than single cells because a hard one-hot target punishes a
    one-cell miss exactly as hard as a whole-screen miss, and the gradient then
    says nothing about which direction was closer.
    """
    B, P, S, _ = proj.shape
    dev = proj.device
    grid = torch.linspace(-RANGE, RANGE, GRID, device=dev)
    gx = grid.view(1, 1, 1, GRID, 1)
    gy = grid.view(1, 1, 1, 1, GRID)
    live = (proj[..., present_idx] > 0.5).float()               # [B,P,S]
    px = (proj[..., dx_idx] * stage_span).unsqueeze(-1).unsqueeze(-1)
    py = (proj[..., dy_idx] * stage_span).unsqueeze(-1).unsqueeze(-1)
    step = 2 * RANGE / (GRID - 1)
    blob = torch.exp(-(((gx - px) ** 2 + (gy - py) ** 2)
                       / (2 * (sigma * step) ** 2)))            # [B,P,S,G,G]
    return (blob * live.unsqueeze(-1).unsqueeze(-1)).amax(2).clamp(0, 1)


class ProjHead(nn.Module):
    """Feature map -> per-owner projectile occupancy, plus a soft-argmax read.

    Two outputs from one map, and they are trained together on purpose: the
    occupancy gives dense supervision everywhere, and the soft-argmax gives the
    single coordinate the policy's observation slot actually wants. Training
    only the coordinate is what failed; training only the map would leave the
    conversion untested.
    """

    def __init__(self, ch: int, owners: int = 2, width: int = 128,
                 coords: bool = True):
        super().__init__()
        self.owners = owners
        # THE FRAME MISMATCH, which is the likelier reason this plateaus.
        #
        # `to_map` is a stack of 3x3 convolutions over a SCREEN-SPACE feature
        # map. `render_targets` renders the occupancy map in PLAYER-RELATIVE
        # coordinates. Nothing converted between them.
        #
        # A convolution is translation-equivariant in its input frame, so
        # asking it to emit at screen cell (i, j) the evidence for RELATIVE
        # offset (i, j) requires subtracting the player's position -- a global
        # operation a 3x3 kernel cannot express, and it was never given the
        # player's position anyway. The camera keeps the midpoint near screen
        # centre, so screen and relative position are correlated, which is why
        # the head reached 17% better than baseline instead of 0%; the
        # correlation decays exactly as the players separate and the view zooms
        # out, which is where the live log already measured positional error
        # tripling.
        #
        # Two extra input planes fix the expressibility: a normalised
        # coordinate grid (CoordConv), and the target player's own predicted
        # position broadcast across the map. With both, "relative" is a
        # subtraction the head can actually compute.
        self.coords = coords
        if coords:
            # grid x, grid y, then BOTH players' (x, y). Both, rather than the
            # one reference player, because owner map p is relative to player
            # 1-p: with both available as planes the final 1x1 conv can route
            # each output channel to the subtraction it needs, and a single
            # shared conv stack still produces both maps in one pass.
            ch = ch + 6

        # A REAL head, not a linear readout.
        #
        # The first version was a single 1x1 conv: 514 parameters mapping 256
        # channels to 2 maps. That cannot learn to FIND a projectile, it can
        # only reweight features the backbone already computes, and it plateaued
        # 11% better than "assume the projectile is on top of the player" at
        # every loss weight. Concluding from that that projectile position is
        # unlearnable -- or that the input resolution is at fault -- would have
        # been blaming the data for a head with no capacity.
        #
        # 3x3 convolutions because detection is a local-neighbourhood question:
        # a projectile is a small bright region, which is a spatial pattern, not
        # a per-cell channel mixture.
        self.to_map = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, owners, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(GRID)

    def forward(self, feat: torch.Tensor, player: torch.Tensor | None = None):
        """feat [B,C,h,w], player [B,2,2] -- both players' (x, y) as the
        encoder itself predicts them -- -> (logits [B,owners,G,G],
        coords [B,owners,2]).

        `player` is the encoder's OWN prediction, not the label. Feeding the
        truth here would train the head on information it will not have at
        inference, which is the same mistake as a probe fitted on one
        checkpoint and read on another.
        """
        if self.coords:
            B, _, h, w = feat.shape
            ys = torch.linspace(-1, 1, h, device=feat.device).view(1, 1, h, 1)
            xs = torch.linspace(-1, 1, w, device=feat.device).view(1, 1, 1, w)
            grid = torch.cat([xs.expand(B, 1, h, w), ys.expand(B, 1, h, w)], 1)
            if player is None:
                # Zeros, not an error: an arm that ablates the player channels
                # is worth running, and it must differ from the full head only
                # in what it is TOLD, not in its parameter count.
                player = feat.new_zeros(B, 2, 2)
            pl = player.reshape(B, 4, 1, 1).expand(B, 4, h, w)
            feat = torch.cat([feat, grid, pl], dim=1)
        logits = self.pool(self.to_map(feat))
        B = logits.shape[0]
        p = torch.softmax(logits.flatten(2), -1).view(B, self.owners, GRID, GRID)
        axis = torch.linspace(-RANGE, RANGE, GRID, device=feat.device)
        dx = (p.sum(-1) * axis).sum(-1)      # expectation over the x axis
        dy = (p.sum(-2) * axis).sum(-1)
        return logits, torch.stack([dx, dy], -1)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-cell BCE. Positives are rare, so they are upweighted by their
        own rarity rather than by a hand-set constant."""
        pos = target.mean().clamp(1e-4, 1 - 1e-4)
        w = torch.where(target > 0.05, (1 - pos) / pos, torch.ones_like(target))
        return (F.binary_cross_entropy_with_logits(
            logits, target, reduction="none") * w).mean()
