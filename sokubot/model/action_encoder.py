"""Action-chunk encoder: one frame-skip block of raw actions -> a condition vector.

A model step spans ``frame_skip`` environment ticks, so the action for one step
is a chunk of shape [ticks, action_dim] (LeWM App. D: "grouping consecutive
actions between frames into a single action block"). This module flattens that
chunk and maps it to the width the predictor's AdaLN modulation expects.

It is deliberately a plain differentiable MLP: the gradient-based planner
optimises actions by backpropagating the goal cost through this module, so
nothing here may quantise or detach its input. During training the Soku channels
are 0/1; during planning they are relaxed to [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class ActionEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.action_chunk_dim, cfg.act_hidden)
        self.fc2 = nn.Linear(cfg.act_hidden, cfg.pred_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """a: [B, T, ticks, action_dim] (or [B, T, ticks*action_dim]) -> [B, T, pred_dim]."""
        if a.dim() == 4:
            a = a.flatten(-2)
        if a.shape[-1] != self.cfg.action_chunk_dim:
            raise ValueError(
                f"action chunk has width {a.shape[-1]}, expected "
                f"{self.cfg.action_chunk_dim} (= {self.cfg.action_ticks} ticks "
                f"x {self.cfg.action_dim} dims)"
            )
        return self.fc2(F.gelu(self.fc1(a)))
