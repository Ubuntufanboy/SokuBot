"""Next-embedding prediction loss (LeWM Eq. 1).

    L_pred = || zhat_{t+1} - z_{t+1} ||^2

Teacher forcing: the predictor always conditions on encoder latents of *real*
observations, never on its own previous outputs. Because the predictor is causal
over the whole window, one forward pass gives every next-step prediction at once
(LeWM Alg. 3: ``mse(emb[:, 1:], next_emb[:, :-1])``).

There is no stop-gradient on the target. The encoder is pulled by this loss from
both sides -- it must produce representations that are predictable *and* that it
can predict -- and SIGReg is what keeps the shared solution from being a
constant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prediction_loss(zhat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """zhat, z: [B, T, D]. Compares zhat[:, :-1] against z[:, 1:]."""
    if zhat.shape != z.shape:
        raise ValueError(f"shape mismatch: zhat {tuple(zhat.shape)} vs z {tuple(z.shape)}")
    if z.shape[1] < 2:
        raise ValueError("need at least 2 timesteps to form a next-step target")
    return F.mse_loss(zhat[:, :-1], z[:, 1:])
