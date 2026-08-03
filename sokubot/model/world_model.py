"""LeWorldModel: the trainable stack (encoder + action encoder + predictor).

    z_t     = enc(o_t)
    zhat_t+1 = pred(z_<=t, a_t)

Trained end to end with ``L_pred + lambda * SIGReg(Z)``. There is deliberately
**no stop-gradient, no EMA, and no target encoder** -- gradients flow into the
prediction target as well as the prediction, which is the whole point of LeWM's
"stable end-to-end JEPA" claim. SIGReg plus the projector's BatchNorm are what
stop the encoder from collapsing to a constant.

(The one place a stop-gradient does appear is AdaJEPA's test-time adaptation;
see ``planning/adajepa.py``.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import Config
from .action_encoder import ActionEncoder
from .encoder import ViTEncoder
from .predictor import LatentPredictor


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@dataclass
class ForwardOut:
    z: torch.Tensor        # [B, T, latent]   encoder latents
    zhat: torch.Tensor     # [B, T, latent]   zhat[:, t] predicts z[:, t+1]


class LeWorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)
        self.predictor = LatentPredictor(cfg)

    # ---------------- training ----------------
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> ForwardOut:
        """obs: [B, T, 3, S, S], actions: [B, T, ticks, action_dim]."""
        z = self.encoder(obs)
        cond = self.action_encoder(actions)
        return ForwardOut(z=z, zhat=self.predictor(z, cond))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    # ---------------- planning ----------------
    def rollout(
        self,
        z_ctx: torch.Tensor,                      # [B, H, latent]
        a_plan: torch.Tensor,                     # [B, P, ticks, action_dim]
        a_hist: Optional[torch.Tensor] = None,    # [B, H-1, ticks, action_dim]
    ) -> torch.Tensor:
        """Autoregressive latent rollout under a candidate action sequence.

        Returns [B, P, latent]: the predicted latents after each planned action.

        The predictor is causal over a fixed-width window, so each step slides
        the window forward by one: the newly predicted latent and the action
        that produced it replace the oldest pair. ``a_hist`` is what was actually
        executed at the context positions; when it is unknown (e.g. at the very
        start of an episode) zeros are used, which biases only the first
        ``H-1`` predictions.
        """
        B, H, _ = z_ctx.shape
        P = a_plan.shape[1]
        if a_hist is None:
            a_hist = torch.zeros(
                B, max(0, H - 1), self.cfg.action_ticks, self.cfg.action_dim,
                device=z_ctx.device, dtype=a_plan.dtype,
            )
        elif a_hist.shape[1] != H - 1:
            raise ValueError(
                f"a_hist has {a_hist.shape[1]} steps, expected H-1 = {H - 1}"
            )

        z_win, a_win = z_ctx, a_hist
        preds = []
        for k in range(P):
            a_full = torch.cat([a_win, a_plan[:, k : k + 1]], dim=1)   # [B, H, ...]
            cond = self.action_encoder(a_full)
            zhat = self.predictor(z_win, cond)[:, -1]                  # [B, latent]
            preds.append(zhat)
            z_win = torch.cat([z_win[:, 1:], zhat.unsqueeze(1)], dim=1)
            if H > 1:
                a_win = torch.cat([a_win[:, 1:], a_plan[:, k : k + 1]], dim=1)
        return torch.stack(preds, dim=1)

    # ---------------- introspection ----------------
    def param_report(self) -> Dict[str, int]:
        rep = {
            "encoder": count_params(self.encoder),
            "action_encoder": count_params(self.action_encoder),
            "predictor": count_params(self.predictor),
        }
        rep["total"] = sum(rep.values())
        return rep
