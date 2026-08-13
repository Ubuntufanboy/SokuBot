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
from ..data.hud import FRAME_HUD_CHANNELS
from .action_encoder import ActionEncoder
from .encoder import ViTEncoder
from .predictor import LatentPredictor
from .inverse_dynamics import InverseDynamicsHead
from .state_head import ProjectileHead, StateHead


N_HUD = len(FRAME_HUD_CHANNELS)


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
        # Supervised readout of the HUD from the latent. Its job is not to be
        # used at inference -- `data/hud.py` reads the HUD from pixels far more
        # accurately than this ever will -- but to put a *gradient* on the
        # encoder that says "carry health, spirit and combo". Without it nothing
        # in the objective asks for them, and they arrive at R^2 0.88 / 0.05 /
        # 0.32 respectively, which is what makes the reward unreadable.
        #
        # A single linear layer on purpose: a deeper head would recover the
        # channels from a latent that does not linearly expose them, which is
        # precisely the property the reward needs and would therefore hide the
        # failure it exists to prevent. This mirrors `probe.py`'s argument for a
        # linear probe.
        self.hud_head = (nn.Linear(cfg.latent_dim, N_HUD)
                         if cfg.hud_coef > 0 else None)
        # Reads a latent *transition* and names the buttons that caused it. See
        # model/inverse_dynamics.py for why its gradient is allowed into the
        # encoder when the counterfactual loss's deliberately is not.
        self.idm_head = (InverseDynamicsHead(cfg, cfg.idm_width)
                         if getattr(cfg, "idm_coef", 0.0) > 0 else None)
        # Supervised game state -- separation, facing, guard, knockdown. Read
        # out of the game's memory at capture time and never available at
        # inference; see model/state_head.py for why the model is told this
        # rather than asked to infer it.
        self.state_head = (StateHead(cfg)
                           if getattr(cfg, "state_coef", 0.0) > 0 else None)
        # What each player has in the air. A separate head rather than more
        # channels on the state head, because the target is a different kind of
        # thing: a variable-length set whose continuous terms are only defined
        # where an object exists. Optional because a corpus captured before the
        # 24-slot walk carries no such labels at all.
        self.proj_head = (ProjectileHead(cfg, getattr(cfg, "proj_slots", 8))
                          if getattr(cfg, "proj_coef", 0.0) > 0 else None)

    # ---------------- training ----------------
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> ForwardOut:
        """obs: [B, T, 3, S, S], actions: [B, T, ticks, action_dim]."""
        z = self.encoder(obs)
        cond = self.action_encoder(actions)
        return ForwardOut(z=z, zhat=self.predictor(z, cond))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_hud(self, z: torch.Tensor) -> torch.Tensor:
        """[..., latent] -> [..., N_HUD] in [0,1]. Requires cfg.hud_coef > 0."""
        if self.hud_head is None:
            raise RuntimeError(
                "this model was built with hud_coef = 0, so it has no HUD head")
        return torch.sigmoid(self.hud_head(z))

    def predict_state(self, z: torch.Tensor) -> torch.Tensor:
        """[..., latent] -> [..., 2, len(STATE_CHANNELS)]. Needs state_coef > 0.

        Continuous channels are raw and binary channels are LOGITS, because
        `state_head.state_loss` applies BCE-with-logits. Anything reading these
        for display has to apply its own sigmoid.
        """
        if self.state_head is None:
            raise RuntimeError(
                "this model was built with state_coef = 0, so it has no state "
                "head")
        return self.state_head(z)

    def predict_projectiles(self, z: torch.Tensor) -> torch.Tensor:
        """[..., latent] -> [..., 2, slots, len(PROJ_FEATURES)].

        `present` and `hb` are LOGITS; the rest are raw. Index 1 is the OWNER,
        so what threatens player p is `out[:, 1 - p]`.
        """
        if self.proj_head is None:
            raise RuntimeError(
                "this model was built with proj_coef = 0, so it has no "
                "projectile head")
        return self.proj_head(z)

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
