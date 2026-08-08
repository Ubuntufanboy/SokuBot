"""Recover the actions from a latent transition, with gradients into the encoder.

WHY THIS TERM EXISTS
--------------------
`scripts/spatial_probe.py` measured that the encoder does not represent where the
characters are: mirroring the play area while holding the HUD fixed is
undetectable from the latent (AUC 0.540) while mirroring the whole frame is easy
(0.956). Position is not weakly held, it is absent -- and blocking, dodging and
spacing are all positional, which is most of Hisoutensoku.

The cause is structural rather than a bad hyper-parameter. JEPA predicts its own
latent, so encoder and predictor can jointly agree to represent only what is easy
to predict and still score a low loss. SIGReg prevents total collapse but says
nothing about *which* content survives. What survived is slow, large and smooth
-- health at R^2 0.88, the KO banner at AUC 0.947. What was dropped is fast or
small: position, spirit at R^2 0.036, projectiles. Dropping the hard-to-predict
parts lowers the loss, so the objective rewarded exactly the wrong thing.

Inverse dynamics inverts that pressure. To name the buttons that produced a
transition, the latent has to keep whatever distinguishes them -- which is pose,
position and contact, the controllable content. Where prediction prefers what is
*predictable*, this prefers what is *controllable*, and those are close to
complements.

BOTH PLAYERS, WHICH MATTERS MORE THAN IT LOOKS
-----------------------------------------------
The action vector is all twenty buttons, ours and the opponent's. So the head
must recover what the *opponent* did too, and a latent that dropped the
opponent's position or their projectile could not. In a single-agent setting an
inverse-dynamics objective is often criticised for discarding everything the
agent cannot control; here the "agent" spans both players, so that failure mode
mostly closes.

WHY THE GRADIENT REACHES THE ENCODER, WHEN `counterfactual_loss` DELIBERATELY
DOES NOT
-----------------------------------------------------------------------------
`scripts/finetune_action.py` takes `z` detached, and says why: letting the term
reshape the encoder "would hand it a second way to cheat, by making latents that
are easy to tell apart rather than predictions that are accurate". That is right
when prediction is the objective. It is exactly wrong when the problem is that
the latent has thrown away the distinctions the policy needs. Here, latents that
are easy to tell apart *is* the objective, so the gradient flows.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class InverseDynamicsHead(nn.Module):
    """[B,T-1,latent] x 2 -> logits over both players' buttons at each tick.

    Deliberately small and shallow. The point is to constrain the *encoder*, and
    a head with enough capacity to reconstruct actions from a poor representation
    would let the encoder off the hook -- the same argument `probe.py` makes for
    keeping the reward probe linear, one level up.
    """

    def __init__(self, cfg: Config, width: int = 512):
        super().__init__()
        self.ticks = cfg.action_ticks
        self.action_dim = cfg.action_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, width),
            nn.GELU(),
            nn.Linear(width, self.ticks * self.action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        # The difference is given explicitly alongside the pair. Nothing stops
        # the MLP computing it, but a transition is what this reads and saying so
        # costs one concatenation.
        x = torch.cat([z_t, z_next - z_t], dim=-1)
        out = self.net(x)
        return out.reshape(*z_t.shape[:-1], self.ticks, self.action_dim)


def inverse_dynamics_loss(head: InverseDynamicsHead, z: torch.Tensor,
                          actions: torch.Tensor,
                          pos_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """z [B,T,latent] (with grad), actions [B,T,ticks,action_dim] in {0,1}.

    Returns (loss, metrics). The loss is a per-button binary cross entropy over
    transitions t -> t+1, scored against the action chunk applied at t.

    Buttons are heavily imbalanced -- a human holds about 9.85% of them at any
    tick -- so accuracy alone would read 90% for a head that always says "not
    pressed". `idm_acc` is therefore balanced: the mean of the pressed-recall and
    the released-recall, which sits at 0.5 for that degenerate head.
    """
    if z.shape[1] < 2:
        raise ValueError(f"need at least two timesteps, got {z.shape[1]}")
    tgt = actions[:, :-1]                                   # action at t
    logits = head(z[:, :-1], z[:, 1:])
    # `pos_weight` because a human holds about 9.85% of buttons at any tick, so
    # plain BCE over eighty button-bits is dominated by correctly saying "not
    # pressed" nine times in ten. The first run showed exactly that signature:
    # the term was the largest in the objective (0.26 against a prediction loss
    # of 0.015) and yet moved barely at all across 30k steps, sitting at ~15% of
    # the reduction available from its 0.3025 chance value. A loss that is big
    # and stuck is not one to turn up; it is one whose gradient is being spent
    # on the easy majority.
    #
    # 1.0 keeps the original behaviour, so the first run stays reproducible.
    pw = (torch.full_like(tgt[:1, :1, :1, :1], pos_weight)
          if pos_weight != 1.0 else None)
    loss = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
    with torch.no_grad():
        pred = (logits > 0).float()
        pos = tgt.sum().clamp(min=1)
        neg = (1 - tgt).sum().clamp(min=1)
        rec_pos = float((pred * tgt).sum() / pos)
        rec_neg = float(((1 - pred) * (1 - tgt)).sum() / neg)
        metrics = {"idm_loss": float(loss.detach()),
                   "idm_acc": 0.5 * (rec_pos + rec_neg),
                   "idm_recall_pressed": rec_pos,
                   "idm_press_rate": float(tgt.mean())}
    return loss, metrics
