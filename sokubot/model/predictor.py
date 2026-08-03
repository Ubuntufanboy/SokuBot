"""Action-conditioned latent predictor.

LeWM Sec. 3.1 / App. D: "The predictor is a transformer with 6 layers, 16
attention heads, and 10% dropout (~10M parameters). Actions are incorporated
into the predictor through Adaptive Layer Normalization (AdaLN) applied at each
layer. The AdaLN parameters are initialized to zero [...] The predictor takes as
input a history of N frame representations and predicts the next frame
representation auto-regressively with temporal causal masking to avoid looking
at future embeddings. The predictor is also followed by a projector network with
the same implementation as the one used for the encoder."

So this is a decoder-only transformer over latents: position ``t`` reads
``z_{<=t}`` and emits ``zhat_{t+1}``. One forward pass over a length-T window
therefore produces T-1 supervised next-step predictions -- teacher forcing for
free, which is what LeWM's Alg. 3 relies on.

At the default config this is 9,917,760 parameters, including the AdaLN
modulation. See ``Config.pred_mlp_ratio`` for how that number is reached.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .layers import AdaLNBlock, Projector, build_causal_pos_embed, init_vit_weights

# Learned positional embeddings are allocated once for the longest sequence the
# model will ever see. Training uses `seq_len` (4); planning uses
# `history + plan_horizon` (8). 64 leaves room for both plus long evaluation
# rollouts, at a cost of 64*384 = 24k parameters.
MAX_SEQ = 64


class LatentPredictor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.pred_dim

        self.in_proj = nn.Linear(cfg.latent_dim, D)
        self.pos_embed = build_causal_pos_embed(MAX_SEQ, D)

        # AdaLN-Zero modulation. One MLP shared across depth (adaLN-single);
        # each block owns a zero-initialised offset on top of it. Zero init here
        # means every gate starts at 0, so the whole stack is the identity at
        # step 0 and action conditioning grows in smoothly.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

        self.blocks = nn.ModuleList(
            AdaLNBlock(D, cfg.pred_heads, cfg.pred_mlp_ratio, cfg.pred_dropout)
            for _ in range(cfg.pred_depth)
        )
        self.projector = Projector(D, cfg.latent_dim)

        # Init the backbone, then re-zero the modulation (apply() would have
        # overwritten it with trunc_normal_).
        self.apply(init_vit_weights)
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z: [B, T, latent] latents, cond: [B, T, pred_dim] action conditions.

        Returns [B, T, latent], where index ``t`` is the prediction of ``z_{t+1}``
        made from ``z_{<=t}`` and the action chunk applied at ``t``.
        """
        B, T, _ = z.shape
        if T > MAX_SEQ:
            raise ValueError(f"sequence length {T} exceeds MAX_SEQ={MAX_SEQ}")
        if cond.shape[:2] != (B, T):
            raise ValueError(
                f"cond has leading shape {tuple(cond.shape[:2])}, expected {(B, T)}"
            )

        h = self.in_proj(z) + self.pos_embed[:, :T]
        mod = self.adaln(cond)                     # [B, T, 6D], per-token
        for blk in self.blocks:
            h = blk(h, mod, causal=True)
        return self.projector(h)
