"""ViT-Tiny observation encoder: pixels -> a single latent per frame.

LeWM Sec. 3.1: "we use the tiny configuration (~5M parameters) with a patch size
of 14, 12 layers, 3 attention heads, and hidden dimensions of 192. The
observation embedding z_t is constructed from the [CLS] token embedding of the
last layer, followed by a projection step."

At the default config this is 5,538,432 parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .layers import Block, Projector, init_vit_weights


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch: int, in_chans: int, dim: int):
        super().__init__()
        self.n = (image_size // patch) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, n, dim]


class ViTEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.enc_dim
        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n + 1, d))
        self.blocks = nn.ModuleList(
            Block(d, cfg.enc_heads, cfg.enc_mlp_ratio) for _ in range(cfg.enc_depth)
        )
        self.norm = nn.LayerNorm(d)
        self.projector = Projector(d, cfg.latent_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 3, S, S] -> z: [..., latent_dim].

        Leading dimensions are flattened, so a whole [B, T, 3, S, S]
        sub-trajectory can be encoded in one call -- which also means the
        projector's BatchNorm sees batch *and* time in the same statistics,
        matching LeWM's Alg. 3 (``emb = encoder(obs)`` over the full sequence).
        """
        # Accept raw uint8 frames and normalise here, on whatever device the
        # tensor is already on. The loader ships uint8 to keep PCIe traffic and
        # worker CPU down (see cfg.loader_uint8), and putting the single
        # conversion at the model boundary means no call site -- training,
        # evaluation, probing, planning -- has to know which dtype it holds.
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        lead = x.shape[:-3]
        flat = x.reshape(-1, *x.shape[-3:])

        B = flat.shape[0]
        tok = self.patch_embed(flat)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)

        z = self.projector(tok[:, 0])            # [CLS] -> MLP + BatchNorm
        return z.reshape(*lead, self.cfg.latent_dim)
