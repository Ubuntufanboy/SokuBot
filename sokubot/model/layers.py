"""Transformer primitives shared by the encoder and the predictor.

Two block flavours:

* :class:`Block` -- a standard pre-norm ViT block, used by the encoder.
* :class:`AdaLNBlock` -- the predictor's block. Actions modulate it through
  Adaptive LayerNorm with zero-initialised gates (AdaLN-Zero, DiT), so at
  initialisation the block is exactly the identity and action conditioning
  ramps in during training rather than perturbing it from step zero.

The modulation *parameters* are produced outside the block and passed in. That
is what lets the predictor share one modulation MLP across all layers (see
``Config.pred_shared_adaln``); each block only owns a small learned offset.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiT-style FiLM: normalised activations scaled and shifted by the condition."""
    return x * (1.0 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)      # each [B, heads, T, head_dim]
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Plain pre-norm transformer block (encoder)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        return x + self.mlp(self.norm2(x))


class AdaLNBlock(nn.Module):
    """Pre-norm block whose norms are modulated by an action condition.

    ``mod`` is [B, T, 6*dim]: (shift, scale, gate) for attention and for the MLP,
    broadcast per token so each timestep is conditioned on its own action chunk.
    The block adds its own learned offset, zero-initialised, so at step 0 every
    block sees the same (all-zero) modulation and both gates are 0 -- the block
    is the identity map.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        # affine=False: the scale/shift comes from the action condition instead.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(dim, mlp_ratio, dropout)
        self.offset = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, x: torch.Tensor, mod: torch.Tensor, causal: bool = True) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            mod + self.offset
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a), causal=causal)
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class Projector(nn.Module):
    """LeWM's projection head: 1-layer MLP followed by BatchNorm.

    The paper is explicit about why this exists (Sec. 3.1): the last ViT layer
    ends in a LayerNorm, which projects embeddings onto a sphere of fixed radius
    and stops SIGReg from being able to shape the distribution. A linear map
    followed by a *non-affine* BatchNorm re-frees the direction while pinning
    each dimension to zero mean and unit variance, which also removes the
    trivial minimiser of the prediction loss (shrink every latent to 0).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., out_dim]. BatchNorm sees the flattened batch."""
        lead = x.shape[:-1]
        y = self.fc(x).reshape(-1, self.fc.out_features)
        return self.bn(y).reshape(*lead, self.fc.out_features)


def init_vit_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_causal_pos_embed(max_len: int, dim: int) -> nn.Parameter:
    p = nn.Parameter(torch.zeros(1, max_len, dim))
    nn.init.trunc_normal_(p, std=0.02)
    return p


__all__ = [
    "Attention",
    "Mlp",
    "Block",
    "AdaLNBlock",
    "Projector",
    "modulate",
    "init_vit_weights",
    "build_causal_pos_embed",
]
