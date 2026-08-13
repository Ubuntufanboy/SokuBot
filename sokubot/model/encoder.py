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
import torch.nn.functional as F

from ..config import Config
from .layers import Block, Projector, init_vit_weights


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch: int, in_chans: int, dim: int):
        super().__init__()
        self.grid = image_size // patch
        self.n = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, n, dim]


class SpatialPool(nn.Module):
    """Patch tokens -> a latent whose LAYOUT is the image's layout.

    WHY THIS EXISTS
    ---------------
    Six objectives failed to make a [CLS]-pooled encoder represent where the
    fighters are: JEPA (spatial AUC 0.540), +IDM (0.651), +class-balanced IDM
    (0.688), direct dx supervision over 2003 replays (0.620), a play-area
    mirror built to forbid the HUD shortcut (0.605), and twelve hours of dx +
    projectile supervision at 448 px (0.553, against 0.500 chance and a 0.958
    ceiling). The one with the most supervision and the highest resolution did
    worst. That is not a loss function that needs another term.

    A [CLS] token is a learned global query: it is free to summarise the frame
    however it likes, and nothing in "predict the next latent" requires that
    summary to retain WHERE anything was. Mean pooling is worse -- it is
    permutation-invariant over patches, so position is destroyed by definition
    rather than merely discarded.

    HOW THIS IS DIFFERENT IN KIND
    -----------------------------
    Here the latent is a grid. Cell (r, c) of the image owns a fixed slice of
    the output vector, so "the fighter is on the left" and "the fighter is on
    the right" are different COORDINATES, not different values of one summary.
    Mirroring the play area permutes the grid columns, which permutes the
    latent's dimensions -- a fixed linear map. A linear probe reads that by
    construction, and the encoder cannot discard it without discarding the
    features themselves.

    The cost is that each cell gets latent_dim / (grid * grid) channels -- 12
    at the default 192 and 4x4 -- so per-cell semantics are thin. That is the
    intended trade: the previous design had 192 channels of semantics and no
    position at all.
    """

    def __init__(self, enc_dim: int, latent_dim: int, grid: int):
        super().__init__()
        if latent_dim % (grid * grid):
            raise ValueError(
                f"latent_dim {latent_dim} must divide by grid^2 {grid*grid}; "
                f"every cell owns an equal slice and a ragged split would make "
                f"the mirror permutation non-linear")
        self.grid = grid
        self.cells = grid * grid
        self.per_cell = latent_dim // self.cells
        # One projection PER CELL, not one shared projection. Written as an
        # explicit [cells, enc_dim, per_cell] weight rather than a grouped conv
        # because the thing that has to be true -- cell k writes latent slice
        # k*per_cell : (k+1)*per_cell and nothing else -- is then visible in the
        # einsum instead of implied by a reshape order that is easy to get
        # silently wrong.
        self.weight = nn.Parameter(torch.empty(self.cells, enc_dim, self.per_cell))
        self.bias = nn.Parameter(torch.zeros(self.cells, self.per_cell))
        nn.init.trunc_normal_(self.weight, std=0.02)
        # Same contract as Projector: a non-affine BatchNorm so SIGReg still
        # sees zero mean and unit variance per dimension, and the trivial
        # minimiser (shrink every latent to 0) stays unavailable.
        self.bn = nn.BatchNorm1d(latent_dim, affine=False)

    def forward(self, tok: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """tok: [B, gh*gw, enc_dim] patch tokens (no [CLS]) -> [B, latent_dim]."""
        B, _, d = tok.shape
        x = tok.transpose(1, 2).reshape(B, d, gh, gw)
        # Average within each cell. Adaptive so the patch grid need not divide
        # the cell grid -- 16x16 patches at 224 px and 32x32 at 448 px both
        # reduce to the same 4x4 latent layout, which is what lets a 224 px run
        # and a 448 px one produce comparable latents.
        x = F.adaptive_avg_pool2d(x, (self.grid, self.grid))      # [B, d, G, G]
        x = x.flatten(2).transpose(1, 2)                          # [B, cells, d]
        # b=batch, k=cell, d=enc_dim, p=per_cell. Cell k uses ONLY weight[k].
        x = torch.einsum("bkd,kdp->bkp", x, self.weight) + self.bias
        return self.bn(x.reshape(B, -1))       # row-major: cell k -> slice k


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
        # `cls` is the original and stays the default, so every checkpoint ever
        # trained still loads. `spatial` is the change of kind described in
        # SpatialPool -- selected by config, not by editing this file, so the
        # two can be run as arms of the same experiment.
        self.pool = getattr(cfg, "encoder_pool", "cls")
        if self.pool == "spatial":
            self.spatial_pool = SpatialPool(d, cfg.latent_dim,
                                            getattr(cfg, "pool_grid", 4))
            self.projector = None
        else:
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

        if self.pool == "spatial":
            g = self.patch_embed.grid
            z = self.spatial_pool(tok[:, 1:], g, g)   # patches -> grid latent
        else:
            z = self.projector(tok[:, 0])        # [CLS] -> MLP + BatchNorm
        return z.reshape(*lead, self.cfg.latent_dim)


def resize_pos_embed(pos_embed: torch.Tensor, old_grid: int,
                     new_grid: int) -> torch.Tensor:
    """Re-grid a ViT positional embedding for a different input resolution.

    ``pos_embed`` is [1, 1 + old_grid**2, dim] with the CLS position first.
    Returns [1, 1 + new_grid**2, dim].

    This is what makes warm-starting a higher-resolution encoder from a trained
    lower-resolution one possible, and it is the *only* tensor that needs it:

      * ``patch_embed.proj`` is a Conv2d with kernel == stride == patch, so it
        sees one patch at a time and does not care how many there are. A 224 px
        model produces 16x16 patches and a 448 px model 32x32, from identical
        weights.
      * the transformer blocks are attention plus MLP over tokens, so they are
        token-count independent.
      * ``cls_token``, ``norm`` and ``projector`` never see the grid at all.
      * the predictor and action encoder consume latents, not pixels.

    Only ``pos_embed`` is tied to the grid, and interpolating it is the standard
    recipe from the original ViT paper's higher-resolution fine-tuning. Bicubic
    rather than nearest because these are smooth learned coordinates, and nearest
    would quantise every position onto its coarse neighbour.
    """
    if pos_embed.ndim != 3 or pos_embed.shape[1] != old_grid ** 2 + 1:
        raise ValueError(
            f"expected [1, 1 + {old_grid}^2, dim], got {tuple(pos_embed.shape)}")
    if old_grid == new_grid:
        return pos_embed.clone()
    cls, grid = pos_embed[:, :1], pos_embed[:, 1:]
    d = grid.shape[-1]
    grid = grid.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid ** 2, d)
    return torch.cat([cls, grid], dim=1)
