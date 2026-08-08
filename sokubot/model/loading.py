"""Load a world model whose architecture is decided by its weights.

    wm, cfg, blob = load_world_model("~/art/wm_cf_bnfix.pt", "cuda")

WHY THIS EXISTS
---------------
`Config` is a dataclass, and a checkpoint stores a *pickled instance* of it. When
a new field is added, every previously saved config unpickles without that field
in its ``__dict__`` -- so reading it silently falls through to the **class**
attribute, i.e. whatever today's default happens to be.

That is not a cosmetic problem, because some of those fields decide the
architecture. Adding ``hud_coef = 0.25`` gave every pre-existing checkpoint a
``hud_head`` that its ``state_dict`` has no weights for, and the failure lands as

    RuntimeError: Missing key(s) in state_dict: "hud_head.weight", ...

on models that were perfectly fine the day before. The tempting fix -- pass
``strict=False`` -- is much worse than the crash: it builds the phantom head with
*random* weights and reports nothing, so `predict_hud` would return noise and
anything reading it would be measuring the initialiser.

The rule this module enforces: **the state dict is the source of truth for
architecture, and the config is reconciled to match it.** A field that cannot be
recovered from the weights is a field that must not change the architecture.

WHAT IS NOT RECONCILED
----------------------
Objective weights that leave no trace in the weights -- ``cf_coef``,
``sigreg_coef``, ``lambda`` -- are left at whatever they resolve to. They affect
only training, and a checkpoint being *loaded* is not being trained by these
callers. Anything that resumes training should set them explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from ..config import Config
from .world_model import LeWorldModel


def reconcile_config(cfg: Config, state: Dict[str, Any]) -> List[str]:
    """Edit `cfg` in place so it describes the architecture stored in `state`.

    Returns a list of human-readable notes, one per correction. An empty list
    means the config already agreed with the weights.
    """
    notes: List[str] = []

    # ---- supervised HUD head -------------------------------------------------
    # Presence is the only thing the weights record; the coefficient itself is a
    # loss weight and is unrecoverable. So only the zero/non-zero distinction is
    # restored, and a positive value is left exactly as found.
    has_hud = "hud_head.weight" in state
    if has_hud and cfg.hud_coef <= 0:
        cfg.hud_coef = Config.hud_coef
        notes.append(
            f"checkpoint has a hud_head but cfg.hud_coef was {0.0:g}; set to "
            f"{cfg.hud_coef:g} so the head is built")
    elif not has_hud and cfg.hud_coef > 0:
        notes.append(
            f"checkpoint has no hud_head but cfg.hud_coef resolved to "
            f"{cfg.hud_coef:g} (this predates the field); set to 0")
        cfg.hud_coef = 0.0

    # ---- input resolution ----------------------------------------------------
    # `pos_embed` is [1, 1 + grid^2, dim], so the grid -- and with it the image
    # size the encoder was actually trained at -- is readable off the weights.
    # This matters for the same reason the HUD head does: `image_size` moved from
    # 224 to 448, and a 224 checkpoint loaded under a 448 default would fail on a
    # shape mismatch rather than a missing key.
    pe = state.get("encoder.pos_embed")
    if pe is not None and pe.ndim == 3:
        n = int(pe.shape[1]) - 1
        grid = int(round(n ** 0.5))
        if grid * grid != n:
            raise ValueError(
                f"encoder.pos_embed has {pe.shape[1]} positions, which is not "
                f"1 + a square grid; this checkpoint is not a plain ViT")
        size = grid * cfg.patch_size
        if size != cfg.image_size:
            notes.append(
                f"checkpoint's positional grid is {grid}x{grid} at patch "
                f"{cfg.patch_size}, i.e. image_size {size}, but cfg said "
                f"{cfg.image_size}; set to {size}")
            cfg.image_size = size
    return notes


def load_world_model(path: Path | str, device: str = "cpu", *,
                     freeze: bool = True,
                     verbose: bool = True) -> Tuple[LeWorldModel, Config, dict]:
    """Load a checkpoint into an eval-mode model, strictly.

    Strict on purpose. Every mismatch this could paper over is a mismatch that
    changes what the model computes, and `docs/BUGS.md` is largely a record of
    such things training happily and reporting nothing.

    ``freeze`` is explicit rather than assumed. RL and evaluation want the world
    model fixed -- a policy that could edit the simulator would optimise the
    simulator -- but `finetune_predictor` legitimately trains part of it, and a
    loader that silently froze the weights would leave that caller stepping an
    optimiser over a graph with no gradients and reporting a flat loss curve.
    """
    path = Path(path).expanduser()
    blob = torch.load(path, map_location=device, weights_only=False)
    if "model" not in blob or "cfg" not in blob:
        raise SystemExit(f"{path} has keys {sorted(blob)}; expected model + cfg")
    cfg: Config = blob["cfg"]
    notes = reconcile_config(cfg, blob["model"])
    if notes and verbose:
        for n in notes:
            print(f"  cfg reconciled: {n}", flush=True)
    cfg.device = device
    wm = LeWorldModel(cfg).to(device)
    wm.load_state_dict(blob["model"])
    wm.eval()
    if freeze:
        for p in wm.parameters():
            p.requires_grad_(False)
    return wm, cfg, blob
