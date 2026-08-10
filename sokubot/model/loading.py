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

    # ---- optional heads ------------------------------------------------------
    # Every entry here is a head whose *existence* is controlled by a loss
    # weight. Presence is the only thing the weights record -- the coefficient
    # itself is unrecoverable -- so only the zero/non-zero distinction is
    # restored, and a positive value is left exactly as found.
    #
    # This is a table rather than two hand-written blocks because the bug it
    # prevents has now happened twice: `hud_coef` broke every earlier checkpoint
    # when it was added with a non-zero default, that was fixed here, and then
    # `idm_coef` was added the same way and broke them again. Anything added to
    # `Config` that gates a module must be added to this list, and the failure
    # if it is not is a confusing "Missing key(s) in state_dict" on artifacts
    # that were fine the day before.
    # The third element is the value to use when the checkpoint HAS the head
    # but the config disabled it. It cannot just be the class default any more:
    # the lesson from those two breakages is that a gating field should default
    # to 0, and "restoring" a 0 default leaves the head unbuilt -- turning this
    # repair into the very missing-key error it exists to prevent. None means
    # "the class default is positive, use it".
    for coef_name, weight_key, rebuild_with in (
            ("hud_coef", "hud_head.weight", None),
            ("idm_coef", "idm_head.net.0.weight", None),
            ("state_coef", "state_head.net.0.weight", 1.0)):
        present = weight_key in state
        value = getattr(cfg, coef_name, 0.0)
        if present and value <= 0:
            fallback = getattr(Config, coef_name) if rebuild_with is None \
                else rebuild_with
            if fallback <= 0:
                raise ValueError(
                    f"{coef_name} would be rebuilt with {fallback:g}, which "
                    f"leaves the head unbuilt and the load failing on a "
                    f"missing key; give it an explicit rebuild_with here")
            setattr(cfg, coef_name, fallback)
            notes.append(
                f"checkpoint has {weight_key.split('.')[0]} but cfg.{coef_name} "
                f"was 0; set to {getattr(cfg, coef_name):g} so the head is built")
        elif not present and value > 0:
            notes.append(
                f"checkpoint has no {weight_key.split('.')[0]} but "
                f"cfg.{coef_name} resolved to {value:g} (this predates the "
                f"field); set to 0")
            setattr(cfg, coef_name, 0.0)

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
