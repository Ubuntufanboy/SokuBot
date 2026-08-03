"""On-disk episode format and action normalisation.

An **episode** is a decision-rate trajectory: one frame per model step, plus the
chunk of ``frame_skip`` raw environment actions applied at that step.

    frames   uint8   [T, H, W, 3]
    actions  float32 [T, ticks, action_dim]   normalised to [-1, 1]

``actions[t]`` is what was applied *at* ``frames[t]``, producing ``frames[t+1]``.
The final entry leads nowhere; it is kept so the two arrays share a length and
windowing stays index-aligned.

Actions are stored **normalised**. PushT's native action range is [0, 512]
pixels, which a network with 0.02-std initialisation would need many steps just
to rescale; and storing raw units would make the planner's CEM variance
(initialised to 1, per LeWM Alg. 2) meaningless. Normalising at record time
keeps one convention everywhere: the model, the dataset, and the planner all
speak [-1, 1], and only the thin env wrapper converts back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..config import Config


def normalize_action(a: np.ndarray, cfg: Config) -> np.ndarray:
    """Native units -> [-1, 1]."""
    lo, hi = cfg.action_low, cfg.action_high
    if hi <= lo:
        raise ValueError(f"action_high ({hi}) must exceed action_low ({lo})")
    return (2.0 * (np.asarray(a, dtype=np.float32) - lo) / (hi - lo) - 1.0).astype(np.float32)


def denormalize_action(a: np.ndarray, cfg: Config) -> np.ndarray:
    """[-1, 1] -> native units."""
    lo, hi = cfg.action_low, cfg.action_high
    return (((np.asarray(a, dtype=np.float32) + 1.0) * 0.5) * (hi - lo) + lo).astype(np.float32)


@dataclass
class Episode:
    frames: np.ndarray                     # uint8 [T, H, W, 3]
    actions: np.ndarray                    # float32 [T, ticks, action_dim]
    # Ground-truth simulator state per frame, when the environment exposes it.
    # Never used for training -- the model is trained from pixels only -- but
    # required for LeWM's physical-latent probing (Sec. 4 / Tab. 1), which is
    # the only direct measure of whether the latent kept any information.
    states: Optional[np.ndarray] = None    # float32 [T, state_dim]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.frames) != len(self.actions):
            raise ValueError(
                f"frames ({len(self.frames)}) and actions ({len(self.actions)}) "
                "must have the same length"
            )
        if self.states is not None and len(self.states) != len(self.frames):
            raise ValueError(
                f"states ({len(self.states)}) and frames ({len(self.frames)}) "
                "must have the same length"
            )

    def __len__(self) -> int:
        return len(self.frames)

    def save(self, path: Union[str, Path]) -> None:
        arrays = dict(
            frames=self.frames,
            actions=self.actions.astype(np.float32),
            meta=np.array(repr(self.meta)),
        )
        if self.states is not None:
            arrays["states"] = self.states.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path: Union[str, Path]) -> "Episode":
        with np.load(path, allow_pickle=False) as d:
            meta = {}
            if "meta" in d:
                try:
                    meta = eval(str(d["meta"]), {"__builtins__": {}}, {})  # literal dict
                except Exception:
                    meta = {}
            return Episode(
                frames=d["frames"],
                actions=d["actions"],
                states=d["states"] if "states" in d else None,
                meta=meta,
            )
