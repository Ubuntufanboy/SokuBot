"""Sub-trajectory windowing over episodes.

LeWM App. D trains on "sub-trajectories of size 4 corresponding to 4 frames and
4 blocks of 5 actions". This dataset produces exactly that: every valid
length-``seq_len`` window of every episode is one training sample.

The index is built once from array headers alone (``np.load`` is lazy about
member data), so constructing the dataset over thousands of shards does not read
a single pixel.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from .episode import Episode


def frames_to_chw(frames: np.ndarray) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float32 [N, 3, H, W] in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


class EpisodeWindowDataset(Dataset):
    """Windows over ``*.npz`` episodes written by :class:`Episode`."""

    def __init__(self, cfg: Config, root: Union[str, Path]):
        self.cfg = cfg
        self.paths: List[str] = sorted(glob.glob(os.path.join(str(root), "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"no .npz episodes under {root}")

        T_win = cfg.seq_len
        self.index: List[Tuple[int, int]] = []
        for ei, p in enumerate(self.paths):
            with np.load(p, allow_pickle=False) as d:
                T = d["frames"].shape[0]
            # A window needs seq_len frames; the last action is unusable as a
            # transition, so stop one short of the end.
            for s in range(0, T - T_win):
                self.index.append((ei, s))
        if not self.index:
            raise ValueError(
                f"no window of length {T_win} fits in any episode under {root}"
            )

        self._cache_ei = -1
        self._cache_ep: Episode | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, ei: int) -> Episode:
        # Windows are drawn in shuffled order, so this one-slot cache only pays
        # off with num_workers=0 and small corpora. It is never a correctness
        # issue -- just a re-read.
        if ei != self._cache_ei:
            self._cache_ep = Episode.load(self.paths[ei])
            self._cache_ei = ei
        return self._cache_ep

    def __getitem__(self, i: int):
        ei, s = self.index[i]
        ep = self._episode(ei)
        T = self.cfg.seq_len
        return {
            "obs": frames_to_chw(ep.frames[s : s + T]),                 # [T,3,S,S]
            "actions": torch.from_numpy(ep.actions[s : s + T]).float(), # [T,ticks,A]
        }
