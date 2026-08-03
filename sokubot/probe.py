"""Physical-latent probing: how much of the true state survives in the latent?

LeWM Sec. 4 / Tab. 1 evaluates representations by fitting a **linear** probe from
the frozen latent to ground-truth simulator state (in PushT: agent position and
block pose) and reporting how well it recovers them.

This is the diagnostic that matters for an anti-collapse claim, and it is
strictly better than watching the latent's effective rank. Effective rank is
ambiguous: it falls both when the encoder collapses *and* when the encoder
correctly discovers that a 5-degree-of-freedom system needs only a few
dimensions. A probe is unambiguous -- a collapsed latent scores R^2 ~ 0 because
there is nothing left to regress from.

The probe is deliberately linear and closed-form (ridge regression). A nonlinear
probe would report how much information is *recoverable in principle* rather
than how well the representation exposes it, which is the property planning
actually depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .config import Config
from .data.window import frames_to_chw
from .model.world_model import LeWorldModel


@dataclass
class ProbeResult:
    r2: Dict[str, float]          # per-target held-out R^2
    r2_mean: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        per = ", ".join(f"{k} {v:.3f}" for k, v in self.r2.items())
        return f"R2 mean {self.r2_mean:.3f} ({per})"


def ridge_probe(
    z: np.ndarray,
    y: np.ndarray,
    names: Optional[Sequence[str]] = None,
    train_frac: float = 0.8,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ProbeResult:
    """Fit z -> y with ridge regression; report held-out R^2 per target.

    z: [N, D] latents. y: [N, K] targets.
    """
    if len(z) != len(y):
        raise ValueError(f"z has {len(z)} rows, y has {len(y)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    n_tr = int(len(z) * train_frac)
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(te) < 2:
        raise ValueError("not enough held-out samples to score a probe")

    Ztr, Zte = z[tr], z[te]
    Ytr, Yte = y[tr], y[te]

    # Standardise inputs and targets on the *training* split only.
    zmu, zsd = Ztr.mean(0), Ztr.std(0) + 1e-8
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xtr = np.concatenate([(Ztr - zmu) / zsd, np.ones((len(Ztr), 1))], axis=1)
    Xte = np.concatenate([(Zte - zmu) / zsd, np.ones((len(Zte), 1))], axis=1)
    Ttr, Tte = (Ytr - ymu) / ysd, (Yte - ymu) / ysd

    D = Xtr.shape[1]
    reg = alpha * np.eye(D)
    reg[-1, -1] = 0.0                               # never penalise the intercept
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Ttr)

    pred = Xte @ W
    ss_res = ((Tte - pred) ** 2).sum(0)
    ss_tot = ((Tte - Tte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    names = list(names) if names is not None else [f"y{i}" for i in range(y.shape[1])]
    return ProbeResult(
        r2={n: float(v) for n, v in zip(names, r2)},
        r2_mean=float(r2.mean()),
        n_train=len(tr),
        n_test=len(te),
    )


@torch.no_grad()
def encode_episodes(
    model: LeWorldModel,
    episodes: Sequence,
    cfg: Config,
    batch_size: int = 64,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, Sequence[str]]:
    """Encode every frame of `episodes` that has a recorded state."""
    was_training = model.training
    model.eval()

    frames, states, names = [], [], None
    for ep in episodes:
        if ep.states is None:
            continue
        names = names or ep.meta.get("state_names")
        frames.append(ep.frames)
        states.append(ep.states)
    if not frames:
        raise ValueError("no episode carries ground-truth states")

    F = np.concatenate(frames)
    S = np.concatenate(states)
    if max_frames and len(F) > max_frames:
        F, S = F[:max_frames], S[:max_frames]

    zs = []
    for i in range(0, len(F), batch_size):
        # The encoder's BatchNorm is in eval mode here, so running statistics
        # are used and each chunk is normalised identically -- otherwise the
        # probe would be fitting features that depend on chunk boundaries.
        x = frames_to_chw(F[i : i + batch_size])
        zs.append(model.encode(x).cpu().numpy())

    if was_training:
        model.train()
    names = names or [f"y{i}" for i in range(S.shape[1])]
    return np.concatenate(zs), S, names


def probe_model(
    model: LeWorldModel, episodes: Sequence, cfg: Config, **kwargs
) -> ProbeResult:
    z, y, names = encode_episodes(model, episodes, cfg, **kwargs)
    return ridge_probe(z, y, names=names)
