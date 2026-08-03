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


# ---------------------------------------------------------------------------
# Inverse dynamics -- the probe to use when there is no simulator state.
# ---------------------------------------------------------------------------
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC. 0.5 is chance; nan if one class is absent."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties, else ties bias the statistic
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class InverseDynamicsResult:
    auc: Dict[str, float]
    auc_mean: float
    base_rate: Dict[str, float]
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return f"inverse-dynamics AUC {self.auc_mean:.3f} (n={self.n_test})"


def inverse_dynamics_probe(
    z_t: np.ndarray,
    z_next: np.ndarray,
    actions: np.ndarray,
    names: Optional[Sequence[str]] = None,
    train_frac: float = 0.8,
    alpha: float = 1e-2,
    seed: int = 0,
) -> InverseDynamicsResult:
    """Can a linear map read the action out of a pair of consecutive latents?

    This is the probe for Soku, where there is no simulator state to regress
    onto. If `(z_t, z_{t+1})` does not determine the buttons pressed between
    them, the latent is not carrying dynamics -- and a world model whose latent
    does not encode what the action did cannot support planning, because
    planning is exactly the inverse question.

    z_t, z_next: [N, D] latents. actions: [N, A] in {0, 1}.
    Scored by AUC per button, which unlike accuracy is not fooled by buttons
    that are pressed 3% of the time.
    """
    if not (len(z_t) == len(z_next) == len(actions)):
        raise ValueError("z_t, z_next and actions must have the same length")
    x = np.concatenate([z_t, z_next, z_next - z_t], axis=1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_tr = int(len(x) * train_frac)
    tr, te = perm[:n_tr], perm[n_tr:]

    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    Xtr = np.concatenate([(x[tr] - mu) / sd, np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([(x[te] - mu) / sd, np.ones((len(te), 1))], axis=1)
    Ytr = actions[tr]

    reg = alpha * np.eye(Xtr.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ Ytr)
    pred = Xte @ W

    names = list(names) if names is not None else [f"a{i}" for i in range(actions.shape[1])]
    auc = {n: _auc(pred[:, i], actions[te][:, i]) for i, n in enumerate(names)}
    base = {n: float(actions[:, i].mean()) for i, n in enumerate(names)}
    vals = [v for v in auc.values() if np.isfinite(v)]
    return InverseDynamicsResult(
        auc=auc,
        auc_mean=float(np.mean(vals)) if vals else float("nan"),
        base_rate=base,
        n_train=len(tr),
        n_test=len(te),
    )
