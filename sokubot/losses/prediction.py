"""Next-embedding prediction loss (LeWM Eq. 1).

    L_pred = || zhat_{t+1} - z_{t+1} ||^2

Teacher forcing: the predictor always conditions on encoder latents of *real*
observations, never on its own previous outputs. Because the predictor is causal
over the whole window, one forward pass gives every next-step prediction at once
(LeWM Alg. 3: ``mse(emb[:, 1:], next_emb[:, :-1])``).

There is no stop-gradient on the target. The encoder is pulled by this loss from
both sides -- it must produce representations that are predictable *and* that it
can predict -- and SIGReg is what keeps the shared solution from being a
constant.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def prediction_loss(zhat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """zhat, z: [B, T, D]. Compares zhat[:, :-1] against z[:, 1:]."""
    if zhat.shape != z.shape:
        raise ValueError(f"shape mismatch: zhat {tuple(zhat.shape)} vs z {tuple(z.shape)}")
    if z.shape[1] < 2:
        raise ValueError("need at least 2 timesteps to form a next-step target")
    return F.mse_loss(zhat[:, :-1], z[:, 1:])


def rollout_loss(zhat: torch.Tensor, z_true: torch.Tensor, z0: torch.Tensor,
                 horizons: Sequence[int] | None = None,
                 eps: float = 1e-8) -> tuple[torch.Tensor, dict[int, float]]:
    """Multi-step loss on an *autoregressive* rollout, normalised per horizon.

    ``zhat`` and ``z_true`` are [B, P, D]; ``z0`` is [B, D], the last observed
    latent the rollout started from.

    WHY THIS EXISTS
    ---------------
    :func:`prediction_loss` is teacher-forced: the predictor is only ever
    conditioned on encoder latents of *real* observations, so it has never once
    been asked to consume its own output. At play time it consumes nothing else.
    That mismatch is the standard reason autoregressive rollouts blur, and here
    it is measurable -- rollout cosine to truth falls 0.9963 -> 0.9717 -> 0.8258
    -> 0.4170 at h = 1, 4, 16, 48, and the action->outcome correlation a policy
    gradient actually consumes falls with it, r = 0.55 -> 0.32 -> 0.09.

    WHY EACH HORIZON IS NORMALISED BY COPY-FORWARD
    ----------------------------------------------
    Raw MSE grows with h, so an unweighted sum is dominated by the longest
    horizon -- the one where the target is least learnable -- and the model buys
    a small improvement there by giving up the short horizons that actually
    drive control. Dividing each term by the error of the trivial predictor
    (copy ``z0`` forward, which is also the denominator of the `skill` metric
    used everywhere else in this project) puts every horizon on the same scale:
    each term is then "fraction of copy-forward error remaining", 1.0 means no
    better than doing nothing, and the horizons compete on equal footing.

    The denominator is detached. It is a yardstick, not something to optimise --
    without the detach the model could lower the loss by making the *baseline*
    worse, which it can do, because ``z0`` is its own input.

    RATIO OF SUMS, NOT MEAN OF RATIOS
    ---------------------------------
    Each horizon's term is ``sum(err) / sum(ident)`` over the batch, not
    ``mean(err / ident)``. The two agree when the baseline is well behaved and
    diverge catastrophically when it is not: Soku has near-static moments --
    round-transition freezes, hitstop, genuinely repeated frames -- where the
    latent barely moves, so copy-forward is nearly perfect and ``ident`` is
    nearly zero. A per-sample ratio then explodes, and the mean is set by a
    handful of those rather than by the model.

    That is not hypothetical: the first run of this loss reported training error
    of 0.43-1.30 while validation on held-out replays read h4 = 364 and h8 = 550,
    non-monotonic in the horizon, purely from a few near-zero denominators.
    Summing first makes an uninformative sample contribute nearly nothing to both
    numerator and denominator, which is what it deserves. It is also exactly how
    `eval_ckpt.predictor_skill` defines skill, so the two metrics stay commensurable.

    Returns (scalar loss, {h: relative error}) with the per-horizon diagnostics
    already floats, since they are logged rather than differentiated.
    """
    if zhat.shape != z_true.shape:
        raise ValueError(
            f"shape mismatch: zhat {tuple(zhat.shape)} vs z_true {tuple(z_true.shape)}")
    P = zhat.shape[1]
    hs = list(horizons) if horizons is not None else list(range(1, P + 1))
    bad = [h for h in hs if not 1 <= h <= P]
    if bad:
        raise ValueError(f"horizons {bad} outside the rollout length {P}")

    terms, report = [], {}
    for h in hs:
        err = ((zhat[:, h - 1] - z_true[:, h - 1]) ** 2).mean(-1)          # [B]
        ident = ((z0 - z_true[:, h - 1]) ** 2).mean(-1).detach()           # [B]
        rel = err.sum() / (ident.sum() + eps)
        terms.append(rel)
        report[h] = float(rel.detach())
    return torch.stack(terms).mean(), report
