"""SIGReg -- Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

The anti-collapse half of the LeWM objective. It pushes a batch of latents toward
an isotropic Gaussian N(0, I) without any stop-gradient, EMA, or target encoder.

Two ingredients, per LeWM App. A:

1. **Cramer-Wold.** A distribution in R^D is N(0, I) iff every 1-D projection of
   it is N(0, 1). So instead of testing normality in D dimensions (where
   classical tests degrade badly), sketch onto M random unit directions and test
   each projection univariately.

2. **Epps-Pulley.** A differentiable normality statistic built from the
   empirical characteristic function:

       T(h) = integral w(t) * |phi_N(t; h) - phi_0(t)|^2 dt

   with phi_N(t; h) = (1/N) sum_n exp(i*t*h_n) the ECF of the projection,
   phi_0(t) = exp(-t^2/2) the CF of N(0, 1), and w(t) = exp(-t^2 / (2*lambda_w))
   the weighting function. The integral is taken over R; here it is evaluated by
   trapezoid quadrature on [-range, range], which is ample because w decays as a
   Gaussian.

   |phi_N - phi_0|^2 expands to (Re phi_N - phi_0)^2 + (Im phi_N)^2, since
   phi_0 is real.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..config import Config


def sigreg(z: torch.Tensor, cfg: Config) -> torch.Tensor:
    """z: [N, D] -> scalar. Mean Epps-Pulley statistic over M random directions.

    Directions are resampled on every call, as in LeJEPA -- a fixed sketch would
    let the encoder learn to satisfy those particular projections and nothing
    else.

    The projections are compared to N(0, 1) *raw*: standardising them per batch
    first would make the test trivially satisfied and remove all regularising
    pressure.
    """
    N, D = z.shape

    A = torch.randn(D, cfg.sigreg_dirs, device=z.device, dtype=z.dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(1e-8)
    h = z @ A                                                   # [N, M]

    t = torch.linspace(
        -cfg.sigreg_range, cfg.sigreg_range, cfg.sigreg_points,
        device=z.device, dtype=z.dtype,
    )                                                           # [P]
    ht = h.unsqueeze(-1) * t.view(1, 1, -1)                     # [N, M, P]

    # ECF -- a mean over samples, so it can be averaged across ranks.
    ecf_re = torch.cos(ht).mean(dim=0)                          # [M, P]
    ecf_im = torch.sin(ht).mean(dim=0)                          # [M, P]

    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(ecf_re, op=dist.ReduceOp.SUM)
        dist.all_reduce(ecf_im, op=dist.ReduceOp.SUM)
        ecf_re, ecf_im = ecf_re / world, ecf_im / world
        N = N * world

    phi0 = torch.exp(-0.5 * t * t)                              # CF of N(0,1)
    w = torch.exp(-0.5 * t * t / cfg.sigreg_lambda_w)           # weighting
    integrand = ((ecf_re - phi0.view(1, -1)) ** 2 + ecf_im ** 2) * w.view(1, -1)
    stat = torch.trapz(integrand, t, dim=1)                     # [M]

    out = stat.mean()
    if cfg.sigreg_scale_n:
        out = out * N
    return out


def sigreg_stepwise(z: torch.Tensor, cfg: Config) -> torch.Tensor:
    """z: [B, T, D] -> scalar, SIGReg applied per timestep and averaged.

    LeWM's Alg. 3 writes this as ``mean(SIGReg(emb.transpose(0, 1)))`` with
    Z in R^{T x B x D}: the statistic is computed *within* each timestep's batch,
    then averaged over time. That matters -- pooling all B*T embeddings into one
    sample would let the model satisfy the Gaussian target only in aggregate
    while each individual timestep stayed degenerate, which is exactly the
    failure mode where a world model looks healthy but every frame at a given
    phase of the trajectory maps to the same point.
    """
    B, T, D = z.shape
    return torch.stack([sigreg(z[:, t], cfg) for t in range(T)]).mean()
