"""How far an imagined latent has drifted from anything the world model has seen.

WHY THIS IS NEEDED *WITH* THE LONGER HORIZON, NOT AFTER IT
-----------------------------------------------------------
`ACConfig.horizon` is 16 decision steps -- four times the horizon GRPO used, on
the argument that a critic bootstraps the tail so imagination only has to be
right over its own length. That argument is sound about *credit assignment* and
says nothing about *fidelity*: `horizon_ablation` puts rollout cosine at 0.9717
at four steps and 0.8258 at sixteen, and `action_effect_test` puts the action
signal at r=0.32 and r=0.09 over the same span. The late steps of a 16-step
rollout are genuinely less trustworthy, and a policy optimising against them will
find whatever the predictor extrapolates rather than whatever the game does.

So the longer horizon and this guard are one change, not two. The guard's job is
to notice when a rollout has left the manifold and hand off to the critic there
instead of continuing to pay out imaginary reward.

WHY MAHALANOBIS OVER THE CORPUS LATENTS, AND WHY IT IS ALMOST FREE
-------------------------------------------------------------------
The encoder is trained with SIGReg, which explicitly pushes the latent
distribution toward `N(0, I)`. So "is this latent plausible" already has a
natural answer -- how far it sits from that distribution in units of its own
spread -- and the fitted covariance should come out near identity, which is
itself a check worth printing.

Scoring is one `[B, D] @ [D, D]` matmul against a precomputed Cholesky factor,
which next to a 9.9M-parameter predictor step is free. That is the whole reason
to try this before the more obvious ensemble: `pred_dropout` is already 0.1, so K
stochastic forward passes would give an epistemic variance, but they also cost K
times the predictor -- the single most expensive thing in the loop. Cheap first,
and only escalate if the cheap one cannot separate.

THE TEST IS TWO-SIDED, AND THAT IS THE WHOLE POINT
---------------------------------------------------
The obvious form of this guard -- flag latents that are *far* from the corpus
mean -- would miss this model's actual failure mode. An autoregressive rollout
here degrades by blurring toward the mean: `horizon_ablation` measures cosine to
truth falling to 0.4170 and relative L2 rising to 0.925 by h=48, and
`HANDOFF.md` section 9 names "multi-step futures collapsing to a blur" as the
thing to fix. A collapsed latent sits *closer* to the mean than a real one, so a
one-sided distance test scores it as more in-distribution the more degraded it
gets -- an instrument that reads success at exactly the failure it exists to
catch, which is the pattern `docs/BUGS.md` opens with.

The fix is to test *typicality* rather than proximity. In D dimensions the
squared Mahalanobis distance of genuine N(0, I) samples concentrates around D:
real latents are neither unusually far from the mean nor unusually close to it.
So both tails are flagged, against thresholds read off the corpus's own score
distribution. Blur trips the lower one; extrapolation trips the upper.

THE THRESHOLDS ARE SET FROM DATA, NOT FROM TASTE
-------------------------------------------------
`fit` records the distribution of scores over the corpus latents themselves and
reads both cuts off it. `quantile = 0.99` means "one per cent of *real* frames
would be called out-of-distribution at each end", which is a statement anyone can
check, unlike a bare number.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class OODConfig:
    # Ridge added to the covariance before inversion. The bank is ~360k latents
    # in 192 dimensions so the estimate is well conditioned, but a fine-tuned
    # encoder could produce a near-degenerate direction and the inverse would
    # then dominate the score with pure estimation noise.
    shrinkage: float = 1e-3
    # Corpus quantile defining "out of distribution", applied at *both* tails:
    # the upper cut is at `quantile`, the lower at `1 - quantile`.
    quantile: float = 0.99
    # Reward penalty per nat of log-ratio outside the band (see `flags`). Being
    # a log ratio it is dimensionless, so this weight means the same thing
    # whatever the latent width or the refit. 0 disables the shaped term and
    # leaves only truncation.
    penalty: float = 0.0
    # Truncate the imagined rollout once the score is this many *times* outside
    # the band, at either end -- 2.0 is "twice the upper cut, or half the lower".
    # A ratio rather than an offset because the score is a positive scale
    # statistic whose two tails sit at very different magnitudes; `flags`
    # explains why an offset cannot trip from the collapsed side at all.
    #
    # Separate from the penalty because they do different jobs: the penalty
    # discourages drifting, truncation refuses to pay for what happens after it.
    hard_mult: float = 2.0
    truncate: bool = True


class LatentOOD(nn.Module):
    """Squared Mahalanobis distance to the corpus latent distribution.

    An `nn.Module` only so the fitted statistics move with `.to(device)` and are
    saved alongside whatever uses it; it has no trainable parameters and is
    registered entirely as buffers.
    """

    def __init__(self, dim: int, cfg: OODConfig | None = None):
        super().__init__()
        self.cfg = cfg or OODConfig()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("L", torch.eye(dim))
        self.register_buffer("hi", torch.tensor(float("inf")))
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, z: np.ndarray | torch.Tensor, batch: int = 65536) -> dict:
        """Fit against a bank of *encoder* latents and set the threshold.

        Encoder latents specifically: they are what real frames produce, so they
        define the manifold. Fitting against predictor outputs would bake the
        drift being measured into the reference and the score would be blind to
        exactly the thing it exists to catch.
        """
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.float()
        if z.ndim != 2:
            raise ValueError(f"expected [N, D] latents, got {tuple(z.shape)}")
        if z.shape[0] <= z.shape[1]:
            raise ValueError(
                f"{z.shape[0]} latents in {z.shape[1]} dimensions cannot "
                f"estimate a covariance; the bank is too small")
        dev = self.mu.device
        z = z.to(dev)
        mu = z.mean(0)
        centred = z - mu
        cov = (centred.T @ centred) / (len(z) - 1)
        cov = cov + self.cfg.shrinkage * torch.eye(len(mu), device=dev)
        # score = ||L (z - mu)||^2 with L L^T = cov^-1, obtained by inverting the
        # Cholesky factor of cov rather than forming cov^-1 explicitly.
        chol = torch.linalg.cholesky(cov)
        L = torch.linalg.solve_triangular(
            chol, torch.eye(len(mu), device=dev), upper=False)
        self.mu.copy_(mu)
        self.L.copy_(L)
        self.fitted.fill_(True)

        scores = torch.cat([self(z[i : i + batch]) for i in range(0, len(z), batch)])
        q = torch.tensor([1.0 - self.cfg.quantile, self.cfg.quantile], device=dev)
        lo, hi = torch.quantile(scores.float(), q)
        self.lo.fill_(float(lo))
        self.hi.fill_(float(hi))
        # A covariance near identity is what SIGReg is supposed to produce, so
        # reporting the deviation turns an assumption into a measurement.
        eye_err = float((cov - torch.eye(len(mu), device=dev)).abs().mean())
        return {"n": int(len(z)), "dim": int(len(mu)),
                "lo": float(lo), "hi": float(hi),
                "median": float(scores.median()),
                "mean": float(scores.mean()),
                # For genuine N(0, I) this sits at the dimension. How far the
                # median is from D says how Gaussian the latent really is.
                "expected_median": int(len(mu)),
                "cov_dev_from_identity": eye_err,
                "latent_var": float(z.var(0).mean())}

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [...] squared Mahalanobis distance."""
        d = (z - self.mu) @ self.L.T
        return (d * d).sum(-1)

    def flags(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (excess outside the band, hard-violation mask). Both [...].

        Excess is measured as a **log ratio** to the nearer edge of the band, not
        as an offset from it, because the score is a positive scale statistic and
        the two tails live on wildly different scales: for a 192-dimensional
        latent the upper cut sits near 250 and the lower near 150, while a fully
        collapsed latent scores ~0. An offset normalised by the band width
        therefore cannot reach a hard threshold from below at all -- the largest
        possible excess from that side is `lo / (hi - lo)`, which for realistic
        quantiles is well under one. A test written that way flags runaway
        latents and silently never flags collapsed ones, which is the exact
        failure this guard was made two-sided to avoid.

        In log space the two ends are symmetric: `hard_mult = 2.0` means "a
        factor of two outside the band", at either end. The result is also
        bounded in practice and grows only logarithmically, which is what a
        reward penalty wants -- a linear one would let a single badly drifted
        step dominate a whole trajectory's return.
        """
        if not bool(self.fitted):
            raise RuntimeError("LatentOOD.fit has not been called")
        s = self(z).clamp_min(1e-8)
        excess = (torch.log(s / self.hi).clamp(min=0.0)
                  + torch.log(self.lo / s).clamp(min=0.0))
        return excess, excess > math.log(max(self.cfg.hard_mult, 1.0 + 1e-9))


def truncate_after(hard: torch.Tensor) -> torch.Tensor:
    """[B, T] hard-violation flags -> [B, T] mask that is 0 from the first one on.

    The step that first goes out of distribution is itself masked out, not merely
    the ones after it: its reward was already read off a latent the probe has no
    business reading. Handing the trajectory to the critic at the *last good*
    state is the conservative choice, and the conservative choice is the right
    one when the alternative is paying imaginary reward.
    """
    if hard.ndim != 2:
        raise ValueError(f"expected [B, T] flags, got {tuple(hard.shape)}")
    return (hard.cumsum(dim=1) == 0).float()


def step_flags(state_hard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T+1] per-*state* violations -> ([B, T] live mask, [B, T] lambda scale).

    Two conversions happen here and both are easy to get subtly wrong.

    **States to steps.** `compute_rewards` pays step t for the transition from
    state t to state t+1, so step t's reward is contaminated if *either* endpoint
    has drifted. The step flag is therefore the OR of the two, not just the
    arrival state -- taking only one endpoint would let the first bad state
    through with a full reward attached.

    **Truncation, not termination.** The returned `lam_scale` is 0 exactly at the
    last live step and 1 before it. Multiplying the configured lambda by it makes
    that step take a pure one-step bootstrap (`r + gamma * v`) instead of
    recursing into imagined futures -- see `critic.lambda_returns`. The episode
    is *not* marked terminal: the match is still going, it is only the
    simulation's licence to describe it that has run out, and telling the policy
    otherwise would price drifting as though it were a KO.
    """
    if state_hard.ndim != 2:
        raise ValueError(f"expected [B, T+1] flags, got {tuple(state_hard.shape)}")
    if state_hard.shape[1] < 2:
        raise ValueError("need at least two states to form one step")
    step_hard = state_hard[:, :-1] | state_hard[:, 1:]        # [B, T]
    ok = truncate_after(step_hard)
    # `ok` shifted left by one, with the tail held: a rollout that never violates
    # keeps lam everywhere and is bootstrapped by the horizon in the usual way.
    nxt = torch.cat([ok[:, 1:], ok[:, -1:]], dim=1)
    return ok, nxt
