"""Latent planners: MPC trajectory optimisation in the world model's latent space.

Given a context of observed latents and a goal latent, find the action sequence
whose imagined rollout lands closest to the goal:

    a* = argmin_a  sum_k alpha_k * || zhat_{t+k} - z_g ||^2        (AdaJEPA Eq. 3)

Two solvers:

* :class:`CEMPlanner` -- Cross-Entropy Method, LeWM App. D's default
  (300 samples, 30 iterations, 30 elites, initial variance 1, horizon 5).
  Derivative-free, so it works unchanged on the binary Soku action space.
* :class:`GDPlanner` -- gradient descent straight through the action encoder.
  Cheaper per iteration; AdaJEPA reports results for both.

Both leave the model's parameters untouched -- only actions are optimised. The
model is put in ``eval()`` mode by the caller so the encoder's BatchNorm uses
running statistics rather than the statistics of a batch of candidate rollouts,
which would make the cost depend on the other candidates.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from ..config import Config
from ..model.world_model import LeWorldModel

CostWeights = Literal["final", "uniform", "linear"]


def require_eval(model: LeWorldModel) -> None:
    """Planning in train mode is silently wrong, so refuse it.

    Both projectors end in a BatchNorm. In train mode it normalises by the
    statistics of whatever batch it is handed -- and during planning that batch
    is the candidate action sequences. Every candidate's cost would then depend
    on the other candidates sampled alongside it, the ranking would shift with
    the sample, and CEM would be optimising a moving target. The failure is
    quiet: the planner still returns plausible-looking actions.
    """
    if model.training:
        raise RuntimeError(
            "planning requires model.eval(): BatchNorm would otherwise normalise "
            "each rollout by the statistics of the candidate batch, making a "
            "candidate's cost depend on the others sampled with it"
        )


def cost_weights(horizon: int, mode: CostWeights, device, dtype) -> torch.Tensor:
    """alpha_k in AdaJEPA Eq. 3.

    "final" matches LeWM (Fig. 4: "a latent cost between the final predicted
    state and the goal embedding"). "linear" ramps toward the horizon, which is
    gentler when the goal is far and early predictions are unreliable.
    """
    if mode == "final":
        w = torch.zeros(horizon, device=device, dtype=dtype)
        w[-1] = 1.0
    elif mode == "uniform":
        w = torch.ones(horizon, device=device, dtype=dtype) / horizon
    elif mode == "linear":
        w = torch.arange(1, horizon + 1, device=device, dtype=dtype)
        w = w / w.sum()
    else:
        raise ValueError(f"unknown cost weighting {mode!r}")
    return w


def rollout_cost(
    model: LeWorldModel,
    z_ctx: torch.Tensor,        # [N, H, latent]
    a_plan: torch.Tensor,       # [N, P, ticks, action_dim]
    z_goal: torch.Tensor,       # [latent] or [1, latent]
    weights: torch.Tensor,      # [P]
    a_hist: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted squared latent distance to the goal. Returns [N]."""
    pred = model.rollout(z_ctx, a_plan, a_hist=a_hist)     # [N, P, latent]
    d = ((pred - z_goal.reshape(1, 1, -1)) ** 2).mean(dim=-1)   # [N, P]
    return (d * weights.view(1, -1)).sum(dim=1)


class CEMPlanner:
    def __init__(self, model: LeWorldModel, cfg: Config, weighting: CostWeights = "final"):
        self.model = model
        self.cfg = cfg
        self.weighting = weighting

    @torch.no_grad()
    def plan(
        self,
        z_ctx: torch.Tensor,                     # [H, latent] or [1, H, latent]
        z_goal: torch.Tensor,                    # [latent]
        a_hist: Optional[torch.Tensor] = None,   # [H-1, ticks, action_dim]
    ) -> torch.Tensor:
        """Returns the optimised action chunk sequence, [P, ticks, action_dim]."""
        require_eval(self.model)
        cfg = self.cfg
        if z_ctx.dim() == 2:
            z_ctx = z_ctx.unsqueeze(0)
        device, dtype = z_ctx.device, z_ctx.dtype
        N, K, P = cfg.cem_samples, cfg.cem_elites, cfg.plan_horizon
        shape = (P, cfg.action_ticks, cfg.action_dim)

        ctx = z_ctx.expand(N, -1, -1)
        hist = None if a_hist is None else a_hist.unsqueeze(0).expand(N, -1, -1, -1)
        w = cost_weights(P, self.weighting, device, dtype)

        if cfg.action_space == "binary":
            # CEM over Bernoulli parameters rather than a Gaussian: sampling
            # continuous values and rounding would put most probability mass on
            # button combinations the model was never trained on.
            p = torch.full(shape, 0.5, device=device, dtype=dtype)
            for _ in range(cfg.cem_iters):
                cand = (torch.rand(N, *shape, device=device, dtype=dtype) < p).to(dtype)
                cost = rollout_cost(self.model, ctx, cand, z_goal, w, hist)
                elite = cand[cost.topk(K, largest=False).indices]
                # Laplace smoothing keeps p off 0/1 so later iterations can still
                # explore; without it the first unanimous elite bit freezes.
                p = (elite.sum(dim=0) + 1.0) / (K + 2.0)
            return (p > 0.5).to(dtype)

        mu = torch.zeros(shape, device=device, dtype=dtype)
        var = torch.full(shape, cfg.cem_init_var, device=device, dtype=dtype)
        for _ in range(cfg.cem_iters):
            std = var.clamp_min(1e-6).sqrt()
            cand = mu + std * torch.randn(N, *shape, device=device, dtype=dtype)
            cand = cand.clamp(-1.0, 1.0)     # actions live in the normalised box
            cost = rollout_cost(self.model, ctx, cand, z_goal, w, hist)
            elite = cand[cost.topk(K, largest=False).indices]
            mu, var = elite.mean(dim=0), elite.var(dim=0, unbiased=False)
        return mu.clamp(-1.0, 1.0)


class GDPlanner:
    """Gradient descent on the action sequence (AdaJEPA's other optimiser).

    Only meaningful for continuous actions; for binary ones it optimises
    relaxed probabilities in [0, 1] which are then thresholded, which is a
    cruder approximation than CEM's Bernoulli search.
    """

    def __init__(
        self,
        model: LeWorldModel,
        cfg: Config,
        steps: int = 50,
        lr: float = 0.1,
        weighting: CostWeights = "final",
    ):
        self.model = model
        self.cfg = cfg
        self.steps = steps
        self.lr = lr
        self.weighting = weighting

    def plan(
        self,
        z_ctx: torch.Tensor,
        z_goal: torch.Tensor,
        a_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        require_eval(self.model)
        cfg = self.cfg
        if z_ctx.dim() == 2:
            z_ctx = z_ctx.unsqueeze(0)
        device, dtype = z_ctx.device, z_ctx.dtype
        shape = (1, cfg.plan_horizon, cfg.action_ticks, cfg.action_dim)

        a = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
        hist = None if a_hist is None else a_hist.unsqueeze(0)
        w = cost_weights(cfg.plan_horizon, self.weighting, device, dtype)
        opt = torch.optim.Adam([a], lr=self.lr)

        # Detach the context: we optimise actions, not the encoder's output.
        ctx = z_ctx.detach()
        goal = z_goal.detach()

        # backward() through the world model would otherwise accumulate into
        # every model parameter. Left there, the next optimiser step -- training
        # resumed, or AdaJEPA's adaptation step -- would apply planning
        # gradients as if they were a learning signal. Freeze for the duration
        # and restore exactly what was frozen before.
        saved = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            for _ in range(self.steps):
                opt.zero_grad(set_to_none=True)
                act = torch.sigmoid(a) if cfg.action_space == "binary" else torch.tanh(a)
                cost = rollout_cost(self.model, ctx, act, goal, w, hist).sum()
                cost.backward()
                opt.step()
        finally:
            for p, flag in saved:
                p.requires_grad_(flag)

        with torch.no_grad():
            act = torch.sigmoid(a) if cfg.action_space == "binary" else torch.tanh(a)
            if cfg.action_space == "binary":
                act = (act > 0.5).to(dtype)
        return act[0].detach()
