"""A symlog twohot critic and the lambda-returns it bootstraps.

WHY A CRITIC AT ALL, WHEN GRPO ALREADY HAS A BASELINE
------------------------------------------------------
GRPO estimates its baseline by rolling one start state `group_size` times and
subtracting the group mean. That works, but it spends G rollouts per start
purely on the baseline: at the settings that produced the best policy,
`256 starts x 16 group`, fifteen sixteenths of the predictor budget buys variance
reduction rather than state coverage. A critic amortises the baseline across the
whole batch, so the same budget covers 16x more distinct start states -- and
low-health starts, the leading suspect for the give-up behaviour, are rare enough
in the corpus that coverage is exactly what is scarce.

The second reason is the one that matters more. A GRPO return is a single number
summarising a whole rollout, so its quality is the quality of the *longest*
rollout -- and `docs/HANDOFF.md` section 3 measures the action signal decaying
from r=0.55 at one step to r=0.09 at sixteen. A critic bootstraps instead:
imagination only has to be right over H short steps and `v(s_H)` carries
everything past it. That is how the 0.27 s trustworthy horizon stops being a
ceiling on credit assignment.

WHY SYMLOG AND TWOHOT RATHER THAN A SCALAR MSE HEAD
----------------------------------------------------
The reward spans two very different scales. A damage exchange over one rollout is
worth about 0.1 in bar units; `RewardConfig.win/lose` are +-5. Under MSE the
squared error of a mispredicted KO is ~2500x that of a mispredicted chip
exchange, so the head spends all of its capacity on the rare terminal event and
reads as noise everywhere the policy actually operates.

DreamerV3's answer, adopted here: regress a *distribution* over a fixed grid of
symlog-spaced bins with cross-entropy, and read the value back as the mean.
symlog compresses the tail so +-5 and +-0.1 are both representable at useful
resolution; twohot keeps the target continuous, so a value of 0.37 between bins
0.3 and 0.5 is encoded as a genuine mixture rather than being rounded. The loss
is then cross-entropy, which is scale-free -- it cannot be dominated by the tail
because it is not measuring squared distance at all.

THE EMA TARGET
--------------
The bootstrap `v(s_{t+1})` inside the lambda-return is read from a slow copy of
the critic rather than the critic itself. Without it the target moves with every
update and the regression chases itself; this is cheap and removes a whole class
of divergence. tau is the *retention* per update, so 0.98 means the target keeps
98% of itself and takes 2% of the online weights each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(|x| + 1). Monotone, invertible, and identity-ish near 0."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(x) * torch.expm1(x.abs())


def twohot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """[...] values -> [..., n_bins] weights on the two bins that bracket each value.

    The two weights are the linear-interpolation coefficients, so the encoded
    distribution has exactly the input as its mean. Values outside the grid are
    clamped onto the end bins, which is why the grid has to be wide enough to
    hold the largest return the reward can produce -- see `CriticConfig.limit`.
    """
    if bins.ndim != 1 or bins.numel() < 2:
        raise ValueError(f"bins must be a 1-D grid of >=2 entries, got {tuple(bins.shape)}")
    n = bins.numel()
    x = x.clamp(float(bins[0]), float(bins[-1]))
    # Index of the bin at or below x. searchsorted with `right=True` returns the
    # insertion point, so subtracting one gives the lower bracket; clamping keeps
    # `hi = lo + 1` in range for x exactly on the last bin.
    lo = (torch.searchsorted(bins, x.detach().contiguous(), right=True) - 1).clamp(0, n - 2)
    hi = lo + 1
    b_lo, b_hi = bins[lo], bins[hi]
    # Equal bins cannot happen on a linspace grid, but a guard here costs nothing
    # and the alternative is a silent NaN that only appears in the loss.
    w_hi = (x - b_lo) / (b_hi - b_lo).clamp_min(1e-8)
    w_lo = 1.0 - w_hi
    out = torch.zeros(*x.shape, n, device=x.device, dtype=w_lo.dtype)
    out.scatter_(-1, lo.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


@dataclass
class CriticConfig:
    n_bins: int = 41
    # Grid half-width in *symlog* units. symlog(5) = 1.79 and symlog(20) = 3.04,
    # so 6.0 covers a return of symexp(6) = 402 bars -- far beyond anything
    # `RewardConfig` can pay out, which is the safe direction to be wrong in:
    # a value clamped onto the end bin is a silent ceiling on the critic.
    limit: float = 6.0
    width: int = 512
    depth: int = 3
    lr: float = 3e-4
    # Retention of the EMA target per update.
    tau: float = 0.98
    gamma: float = 0.99
    lam: float = 0.95


class SokuCritic(nn.Module):
    """[B, H, latent] + side -> a distribution over symlog-spaced return bins.

    Shaped deliberately like `SokuPolicy`: same input, same trunk, same side
    embedding. The side matters for the same reason it does there -- the probe's
    channels are P1-first, so "my health" is a different column depending on the
    chair, and a critic that could not tell the chairs apart would have to
    average two anti-correlated value functions.
    """

    def __init__(self, latent_dim: int = 192, history: int = 3,
                 cfg: CriticConfig | None = None):
        super().__init__()
        self.cfg = cfg or CriticConfig()
        c = self.cfg
        self.side_emb = nn.Embedding(2, c.width)
        layers: list[nn.Module] = [nn.Linear(latent_dim * history, c.width), nn.GELU()]
        for _ in range(c.depth - 1):
            layers += [nn.Linear(c.width, c.width), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(c.width, c.n_bins)
        # Zero init so the critic predicts a uniform distribution over bins at
        # step 0, whose mean is 0 by symmetry of the grid. A critic that starts
        # with a large arbitrary value injects that error into every advantage
        # in the first batches, which is when the policy is most movable.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("bins", torch.linspace(-c.limit, c.limit, c.n_bins))

    def logits(self, z_hist: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        B = z_hist.shape[0]
        h = self.trunk(z_hist.reshape(B, -1)) + self.side_emb(side)
        return self.head(h)

    def forward(self, z_hist: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """-> [B] value in reward units."""
        probs = F.softmax(self.logits(z_hist, side), dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def loss(self, z_hist: torch.Tensor, side: torch.Tensor,
             target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Cross-entropy against the twohot encoding of symlog(target).

        `target` is in reward units; the transform to symlog happens here so no
        caller has to remember it. Returns (per-sample loss [B], diagnostics).
        """
        logits = self.logits(z_hist, side)
        with torch.no_grad():
            tgt = twohot(symlog(target), self.bins)
        loss = -(tgt * F.log_softmax(logits, dim=-1)).sum(-1)
        with torch.no_grad():
            pred = symexp((F.softmax(logits, -1) * self.bins).sum(-1))
            stats = {"value_mean": float(pred.mean()),
                     "value_std": float(pred.std()),
                     "target_mean": float(target.mean()),
                     "target_std": float(target.std()),
                     # Fraction of targets pinned to an end bin. Anything above
                     # ~0 means `limit` is too small and the critic has a ceiling.
                     "clipped": float(((symlog(target).abs()
                                        >= self.cfg.limit).float()).mean())}
        return loss, stats


class TargetCritic:
    """A slow copy of the critic, used only for the bootstrap value.

    Not an `nn.Module`: it holds no gradients and must never be optimised, and
    registering it as a submodule would put it in the optimiser's parameter list
    and in the saved state dict as if it were trainable.
    """

    def __init__(self, critic: SokuCritic, tau: float):
        import copy
        self.tau = tau
        self.net = copy.deepcopy(critic).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, critic: SokuCritic) -> None:
        for t, s in zip(self.net.parameters(), critic.parameters()):
            t.lerp_(s.detach(), 1.0 - self.tau)
        # Buffers (the bin grid) are constant, but copying them keeps the target
        # correct if a future config ever makes them depend on training.
        for t, s in zip(self.net.buffers(), critic.buffers()):
            t.copy_(s)

    @torch.no_grad()
    def __call__(self, z_hist: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        return self.net(z_hist, side)


def lambda_returns(reward: torch.Tensor, value: torch.Tensor,
                   cont: torch.Tensor, gamma: float,
                   lam: float | torch.Tensor) -> torch.Tensor:
    """The recursion, computed backwards through an imagined rollout.

        V_t = r_t + gamma * c_t * [ (1 - lam_t) * v(s_{t+1}) + lam_t * V_{t+1} ]
        V_H = v(s_H)

    reward [B, T]      r_t, paid for the transition from state t to t+1
    value  [B, T+1]    v(s_t) for every state including the bootstrap at T
    cont   [B, T]      1 while the episode continues past step t, 0 at a terminal
    lam                scalar, or [B, T] to vary it per step

    TERMINATION AND TRUNCATION ARE DIFFERENT, AND THE TWO KNOBS ARE DIFFERENT
    -------------------------------------------------------------------------
    `cont = 0` is **termination**: the episode is over, nothing follows, and the
    recursion stops carrying value across the boundary. A KO is this.

    `lam = 0` is **truncation**: the episode continues but this rollout stops
    trusting itself, so the step takes a pure one-step bootstrap `r_t + gamma *
    v(s_{t+1})` instead of recursing into imagined futures. Running out of
    horizon is this, and so is the OOD guard in `rl/ood.py` deciding the latent
    has drifted off the manifold -- there the game has not ended, only the
    simulation's licence to describe it.

    Collapsing the two would be wrong in both directions: terminating on an OOD
    step would teach the policy that drifting is worth whatever a KO is worth,
    and bootstrapping through a KO would credit a finished match with a value for
    a state that does not exist.

    `cont` is what makes a KO terminal rather than merely unrewarding: at c_t = 0
    the recursion stops carrying value across the boundary, so a trajectory that
    ends does not get credited with whatever the critic happens to predict for
    the state after it ended. GRPO handled this by masking the step out of the
    loss entirely, which also removed its gradient -- the mechanism proposed in
    `docs/HANDOFF.md` section 10 for why the agent gives up at low health.
    Bootstrapping with cont = 0 pays the terminal value once and keeps the steps
    leading up to it fully weighted.
    """
    B, T = reward.shape
    if value.shape != (B, T + 1):
        raise ValueError(
            f"value must be [B, T+1] = {(B, T + 1)} to bootstrap a {T}-step "
            f"rollout, got {tuple(value.shape)}")
    if cont.shape != (B, T):
        raise ValueError(f"cont must be [B, T] = {(B, T)}, got {tuple(cont.shape)}")
    if isinstance(lam, torch.Tensor):
        if lam.shape != (B, T):
            raise ValueError(
                f"per-step lam must be [B, T] = {(B, T)}, got {tuple(lam.shape)}")
        lam_t = lam
    else:
        lam_t = torch.full_like(reward, float(lam))

    out = torch.empty_like(reward)
    nxt = value[:, -1]
    for t in range(T - 1, -1, -1):
        l = lam_t[:, t]
        nxt = reward[:, t] + gamma * cont[:, t] * (
            (1.0 - l) * value[:, t + 1] + l * nxt)
        out[:, t] = nxt
    return out


def continuation(terminal: torch.Tensor) -> torch.Tensor:
    """`reward.terminal_mask` -> the `cont` flag lambda_returns wants.

    Deliberately *not* derived from `compute_rewards`' `alive` mask. `alive` is 1
    up to and including the KO step, so `alive[t+1]` gives the right answer
    everywhere except the last step of the rollout -- where a KO landing exactly
    on the horizon is indistinguishable from a rollout that merely ran out of
    steps. Those want opposite treatment: the first must not bootstrap, the
    second must. Reading the terminal flag directly removes the ambiguity rather
    than resolving it by whichever guess is wrong less often.
    """
    if terminal.ndim != 2:
        raise ValueError(f"terminal must be [B, T], got {tuple(terminal.shape)}")
    return 1.0 - terminal
