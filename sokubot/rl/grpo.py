"""GRPO inside the imagined world.

The world model and the probe are frozen. Only the policy learns. A step is:

  1. draw start states from real gameplay, and a side for each,
  2. expand every start into a *group* of G rollouts,
  3. roll the predictor forward under the policy's sampled actions, with an
     opponent filling the other player's buttons,
  4. read game state off each imagined latent with the calibrated probe and
     score it with `reward.compute_rewards`,
  5. centre each rollout's return against its own group and take a clipped
     policy-gradient step.

WHY THE GROUP IS (START, SIDE) AND NOT JUST START
-------------------------------------------------
GRPO replaces a learned critic with the mean return of a group of rollouts that
share a prompt. That only works when the group members are exchangeable samples
of the same situation. P1's and P2's trajectories through one match are not:
they are opposite sides of a nearly zero-sum game, so their returns are
anti-correlated and pooling them makes the baseline measure which side got the
better draw rather than which rollout played better. Every group here therefore
fixes both the start state and the side, and the two sides are separate groups
even when they come from the same starting frame.

WHY THE OPPONENT IS NOT ALWAYS THE CURRENT POLICY
-------------------------------------------------
Pure self-play converges to a private equilibrium: two copies of one policy find
a single line that beats each other and never explore anything else. It scores
beautifully against itself and loses to a human immediately. Sampling the
opponent from a pool -- the current policy, frozen snapshots of it, and recorded
human input -- keeps a spread of styles in the population.

The human share is the recorded `.rep` input itself, not a model of it. The
behaviour-cloned head reached 0.910 accuracy against a 0.896 base rate, which is
not distinguishable from always predicting "no button pressed", so it is not
used. Replaying the buttons a person actually pressed costs nothing, carries no
model error, and is the same distribution the agent has to beat online.

Its one weakness is that it is open-loop: the recorded human cannot react to
what our agent does, so the further a rollout diverges from what really
happened, the less the replayed input is a response to the situation on screen.
That is tolerable at the horizon this trains at (roughly a second and a half)
and it is why the pool mixes in reactive opponents rather than relying on
replay alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import SokuPolicy, jitter_actions, to_joint
from .reward import RewardConfig, compute_rewards


@dataclass
class GRPOConfig:
    horizon: int = 24                # decision steps; set from the ablation
    group_size: int = 8              # rollouts sharing one (start, side)
    starts_per_batch: int = 48
    gamma: float = 0.99
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    # Raised to 0.02 after the first run lost 40% of its entropy in 25 steps,
    # which was the right diagnosis of the wrong cause: that collapse came from
    # per-group advantage scaling amplifying tiny return differences, and is
    # fixed by `advantage_scale`. At 0.02 the entropy bonus contributes about 0.5
    # to the loss while the policy-gradient term is far smaller, so the policy
    # sat at 24.98 of a possible 25.4 for 625 steps -- pinned uniform, unable to
    # commit to anything.
    #
    # Worth being clear that the rollout is deterministic: one start plus one
    # action sequence gives exactly one return, so differences within a group are
    # entirely caused by the actions sampled. There is no observation noise for
    # exploration to average out, and the gradient is small rather than noisy.
    # Anchor to the initial policy, not to the uniform distribution. An entropy
    # bonus rewards spreading mass evenly, which actively undoes the corpus
    # prior: the first prior-initialised run started at 15.0 nats and climbed to
    # 18.0 within fifty steps, walking back out of the distribution the world
    # model was trained on. A KL penalty toward the reference keeps the policy
    # near human-like play while leaving it free to prefer better actions within
    # that neighbourhood -- the same construction RLHF uses, for the same reason.
    entropy_coef: float = 0.0
    kl_ref_coef: float = 0.05
    # Passes over each batch of rollouts. With one pass the sampling policy *is*
    # the current policy, so the ratio is identically 1 and both the clipping and
    # the KL penalty are inert -- the update degenerates to REINFORCE with a
    # baseline. More than one makes them do their job.
    epochs: int = 2
    advantage_scale: str = "batch"
    # Bounds exp() in the surrogate and the KL. exp(5) ~ 148 is already far
    # outside the clipped region, so this costs nothing the objective was using.
    max_log_ratio: float = 5.0
    lr: float = 1e-4
    grad_clip: float = 1.0
    jitter_sigma: float = 1.0        # frames; the user's "acts more human" noise
    # Probability of drawing the opponent from (current policy, snapshot,
    # recorded human input). Falls back to the current policy when no snapshot
    # exists yet.
    opponent_mix: tuple[float, float, float] = (0.35, 0.35, 0.30)
    snapshot_every: int = 250
    max_snapshots: int = 8
    reward: RewardConfig = field(default_factory=RewardConfig)


class ProbeHead(nn.Module):
    """The fitted LinearProbe as a frozen torch module, for batched GPU reads."""

    def __init__(self, probe):
        super().__init__()
        self.register_buffer("zmu", torch.tensor(probe.zmu, dtype=torch.float32))
        self.register_buffer("zsd", torch.tensor(probe.zsd, dtype=torch.float32))
        self.register_buffer("ymu", torch.tensor(probe.ymu, dtype=torch.float32))
        self.register_buffer("ysd", torch.tensor(probe.ysd, dtype=torch.float32))
        self.register_buffer("W", torch.tensor(probe.W, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[..., latent] -> [..., n_targets] in bar units."""
        x = (z - self.zmu) / self.zsd
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        return (x @ self.W) * self.ysd + self.ymu


class ReplayOpponent:
    """Plays back the buttons a real person pressed, from the start's own replay.

    `future` is [B, T, ticks, 20] lifted straight from the capture the start
    state was drawn from, so step t returns what the human on that side actually
    did t steps after that frame. It ignores `z_hist` entirely -- that is the
    open-loop caveat in the module docstring, not an oversight.
    """

    def __init__(self, future: torch.Tensor):
        self.future = future
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    @torch.no_grad()
    def act(self, z_hist: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        step = self.future[:, min(self.t, self.future.shape[1] - 1)]   # [B,ticks,20]
        self.t += 1
        # The opponent holds the chair the agent does not.
        idx = ((1 - side) * 10)[:, None, None] + torch.arange(10, device=step.device)
        return step.gather(-1, idx.expand(-1, step.shape[1], 10))


class PolicyOpponent:
    """A (possibly frozen) SokuPolicy playing the other side."""

    def __init__(self, policy: SokuPolicy):
        self.policy = policy

    @torch.no_grad()
    def act(self, z_hist: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        # The opponent occupies the *other* chair, so its side is flipped.
        return self.policy(z_hist, 1 - side, sample=True).actions


class ImaginedArena:
    """Rolls a policy against an opponent inside the frozen world model."""

    def __init__(self, world_model, probe_head: ProbeHead, cfg: GRPOConfig,
                 history: int, ticks: int):
        self.wm = world_model.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)
        self.probe = probe_head
        self.cfg = cfg
        self.H = history
        self.ticks = ticks

    @torch.no_grad()
    def rollout(self, z_ctx: torch.Tensor, a_hist: torch.Tensor,
                side: torch.Tensor, policy: SokuPolicy, opponent) -> dict:
        """z_ctx [B,H,latent], a_hist [B,H-1,ticks,20], side [B] -> trajectory.

        Returns the policy inputs and sampled actions at every step, so the
        update can re-score them, plus the probed state sequence the reward
        reads. Nothing here carries gradient: the update recomputes log-probs.
        """
        cfg, T = self.cfg, self.cfg.horizon
        if hasattr(opponent, "reset"):
            opponent.reset()          # replay opponents are stateful in time
        z_win, a_win = z_ctx, a_hist
        # Deliberately not seeded with z_ctx[:, -1]. That is an *encoder* latent
        # and every later state is a *predictor* output, and the reward probe is
        # calibrated for one of those two distributions, not both. Reading the
        # pair through one probe compares numbers on different scales: on encoder
        # latents the calibrated probe emits health with standard deviation 0.85
        # against a label spread of 0.32, which manufactures a large fake change
        # at the first step and puts every later comparison against a corrupted
        # baseline. One extra predictor step costs less than that.
        zs, mine_all, joint_all, obs_all = [], [], [], []

        for _ in range(T + 1):
            obs_all.append(z_win)
            out = policy(z_win, side, sample=True)
            mine = jitter_actions(out.actions, cfg.jitter_sigma)
            theirs = jitter_actions(opponent.act(z_win, side), cfg.jitter_sigma)
            joint = to_joint(mine, theirs, side)

            a_full = torch.cat([a_win, joint[:, None]], dim=1)
            zhat = self.wm.predictor(z_win, self.wm.action_encoder(a_full))[:, -1]

            mine_all.append(mine)
            joint_all.append(joint)
            zs.append(zhat)
            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            if self.H > 1:
                a_win = torch.cat([a_win[:, 1:], joint[:, None]], dim=1)

        # T+1 predicted states against the T actions that produced the last T of
        # them, so the state sequence is homogeneous and the action alignment is
        # unchanged: joint_all[k] carries zs[k] to zs[k+1].
        states = self.probe(torch.stack(zs, dim=1))          # [B, T+1, K]
        joint_seq = torch.stack(joint_all[1:], dim=1)        # [B, T, ticks, 20]
        reward, alive, terms = compute_rewards(states, joint_seq, side, cfg.reward)
        return {"obs": torch.stack(obs_all[1:], dim=1),      # [B, T, H, latent]
                "mine": torch.stack(mine_all[1:], dim=1),    # [B, T, ticks, 10]
                "reward": reward, "alive": alive, "terms": terms,
                "states": states, "side": side}


def group_advantages(reward: torch.Tensor, alive: torch.Tensor,
                     group_size: int, gamma: float, scale: str = "batch",
                     eps: float = 1e-6) -> torch.Tensor:
    """Discounted return-to-go, centred within each group of `group_size`.

    Each group shares a start state and a side, so subtracting the group mean
    removes exactly the part of the return that came from the situation rather
    than from the policy's choices -- which is the baseline a critic would
    otherwise have to learn. That part is not optional; it is what GRPO is.

    The *scaling* is. Dividing by each group's own standard deviation, as the
    original formulation does, is actively harmful when rewards are read from a
    noisy probe: a group whose eight rollouts genuinely played out the same way
    has a return spread of almost nothing, and dividing by it rescales pure
    measurement noise up to unit magnitude. The policy then receives a confident
    gradient telling it to prefer whichever rollout the probe happened to
    misread upward. Scaling by the batch's spread instead leaves such a group
    with the near-zero advantage it deserves, while keeping the overall gradient
    scale stable. Pass ``scale="group"`` for the original behaviour.
    """
    B, T = reward.shape
    ret = torch.zeros_like(reward)
    run = torch.zeros(B, device=reward.device)
    for t in range(T - 1, -1, -1):
        run = reward[:, t] + gamma * run * alive[:, t]
        ret[:, t] = run
    g = ret.view(-1, group_size, T)
    centred = g - g.mean(dim=1, keepdim=True)
    if scale == "group":
        adv = centred / (g.std(dim=1, keepdim=True) + eps)
    elif scale == "batch":
        # Per timestep: return-to-go shrinks as the horizon runs out, so one
        # scalar for the whole trajectory would over-weight early steps.
        adv = centred / (centred.std(dim=(0, 1), keepdim=True) + eps)
    elif scale == "none":
        adv = centred
    else:
        raise ValueError(f"unknown scale {scale!r}; want group, batch or none")
    return adv.view(B, T)


def grpo_loss(policy: SokuPolicy, traj: dict, adv: torch.Tensor,
              old_logp: torch.Tensor, cfg: GRPOConfig,
              ref_logp: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, dict]:
    """Clipped surrogate, a k3 KL back to the sampling policy, and optionally a
    second KL back to a fixed reference.

    The two KLs do different jobs. The one against `old_logp` is a trust region:
    it keeps each update small relative to the policy the rollouts were drawn
    from. The one against `ref_logp` is an anchor: it keeps the policy near the
    behaviour the world model can be trusted to simulate, however many updates
    accumulate.
    """
    B, T = adv.shape
    H, latent = traj["obs"].shape[2], traj["obs"].shape[3]
    obs = traj["obs"].reshape(B * T, H, latent)
    side = traj["side"][:, None].expand(B, T).reshape(B * T)
    acts = traj["mine"].reshape(B * T, *traj["mine"].shape[2:])

    logp, ent = policy.log_prob_of(obs, side, acts)
    logp, ent = logp.view(B, T), ent.view(B, T)
    alive = traj["alive"]

    # One decision step's log-probability is a sum over ticks x (two categoricals
    # plus six Bernoullis) -- about thirty terms. Small per-term movements
    # compound, so the joint log-ratio swings far more than a scalar action's
    # would, and exp() of it overflows long before the policy has done anything
    # unreasonable. The first run reached a k3 KL of 5.8e8 at step 1350 and
    # produced NaN logits shortly after. Clamping the exponent bounds both the
    # surrogate and the penalty without changing either where they matter, since
    # the clipped objective already ignores ratios outside +-clip_eps.
    log_ratio = (logp - old_logp).clamp(-cfg.max_log_ratio, cfg.max_log_ratio)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    pg = -torch.min(unclipped, clipped)

    # k3 estimator: always positive, low variance, and unbiased for the KL.
    log_r = -log_ratio
    kl = torch.exp(log_r) - 1 - log_r

    if ref_logp is not None:
        log_r_ref = (ref_logp - logp).clamp(-cfg.max_log_ratio, cfg.max_log_ratio)
        kl_ref = torch.exp(log_r_ref) - 1 - log_r_ref
    else:
        kl_ref = torch.zeros_like(kl)

    denom = alive.sum().clamp(min=1.0)
    loss = (((pg + cfg.kl_coef * kl + cfg.kl_ref_coef * kl_ref
              - cfg.entropy_coef * ent) * alive).sum() / denom)
    with torch.no_grad():
        stats = {
            "pg": float((pg * alive).sum() / denom),
            "kl": float((kl * alive).sum() / denom),
            "entropy": float((ent * alive).sum() / denom),
            "ratio": float((ratio * alive).sum() / denom),
            "clip_frac": float((((ratio - 1).abs() > cfg.clip_eps) * alive).sum() / denom),
            "kl_ref": float((kl_ref * alive).sum() / denom),
        }
    return loss, stats


class SnapshotPool:
    """Frozen past copies of the policy, kept as opponents."""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.snaps: list[SokuPolicy] = []

    def maybe_add(self, policy: SokuPolicy, step: int) -> None:
        if step and step % self.cfg.snapshot_every == 0:
            import copy
            snap = copy.deepcopy(policy).eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            self.snaps.append(snap)
            if len(self.snaps) > self.cfg.max_snapshots:
                self.snaps.pop(0)

    def sample(self, rng: np.random.Generator) -> Optional[SokuPolicy]:
        if not self.snaps:
            return None
        return self.snaps[rng.integers(len(self.snaps))]
