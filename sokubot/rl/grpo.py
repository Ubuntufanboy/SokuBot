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
    # Entropy floor, as a fraction of the policy's entropy at initialisation.
    #
    # A fixed entropy *bonus* is the wrong shape for this: at 0.02 it pinned the
    # policy at 24.98 of a possible 25.4 nats for 625 steps, and at 0 the policy
    # collapsed to 0.05 and started mashing. What is wanted is not a preference
    # for entropy but a floor under it, so the term costs nothing while the
    # policy explores and only bites at the boundary. `entropy_coef` becomes a
    # learned multiplier on the violation, exactly as SAC tunes its temperature.
    #
    # Anchored to the initial entropy rather than to an absolute number because
    # the policy is initialised from the corpus action prior -- roughly 12.5 nats
    # against a uniform 25.4 -- so "80% of where human-like play starts" is a
    # meaningful floor and "10 nats" is an arbitrary one.
    entropy_floor_frac: float = 0.8
    # The multiplier starts near zero, not at one. Starting at alpha = 1 makes
    # the floor a full-strength entropy *bonus* from step one: the first attempt
    # drove the policy from its corpus-prior 10.76 nats to the 25.40 maximum
    # within fifty steps and a 0.43 press rate, destroying the initialisation
    # before the dual could decay. A floor has to cost nothing until it is
    # violated, which means starting inert and rising only if entropy falls.
    entropy_log_alpha_init: float = -4.0
    # Every bound here is set from a measurement, because the first version got
    # all three wrong and failed in both directions at once.
    #
    # The multiplier wound down to its -10 floor over the two thousand steps the
    # policy spent healthy, and Adam moves log_alpha by about `entropy_lr` per
    # step regardless of how badly the constraint is violated. So when entropy
    # finally fell it needed ~500 steps to climb back to a value that does
    # anything: one arm sat at entropy 3.89 against a floor of 8.59 with alpha
    # still 0.000, and the other took so long that it overshot to alpha 54.6.
    # Classic integrator wind-up, in a controller that has to act faster than
    # the collapse it is catching.
    #
    # -4 bounds the wind-up (alpha never drops below 0.018, still negligible)
    # and 0.0 caps it at alpha 1.0, because alpha = 1.0 is already violent: it
    # drove entropy from 10.76 to the 25.40 maximum in fifty steps. 54.6 was
    # never a sensible number. At lr 0.05 the whole range takes 80 steps.
    entropy_min_log_alpha: float = -4.0
    entropy_lr: float = 0.05
    max_log_alpha: float = 0.0
    # Passes over each batch of rollouts. With one pass the sampling policy *is*
    # the current policy, so the ratio is identically 1 and both the clipping and
    # the KL penalty are inert -- the update degenerates to REINFORCE with a
    # baseline. More than one makes them do their job.
    epochs: int = 2
    # Abandon the rest of a batch's inner epochs once the policy has already
    # moved this far from the one that sampled it. Standard PPO practice, and
    # the thing whose absence killed every run so far: with kl_coef alone the
    # measured KL went 0.48 -> 15 -> 42 -> 77 over 250 steps, and entropy
    # collapsed *after* that, as a symptom. A penalty argues with the gradient;
    # a target stops it. 0.05 is two orders of magnitude above the 0.0005 a
    # healthy step shows here and still far below the blow-out.
    target_kl: float = 0.05
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
        # Deliberately not seeded with z_ctx[:, -1]: that is an *encoder* latent
        # and every later state is a *predictor* output, so the state sequence
        # stays homogeneous and one probe reads all of it. One extra predictor
        # step is cheaper than reasoning about whether the two agree.
        #
        # This was originally written to fix a much larger effect -- the probe
        # read encoder latents at standard deviation 0.85 against 0.32 for
        # predictor outputs, fabricating damage at the first step. That gap was
        # not a property of the two distributions. It was `ckpt_cf/best.pt`
        # carrying BatchNorm running statistics that were never recalibrated,
        # which put its predictor outputs on the wrong scale entirely (one-step
        # skill -6.15). On a correctly saved checkpoint the same measurement
        # reads 0.3470 against 0.3438, i.e. no gap at all. Keeping the sequence
        # homogeneous is still the right call; the alarming number was a bug
        # elsewhere.
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


def _anchored_kl(log_r: torch.Tensor, fence: float) -> torch.Tensor:
    """k3 KL that keeps pulling once the policy is outside the fence.

    `exp(log_r) - 1 - log_r` on a hard-clamped `log_r` has *exactly zero
    gradient* beyond the clamp. As an anchor that is a fence whose gate opens
    outward: it holds while the policy stays near the reference and stops
    existing the moment the policy escapes, which is the one situation it was
    added for. The first run at the corrected reward drifted past it around step
    1000 and then ran away unopposed -- entropy 5.72 -> 0.05, KL to reference 87
    and still climbing, the policy pinned on deterministic button-mashing, and
    the eval gains it had accumulated to that point wiped out.

    Inside the fence this is unchanged. Outside, the exponential stays clamped
    (it is what overflows) and a linear term takes over, so the penalty grows
    without bound and its gradient keeps pointing home at constant magnitude.

    That is necessary and, on its own, not sufficient. It moved the collapse from
    step ~1200 to step ~2000 rather than preventing it. The reason is visible in
    the asymptotics: the policy collapses by becoming *confident*, so on its own
    sampled actions `logp -> 0` while `ref_logp` stays very negative, and the log
    ratio runs to minus infinity. On that side the true k3 KL is
    `exp(r) - 1 - r ~ -1 - r`, already only linear, so its restoring gradient is
    a constant 1 no matter how far the policy has gone. At `kl_ref_coef 0.05`
    that is a pull of 0.05 against a policy gradient that grows with confidence,
    and the anchor loses. Raising the coefficient to 0.3 does hold it -- and
    freezes the policy from step one, net gain 0.00002 over 1000 steps.

    So the KL anchor is the wrong instrument for this particular failure. What
    collapses is the entropy, and the thing to constrain is the entropy: a floor
    with a Lagrange multiplier, as SAC does, costs nothing while the policy is
    exploring and only bites at the boundary. Left undone deliberately rather
    than guessed at.
    """
    safe = log_r.clamp(-fence, fence)
    core = torch.exp(safe) - 1 - safe
    return core + (log_r.abs() - fence).clamp(min=0.0)


class EntropyFloor:
    """A learned multiplier that only pays attention below a target entropy.

    The dual of `maximise return subject to H(pi) >= H_floor`. `alpha` rises
    while the constraint is violated and decays to zero once it is satisfied,
    so an exploring policy is not taxed and a collapsing one is caught. This is
    SAC's temperature tuning with the inequality pointing the same way.

    The failure it exists for: with no entropy term the policy fell from 12.5
    nats to 0.05 and pinned itself on deterministic button-mashing at a 33%
    press rate, wiping out gains it had already banked. With a *fixed* bonus it
    sat at maximum entropy instead, unable to commit to anything. Neither is a
    tuning problem; a bonus and a floor are different objectives.
    """

    def __init__(self, cfg: GRPOConfig, device):
        self.cfg = cfg
        self.log_alpha = torch.tensor(cfg.entropy_log_alpha_init, device=device,
                                      requires_grad=True)
        self.opt = torch.optim.Adam([self.log_alpha], lr=cfg.entropy_lr)
        self.floor: Optional[float] = None

    def set_floor_from(self, initial_entropy: float) -> float:
        self.floor = self.cfg.entropy_floor_frac * initial_entropy
        return self.floor

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.detach().clamp(max=self.cfg.max_log_alpha).exp()

    def update(self, entropy: float) -> dict:
        """Push alpha up while entropy sits under the floor, down while over."""
        if self.floor is None:
            return {"alpha": 0.0, "floor": float("nan"), "ent_violation": 0.0}
        # Relative violation, so `entropy_lr` means the same thing whatever the
        # action space's entropy scale happens to be.
        loss = self.log_alpha.clamp(max=self.cfg.max_log_alpha) * (
            (entropy - self.floor) / max(abs(self.floor), 1e-6))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(self.cfg.entropy_min_log_alpha,
                                  self.cfg.max_log_alpha)
        return {"alpha": float(self.alpha), "floor": self.floor,
                "ent_violation": self.floor - entropy}


def grpo_loss(policy: SokuPolicy, traj: dict, adv: torch.Tensor,
              old_logp: torch.Tensor, cfg: GRPOConfig,
              ref_logp: Optional[torch.Tensor] = None,
              ent_alpha: float = 0.0) -> tuple[torch.Tensor, dict]:
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
        kl_ref = _anchored_kl(ref_logp - logp, cfg.max_log_ratio)
    else:
        kl_ref = torch.zeros_like(kl)

    denom = alive.sum().clamp(min=1.0)
    # `ent_alpha` is the EntropyFloor's multiplier: zero while the policy is
    # above its floor, growing while it is below. `entropy_coef` is the old
    # fixed bonus and stays at 0 unless something explicitly wants it.
    loss = (((pg + cfg.kl_coef * kl + cfg.kl_ref_coef * kl_ref
              - (cfg.entropy_coef + ent_alpha) * ent) * alive).sum() / denom)
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
