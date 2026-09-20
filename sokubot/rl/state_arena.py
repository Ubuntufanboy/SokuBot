"""GRPO's environment, rolled forward in state space instead of latent space.

`rl/grpo.py`'s `ImaginedArena` rolls a JEPA predictor forward and reads game
state off each imagined latent with a linear probe. This does the same job
against `model/state_dynamics.StateDynamics`, and the difference is not an
implementation detail:

    latent arena   z -> z', reward = probe(z'),  residual 0.116 of a bar,
                   position unreadable (mirror AUC 0.51 = chance)
    state arena    s -> s', reward = s'.hp,      no probe at all,
                   position is an input column

Everything else is deliberately shared. `SokuPolicy`, `jitter_actions`,
`to_joint`, `group_advantages`, `grpo_loss`, `EntropyFloor`, `SnapshotPool` and
`ReplayOpponent` are imported from the latent stack unchanged, because none of
them ever cared what the observation meant -- only its width.

THE OBSERVATION IS EGO-ORDERED, AND THAT IS FREE ACCURACY
---------------------------------------------------------
The simulator is fed raw, player-major, P1-first state, because that is what it
was trained on. The POLICY is fed the same numbers reordered so the agent's own
row comes first, its opponent's second, the objects flying AT it third and the
ones it owns fourth.

That costs one gather and removes a whole symmetry the network would otherwise
have to learn twice. `data/state.py` already computes `dx`, `dy` and `facing`
per player from that player's point of view, so a reordered row is genuinely
ego-relative; only absolute `x` stays a fact about the stage, which is what it
is for. The side embedding is kept anyway -- it costs a lookup and leaves the
policy able to express anything genuinely P1-specific.

THE PROJECTILE FEEDBACK PATH HAS A DEFECT, AND IT IS REPRODUCED ON PURPOSE
--------------------------------------------------------------------------
`state_dynamics._unroll_impl` sigmoids the WHOLE projectile tensor before
feeding it back, not just the two logit features (`present`, `hb`). So from
step 1 onward a bullet's `dx`, `dy`, `vx`, `vy` are squashed into (0, 1) and
their sign -- which is the entire content of "is it coming at me" -- is gone.

The fix is obvious and is NOT applied by default, because that path is the one
the simulator was trained through: `unroll` and `rollout` share `_unroll_impl`,
so the model has learned to consume squashed projectiles and handing it raw
ones is off-distribution. `proj_feedback="raw"` selects the fixed behaviour so
the two can be measured against each other on the same checkpoint
(`scripts/state_preflight.py`), and the honest resolution is to retrain with
the fix rather than to flip a flag under a model that never saw it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..data.state import PROJ_FEATURES, STATE_CHANNELS
from ..model.state_dynamics import feed_proj
from ..model.state_head import BINARY
from .policy import SokuPolicy, jitter_actions, to_joint
from .state_reward import StateRewardConfig, compute_rewards, terminal_mask

N_STATE = len(STATE_CHANNELS)
N_PROJF = len(PROJ_FEATURES)


@dataclass
class StateGRPOConfig:
    """GRPO settings for the state arena.

    Every default that differs from `rl.grpo.GRPOConfig` differs for a reason
    recorded here; everything else is deliberately identical so the two runs
    stay comparable.
    """

    # 24 was set from the latent model's horizon ablation. The state simulator
    # was measured directly: kinematic skill +0.613 at one step (83 ms), +0.588
    # at four, +0.268 at sixteen, and in game units the edge over standing
    # still narrows from 7.7-vs-14.3 units at one step to 124-vs-168 at
    # sixteen. 8 steps is 667 ms and sits where the model is still clearly
    # better than a no-op.
    horizon: int = 8
    group_size: int = 8
    starts_per_batch: int = 64
    gamma: float = 0.99
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    kl_ref_coef: float = 0.05
    entropy_floor_frac: float = 0.8
    entropy_log_alpha_init: float = -4.0
    entropy_min_log_alpha: float = -4.0
    entropy_lr: float = 0.05
    max_log_alpha: float = 0.0
    epochs: int = 2
    target_kl: float = 0.05
    advantage_scale: str = "batch"
    max_log_ratio: float = 5.0
    lr: float = 1e-4
    grad_clip: float = 1.0
    jitter_sigma: float = 1.0
    snapshot_every: int = 250
    max_snapshots: int = 8
    reward: StateRewardConfig = field(default_factory=StateRewardConfig)


class StateObs(nn.Module):
    """Raw state + projectiles -> the flat, standardised, ego-ordered vector.

    A module rather than a function so the normalisation travels inside the
    checkpoint. The alternative -- recomputing corpus statistics at load time --
    silently changes the policy's input space whenever the corpus changes, and
    a policy whose observation drifted is wrong in a way that trains happily.

    Standardisation is not optional here. `untech` and `action_frame` are
    normalised by 60 frames but reach magnitudes of 441 (untech holds stale
    readings up to 26 507), so an unstandardised trunk would see one input four
    hundred times the size of health and learn that column.
    """

    def __init__(self, s_mu, s_sd, p_mu, p_sd, slots: int, clip: float = 8.0):
        super().__init__()
        self.slots = int(slots)
        self.clip = float(clip)
        self.register_buffer("s_mu", torch.as_tensor(s_mu, dtype=torch.float32))
        self.register_buffer("s_sd", torch.as_tensor(s_sd, dtype=torch.float32))
        self.register_buffer("p_mu", torch.as_tensor(p_mu, dtype=torch.float32))
        self.register_buffer("p_sd", torch.as_tensor(p_sd, dtype=torch.float32))

    @property
    def dim(self) -> int:
        return 2 * N_STATE + 2 * self.slots * N_PROJF

    def forward(self, s: torch.Tensor, p: torch.Tensor,
                side: torch.Tensor) -> torch.Tensor:
        """s [B,H,2,C], p [B,H,2,K,F], side [B] -> [B,H,dim]."""
        B, H = s.shape[0], s.shape[1]
        me = side.view(B, 1, 1, 1)
        s_me = s.gather(2, me.expand(B, H, 1, N_STATE))
        s_th = s.gather(2, (1 - me).expand(B, H, 1, N_STATE))
        # `proj[:, q]` holds what player q OWNS, already expressed relative to
        # whoever it is flying at -- so the objects threatening ME are the
        # OPPONENT'S, and they go first because that is the urgent half.
        pm = side.view(B, 1, 1, 1, 1)
        p_in = p.gather(2, (1 - pm).expand(B, H, 1, self.slots, N_PROJF))
        p_out = p.gather(2, pm.expand(B, H, 1, self.slots, N_PROJF))

        st = ((torch.cat([s_me, s_th], dim=2) - self.s_mu) / self.s_sd)
        pr = ((torch.cat([p_in, p_out], dim=2) - self.p_mu) / self.p_sd)
        return torch.cat([st.clamp(-self.clip, self.clip).reshape(B, H, -1),
                          pr.clamp(-self.clip, self.clip).reshape(B, H, -1)],
                         dim=-1)


def corpus_stats(S: np.ndarray, P: np.ndarray, eps: float = 1e-3):
    """Per-channel mean/std over the corpus, pooled across both players.

    Pooled deliberately: the observation is ego-ordered, so a given column is
    fed by P1 on half the samples and P2 on the other half, and two separate
    normalisations would make the same physical quantity arrive on two
    different scales depending on which chair the agent drew.

    NumPy warns `overflow encountered in square` here. It was checked rather
    than silenced: every returned statistic is finite, and each is within 0.5%
    of the same figure computed in float64 -- the warning comes from an
    intermediate in float32's two-pass variance over `untech`, which reaches
    500 in channel units against every other channel's O(1). Left in float32
    because changing the normalisation would move the input space of every
    policy already trained against it, which is a larger error than 0.5%.

    `ax` is exactly constant (float64 std 0.0), so `eps` becomes its divisor
    and the channel contributes exactly zero -- `x - mean` is identically 0, so
    nothing is amplified. It is one dead input of 178, the same channel
    `train_state_dynamics.delta_scale` drops from the loss.
    """
    s = S.reshape(-1, S.shape[-1])
    p = P.reshape(-1, P.shape[-1])
    return (s.mean(0).astype(np.float32),
            np.maximum(s.std(0), eps).astype(np.float32),
            p.mean(0).astype(np.float32),
            np.maximum(p.std(0), eps).astype(np.float32))


class StatePolicyOpponent:
    """A (possibly frozen) SokuPolicy playing the other chair."""

    def __init__(self, policy: SokuPolicy):
        self.policy = policy

    @torch.no_grad()
    def act(self, obs: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        # The opponent's observation is ego-ordered for the AGENT, so it is
        # reading the board from the wrong chair. That is a genuine handicap
        # and the reason `StateArena.rollout` builds a second observation for
        # it rather than reusing the agent's -- see `obs_opp` there.
        return self.policy(obs, 1 - side, sample=True).actions


class StateArena:
    """Rolls a policy against an opponent inside the frozen state simulator."""

    def __init__(self, sim, obs: StateObs, cfg: StateGRPOConfig,
                 history: int, ticks: int, proj_feedback: str | None = None,
                 button_mask: tuple[int, ...] | None = None):
        # Buttons forced to 0 for BOTH chairs after sampling. Not a training
        # device -- it exists to ask how much of a measured advantage survives
        # when a particular input is unavailable to everyone.
        #
        # The question it was built for: the first run's policy pressed `spell`
        # 12.3x and `change` 7.4x more often than any human in the corpus, and
        # those are the two rarest buttons in it -- so they are where the
        # simulator has seen least and is freest to be wrong. Masking them
        # symmetrically separates "learned to fight" from "learned to press the
        # button the model does not understand".
        self.button_mask = tuple(button_mask or ())
        self.sim = sim.eval()
        for p in self.sim.parameters():
            p.requires_grad_(False)
        self.obs = obs
        self.cfg = cfg
        self.H = history
        self.ticks = ticks
        # Defaults to the MODEL's own setting, so the arena cannot roll a
        # checkpoint through a path it was not trained through by omission.
        # Passing it explicitly is for the one job that legitimately wants the
        # other convention: measuring the two against each other.
        self.proj_feedback = proj_feedback or sim.proj_feedback

    def _advance(self, s_win, p_win, a_full):
        """One simulator step -> the next (state, proj), ready to feed back.

        Deliberately the same arithmetic as `state_dynamics._unroll_impl`, via
        the same helper: what RL rolls forward in has to be the thing that was
        trained and the thing that was measured, or the skill numbers on record
        describe a different function.
        """
        ns, np_ = self.sim(s_win, p_win, a_full)
        binr = torch.tensor(BINARY, device=ns.device)
        nxt_s = ns[:, -1:].index_copy(
            -1, binr, torch.sigmoid(ns[:, -1:].index_select(-1, binr)))
        return nxt_s, feed_proj(np_[:, -1:], self.proj_feedback)

    @torch.no_grad()
    def rollout(self, s_ctx: torch.Tensor, p_ctx: torch.Tensor,
                a_hist: torch.Tensor, side: torch.Tensor,
                policy: SokuPolicy, opponent, two_sided: bool = False) -> dict:
        """s_ctx [B,H,2,C], p_ctx [B,H,2,K,F], a_hist [B,H-1,ticks,20], side [B].

        Returns the policy inputs and sampled actions at every step so the
        update can re-score them, plus the state sequence the reward reads.
        Nothing carries gradient; the update recomputes log-probs.

        ACTION t DRIVES STATE t -> t+1
        ------------------------------
        The simulator at position t consumes `(s_t, p_t, a_t)` and emits
        `s_{t+1}`, which is exactly how `train_state_dynamics.step_loss` pairs
        them (`model(s[:, :-1], p[:, :-1], a[:, :-1])` against `s[:, 1:]`).
        `a_hist` is therefore the H-1 REAL past actions, and the H'th slot is
        the one the policy is choosing now. Getting this off by one trains the
        policy on a simulator being asked a different question than the one it
        was fit to.
        """
        cfg, T = self.cfg, self.cfg.horizon
        if hasattr(opponent, "reset"):
            opponent.reset()                # replay opponents are stateful
        s_win, p_win, a_win = s_ctx, p_ctx, a_hist
        states, obs_all, mine_all, theirs_all, joint_all = [], [], [], [], []

        # T+1 predicted states against the T actions that produced the last T
        # of them: the state sequence stays homogeneous (all simulator output,
        # never a mix of a real start state and predicted successors) and
        # joint_all[k] carries states[k] to states[k+1].
        for _ in range(T + 1):
            obs = self.obs(s_win, p_win, side)
            obs_all.append(obs)
            mine = jitter_actions(policy(obs, side, sample=True).actions,
                                  cfg.jitter_sigma)
            if isinstance(opponent, StatePolicyOpponent):
                # The opponent sits in the other chair, so it gets the board
                # ego-ordered for ITSELF. Handing it the agent's view would
                # make self-play asymmetric in a way that has nothing to do
                # with skill: one copy would read "my health" out of the column
                # holding the other's.
                theirs = opponent.policy(self.obs(s_win, p_win, 1 - side),
                                         1 - side, sample=True).actions
            else:
                theirs = opponent.act(obs, side)
            theirs = jitter_actions(theirs, cfg.jitter_sigma)
            if self.button_mask:
                # After jitter, so a shifted chunk cannot reintroduce a masked
                # press from a neighbouring tick.
                bm = torch.tensor(self.button_mask, device=mine.device)
                mine = mine.index_fill(-1, bm, 0.0)
                theirs = theirs.index_fill(-1, bm, 0.0)
            joint = to_joint(mine, theirs, side)

            a_full = torch.cat([a_win, joint[:, None]], dim=1)
            nxt_s, nxt_p = self._advance(s_win, p_win, a_full)

            mine_all.append(mine)
            theirs_all.append(theirs)
            joint_all.append(joint)
            states.append(nxt_s[:, 0])
            s_win = torch.cat([s_win[:, 1:], nxt_s], dim=1)
            p_win = torch.cat([p_win[:, 1:], nxt_p], dim=1)
            if self.H > 1:
                a_win = torch.cat([a_win[:, 1:], joint[:, None]], dim=1)

        seq = torch.stack(states, dim=1)                     # [B, T+1, 2, C]
        joint_seq = torch.stack(joint_all[1:], dim=1)        # [B, T, ticks, 20]
        reward, alive, terms = compute_rewards(seq, joint_seq, side, cfg.reward)
        out = {"obs": torch.stack(obs_all[1:], dim=1),       # [B, T, H, dim]
               "mine": torch.stack(mine_all[1:], dim=1),     # [B, T, ticks, 10]
               # Both players' buttons, always: a counterfactual has to replay
               # the opponent exactly, or the difference contains its reaction
               # as well as the agent's choice.
               "joint": joint_seq,
               "reward": reward, "alive": alive, "terms": terms,
               "states": seq, "side": side,
               "terminal": terminal_mask(seq, side, cfg.reward)}
        if two_sided:
            # One imagined rollout already IS a two-player game -- the
            # simulator is conditioned on both players' twenty buttons -- so
            # the opponent's experience costs no extra simulator steps. Only
            # on-policy when the opponent IS the current policy; the caller
            # owns that condition.
            opp_side = 1 - side
            r_o, alive_o, terms_o = compute_rewards(seq, joint_seq, opp_side,
                                                    cfg.reward)
            out.update({"mine_opp": torch.stack(theirs_all[1:], dim=1),
                        "reward_opp": r_o, "alive_opp": alive_o,
                        "terms_opp": terms_o, "side_opp": opp_side})
        return out
