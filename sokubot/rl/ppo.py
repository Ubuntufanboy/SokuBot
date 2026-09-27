"""PPO self-play inside the state simulator: the pieces, kept testable without a GPU or data.

    cfg = PPOConfig(...)
    league = League(cfg, template_policy)
    traj = arena.rollout(ctx..., policy, opponent, two_sided=True)
    batch = assemble(traj, critic, cfg, both_chairs=True)
    stats = ppo_update(policy, critic, opt, critic_opt, batch, cfg, reference, ...)

WHAT IS DIFFERENT FROM `train_state_grpo`, AND WHY
--------------------------------------------------
GRPO rolls every start G times and scores each rollout against its own group. That spends G-1 of
every G simulator rollouts on a baseline, and it can only credit what lands inside the 8-step
imagined horizon. Here a learned critic is the baseline (GAE over the horizon, bootstrapped from
v(s_T)), so every rollout is a distinct start state and credit can outlive the horizon. The clipped
surrogate, the trust-region and anchor KLs, the entropy floor and the button-rate floor are REUSED
from `rl/grpo.py` unchanged (`grpo_loss` is already PPO's objective), so the two trainers differ in
exactly the advantage estimator, the minibatching and the opponent schedule.

SELF-PLAY, AND WHAT IT HARVESTS
-------------------------------
Each update draws ONE opponent for the whole batch:

    self        the current policy in the other chair. BOTH chairs are on-policy then, so both are
                trained on (`two_sided`): twice the samples for no extra simulator steps.
    league      a frozen past self (`League`), chosen by prioritised fictitious self-play: the ones
                the current policy does worst against are drawn more often. Only the agent's chair
                is trained -- the opponent's actions came from a different policy.
    reference   the frozen prior-initialised policy: the fixed point every `net` is measured from.
    replay      the recorded human buttons from the start's own replay, open-loop. The only
                opponent that is not a copy of ourselves.

WHAT THIS CANNOT FIX
--------------------
Imagination is trusted over ~8 decision steps (667 ms) and no further (`state_arena.py`). A policy
can learn to beat the simulator rather than the game; the anchor KL and the button-rate floor exist
for that and are on by default. And self-play's training return is zero-sum by construction, so it
is never a progress signal: progress is `net` against the FIXED reference on FIXED starts, both
chairs (`evaluate_vs`), exactly as `train_state_grpo` defines it, so the two are comparable.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .grpo import grpo_loss
from .policy import SokuPolicy, representable_prior
from .state_arena import StateGRPOConfig, StatePolicyOpponent
from .state_critic import StateCritic, lambda_returns
from .state_reward import StateRewardConfig, compute_rewards

OPPONENTS = ("self", "league", "reference", "replay")

# The yardstick. Per-step damage exchange and nothing else, so `net` means the same thing in every
# run whatever that run was paid for -- identical to train_state_grpo's EVAL_RCFG.
EVAL_RCFG = StateRewardConfig(damage_mode="step", combo=0.0, idle=0.0, crush=0.0, win=0.0,
                              lose=0.0)


@dataclass
class PPOConfig(StateGRPOConfig):
    """PPO on top of the state arena's GRPO config (the arena and `grpo_loss` read that one)."""

    starts_per_batch: int = 512         # distinct start states per update; no groups
    group_size: int = 1
    lam: float = 0.95
    epochs: int = 4
    minibatches: int = 4
    critic_lr: float = 3e-4
    critic_grad_clip: float = 10.0
    two_sided: bool = True              # train on the opponent's chair when it is the policy
    # Opponent mixture, per update. Normalised at use.
    p_self: float = 0.50
    p_league: float = 0.35
    p_reference: float = 0.05
    p_replay: float = 0.10
    snapshot_every: int = 100
    max_snapshots: int = 16
    pfsp_temperature: float = 1.0       # in units of the league's own spread of scores
    score_ema: float = 0.9
    reward: StateRewardConfig = field(default_factory=StateRewardConfig)

    def opponent_probs(self) -> np.ndarray:
        p = np.array([self.p_self, self.p_league, self.p_reference, self.p_replay], float)
        if (p < 0).any() or p.sum() <= 0:
            raise ValueError(f"opponent mixture must be non-negative and not all zero: {p}")
        return p / p.sum()


# --------------------------------------------------------------------------------------------
# the league of past selves
# --------------------------------------------------------------------------------------------
class League:
    """Frozen past policies as CPU state dicts, one reusable module to play them, and a score each.

    Kept as state dicts rather than module copies so the whole league goes into a checkpoint and
    comes back on resume -- a job requeued at its time limit must not restart self-play against an
    empty pool. `score` is an EMA of the current policy's net damage exchange against that entry
    (positive = we win the exchange), updated from training batches and evaluations alike.
    """

    def __init__(self, cfg: PPOConfig, template: SokuPolicy):
        self.cfg = cfg
        self.entries: list[dict] = []
        self.player = copy.deepcopy(template).eval()
        for p in self.player.parameters():
            p.requires_grad_(False)
        self._loaded: Optional[int] = None

    def __len__(self) -> int:
        return len(self.entries)

    def maybe_add(self, policy: SokuPolicy, step: int) -> bool:
        if not step or step % self.cfg.snapshot_every:
            return False
        self.entries.append({"step": int(step), "score": None, "games": 0,
                             "state": {k: v.detach().cpu().clone()
                                       for k, v in policy.state_dict().items()}})
        if len(self.entries) > self.cfg.max_snapshots:
            self.entries.pop(0)
            self._loaded = None
        return True

    def weights(self) -> np.ndarray:
        """PFSP: favour the opponents we do WORST against; an unplayed entry gets the top weight."""
        n = len(self.entries)
        scores = [e["score"] for e in self.entries]
        known = np.array([s for s in scores if s is not None], float)
        if len(known) < 2:
            return np.full(n, 1.0 / n)
        spread = max(float(known.std()), 1e-9) * self.cfg.pfsp_temperature
        worst = float(known.min())
        z = np.array([(worst if s is None else s) for s in scores], float)
        w = np.exp(-(z - worst) / spread)
        return w / w.sum()

    def sample(self, rng: np.random.Generator) -> Optional[int]:
        if not self.entries:
            return None
        return int(rng.choice(len(self.entries), p=self.weights()))

    def policy(self, i: int) -> SokuPolicy:
        """Entry `i` loaded into the league's ONE player module. Valid until the next call: holding
        two league opponents at once means both are whichever was loaded last."""
        if self._loaded != id(self.entries[i]):
            self.player.load_state_dict(self.entries[i]["state"])
            self._loaded = id(self.entries[i])
        return self.player

    def record(self, i: int, net: float) -> None:
        e = self.entries[i]
        e["score"] = net if e["score"] is None else (
            self.cfg.score_ema * e["score"] + (1 - self.cfg.score_ema) * net)
        e["games"] += 1

    def state_dict(self) -> dict:
        return {"entries": self.entries}

    def load_state_dict(self, d: dict) -> None:
        self.entries = list(d["entries"])
        self._loaded = None


# --------------------------------------------------------------------------------------------
# batch assembly: GAE per chair, flattened to the samples that are alive
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def advantages(critic: StateCritic, obs: torch.Tensor, obs_last: torch.Tensor,
               reward: torch.Tensor, alive: torch.Tensor, terminal: torch.Tensor,
               gamma: float, lam: float):
    """-> (returns [B,T], advantages [B,T], values [B,T]). GAE(lambda) with a TRUE bootstrap.

    `obs_last` is the observation of the final imagined state, so v(s_T) is valued where the
    rollout actually ended. A KO (`terminal`) stops the bootstrap; running out of horizon does not.
    """
    v = critic.value(obs)                                   # [B, T]
    v_boot = critic.value(obs_last)                         # [B]
    ret = lambda_returns(reward, torch.cat([v, v_boot[:, None]], 1), alive, terminal,
                         gamma, lam)
    return ret, (ret - v) * alive, v


def assemble(traj: dict, critic: StateCritic, cfg: PPOConfig, both_chairs: bool) -> dict:
    """Rollout -> flat PPO samples [N, ...], dead steps dropped, both chairs if on-policy for both."""
    chairs = [("obs", "obs_last", "mine", "reward", "alive", "terminal", "side")]
    if both_chairs:
        if "obs_opp" not in traj:
            raise ValueError("both_chairs needs a two-sided rollout against a policy opponent")
        chairs.append(("obs_opp", "obs_last_opp", "mine_opp", "reward_opp", "alive_opp",
                       "terminal_opp", "side_opp"))
    parts = []
    for o, ol, m, r, al, te, sd in chairs:
        ret, adv, v = advantages(critic, traj[o], traj[ol], traj[r], traj[al], traj[te],
                                 cfg.gamma, cfg.lam)
        B, T = adv.shape
        keep = traj[al].reshape(-1) > 0
        side = traj[sd][:, None].expand(B, T).reshape(-1)
        parts.append({"obs": traj[o].reshape(B * T, *traj[o].shape[2:])[keep],
                      "act": traj[m].reshape(B * T, *traj[m].shape[2:])[keep],
                      "side": side[keep], "ret": ret.reshape(-1)[keep],
                      "adv": adv.reshape(-1)[keep], "value": v.reshape(-1)[keep]})
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    a = out["adv"]
    # Normalised over the whole batch, both chairs together: they are samples of one policy.
    out["adv"] = (a - a.mean()) / a.std().clamp(min=1e-6) if len(a) > 1 else a * 0
    return out


# --------------------------------------------------------------------------------------------
# the update
# --------------------------------------------------------------------------------------------
def ppo_update(policy: SokuPolicy, critic: StateCritic, opt, critic_opt, batch: dict,
               cfg: PPOConfig, reference: SokuPolicy, rng: np.random.Generator,
               sched=None, ent_alpha: float = 0.0, button_floor=None) -> dict:
    """Clipped PPO over minibatches for `cfg.epochs`, stopping early past `cfg.target_kl`.

    `grpo_loss` does the actor objective, called on [n, 1]-shaped slices so its per-step alive mask
    is all ones. The critic is fitted to the same lambda-returns the advantages were built from.
    """
    N = len(batch["adv"])
    if N == 0:
        return {"samples": 0}
    with torch.no_grad():
        old_logp, _ = policy.log_prob_of(batch["obs"], batch["side"], batch["act"])
        ref_logp, _ = reference.log_prob_of(batch["obs"], batch["side"], batch["act"])
    mb = max(1, N // max(1, cfg.minibatches))
    agg: dict[str, list] = {}
    epochs_run = skipped = 0
    stopped_early = 0
    for ep in range(cfg.epochs):
        perm = torch.from_numpy(rng.permutation(N)).to(batch["adv"].device)
        ep_kl = []
        for start in range(0, N, mb):
            idx = perm[start:start + mb]
            n = len(idx)
            traj = {"obs": batch["obs"][idx][:, None], "side": batch["side"][idx],
                    "mine": batch["act"][idx][:, None],
                    "alive": torch.ones(n, 1, device=idx.device)}
            loss, st = grpo_loss(policy, traj, batch["adv"][idx][:, None],
                                 old_logp[idx][:, None], cfg, ref_logp[idx][:, None],
                                 ent_alpha=ent_alpha, button_floor=button_floor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                # One non-finite minibatch would write NaN into every weight. Skip it.
                skipped += 1
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
                if sched is not None:
                    sched.step()
            closs = critic.loss(batch["obs"][idx], batch["ret"][idx]).mean()
            critic_opt.zero_grad(set_to_none=True)
            closs.backward()
            cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_clip)
            if torch.isfinite(closs) and torch.isfinite(cgn):
                critic_opt.step()
            st["critic_loss"] = float(closs.detach())
            st["grad_norm"] = float(gn)
            for k, v in st.items():
                agg.setdefault(k, []).append(v)
            ep_kl.append(st["kl"])
        epochs_run += 1
        # Measured on the epoch just run, so the NEXT epoch is refused once the policy has moved
        # too far from the one that sampled the data.
        if float(np.mean(ep_kl)) > cfg.target_kl:
            stopped_early = 1
            break
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    with torch.no_grad():
        v_now = critic.value(batch["obs"])
        r = batch["ret"]
        out["v_r2"] = float(1 - ((r - v_now) ** 2).sum() / ((r - r.mean()) ** 2).sum().clamp(min=1e-9))
    out.update(samples=N, epochs_run=epochs_run, kl_early_stop=stopped_early,
               skipped=skipped)
    return out


# --------------------------------------------------------------------------------------------
# evaluation against a FIXED opponent
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_vs(arena, policy: SokuPolicy, opponent: SokuPolicy, ctx: tuple) -> dict:
    """Net damage exchange of `policy` vs `opponent` on fixed starts, both chairs, fixed reward.

    Averaging the chairs cancels whatever bias the simulator has toward one seat. `net` here is
    defined exactly as in train_state_grpo, so the two trainers' numbers are comparable.
    """
    s_ctx, p_ctx, a_hist = ctx
    out = {}
    for tag, s0 in (("p1", 0), ("p2", 1)):
        side = torch.full((len(s_ctx),), s0, device=s_ctx.device, dtype=torch.long)
        tr = arena.rollout(s_ctx, p_ctx, a_hist, side, policy, StatePolicyOpponent(opponent))
        _, al, terms = compute_rewards(tr["states"], tr["joint"], side, EVAL_RCFG)
        n = al.sum().clamp(min=1)
        out[f"{tag}_dealt"] = float((terms["dealt"] * al).sum() / n)
        out[f"{tag}_taken"] = float((terms["taken"] * al).sum() / n)
    out["net"] = ((out["p1_dealt"] + out["p1_taken"]) + (out["p2_dealt"] + out["p2_taken"])) / 2
    return out


# --------------------------------------------------------------------------------------------
# small helpers the driver and the tests share
# --------------------------------------------------------------------------------------------
def init_policy_from_corpus(policy: SokuPolicy, A: np.ndarray, button_names) -> None:
    """Start the policy at the corpus's own button statistics (same as train_state_grpo).

    A uniform policy holds ~44% of buttons per tick against a human's ~10%, so every rollout from a
    uniform start is a regime the simulator never saw.
    """
    p1 = A.reshape(-1, 20)[:, :10].astype(np.float32)
    lr = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                   float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                   float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    policy.set_action_prior(lr / lr.sum(), ud / ud.sum(),
                            representable_prior(p1[:, 4:10].mean(0), policy.logit_bound,
                                                button_names))


def file_fingerprint(path: Path) -> str:
    """sha256 of a file's bytes, first 16 hex digits. Two simulators saved under the same name are
    different models; the name proves nothing (memory: serve_policy has no fingerprint check)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def bank_fingerprint(*arrays: np.ndarray) -> str:
    """Shape + dtype + a strided sample of bytes, so a 1 GB bank is not read twice to identify it."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        flat = a.reshape(-1)
        h.update(np.ascontiguousarray(flat[::max(1, len(flat) // 4096)]).tobytes())
    return h.hexdigest()[:16]
