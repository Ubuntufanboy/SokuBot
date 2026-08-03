"""AdaJEPA (arXiv:2606.32026): test-time adaptation inside the MPC loop.

A pretrained world model is frozen at deployment, so when its predictions are
wrong the planner optimises the wrong objective. AdaJEPA closes that loop: after
executing an action it uses the transition it just observed as a self-supervised
target, takes a small number of gradient steps, and replans with the updated
model (Algorithm 1).

Defaults follow the paper's Sec. 4.1:

  * adapt **only the final layers** of the visual encoder and the predictor
  * one gradient step per replanning step (U = 1)
  * eta_pred = 5e-4, eta_enc = 1e-5 -- the same learning rates as training
  * a recent-5 replay buffer
  * stop-gradient on the adaptation target

The stop-gradient is worth flagging: main training deliberately has none (SIGReg
is the anti-collapse mechanism there). Online, with five samples and a handful of
parameters, there is no batch to compute SIGReg over, so the paper falls back to
sg() as the stabiliser -- and notes that restricting the update to the last
layers largely prevents collapse on its own.

Each episode gets its **own copy** of the pretrained model, so adaptation never
leaks between episodes.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config
from ..data.window import frames_to_chw
from ..model.world_model import LeWorldModel
from .cem import CEMPlanner


class TestTimeAdapter:
    """Owns a private copy of the model and adapts it from observed transitions."""

    def __init__(self, model: LeWorldModel, cfg: Config, clone: bool = True):
        self.cfg = cfg
        self.model = copy.deepcopy(model) if clone else model
        # eval(): the encoder's BatchNorm must use its running statistics. In
        # train mode a five-sample buffer would both normalise by meaningless
        # batch statistics and overwrite the running ones.
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        enc_params = list(self.model.encoder.projector.parameters())
        pred_params = (
            list(self.model.predictor.blocks[-1].parameters())
            + list(self.model.predictor.projector.parameters())
        )
        for p in enc_params + pred_params:
            p.requires_grad_(True)
        self.adapted_params = enc_params + pred_params

        self.opt = torch.optim.AdamW([
            {"params": pred_params, "lr": cfg.tta_lr_pred},
            {"params": enc_params, "lr": cfg.tta_lr_enc},
        ])

        # Rolling trajectory of what actually happened this episode. The
        # invariant these two must preserve is that ``actions[i]`` is the action
        # applied at ``frames[i]``.
        #
        # That is why the two maxlens differ by one. A frame arrives before its
        # action does, so `frames` is always one ahead; with equal maxlens it
        # would reach its bound one step earlier, drop its oldest entry while
        # `actions` kept all of its own, and every pair would be off by one
        # from then on. Sized one apart, the deques saturate on the same call
        # and roll together. This only bites after `tta_buffer + seq_len` steps,
        # so a short episode looks perfectly fine.
        M = cfg.tta_buffer + cfg.seq_len
        self.frames: deque = deque(maxlen=M)
        self.actions: deque = deque(maxlen=M - 1)

    def observe(self, frame: torch.Tensor, action: Optional[torch.Tensor]) -> None:
        """frame: [3, S, S]; action: [ticks, action_dim] applied at that frame."""
        self.frames.append(frame)
        if action is not None:
            self.actions.append(action)

    def _windows(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """The most recent complete prediction windows, up to `tta_buffer` of them."""
        T = self.cfg.seq_len
        # A window at start s reads frames[s .. s+T-1] *and* actions[s .. s+T-1]
        # (the predictor needs a condition for every token it is given), so the
        # usable prefix is bounded by the shorter deque, not by frames alone.
        n = min(len(self.frames), len(self.actions))
        if n < T:
            return None
        starts = list(range(n - T + 1))[-self.cfg.tta_buffer :]
        obs = torch.stack([torch.stack([self.frames[s + k] for k in range(T)]) for s in starts])
        act = torch.stack([torch.stack([self.actions[s + k] for k in range(T)]) for s in starts])
        return obs, act

    def adapt(self, steps: Optional[int] = None) -> float:
        """Take U gradient steps on the buffered transitions. Returns the last loss."""
        win = self._windows()
        if win is None:
            return float("nan")
        obs, act = win
        device = next(self.model.parameters()).device
        obs, act = obs.to(device), act.to(device)

        last = float("nan")
        for _ in range(steps or self.cfg.tta_steps):
            z = self.model.encode(obs)
            cond = self.model.action_encoder(act)
            zhat = self.model.predictor(z, cond)
            target = z[:, 1:].detach() if self.cfg.tta_stop_grad else z[:, 1:]
            loss = F.mse_loss(zhat[:, :-1], target)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            last = float(loss.item())
        return last


@dataclass
class EpisodeResult:
    steps: int = 0
    success: bool = False
    goal_distance: List[float] = field(default_factory=list)
    adapt_loss: List[float] = field(default_factory=list)
    info: Dict = field(default_factory=dict)


def plan_and_adapt(
    runner,
    model: LeWorldModel,
    cfg: Config,
    goal_frame: np.ndarray,
    max_steps: int = 20,
    adapt: bool = True,
    success_fn: Optional[Callable[[dict], bool]] = None,
    planner_cls=CEMPlanner,
    verbose: bool = False,
) -> EpisodeResult:
    """AdaJEPA Algorithm 1 over a `runner` exposing reset() / step(chunk) -> (frame, done, info).

    With ``adapt=False`` this is the frozen-model MPC baseline, which is exactly
    the comparison the paper reports.
    """
    device = next(model.parameters()).device
    adapter = TestTimeAdapter(model, cfg)
    wm = adapter.model
    planner = planner_cls(wm, cfg)

    def enc(frame_hwc: np.ndarray) -> torch.Tensor:
        t = frames_to_chw(frame_hwc[None])[0].to(device)
        return t

    with torch.no_grad():
        z_goal = wm.encode(enc(goal_frame).unsqueeze(0)).squeeze(0)

    frame = runner.reset()
    # Prime the planning context. Nothing has been executed yet, so the history
    # is the opening frame repeated -- a still state, which holds for both PushT
    # (the agent is stationary at reset) and a Soku round start.
    #
    # The adapter is given that frame exactly once: its deques must stay
    # index-aligned, and a repeated frame has no action to pair with.
    first = enc(frame)
    hist_frames = [first for _ in range(cfg.history)]
    adapter.observe(first, None)

    result = EpisodeResult()
    for step in range(max_steps):
        with torch.no_grad():
            z_ctx = wm.encode(torch.stack(hist_frames).unsqueeze(0))[0]    # [H, latent]
            result.goal_distance.append(
                float(((z_ctx[-1] - z_goal) ** 2).mean().item())
            )

        chunk = planner.plan(z_ctx, z_goal)                    # [P, ticks, A]
        first = chunk[0]                                       # execute one chunk
        frame, done, info = runner.step(first.detach().cpu().numpy())

        f = enc(frame)
        adapter.observe(f, first.detach().cpu())
        hist_frames = hist_frames[1:] + [f]
        result.steps = step + 1

        if adapt:
            result.adapt_loss.append(adapter.adapt())
        if verbose:
            print(f"  step {step:3d} | d(goal) {result.goal_distance[-1]:.4f}"
                  + (f" | ada {result.adapt_loss[-1]:.4f}" if adapt else ""))

        if success_fn is not None and success_fn(info):
            result.success = True
            result.info = dict(info)
            break
        if done:
            result.info = dict(info)
            break

    with torch.no_grad():
        z_last = wm.encode(hist_frames[-1].unsqueeze(0))[0]
        result.goal_distance.append(float(((z_last - z_goal) ** 2).mean().item()))
    return result
