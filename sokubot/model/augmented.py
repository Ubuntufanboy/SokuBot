"""A predictor whose state carries the HUD explicitly, alongside the latent.

WHY
---
`scripts/horizon_ablation.py` measures spirit at R^2 0.05 out of the latent and
cards at 0.156/0.051, and `scripts/probe_reliability.py` measures the health
probe at 0.25 noise near zero, where the KO detector built on it then runs at
precision 0.003. The natural reading is "the reader is bad", and it is wrong:
48 human-annotated frames put `data/hud.py` at **MAE 0.012 for health and 0.025
for spirit**, at low health as much as anywhere else.

So the labels were never the problem. The problem is that the encoder is shown a
224 px downsample, and the spirit gauge is five six-pixel hexagons that become
about three pixels. The information is destroyed *before* the encoder, and no
probe, however good, recovers what was thrown away.

This module carries it instead of trying to recover it. The world-model state
becomes

    z_aug = [ z_vision (latent_dim, frozen ViT on 224 px) ; hud (n_hud, from 480 px) ]

so the reward reads health and spirit off the imagined state directly. `hud.py`
runs on the live 480 px capture at inference exactly as it does on the corpus, so
this stays pixels-only -- see `docs/HANDOFF.md` section 8.

WHAT IS DELIBERATE HERE
-----------------------
**The new input weights are zero-initialised.** `from_pretrained` copies the base
predictor's `in_proj` into the first `latent_dim` columns and zeroes the `n_hud`
new ones, so at step zero the augmented model computes *exactly* what the base
model computed. That makes the A/B honest: arm B starts as arm A and any
difference is something it learned, not a different initialisation.

**The HUD head bypasses the projector.** The projector ends in a non-affine
BatchNorm, which forces every output dimension to zero mean and unit variance --
correct for an abstract latent, wrong for a channel that means "fraction of a
health bar". HUD gets its own linear head with a sigmoid, so its outputs stay in
[0, 1] where they are interpretable and where the KO detector's thresholds mean
what they say.

**The HUD head predicts a residual, and this was learned the hard way.** The
first version emitted `sigmoid(Linear(h))` with no path from the input HUD to the
output, so the model had to *reconstruct* health from the transformer state every
step rather than carry it forward and predict a change. Health barely moves
between two 15 Hz frames, which makes copy-forward a very strong baseline, and
after 15000 steps that arm scored **1.53 at h=1 where 1.0 is copy-forward** --
half again worse than assuming nothing happens. It also dragged latent fidelity
down at every horizon.

So the head now outputs a delta in logit space on top of the incoming HUD, with
the delta zero-initialised. At step zero the model reproduces its input exactly,
which is copy-forward, which scores exactly 1.0 -- the same trick as zeroing the
new `in_proj` columns, applied to the other end. The arm then starts at the
trivial baseline and can only improve on it, instead of starting far below it and
spending its whole budget climbing back.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .predictor import MAX_SEQ, LatentPredictor

# The channels data/hud.py produces, in the order scripts/horizon_ablation.py
# lists them. rl/reward.py indexes the first six positionally, so this order is
# load-bearing and new channels are appended, never inserted.
HUD_CHANNELS = ("hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2",
                "cards1", "cards2")
N_HUD = len(HUD_CHANNELS)

# Keeps logit() finite at a gauge reading exactly 0 or 1, both of which really
# occur. Small enough that a full bar still round-trips to 0.999.
HUD_EPS = 1e-3


class AugmentedPredictor(nn.Module):
    """LatentPredictor over [latent ; hud], emitting both."""

    def __init__(self, cfg: Config, n_hud: int = N_HUD):
        super().__init__()
        self.cfg = cfg
        self.n_hud = n_hud
        self.latent_dim = cfg.latent_dim
        self.base = LatentPredictor(cfg)
        # Widened input projection. Everything else in `base` is reused as-is.
        self.in_proj = nn.Linear(cfg.latent_dim + n_hud, cfg.pred_dim)
        self.hud_head = nn.Linear(cfg.pred_dim, n_hud)
        nn.init.zeros_(self.hud_head.weight)
        nn.init.zeros_(self.hud_head.bias)

    @classmethod
    def from_pretrained(cls, base: LatentPredictor, cfg: Config,
                        n_hud: int = N_HUD,
                        hud_prior: torch.Tensor | None = None) -> "AugmentedPredictor":
        """Wrap a trained predictor so that it starts numerically unchanged.

        Both ends are zeroed: the new `in_proj` columns, so the latent path is
        bit-identical to the base predictor, and the `hud_head`, so the HUD path
        is exactly copy-forward. The arm therefore starts as "arm A, plus a HUD
        channel that predicts no change", which is the honest control.

        `hud_prior` is accepted and ignored for the bias, which is deliberate:
        biasing toward the corpus mean made sense when the head predicted HUD
        outright, but a residual head must start at *zero* or it no longer starts
        at copy-forward.
        """
        m = cls(cfg, n_hud)
        m.base.load_state_dict(base.state_dict())
        with torch.no_grad():
            m.in_proj.weight.zero_()
            m.in_proj.weight[:, : cfg.latent_dim].copy_(base.in_proj.weight)
            m.in_proj.bias.copy_(base.in_proj.bias)
        return m

    def forward(self, z: torch.Tensor, hud: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z [B,T,latent], hud [B,T,n_hud], cond [B,T,pred_dim].

        Returns (zhat, hud_hat), each aligned like `LatentPredictor.forward`:
        index ``t`` predicts step ``t+1``.
        """
        B, T, _ = z.shape
        if T > MAX_SEQ:
            raise ValueError(f"sequence length {T} exceeds MAX_SEQ={MAX_SEQ}")
        if hud.shape[:2] != (B, T) or hud.shape[-1] != self.n_hud:
            raise ValueError(
                f"hud has shape {tuple(hud.shape)}, expected {(B, T, self.n_hud)}")
        b = self.base
        h = self.in_proj(torch.cat([z, hud], dim=-1)) + b.pos_embed[:, :T]
        mod = b.adaln(cond)
        for blk in b.blocks:
            h = blk(h, mod, causal=True)
        # Residual in logit space on top of the incoming HUD. With `hud_head`
        # zeroed this returns `hud` exactly, i.e. copy-forward, which is the
        # baseline the loss normalises against -- so the arm starts at 1.0 and
        # improves, rather than starting at 1.53 and climbing back. The clamp
        # keeps logit() finite at a gauge that reads exactly 0 or 1, both of
        # which occur (a spent spirit gauge, a full health bar).
        p = hud.clamp(HUD_EPS, 1.0 - HUD_EPS)
        return b.projector(h), torch.sigmoid(
            torch.log(p / (1 - p)) + self.hud_head(h))


class AugmentedWorldModel(nn.Module):
    """The frozen encoder and action encoder, with the augmented predictor.

    Deliberately not a subclass of `LeWorldModel`: the encoder is frozen and the
    only thing that trains is the predictor, so inheriting a `forward` that
    trains the encoder end to end would invite exactly the mistake this avoids.
    """

    def __init__(self, wm, predictor: AugmentedPredictor):
        super().__init__()
        self.encoder = wm.encoder
        self.action_encoder = wm.action_encoder
        self.predictor = predictor
        self.cfg = wm.cfg

    def rollout(self, z_ctx: torch.Tensor, hud_ctx: torch.Tensor,
                a_plan: torch.Tensor, a_hist: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout over the augmented state.

        Mirrors `LeWorldModel.rollout` step for step -- the window slides by one
        each iteration and the newly predicted pair replaces the oldest -- so the
        two arms of the fine-tune differ only in what the state contains.
        """
        B, H, _ = z_ctx.shape
        P = a_plan.shape[1]
        if a_hist is None:
            a_hist = torch.zeros(B, max(0, H - 1), self.cfg.action_ticks,
                                 self.cfg.action_dim, device=z_ctx.device,
                                 dtype=a_plan.dtype)
        z_win, h_win, a_win = z_ctx, hud_ctx, a_hist
        zs, hs = [], []
        for k in range(P):
            a_full = torch.cat([a_win, a_plan[:, k : k + 1]], dim=1)
            cond = self.action_encoder(a_full)
            zhat, hhat = self.predictor(z_win, h_win, cond)
            zhat, hhat = zhat[:, -1], hhat[:, -1]
            zs.append(zhat)
            hs.append(hhat)
            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            h_win = torch.cat([h_win[:, 1:], hhat[:, None]], dim=1)
            if H > 1:
                a_win = torch.cat([a_win[:, 1:], a_plan[:, k : k + 1]], dim=1)
        return torch.stack(zs, dim=1), torch.stack(hs, dim=1)


class HudDeltaHead(nn.Module):
    """Predict the HUD's *change* from frozen trunk features plus the current HUD.

    WHY THIS SHAPE, AND NOT A FINE-TUNE
    ------------------------------------
    Phase 1 measured that fine-tuning this predictor for predictive accuracy
    **destroys** what the model is for: at h=4 the action->return correlation on
    held-out starts fell from +0.2622 (base) to +0.1938 (unrolled loss) and
    +0.1545 (one-step loss), and only the untouched base earns the verdict that
    the rule transfers to unseen starts at all. So nothing here touches a single
    predictor weight. The trunk is read, never written, and the latent rollout is
    byte-identical to the base model by construction rather than by measurement.

    WHY IT CAN WORK ANYWAY
    ----------------------
    The head is given two things the probe never had together: the **exact**
    current HUD, from `data/hud.py` at native resolution (MAE 0.012 for health,
    human-validated), and the trunk's **action-conditioned** state. Damage depends
    on whether a hit is landing, and the latent does represent the characters --
    matched-HUD frames whose characters differ sit at cosine 0.78 against 0.055
    for arbitrary pairs. So "a hit is landing now" is plausibly in `h` even though
    "health is 0.47" is not.

    That division matters. `scripts/anchored_ko_test.py` showed the probe's health
    error is 0.116 of which only ~0.025 is a fixed offset: the rest is drift
    through its per-step deltas, against a KO threshold of 0.06. Anchoring the
    level fixed the offset and changed nothing. This attacks the deltas instead.

    The output is a residual in logit space with the last layer zero-initialised,
    so an untrained head is exactly copy-forward -- the baseline the loss
    normalises against. It starts at 1.0 and can only improve.
    """

    def __init__(self, cfg: Config, n_hud: int = N_HUD, width: int = 256,
                 depth: int = 2):
        super().__init__()
        self.n_hud = n_hud
        layers: list[nn.Module] = [nn.Linear(cfg.pred_dim + n_hud, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, n_hud)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, hud: torch.Tensor) -> torch.Tensor:
        """h [B,T,pred_dim] frozen trunk features, hud [B,T,n_hud] -> [B,T,n_hud]."""
        if hud.shape[-1] != self.n_hud:
            raise ValueError(
                f"hud has {hud.shape[-1]} channels, expected {self.n_hud}")
        p = hud.clamp(HUD_EPS, 1.0 - HUD_EPS)
        delta = self.out(self.net(torch.cat([h, hud], dim=-1)))
        return torch.sigmoid(torch.log(p / (1 - p)) + delta)


class HudWorldModel(nn.Module):
    """Base world model, unmodified, with a HUD delta head bolted on.

    The predictor is frozen and used exactly as `LeWorldModel` uses it, so the
    latent half of every rollout is bit-identical to the base model's. Only
    `hud_head` has gradients.
    """

    def __init__(self, wm, head: HudDeltaHead):
        super().__init__()
        self.encoder = wm.encoder
        self.action_encoder = wm.action_encoder
        self.predictor = wm.predictor
        self.hud_head = head
        self.cfg = wm.cfg
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.action_encoder.parameters():
            p.requires_grad_(False)

    def rollout(self, z_ctx: torch.Tensor, hud_ctx: torch.Tensor,
                a_plan: torch.Tensor, a_hist: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same sliding window as `LeWorldModel.rollout`, carrying HUD alongside."""
        B, H, _ = z_ctx.shape
        P = a_plan.shape[1]
        if a_hist is None:
            a_hist = torch.zeros(B, max(0, H - 1), self.cfg.action_ticks,
                                 self.cfg.action_dim, device=z_ctx.device,
                                 dtype=a_plan.dtype)
        z_win, h_win, a_win = z_ctx, hud_ctx, a_hist
        zs, hs = [], []
        for k in range(P):
            a_full = torch.cat([a_win, a_plan[:, k : k + 1]], dim=1)
            cond = self.action_encoder(a_full)
            with torch.no_grad():
                zhat, feat = self.predictor(z_win, cond, return_hidden=True)
            hhat = self.hud_head(feat, h_win)[:, -1]
            zhat = zhat[:, -1]
            zs.append(zhat)
            hs.append(hhat)
            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            h_win = torch.cat([h_win[:, 1:], hhat[:, None]], dim=1)
            if H > 1:
                a_win = torch.cat([a_win[:, 1:], a_plan[:, k : k + 1]], dim=1)
        return torch.stack(zs, dim=1), torch.stack(hs, dim=1)
