"""Predict the game's own state from the latent, to force the latent to keep it.

WHY THIS HEAD EXISTS
--------------------
The encoder does not represent where the characters are. `spatial_probe.py`
puts a linear probe at AUC 0.540 for "did the characters swap sides" against a
0.956 ceiling for "did the HUD swap sides", and `block_effect.py` finds the
predictor forecasting *less* damage when the defender holds no direction at
all -- the sign of guarding, inverted. Blocking in Hisoutensoku is holding away
from the opponent, so a model that cannot see which side they are on cannot
represent the mechanic the matchup runs through.

Two objectives failed to recover it from pixels alone at this scale: plain
JEPA prediction, and JEPA plus inverse dynamics. Both improved the
representation and neither flipped the sign:

    objective          spatial AUC   inv_dyn_auc   guarding predicts less damage?
    JEPA 225k              0.540        0.7347                 no
    + IDM 1.0              0.651        0.7733                 no
    + balanced 9.0         0.688        0.7898                 no

Two independent levers moving the proxy and leaving the target untouched is
the argument for stopping: "away" is a fact about relative position, and
nothing in pixels-plus-inputs ever states it. So this stops asking the model
to infer what the game can simply be asked.

THE CONSTRAINT IS UNCHANGED
---------------------------
It was always about *inference*: the policy at play time consumes pixels and
its own inputs, nothing else. These labels never reach it. They shape a world
model, which is the same asymmetric arrangement `hud_coef` already uses -- the
only difference being that the HUD is legible in pixels and position is not.

TEACHING AND MEASURING ARE DIFFERENT JOBS
-----------------------------------------
The head was linear at first so that it could only succeed if the latent held
position linearly -- `probe.py`'s argument for the reward probe. On real data
that produced a head which could not fit its target and therefore could not
teach it either; see the class docstring for the measurement. The head now has
a hidden layer, and the linear-decodability question is asked afterwards by a
separate probe on the frozen latent, which is where that argument belongs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.state import STATE_CHANNELS

N_PLAYERS = 2

# The channels are not homogeneous and must not share a loss. `dx` is a signed
# continuous separation and `facing` is +-1; the rest are indicator flags whose
# base rates run from 5.5% (guarding) down to 0.04% (crushed), where a plain
# mean-squared error is minimised by predicting "never".
CONTINUOUS = tuple(i for i, c in enumerate(STATE_CHANNELS)
                   if c in ("dx", "facing"))
BINARY = tuple(i for i in range(len(STATE_CHANNELS)) if i not in CONTINUOUS)

# Measured over the first 941 captures of the corpus re-run: guarding 5.5%,
# knockdown 2.5%, wrongblock 1.1%, crushed 0.04%, airborne 45%. A single
# pos_weight across such different rates would over-correct `airborne` while
# barely touching `crushed`, so it is per channel and derived from the rate.
DEFAULT_POS_RATE = {"guarding": 0.055, "wrongblock": 0.011, "crushed": 0.0004,
                    "knockdown": 0.025, "airborne": 0.45}


def default_pos_weight() -> torch.Tensor:
    """(1 - p) / p per binary channel, clamped so `crushed` cannot dominate.

    The uncorrected weight for crushed is 2500, which would make a channel
    worth 0.04% of frames the largest term in the objective. 50 keeps it
    present without letting it steer.
    """
    w = [min((1 - DEFAULT_POS_RATE[STATE_CHANNELS[i]])
             / DEFAULT_POS_RATE[STATE_CHANNELS[i]], 50.0) for i in BINARY]
    return torch.tensor(w, dtype=torch.float32)


class StateHead(nn.Module):
    """[..., latent] -> [..., 2, len(STATE_CHANNELS)], per player.

    Continuous channels come out raw; binary channels come out as logits, so
    the loss can apply BCE-with-logits and stay numerically sane.

    ONE HIDDEN LAYER, AND WHY THAT REVERSES AN EARLIER ARGUMENT
    ----------------------------------------------------------
    This head was linear at first, on the reasoning `probe.py` uses for the
    reward probe: a head with enough capacity to recover position from a
    representation that does not hold it lets the encoder off the hook.

    That argument is right about *measurement* and wrong about *teaching*, and
    16 000 steps of supervision on the real corpus showed the difference. The
    linear head reached dx R2 -0.09 with predictions of standard deviation
    0.15 against the truth's 0.30 -- it was regressing toward the mean because
    it could not express the mapping. A head that cannot fit its target emits a
    weak and uninformative gradient, so the encoder was pushed hard by
    everything else (its weights moved 24.6%) and barely at all by this.

    So the head gets capacity to learn from, and the *measurement* moves to a
    separate linear probe fit after the fact, which is the honest split. Set
    `state_width = 0` for the original linear head.
    """

    def __init__(self, cfg: Config, width: int | None = None):
        super().__init__()
        self.n_channels = len(STATE_CHANNELS)
        out_dim = N_PLAYERS * self.n_channels
        w = getattr(cfg, "state_width", 512) if width is None else width
        if w and w > 0:
            self.net = nn.Sequential(
                nn.Linear(cfg.latent_dim, w), nn.GELU(), nn.Linear(w, out_dim))
        else:
            self.net = nn.Linear(cfg.latent_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(*z.shape[:-1], N_PLAYERS, self.n_channels)


def state_loss(pred: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict]:
    """pred/target [..., 2, C]; `valid` [...] marks frames that are real labels.

    `valid` is not optional in spirit. Frames the re-capture did not cover are
    filled by repeating the nearest real row (`pipeline/align_sidecar.py`), and
    training on them as if measured is exactly the quiet error the flag exists
    to prevent. Passing None means "every frame is real", which is true only of
    a capture that was never aligned.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target "
                         f"{tuple(target.shape)}")

    if valid is None:
        w = torch.ones(pred.shape[:-2], device=pred.device, dtype=pred.dtype)
    else:
        w = valid.to(pred.dtype)
    denom = w.sum().clamp(min=1.0)
    # [..., 1, 1] so it broadcasts over player and channel.
    wb = w[..., None, None]

    cont_idx = torch.tensor(CONTINUOUS, device=pred.device)
    bin_idx = torch.tensor(BINARY, device=pred.device)

    cont_err = (pred.index_select(-1, cont_idx)
                - target.index_select(-1, cont_idx)) ** 2
    l_cont = (cont_err * wb).sum() / (denom * N_PLAYERS * len(CONTINUOUS))

    bin_logits = pred.index_select(-1, bin_idx)
    bin_target = target.index_select(-1, bin_idx)
    pw = pos_weight.to(pred.device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(
        bin_logits, bin_target, reduction="none", pos_weight=pw)
    l_bin = (bce * wb).sum() / (denom * N_PLAYERS * len(BINARY))

    loss = l_cont + l_bin
    with torch.no_grad():
        metrics = {"state_loss": float(loss.detach()),
                   "state_cont_mse": float(l_cont.detach()),
                   "state_bin_bce": float(l_bin.detach()),
                   "state_frames": float(denom)}
        # dx is the channel the whole exercise is about, so it is reported on
        # its own and as a variance-explained figure rather than an error --
        # an MSE tells you nothing without knowing the spread it is against.
        dx = STATE_CHANNELS.index("dx")
        t = target[..., dx][w.bool()] if valid is not None else target[..., dx]
        p = pred[..., dx][w.bool()] if valid is not None else pred[..., dx]
        if t.numel() > 1:
            var = t.var()
            metrics["state_dx_r2"] = float(
                1.0 - ((p - t) ** 2).mean() / var.clamp(min=1e-8))
        for k, i in enumerate(BINARY):
            name = STATE_CHANNELS[i]
            tt = target[..., i][w.bool()] if valid is not None else target[..., i]
            pp = pred[..., i][w.bool()] if valid is not None else pred[..., i]
            if tt.numel() == 0:
                continue
            hit = ((pp > 0).float() == tt).float().mean()
            metrics[f"state_{name}_acc"] = float(hit)
    return loss, metrics


# Display-orientation rows of the play area; health/names sit at 34-74 and
# spirit/cards at 428-470 (data/hud.py). Frames are stored vertically flipped,
# so these convert to stored rows below. Kept identical to
# scripts/spatial_probe.py, which measures exactly this transform -- if the two
# ever disagree the training signal and the metric stop being the same question.
DISP_PLAY_Y = (80, 420)
FRAME = 480


def mirror_play(obs: torch.Tensor) -> torch.Tensor:
    """Mirror ONLY the play area of [..., C, H, W] frames, HUD untouched.

    This is the whole trick. The encoder currently reads which way round the
    fight is off the HUD: mirroring the entire frame is detectable at AUC 0.958
    while mirroring only the play area sits at 0.620, barely above chance. The
    HUD is in every frame and trivially legible, so it is the shortcut any
    objective will take -- including 2003 replays of direct dx supervision,
    which did not dislodge it.

    Mirroring the characters while leaving the HUD *identical* removes the
    shortcut by construction: the two views differ only in where the fighters
    are, so a latent that ignores them cannot tell the pair apart, and a label
    that flips sign between them is unlearnable without looking.
    """
    out = obs.clone()
    lo, hi = FRAME - DISP_PLAY_Y[1], FRAME - DISP_PLAY_Y[0]
    h = obs.shape[-2]
    if h != FRAME:                       # 224 px training frames, say
        lo, hi = int(lo * h / FRAME), int(hi * h / FRAME)
    out[..., lo:hi, :] = torch.flip(out[..., lo:hi, :], dims=(-1,))
    return out


def mirror_targets(state: torch.Tensor) -> torch.Tensor:
    """The same labels as seen from the mirrored world.

    `dx` is a signed separation and `facing` a signed direction, so both negate.
    Guard, knockdown and airborne are properties of a player, not of a side, so
    they are untouched -- and swapping them would teach the opposite of the
    truth.
    """
    out = state.clone()
    out[..., CONTINUOUS[0]] = -out[..., CONTINUOUS[0]]     # dx
    out[..., CONTINUOUS[1]] = -out[..., CONTINUOUS[1]]     # facing
    return out
