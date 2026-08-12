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

Five objectives failed to recover it from pixels at this scale: JEPA (0.540),
plus inverse dynamics (0.651), plus class-balanced IDM (0.688), direct dx
supervision over 2003 replays (0.620), and a play-area-mirror augmentation
built specifically to forbid the HUD shortcut (0.6047). Independent levers
moving the proxy and leaving the target untouched is the argument for stopping:
"away" is a fact about relative position, and nothing in pixels-plus-inputs
ever states it. So this stops asking the model to infer what the game can
simply be asked.

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
teach it either: dx R2 -0.09, predictions of standard deviation 0.15 against
the truth's 0.30, regressing toward the mean because it could not express the
mapping. So the head gets capacity to learn from, and the linear-decodability
question moves to a separate probe on the frozen latent, which is where that
argument belongs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from ..data.state import (CH, PF, PROJ_FEATURES, STAGE_SPAN, STATE_CHANNELS)

N_PLAYERS = 2

# ---------------------------------------------------------------------------
# Which channels are indicator flags
#
# Stated BY NAME and explicitly. The previous version derived this by
# exclusion -- everything that is not `dx` or `facing` is binary -- which was
# true of a seven-channel state and became silently wrong the moment the
# sidecar grew to thirty-three: positions, velocities, health and spirit were
# all classified as indicator flags, and the only thing that stopped it
# training nonsense was a missing pos_weight key raising KeyError.
#
# Listed positively, a new channel defaults to regression, which is the safe
# direction to be wrong in. A new FLAG has to be named here and given a base
# rate, and that is a decision worth forcing someone to make.
# ---------------------------------------------------------------------------
BINARY_NAMES: tuple[str, ...] = ("guarding", "wrongblock", "crushed",
                                 "knockdown", "airborne")
_unknown = [n for n in BINARY_NAMES if n not in STATE_CHANNELS]
if _unknown:
    raise ImportError(f"BINARY_NAMES lists channels that do not exist: "
                      f"{_unknown}. STATE_CHANNELS is the contract.")

BINARY = tuple(STATE_CHANNELS.index(n) for n in BINARY_NAMES)
CONTINUOUS = tuple(i for i in range(len(STATE_CHANNELS)) if i not in BINARY)

# Measured over 336 926 frames of the re-captured corpus (30 replays, both
# players). A single pos_weight across rates this different would over-correct
# `airborne` while barely touching `crushed`, so it is per channel.
DEFAULT_POS_RATE = {"guarding": 0.0466, "wrongblock": 0.0115,
                    "crushed": 0.0011, "knockdown": 0.0351,
                    "airborne": 0.4299}

# Regression targets are clipped to this many units before the error is taken.
#
# Not cosmetic. `untech` is normalised by 60 frames but reaches 26 507 raw --
# 441 in channel units, against every other channel's O(1) -- so its squared
# error would be five orders of magnitude larger than health's and would be the
# objective. The field's semantics are unconfirmed and its large values are not
# meaningful (it holds stale readings; see data/state.py), so nothing real is
# lost by declining to predict past five seconds.
#
# Clipping happens here rather than in data/state.py on purpose: the sidecar
# reader is the faithful record of what the game said, and what a model chooses
# to spend capacity on is a modelling decision. If more channels misbehave the
# right answer is per-channel standardisation from corpus statistics, not more
# entries here.
TARGET_CLIP: dict[str, float] = {
    "untech": 5.0, "hitstop": 5.0, "action_frame": 5.0,
    "spirit_delay": 5.0, "timestop": 5.0,
}

# How many projectile slots the head predicts, of the 24 the sidecar carries.
# Occupancy past a handful is rare -- live-hitbox counts are p90 4-5 -- so
# predicting all 24 would spend most of the loss learning to emit zeros.
DEFAULT_PROJ_SLOTS = 8
# Binary features of a projectile slot; the rest are regressed.
PROJ_BINARY_NAMES: tuple[str, ...] = ("present", "hb")
PROJ_BINARY = tuple(PROJ_FEATURES.index(n) for n in PROJ_BINARY_NAMES)
PROJ_CONT = tuple(i for i in range(len(PROJ_FEATURES)) if i not in PROJ_BINARY)


def default_pos_weight() -> torch.Tensor:
    """(1 - p) / p per binary channel, clamped so `crushed` cannot dominate.

    The uncorrected weight for crushed is about 900, which would make a channel
    worth 0.1% of frames the largest term in the objective. 50 keeps it present
    without letting it steer.
    """
    w = [min((1 - DEFAULT_POS_RATE[n]) / DEFAULT_POS_RATE[n], 50.0)
         for n in BINARY_NAMES]
    return torch.tensor(w, dtype=torch.float32)


def _clip_vector(device=None) -> torch.Tensor:
    """Per-continuous-channel clip magnitude; `inf` where none applies."""
    v = [TARGET_CLIP.get(STATE_CHANNELS[i], float("inf")) for i in CONTINUOUS]
    return torch.tensor(v, dtype=torch.float32, device=device)


class StateHead(nn.Module):
    """[..., latent] -> [..., 2, len(STATE_CHANNELS)], per player.

    Continuous channels come out raw; binary channels come out as logits, so
    the loss can apply BCE-with-logits and stay numerically sane.
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


class ProjectileHead(nn.Module):
    """[..., latent] -> [..., 2, slots, len(PROJ_FEATURES)].

    A SET, PREDICTED AS AN ORDERED LIST, WHICH IS ONLY LEGITIMATE BECAUSE THE
    ORDER IS A FUNCTION OF THE SCENE
    -------------------------------------------------------------------------
    Predicting an unordered set with a fixed-slot head normally needs matching,
    because slot k has no meaning. Here it does: the extractor fills slots
    danger-first -- live hitbox, then nearest to the player being shot at -- so
    slot k is a well-defined question ("the k-th most threatening object") that
    the pixels determine. No Hungarian matching, and the target is stable
    across frames in the way an allocation-ordered list would not be.

    `proj[:, p]` is what player p OWNS, positioned against the player they are
    flying at, so the head learns "what is in the air and where is it going",
    which is the input a dodging or blocking decision is a response to.
    """

    def __init__(self, cfg: Config, slots: int = DEFAULT_PROJ_SLOTS,
                 width: int | None = None):
        super().__init__()
        self.slots = slots
        self.n_features = len(PROJ_FEATURES)
        out_dim = N_PLAYERS * slots * self.n_features
        w = getattr(cfg, "state_width", 512) if width is None else width
        if w and w > 0:
            self.net = nn.Sequential(
                nn.Linear(cfg.latent_dim, w), nn.GELU(), nn.Linear(w, out_dim))
        else:
            self.net = nn.Linear(cfg.latent_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).reshape(*z.shape[:-1], N_PLAYERS, self.slots,
                                   self.n_features)


def state_loss(pred: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor | None = None,
               pos_weight: torch.Tensor | None = None,
               channel_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict]:
    """pred/target [..., 2, C]; `valid` [...] marks frames that are real labels.

    `valid` is not optional in spirit. Frames the re-capture did not cover are
    filled by repeating the nearest real row (`pipeline/align_sidecar.py`), and
    training on them as if measured is exactly the quiet error the flag exists
    to prevent. Passing None means "every frame is real", which is true only of
    a capture that was never aligned.

    `channel_mask` [C] zeroes individual channels. It exists for the mirrored
    copy of a batch, where some channels do not survive the transform -- see
    `mirror_targets`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target "
                         f"{tuple(target.shape)}")

    if valid is None:
        w = torch.ones(pred.shape[:-2], device=pred.device, dtype=pred.dtype)
    else:
        w = valid.to(pred.dtype)
    denom = w.sum().clamp(min=1.0)
    wb = w[..., None, None]        # broadcasts over player and channel

    cont_idx = torch.tensor(CONTINUOUS, device=pred.device)
    bin_idx = torch.tensor(BINARY, device=pred.device)

    cm_cont = cm_bin = None
    if channel_mask is not None:
        cm = channel_mask.to(pred.device, pred.dtype)
        cm_cont = cm.index_select(0, cont_idx)
        cm_bin = cm.index_select(0, bin_idx)

    # Clip both sides identically: an error is only counted where the target is
    # a quantity worth predicting, and a prediction beyond the clip is not
    # punished for overshooting a value nobody asked for.
    clip = _clip_vector(pred.device)
    p_cont = pred.index_select(-1, cont_idx).clamp(-clip, clip)
    t_cont = target.index_select(-1, cont_idx).clamp(-clip, clip)
    cont_err = (p_cont - t_cont) ** 2
    if cm_cont is not None:
        cont_err = cont_err * cm_cont
        n_cont = cm_cont.sum().clamp(min=1.0)
    else:
        n_cont = torch.tensor(float(len(CONTINUOUS)), device=pred.device)
    l_cont = (cont_err * wb).sum() / (denom * N_PLAYERS * n_cont)

    bin_logits = pred.index_select(-1, bin_idx)
    bin_target = target.index_select(-1, bin_idx)
    pw = pos_weight.to(pred.device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(
        bin_logits, bin_target, reduction="none", pos_weight=pw)
    if cm_bin is not None:
        bce = bce * cm_bin
        n_bin = cm_bin.sum().clamp(min=1.0)
    else:
        n_bin = torch.tensor(float(len(BINARY)), device=pred.device)
    l_bin = (bce * wb).sum() / (denom * N_PLAYERS * n_bin)

    loss = l_cont + l_bin
    with torch.no_grad():
        metrics = {"state_loss": float(loss.detach()),
                   "state_cont_mse": float(l_cont.detach()),
                   "state_bin_bce": float(l_bin.detach()),
                   "state_frames": float(denom)}
        m = w.bool()
        # dx is the channel the whole exercise is about, so it is reported on
        # its own and as a variance-explained figure rather than an error -- an
        # MSE tells you nothing without knowing the spread it is against.
        for name in ("dx", "dy", "x", "hp"):
            i = CH[name]
            t = target[..., i][m] if valid is not None else target[..., i]
            p = pred[..., i][m] if valid is not None else pred[..., i]
            if t.numel() > 1:
                var = t.var()
                metrics[f"state_{name}_r2"] = float(
                    1.0 - ((p - t) ** 2).mean() / var.clamp(min=1e-8))
        for name in BINARY_NAMES:
            i = CH[name]
            tt = target[..., i][m] if valid is not None else target[..., i]
            pp = pred[..., i][m] if valid is not None else pred[..., i]
            if tt.numel() == 0:
                continue
            metrics[f"state_{name}_acc"] = float(
                ((pp > 0).float() == tt).float().mean())
    return loss, metrics


def projectile_loss(pred: torch.Tensor, target: torch.Tensor,
                    valid: torch.Tensor | None = None,
                    presence_pos_weight: float = 4.0
                    ) -> tuple[torch.Tensor, dict]:
    """pred/target [..., 2, slots, F]. Continuous terms are PRESENCE-MASKED.

    The masking is the correctness point, not an optimisation. An empty slot is
    written as all zeros, so an unmasked regression would spend most of its
    capacity learning to emit the coordinates of bullets that do not exist --
    and, worse, would punish a correct "nothing there" prediction for having
    the wrong position. Position is only defined where something is.

    So presence is a classification over every slot, and everything else is
    scored only where the target says an object is really there.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target "
                         f"{tuple(target.shape)}")
    if valid is None:
        w = torch.ones(pred.shape[:-3], device=pred.device, dtype=pred.dtype)
    else:
        w = valid.to(pred.dtype)
    wb = w[..., None, None, None]
    denom = w.sum().clamp(min=1.0)

    present = target[..., PF["present"]]                     # [..., 2, slots]
    p_idx = torch.tensor(PROJ_BINARY, device=pred.device)
    c_idx = torch.tensor(PROJ_CONT, device=pred.device)

    # Presence and hb: classification over all slots. hb is additionally
    # masked, because "does it have a hitbox" is meaningless for a slot with
    # nothing in it.
    bce = F.binary_cross_entropy_with_logits(
        pred.index_select(-1, p_idx), target.index_select(-1, p_idx),
        reduction="none",
        pos_weight=torch.tensor([presence_pos_weight, presence_pos_weight],
                                device=pred.device))
    hb_col = PROJ_BINARY_NAMES.index("hb")
    bce = bce.clone()
    bce[..., hb_col] = bce[..., hb_col] * present
    l_bin = (bce * wb).sum() / (denom * N_PLAYERS * pred.shape[-2]
                                * len(PROJ_BINARY))

    err = (pred.index_select(-1, c_idx)
           - target.index_select(-1, c_idx)) ** 2 * present[..., None]
    # Normalise by how many objects there actually were, not by slot count:
    # otherwise a frame with one bullet and a frame with eight contribute the
    # same, and the busy frames -- the ones that matter -- are diluted.
    n_present = (present * w[..., None, None]).sum().clamp(min=1.0)
    l_cont = (err * wb).sum() / (n_present * len(PROJ_CONT))

    loss = l_bin + l_cont
    with torch.no_grad():
        occ = float(present.mean())
        acc = float(((pred[..., PF["present"]] > 0).float()
                     == present).float().mean())
        metrics = {"proj_loss": float(loss.detach()),
                   "proj_bin_bce": float(l_bin.detach()),
                   "proj_cont_mse": float(l_cont.detach()),
                   "proj_occupancy": occ, "proj_present_acc": acc}
    return loss, metrics


# ---------------------------------------------------------------------------
# The play-area mirror
# ---------------------------------------------------------------------------
# Display-orientation rows of the play area; health/names sit at 34-74 and
# spirit/cards at 428-470 (data/hud.py). Frames are stored vertically flipped,
# so these convert to stored rows below. Kept identical to
# scripts/spatial_probe.py, which measures exactly this transform -- if the two
# ever disagree the training signal and the metric stop being the same question.
DISP_PLAY_Y = (80, 420)
FRAME = 480

# Channels that negate under a horizontal mirror: signed quantities along x.
MIRROR_NEGATE = ("dx", "facing", "vx", "ax")
# Channels the mirror CANNOT produce a correct label for. See mirror_targets.
MIRROR_UNDEFINED = ("x",)


def mirror_play(obs: torch.Tensor) -> torch.Tensor:
    """Mirror ONLY the play area of [..., C, H, W] frames, HUD untouched.

    The encoder reads which way round the fight is off the HUD: mirroring the
    entire frame is detectable at AUC 0.958 while mirroring only the play area
    sits at 0.620, barely above chance. The HUD is in every frame and trivially
    legible, so it is the shortcut any objective will take -- including 2003
    replays of direct dx supervision, which did not dislodge it.

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


def mirror_targets(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The same labels as seen from the mirrored world, plus a channel mask.

    Returns (state', mask [C]) where mask is 0 for channels the mirror cannot
    produce a correct label for.

    WHY A MASK, AND WHY `x` IS IN IT
    --------------------------------
    Relative quantities negate cleanly: `dx` is a signed separation, `facing` a
    signed direction, `vx` and `ax` signed motion along the same axis. Those
    are properties of the fighters relative to each other and survive the
    transform exactly.

    Absolute `x` does NOT. Mirroring the IMAGE reflects about the centre of the
    camera view, and the camera pans with the action -- so the world coordinate
    a mirrored pixel corresponds to depends on where the camera was, which the
    label does not carry. Emitting `STAGE - x` here would be right only for a
    centred camera and quietly wrong the rest of the time, which is worse than
    not training on it: the channel would be teaching the encoder a false
    relationship on every off-centre frame.

    So `x` is masked out of the mirrored copy and learned from the unmirrored
    one only. `y`, `dy`, `vy` and `ay` are untouched because nothing vertical
    is reflected. Guard, knockdown and airborne are properties of a player
    rather than of a side.
    """
    out = state.clone()
    for name in MIRROR_NEGATE:
        out[..., CH[name]] = -out[..., CH[name]]
    mask = torch.ones(len(STATE_CHANNELS), dtype=state.dtype,
                      device=state.device)
    for name in MIRROR_UNDEFINED:
        mask[CH[name]] = 0.0
    return out, mask


def mirror_projectiles(proj: torch.Tensor) -> torch.Tensor:
    """Projectile features as seen from the mirrored world.

    Every projectile feature here is already relative to the player it is
    flying at, so all of them survive the mirror -- `dx` and `vx` negate,
    `closing` does not (it is a sign about approach, which the mirror
    preserves), and the rest are vertical or categorical.
    """
    out = proj.clone()
    out[..., PF["dx"]] = -out[..., PF["dx"]]
    out[..., PF["vx"]] = -out[..., PF["vx"]]
    return out
