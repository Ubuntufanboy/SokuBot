"""Read game state off the HUD pixels.

The world model predicts latents, and a latent carries no readable notion of
health, spirit, or who won. RL needs those, and the 200 h corpus has only frames
and button states -- no labels anywhere. Rather than re-running 12,310 replays
under an instrumented DLL, this reads the state the game already draws, which
labels every existing capture retroactively.

ORIENTATION
-----------
Captured video is **vertically flipped**: `runner/encode.py` applies `vflip` on
the assumption that glReadPixels is bottom-up, and over-corrects -- the round
timer reads as mirrored glyphs in the raw mp4. Harmless for the world model,
which saw all 200 h flipped identically and learned a self-consistent world, but
everything here works in screen space, so frames are flipped back first. **Any
live inference path must apply the same flip**, or the model sees an orientation
it never trained on.

THE HEALTH BAR HAS TWO COLOURS
------------------------------
Yellow is health remaining. Red is damage dealt during the current combo, and
its width is how big that combo was -- which is the reward signal for "land long
combos". Both live in the same bar, so they are separated per column by the mean
green channel over the lit rows (yellow ~rgb(255,210,8), red is much lower in
green). Classifying per *pixel* and OR-ing down the column double-counts, since
one column contains both near a boundary; that produced totals above 1.0 and red
in 85% of frames instead of 23%.

    fill band  rows 34..49        empty reads ~rgb(60,62,62)
    P1         x 8..197           right edge fixed, drains outward
    P2         x 282..471         left edge fixed, drains outward
    full width 189 px

THE END-OF-MATCH HEAL
---------------------
When a match ends, *all* health lost during it is shown as red and then heals
back to full. That is not a combo, and counting it as one would hand the agent a
large fictitious reward exactly when it had just lost. Samples where yellow
jumps upward are flagged `healing`, and combo reward must be read only where it
is False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

FILL_ROWS = (34, 50)
P1_HP_X = (8, 198)
P2_HP_X = (282, 472)
HP_FULL_PX = 189
MIN_LIT_ROWS = 8            # half the band; rejects one-row specular glints
GREEN_SPLIT = 130           # yellow (health) above, red (combo damage) below

SPIRIT_ROWS = (440, 468)
P1_SPIRIT_X = (115, 222)
P2_SPIRIT_X = (260, 347)

HEAL_JUMP = 0.10            # yellow rising by more than this is a heal, not play


@dataclass
class HudTrace:
    """Per-frame HUD readings for one capture. All arrays are [N]."""
    hp1: np.ndarray          # yellow, 0..1
    hp2: np.ndarray
    combo1: np.ndarray       # red on P1's bar = damage P2 is dealing to P1
    combo2: np.ndarray
    spirit1: np.ndarray
    spirit2: np.ndarray
    healing: np.ndarray      # bool; end-of-match heal, exclude from rewards

    def __len__(self) -> int:
        return len(self.hp1)


def _split_bar(band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """band: [N, rows, cols, 3] -> (yellow_fraction, red_fraction), each [N]."""
    r = band[..., 0].astype(np.int16)
    g = band[..., 1].astype(np.int16)
    b = band[..., 2].astype(np.int16)
    bright = (r > 150) & (b < 120)
    lit = bright.sum(axis=1)                                  # [N, cols]
    gmean = np.where(lit > 0,
                     np.where(bright, g, 0).sum(axis=1) / np.maximum(lit, 1), 0)
    filled = lit >= MIN_LIT_ROWS
    yellow = (filled & (gmean >= GREEN_SPLIT)).sum(axis=1) / HP_FULL_PX
    red = (filled & (gmean < GREEN_SPLIT)).sum(axis=1) / HP_FULL_PX
    return yellow, red


def _orb_fraction(band: np.ndarray, width: int) -> np.ndarray:
    r = band[..., 0].astype(np.int16)
    g = band[..., 1].astype(np.int16)
    b = band[..., 2].astype(np.int16)
    lit = ((b > 140) & ((b - r) > 45) & (g > 60)).any(axis=1)
    return lit.sum(axis=1) / width


def read_trace(frames: np.ndarray, flip: bool = True, smooth: int = 3) -> HudTrace:
    """frames: uint8 [N, 480, 480, 3] in capture orientation.

    `smooth` applies a median over that many consecutive samples. Sprites and
    hit effects cross the bar for a frame or two and briefly read as empty; a
    3-sample median removes them without blunting real damage, which persists.
    """
    if frames.ndim != 4 or frames.shape[1:3] != (480, 480):
        raise ValueError(f"expected [N,480,480,3], got {frames.shape}")
    if flip:
        frames = frames[:, ::-1]

    fr, fc = FILL_ROWS
    y1, r1 = _split_bar(frames[:, fr:fc, P1_HP_X[0]:P1_HP_X[1]])
    y2, r2 = _split_bar(frames[:, fr:fc, P2_HP_X[0]:P2_HP_X[1]])
    sr0, sr1 = SPIRIT_ROWS
    s1 = _orb_fraction(frames[:, sr0:sr1, P1_SPIRIT_X[0]:P1_SPIRIT_X[1]],
                       P1_SPIRIT_X[1] - P1_SPIRIT_X[0])
    s2 = _orb_fraction(frames[:, sr0:sr1, P2_SPIRIT_X[0]:P2_SPIRIT_X[1]],
                       P2_SPIRIT_X[1] - P2_SPIRIT_X[0])

    if smooth and smooth > 1 and len(y1) >= smooth:
        def med(a):
            pad = smooth // 2
            p = np.pad(a, pad, mode="edge")
            return np.median(np.stack([p[i:i + len(a)] for i in range(smooth)]), axis=0)
        y1, r1, y2, r2, s1, s2 = (med(v) for v in (y1, r1, y2, r2, s1, s2))

    # Health only falls during play, so any rise is the between-match heal.
    healing = np.zeros(len(y1), dtype=bool)
    d1, d2 = np.diff(y1, prepend=y1[0]), np.diff(y2, prepend=y2[0])
    healing[(d1 > HEAL_JUMP) | (d2 > HEAL_JUMP)] = True
    # The heal is an animation, not an instant: mark the whole ramp.
    for i in np.where(healing)[0]:
        healing[max(0, i - 2):min(len(healing), i + 4)] = True

    return HudTrace(hp1=y1, hp2=y2, combo1=r1, combo2=r2,
                    spirit1=s1, spirit2=s2, healing=healing)


def damage_events(t: HudTrace, who: int, min_drop: float = 0.01) -> np.ndarray:
    """Per-sample health lost by `who` (1 or 2), zero during heals.

    This is the damage-dealt / damage-taken reward term. Rises are clipped to
    zero rather than counted as negative damage.
    """
    hp = t.hp1 if who == 1 else t.hp2
    d = -np.diff(hp, prepend=hp[0])
    d[t.healing] = 0.0
    d[d < min_drop] = 0.0
    return d


def combo_size(t: HudTrace, who: int) -> np.ndarray:
    """Red width on `who`'s bar -- how much damage the current combo has done.

    NOT YET VALIDATED AS A REWARD TERM. The separation itself is sound (yellow
    and red never double-count: totals exceed 1.02 in 0.00% of samples), but the
    semantics are unconfirmed. Correlation between red and damage over the
    preceding N samples rises with N (+0.13 at 4, +0.27 at 32), which is the
    right direction, yet samples with red > 0.15 show *less* damage in the
    previous 16 samples than average -- the opposite of what "red means a combo
    just landed" predicts. The likely explanation is that red lingers after the
    combo ends, putting the damage outside the window, but that is a hypothesis.
    Overlay the detected yellow/red spans on frames and watch a combo before
    wiring this into a reward.

    `hp1`/`hp2` and `damage_events` are validated and safe to use: health is
    non-increasing within a round in 96-98% of samples, spans the full 0..1
    range, and totals 2.2-3.5 bars of damage per match.
    """
    c = (t.combo1 if who == 1 else t.combo2).copy()
    c[t.healing] = 0.0
    return c
