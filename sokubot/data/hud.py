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

# Screen-wide effects (weather transitions, supers, spellcard flashes) wash the
# whole frame toward white. Measured during a "Typhoon" transition, the bar's
# brightest pixels went from rgb(252,215,38) to rgb(250,236,195) -- blue rising
# 38 -> 195. Any fixed colour test then rejects the bar and reports it empty,
# which read as a 64% health loss in ten frames. Saturation collapses too
# (0.85 -> 0.35), so switching colour spaces does not rescue it either.
#
# The reliable tell is that the dark parts stop being dark. In normal play the
# band always contains near-black border pixels; under a flash nothing is dark.
# A dip that RECOVERS is the signature of a flash, because real damage never
# heals mid-round. Measured across a "Typhoon" transition the lit-column count
# went 190 -> 79 -> 188 while the band brightened 145 -> 175; a genuine KO
# instead sits at 0 lit columns with the band near black (2-17). So the rule is
# not a colour threshold at all: take a forward maximum over a window longer
# than any effect but shorter than real damage persists. Transient dips are
# erased, real drops survive untouched.
#
# Colour thresholds were tried first and are not sufficient -- under a flash the
# bar washes to rgb(250,236,195), which fails a blue test, and its saturation
# collapses 0.85 -> 0.35, which fails an HSV test too.
DIP_WINDOW = 36             # 0.6 s at 60 fps; longer than any flash observed
KO_LEVEL = 0.03             # health this low means the round is over
KO_MIN_FRAMES = 25          # ...and it has to stay there; flashes recover in ~20
HEAL_TAIL = 90              # the refill animation after a KO, excluded from rewards


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
    flash: np.ndarray        # bool; screen effect washed out the HUD, reading held
    clamped: np.ndarray      # bool; a physically impossible drop was rejected

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


def _fill_dips(x: np.ndarray, window: int) -> np.ndarray:
    """Forward maximum filter: erase downward excursions that recover."""
    n = len(x)
    out = x.copy()
    for i in range(n):
        out[i] = x[i:min(n, i + window)].max()
    return out


def _monotone_within_rounds(hp: np.ndarray, heal: np.ndarray) -> np.ndarray:
    """Health is non-increasing inside a round; enforce it, resetting on heals."""
    out = hp.copy()
    run_min = out[0]
    for i in range(1, len(out)):
        if heal[i]:
            run_min = out[i]
        else:
            run_min = min(run_min, out[i])
            out[i] = run_min
    return out


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

    # Rounds are delimited by KOs, not by heals. Detecting them from the heal
    # rise is circular: a screen flash makes health dip and recover, the recovery
    # reads as a heal, spurious heals fragment the rounds, and the fragments stop
    # the dip filter from spanning the very flash that caused them. That loop
    # reported 16 rounds in a 3.6-minute replay and 6.6 bars of damage per player.
    #
    # A KO is health pinned near zero for a sustained stretch. No flash does
    # that -- the longest observed dip recovers within ~20 frames -- so this is
    # the one round signal the effects cannot forge.
    raw1, raw2 = y1.copy(), y2.copy()

    def ko_runs(hp):
        dead = hp < KO_LEVEL
        runs, start = [], None
        for i, d in enumerate(dead):
            if d and start is None:
                start = i
            elif not d and start is not None:
                if i - start >= KO_MIN_FRAMES:
                    runs.append((start, i))
                start = None
        if start is not None and len(dead) - start >= KO_MIN_FRAMES:
            runs.append((start, len(dead)))
        return runs

    healing = np.zeros(len(y1), dtype=bool)
    bounds = {0, len(y1)}
    for hp in (raw1, raw2):
        for a_, b_ in ko_runs(hp):
            healing[a_:min(len(healing), b_ + HEAL_TAIL)] = True
            bounds.add(min(len(y1), b_))
    bounds = sorted(bounds)

    for a_, b_ in zip(bounds[:-1], bounds[1:]):
        if b_ - a_ < 2:
            continue
        for arr in (y1, y2):
            # Only transient dips are removed. Health is NOT monotone within a
            # round -- calm weather regenerates it -- so forcing a running
            # minimum here erased every legitimate heal and rewrote 64% of
            # samples. Persistence is the discriminator, not direction: a flash
            # recovers inside ~20 frames, real damage stays, real regen is slow.
            arr[a_:b_] = _fill_dips(arr[a_:b_], DIP_WINDOW)

    repaired = (np.abs(y1 - raw1) > 0.02) | (np.abs(y2 - raw2) > 0.02)

    return HudTrace(hp1=y1, hp2=y2, combo1=r1, combo2=r2,
                    spirit1=s1, spirit2=s2, healing=healing,
                    flash=repaired, clamped=repaired)


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

    Health is NOT monotone within a round -- calm weather regenerates it -- so
    do not use direction as a correctness check.

    STILL TO IMPLEMENT (spellcards):
      * A player's card slots all turn grey the moment they activate a spell
        card, and become visible again when it ends -- but the grey lingers
        past the end, so the window overruns the card and must be trimmed.
      * A card showing a question mark is a weather effect, not a spell card,
        and must not be counted.
      * Spell-card damage is then red-bar growth inside the (trimmed) grey
        window, which is the 2x reward term.
      * Card cost can be mined from the spirit level at which a card's border
        starts glowing.
    """
    c = (t.combo1 if who == 1 else t.combo2).copy()
    c[t.healing] = 0.0
    return c
