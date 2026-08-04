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

# Spirit is five discrete hexagons per player, NOT a continuous bar, and they
# sit in a squiggle -- even indices ride ~5 px higher than odd ones. Measuring
# the strip as one bar under-reports badly and never reaches 1.0 (a full gauge
# read 0.72-0.77), because the gaps between hexagons are always unlit.
#
# Each hexagon is one of: blue (full, 1.0), a purple/grey mix (recovering, worth
# the purple share), or fully grey (crushed or spent, 0.0).
P1_SPIRIT_X = (120, 218)
P2_SPIRIT_X = (263, 363)
SPIRIT_N = 5
SPIRIT_CY_EVEN = 452
SPIRIT_CY_ODD = 457
SPIRIT_BOX = (5, 6)          # half-height, half-width of the sample inside a hexagon

# Card stock. Casting consumes every lit card, so all slots dim at once -- that
# is the spell-card activation signal. The gold borders on the stock show the
# cost of the currently selected card.
P1_CARD_X = (42, 112)
P2_CARD_X = (370, 436)
CARD_ROWS = (428, 470)

# Red never reads exactly zero: the bar's own border columns are misclassified,
# giving a floor of 0.011-0.016 (about two columns of 189) whenever the player is
# alive. Confirmed by a player who checked the footage -- red only truly hits 0
# on a KO. Subtracting the floor makes "no combo" read as no combo.
RED_FLOOR = 0.018

# Spell cards are detected by the card's NAME, which the game draws in red under
# the casting player's health bar -- left-aligned for P1, right-aligned for P2.
# It rises from near the card slots and then PARKS just below the win indicator
# for the card's duration.
#
# Measured on a confirmed cast (replay 5278716, P2): the text enters at y=106,
# climbs ~6 px/frame for seven frames, and holds at y=60-62 from frame 5246
# until the card ends. Parking is the signal worth using -- it is long-lived and
# gives the window directly -- while the rise pins down the start.
#
# Colour alone is a trap: a red character flying through the band matches it.
# What a character cannot do is sit at exactly this height for tens of frames.
NAME_BAND = (56, 74)         # where the name parks
# "Red" is not the signal -- a red stage fills the band and every colour-only
# detector built here failed on exactly that. Text is *thin strokes*: measured on
# real spell-card banners, a row crossing the text breaks into 5-18 separate red
# runs with a median length of 1-3.5 px, covering only 13-25% of the row. A solid
# red background is one run covering nearly all of it.
TEXT_MIN_RUNS = 4            # separate strokes in a row
TEXT_MAX_MEDIAN_RUN = 5      # px; strokes are thin
TEXT_MAX_COVERAGE = 0.55     # text is sparse, a background is not
TEXT_MIN_COVERAGE = 0.04
# Real casts park the banner for seconds; the false positives were 0.3-0.9 s
# blips from red effects crossing the band. 90 frames (1.5 s) separates them.
NAME_MIN_FRAMES = 90

# Only the OUTER half of each side can hold the banner: it is left-aligned for
# P1 and right-aligned for P2. Searching the full half reached into the middle
# of the screen and fired on the weather bubble -- whose text is sometimes red --
# and on the round-win rectangles, which sit at the inner edge beside the timer.
# Neither may ever be inside the search box.
P1_NAME_X = (8, 124)
P2_NAME_X = (356, 472)

# The rise is the signal. Measured: enters around y=106, climbs ~6 px/frame for
# seven frames, stops at 60-62 and holds there for the card.
RISE_TOP, RISE_BOTTOM = 52, 300
# The cap was first set from the average speed (~6 px/frame) and rejected the
# very cast it was derived from: that trajectory steps 100 -> 82 in one frame.
# Bounds must come from the per-frame extremes, not the mean.
RISE_MIN_SPEED, RISE_MAX_SPEED = 1.0, 30.0
RISE_MIN_FRAMES = 4
RISE_MIN_TRAVEL = 20
PARK_LO, PARK_HI = 54, 76
PARK_MIN_FRAMES = 15

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
    cards1: np.ndarray       # lit fraction of P1's card stock
    cards2: np.ndarray
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


def _spirit(frames: np.ndarray, xs) -> np.ndarray:
    """Per-hexagon spirit: blue = 1.0, purple/grey mix = purple share, grey = 0."""
    x0, x1 = xs
    pitch = (x1 - x0) / SPIRIT_N
    hy, hx = SPIRIT_BOX
    out = np.zeros(len(frames), dtype=np.float32)
    for k in range(SPIRIT_N):
        cx = int(x0 + pitch * (k + 0.5))
        cy = SPIRIT_CY_EVEN if k % 2 == 0 else SPIRIT_CY_ODD
        box = frames[:, cy - hy:cy + hy + 1, cx - hx:cx + hx + 1].astype(np.int16)
        r, g, b = box[..., 0], box[..., 1], box[..., 2]
        blue = ((b > 120) & ((b - r) > 40)).mean(axis=(1, 2))
        purple = ((b > 70) & (r > 60) & ((b - g) > 25) & ((r - g) > 15)).mean(axis=(1, 2))
        grey = ((np.abs(r - g) < 28) & (np.abs(g - b) < 28) & (b < 150)).mean(axis=(1, 2))
        denom = purple + grey
        partial = np.where(denom > 0.05, purple / np.maximum(denom, 1e-6), 0.0)
        out += np.where(blue > 0.60, 1.0, partial)
    return out / SPIRIT_N


def _cards_lit(frames: np.ndarray, xs) -> np.ndarray:
    """Fraction of the card-stock strip that is lit (coloured rather than dim).

    Every lit card is consumed on cast, so this collapses when a spell card is
    activated and recovers when it ends -- with a lag, since the slots stay dim
    for a moment after.
    """
    band = frames[:, CARD_ROWS[0]:CARD_ROWS[1], xs[0]:xs[1]].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    lit = (mx > 90) & ((mx - mn) > 45)          # bright and saturated
    return lit.mean(axis=(1, 2))


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
    s1 = _spirit(frames, P1_SPIRIT_X)
    s2 = _spirit(frames, P2_SPIRIT_X)
    k1 = _cards_lit(frames, P1_CARD_X)
    k2 = _cards_lit(frames, P2_CARD_X)

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
                    spirit1=s1, spirit2=s2, cards1=k1, cards2=k2, healing=healing,
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
    return np.maximum(0.0, c - RED_FLOOR)


def banner_top(frames: np.ndarray, who: int, flip: bool = True) -> np.ndarray:
    """Top-most row holding a wide red run, per frame. NaN where none."""
    if flip:
        frames = frames[:, ::-1]
    x0, x1 = P1_NAME_X if who == 1 else P2_NAME_X
    band = frames[:, RISE_TOP:RISE_BOTTOM, x0:x1].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    red = (r > 110) & ((r - g) > 45) & ((r - b) > 45)
    n, rowsn, cols = red.shape
    pad = np.zeros((n, rowsn, 1), dtype=np.int8)
    d = np.diff(np.concatenate([pad, red.astype(np.int8), pad], axis=2), axis=2)
    starts = (d == 1).sum(axis=2)                    # runs per row
    cover = red.mean(axis=2)
    # median run length ~= covered px / number of runs
    mean_run = np.where(starts > 0, red.sum(axis=2) / np.maximum(starts, 1), 0.0)
    istext = ((starts >= TEXT_MIN_RUNS) & (mean_run <= TEXT_MAX_MEDIAN_RUN)
              & (cover <= TEXT_MAX_COVERAGE) & (cover >= TEXT_MIN_COVERAGE))
    out = np.full(n, np.nan)
    for i in range(n):
        w = np.where(istext[i])[0]
        if len(w):
            out[i] = w.min() + RISE_TOP
    return out


def spellcard_events(frames: np.ndarray, t: HudTrace, who: int, flip: bool = True):
    """BLOCKED BY CAPTURE RESOLUTION. Not solvable from this corpus as recorded.

    Measured on the hand-verified cast, inside the correct band, against a
    red-background frame in the same band:

        actual banner (f5252)      runs/row 1-4   mean run 7-88 px  coverage <=0.76
        red background (f5352)     runs/row 1-7   mean run 1-3.7 px coverage <=0.16

    The discriminators are inverted. Real spell text has *fewer and fatter* runs
    than the red background, so by any thin-stroke test the background scores
    more text-like than the text. The game draws 640x480; capture squashes to
    480x480 and encodes h264 at ~640 MB/h, and small text does not survive that.
    The reference screenshots show letters; this corpus holds a blob.

    Six approaches failed, and the last four failed for the same reason -- every
    threshold was set from something other than the corpus: from the average
    rather than the extremes, from the pre-narrowing band, from clean
    screenshots. The measurement above is what should have come first.

    OPTIONS, in order of cost:
      * Drop the 2x spell-card bonus and whiff penalty. Everything else in the
        reward is validated and unaffected; these are additive terms.
      * Re-encode a subset at native 640x480 and higher bitrate. The 12,310
        source .rep files are preserved on the Hugging Face staging repos, so a
        few hundred replays can be re-captured without touching the fleet.
      * Read the state from memory. SokuLib exposes it; the DLL already calls
        game functions directly for the replay bypass, so logging a cast flag is
        a small addition -- and would give exact ground truth rather than an
        inference from pixels.
    """

    Four approaches have failed here, and the history is worth keeping because
    each failed differently:

      1. Card-stock dimming. Documented and true (casting consumes every lit
         card) but not specific: the stock dims in many states. 0/49 correct.
      2. Same, plus a lit-to-dark transition. Removed round intros, still 0%.
      3. Red text parked under the health bar. Colour-only, so a red stage and
         red effects trigger it. 0% again -- which is exactly what the player who
         suggested this signal warned about when they said to track movement.
      4. This one: the banner's rise. Correct in principle, but it misses the
         one cast verified frame by frame (replay 5278716, P2, frame 5239).

    Why 4 currently fails, most likely: the trajectory was measured over
    x 300-472, then the band was narrowed to the outer half (356-472) to exclude
    the weather bubble and the win markers, and the narrowing was never
    re-validated. The banner spans roughly x 305-460, so the narrowed box clips
    its left portion and may no longer contain a wide enough red run.

    NEXT STEP, and do it in this order: re-measure `banner_top` against frames
    5230-5260 of replay 5278716 inside the *narrowed* band, confirm the rise is
    still visible there, and only then set the speed and travel bounds from what
    that measurement shows. Every failure above came from setting a parameter
    before measuring the thing it constrains.

    Spell-card casts, detected by the name banner's RISE.

    Colour is not the signal and never was. The stage can be red, effects can be
    red, and a red character can drift through the band -- every colour-only
    detector built here scored 0% against footage, twice.

    What is specific to a cast is the motion: the name climbs at a near-constant
    rate and then stops just under the win indicator. Measured on a confirmed
    cast (replay 5278716, P2): y = 106 -> 64 over seven frames, about 6 px per
    frame, then parked at 60-62. Requiring a monotone climb of roughly that speed
    that *terminates in the park band* is a trajectory a background cannot fake.

    Returns [(start, end, damage_dealt)].
    """
    y = banner_top(frames, who, flip=flip)
    foe = damage_events(t, 2 if who == 1 else 1)
    n = len(y)
    out = []
    i = 1
    while i < n:
        if np.isnan(y[i]) or np.isnan(y[i - 1]):
            i += 1
            continue
        # a rise: consecutive frames moving up within the plausible speed range
        j = i
        while (j + 1 < n and not np.isnan(y[j + 1])
               and RISE_MIN_SPEED <= (y[j] - y[j + 1]) <= RISE_MAX_SPEED):
            j += 1
        span = j - i
        if span >= RISE_MIN_FRAMES and (y[i] - y[j]) >= RISE_MIN_TRAVEL \
                and PARK_LO <= y[j] <= PARK_HI:
            # the card lasts while the banner stays parked
            k = j
            while k + 1 < n and not np.isnan(y[k + 1]) and PARK_LO <= y[k + 1] <= PARK_HI:
                k += 1
            # The rise is what identifies a cast; the park only measures how long
            # the card lasts. Requiring 90 parked frames was left over from when
            # parking was the detector, and it discarded valid casts whose
            # banner clears quickly -- including the one confirmed by hand.
            if k - j >= PARK_MIN_FRAMES:
                out.append((i, k, float(foe[i:min(k, len(foe))].sum())))
            i = k + 1
            continue
        i = max(j + 1, i + 1)
    return out
