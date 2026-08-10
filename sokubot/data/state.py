"""Ground-truth game state from the extractor's sidecar, for supervising the
world model.

WHY THESE LABELS EXIST
----------------------
The encoder does not represent where the characters are. `spatial_probe.py`
measures a linear probe at AUC 0.540 for "did the characters swap sides" against
a 0.956 ceiling for "did the HUD swap sides", and `block_effect.py` measures the
predictor forecasting *less* damage when the defender holds no direction at all
-- the sign of guarding, inverted. Guarding in Hisoutensoku is holding away from
the opponent, so a model that cannot see which side they are on cannot represent
the mechanic the whole matchup runs through.

Two objectives failed to recover it from pixels alone: plain JEPA prediction,
and JEPA plus inverse dynamics. Both improved the representation and neither
flipped the sign. So these columns stop asking the model to infer what the game
can simply be asked.

THE CONSTRAINT IS UNCHANGED
---------------------------
It was always about *inference*: the policy at play time consumes pixels and its
own inputs, nothing else. These labels never reach it. They shape a world model,
which is the same asymmetric arrangement `hud_coef` already uses -- the only
difference is that the HUD is legible in pixels and position is not.

WHAT IS DERIVED HERE AND WHY
----------------------------
`facing` and `dx` are the two the model actually needs, and neither is a raw
column. Blocking is holding away from the opponent, so what matters is the
*sign of their separation*, not two absolute coordinates -- and a signed
difference is also what survives the mirroring the probe tests with. The raw
positions are kept because a future question may want them, but the supervision
target is the relative quantity.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import List

import numpy as np

# Order is the contract with the supervised head. Appending is safe; inserting
# silently permutes every previously trained head, which is the failure
# `build_hud_bank` guards against for the HUD channels.
STATE_CHANNELS: tuple[str, ...] = (
    "dx",            # x_opponent - x_me, normalised; sign is "which way is away"
    "facing",        # +1 if I face right, -1 if left, from the game's own flag
    "guarding",      # correct guard, ground or air
    # Blocked in the right direction at the wrong HEIGHT, not in the wrong
    # direction: SokuLib names the range ACTION_WRONGBLOCK_{HIGH,LOW}_*_
    # BLOCKSTUN, beside ACTION_RIGHTBLOCK_{HIGH,LOW}_*. Measured on a real
    # capture, 96.4% of these frames hold away, the same as a right block.
    # It is still a gap-detection failure and still the road to a crush; it is
    # simply not a positional error, so a model can only tell the two apart
    # from the attack, never from the defender's stick.
    "wrongblock",
    "crushed",       # the guard broke
    "knockdown",     # knocked down, or grabbed
    "airborne",      # y above the floor; free from position, and dodging needs it
)

# Maximum separation in game units, measured rather than assumed. A capture of
# a real match (10 073 frames, replay 5262777) puts x in [40, 1240] and the
# signed separation in [-1200, +839], so the stage is about 1280 units wide with
# the origin at one edge -- NOT centred, which is what an earlier value of 600
# here assumed. Dividing by the full span is what puts `dx` in [-1, 1]; at 600
# it reached +-2. Only the scale is affected -- the sign is what carries the
# mechanic -- but a channel that silently exceeds its stated range is the kind
# of thing that later reads as a bug in something else.
STAGE_SPAN = 1200.0
FLOOR_EPS = 1.0            # y above this counts as airborne; the floor is 0.0

# Columns the extractor writes per player, from dll/src/video_encoder.cpp.
_PER_PLAYER = ("x", "y", "dir", "action",
               "guarding", "wrongblock", "crushed", "knockdown")


def _open(path: Path):
    """Text handle for a sidecar, gzipped or not.

    `align_sidecar.py` writes `state.csv.gz` by default: the full sidecars are
    1.42 GB across the corpus against 1.1 GB free on the machine that holds it,
    and gzip takes that to about 0.1 GB. Deciding by suffix rather than by a
    flag means a caller cannot pass the wrong one.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def has_state_columns(path: Path) -> bool:
    """True if this sidecar was written by an extractor that logs game state.

    Captures made before the state columns existed are still perfectly good for
    everything else, so a corpus mixing both must load rather than fail -- the
    columns were appended for exactly this reason.
    """
    with _open(path) as fh:
        cols = set(csv.DictReader(fh).fieldnames or [])
    return all(f"p{i}_{c}" in cols for i in (1, 2) for c in _PER_PLAYER)


LABEL_VALID = "label_valid"


def read_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """state csv -> (float32 [N, 2, len(STATE_CHANNELS)], bool [N]).

    THE SECOND RETURN IS NOT OPTIONAL, ON PURPOSE
    ---------------------------------------------
    Labels produced by `pipeline/align_sidecar.py` are a *re-capture* of the
    replay attached to the video already on disk, and the two captures do not
    begin on exactly the same engine tick. Frames at the very start or end that
    the re-capture did not cover are filled by repeating the nearest real row
    and marked `label_valid = 0`, so that row i stays video frame i. Measured
    over 941 real pairs that is 420 frames in total, worst case 8 -- 0.004% --
    but they are invented and must not be trained on as if measured.

    This returns a tuple rather than an array with an optional flag because a
    mask that must be remembered will eventually not be (docs/BUGS.md 8). A
    caller that ignores it has to write `state, _ = read_state(...)`, which is
    a decision; `state = read_state(...)` no longer type-checks or unpacks into
    an array of the right shape.

    Captures with no `label_valid` column -- anything taken directly rather
    than aligned -- report every frame valid, which is correct: nothing was
    filled in.

    Player-major rather than one flat row, because every consumer wants "my
    state" and "their state" chosen by `side`, exactly as `rl/reward.py::_sides`
    does for the HUD. Flattening it here would push that gather into every
    caller.
    """
    with _open(path) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        for i in (1, 2):
            missing = [c for c in _PER_PLAYER if f"p{i}_{c}" not in cols]
            if missing:
                raise ValueError(
                    f"{path}: missing state columns for p{i}: {missing}. "
                    f"This capture predates game-state logging; use "
                    f"has_state_columns() to skip it.")

        has_valid = LABEL_VALID in cols
        rows: List[np.ndarray] = []
        valid: List[bool] = []
        for n, row in enumerate(reader):
            # A truncated final row is a real thing: the extractor is killed at
            # MAX_FRAMES and on scene changes, so the last line can be a partial
            # write. DictReader fills the missing fields with None, which
            # otherwise surfaces as `float() argument must be ... not NoneType`
            # several lines later and says nothing about the cause.
            if any(row.get(c) is None for c in ("p1_x", "p2_x", "p1_crushed",
                                                "p2_crushed")):
                raise ValueError(
                    f"{path}: row {n} is short ({len(row)} of {len(cols)} "
                    f"fields). A truncated last line usually means the capture "
                    f"was killed mid-write; drop it and re-read.")
            x = [float(row["p1_x"]), float(row["p2_x"])]
            y = [float(row["p1_y"]), float(row["p2_y"])]
            out = np.zeros((2, len(STATE_CHANNELS)), dtype=np.float32)
            for me in (0, 1):
                them = 1 - me
                # The signed separation, from my point of view. This is the
                # whole reason the labels exist: "away" is the sign of this.
                out[me, 0] = (x[them] - x[me]) / STAGE_SPAN
                # The game stores direction as a signed flag; normalise it to
                # +-1 so a model never has to learn the encoding.
                d = float(row[f"p{me+1}_dir"])
                out[me, 1] = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
                out[me, 2] = float(row[f"p{me+1}_guarding"])
                out[me, 3] = float(row[f"p{me+1}_wrongblock"])
                out[me, 4] = float(row[f"p{me+1}_crushed"])
                out[me, 5] = float(row[f"p{me+1}_knockdown"])
                out[me, 6] = 1.0 if y[me] > FLOOR_EPS else 0.0
            rows.append(out)
            valid.append(bool(int(float(row[LABEL_VALID]))) if has_valid
                         else True)

    if not rows:
        raise ValueError(f"{path}: no rows")
    arr = np.stack(rows)
    mask = np.array(valid, dtype=bool)
    # `dx` is the only unbounded channel and the only one that can reveal a
    # wrong offset. A stage is about 1200 units across, so anything far outside
    # that means the read is not a position at all -- which is the failure mode
    # a hand-picked struct offset actually has.
    lo, hi = float(arr[..., 0].min()), float(arr[..., 0].max())
    if not (-4.0 < lo and hi < 4.0):
        raise ValueError(
            f"{path}: dx spans [{lo:.2f}, {hi:.2f}] in stage widths, which is "
            f"not a position. Check CHAR_POSITION_X_OFFSET against the game "
            f"build before trusting any of these labels.")
    return arr, mask
