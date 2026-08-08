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
    "wrongblock",    # guarded the wrong way -- a gap-detection false negative
    "crushed",       # the guard broke
    "knockdown",     # knocked down, or grabbed
    "airborne",      # y above the floor; free from position, and dodging needs it
)

# Stage width in game units. Positions run roughly -600..600 in Soku's
# coordinates, so this maps `dx` into about [-1, 1] without clipping real play.
# Only the scale matters -- the sign is what carries the mechanic.
STAGE_HALF_WIDTH = 600.0
FLOOR_EPS = 1.0            # y above this counts as airborne

# Columns the extractor writes per player, from dll/src/video_encoder.cpp.
_PER_PLAYER = ("x", "y", "dir", "action",
               "guarding", "wrongblock", "crushed", "knockdown")


def has_state_columns(path: Path) -> bool:
    """True if this sidecar was written by an extractor that logs game state.

    Captures made before the state columns existed are still perfectly good for
    everything else, so a corpus mixing both must load rather than fail -- the
    columns were appended for exactly this reason.
    """
    with path.open(newline="") as fh:
        cols = set(csv.DictReader(fh).fieldnames or [])
    return all(f"p{i}_{c}" in cols for i in (1, 2) for c in _PER_PLAYER)


def read_state(path: Path) -> np.ndarray:
    """inputs.csv -> float32 [N, 2, len(STATE_CHANNELS)], indexed [frame, player].

    Player-major rather than one flat row, because every consumer wants "my
    state" and "their state" chosen by `side`, exactly as `rl/reward.py::_sides`
    does for the HUD. Flattening it here would push that gather into every
    caller.
    """
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        for i in (1, 2):
            missing = [c for c in _PER_PLAYER if f"p{i}_{c}" not in cols]
            if missing:
                raise ValueError(
                    f"{path}: missing state columns for p{i}: {missing}. "
                    f"This capture predates game-state logging; use "
                    f"has_state_columns() to skip it.")

        rows: List[np.ndarray] = []
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
                out[me, 0] = (x[them] - x[me]) / STAGE_HALF_WIDTH
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

    if not rows:
        raise ValueError(f"{path}: no rows")
    arr = np.stack(rows)
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
    return arr
