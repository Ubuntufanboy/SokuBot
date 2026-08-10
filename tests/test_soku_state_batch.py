"""State labels reaching the batch, at the right frames.

The one thing that cannot be checked by eye is the indexing. A clip is T
decision steps `frame_skip` source frames apart, while `state.csv` is indexed
by source frame, so decision step d must pick up row d*skip. Off by a factor
of `skip` still produces batches of the right shape full of plausible numbers,
and the model would learn a slightly wrong association with nothing to show
for it -- the same class of silent error the alignment step exists to prevent.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from sokubot.config import Config
from sokubot.data.state import STATE_CHANNELS

BUTTONS = ["up", "down", "left", "right", "a", "b", "c", "d", "change", "spell"]
HEADER = (["frame", "game_frame", "p1_input", "p2_input"]
          + [f"p{p}_{b}" for p in (1, 2) for b in BUTTONS]
          + [f"p{p}_{c}" for p in (1, 2) for c in
             ("x", "y", "dir", "action", "guarding", "wrongblock", "crushed",
              "knockdown")]
          + ["label_valid"])


def write_state(path, n, *, invalid=()):
    """dx is made a known function of the frame index so the row picked up by
    a given decision step is identifiable from its value alone."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i in range(n):
            row = [i, i, 0, 0] + [0] * 20
            # p1 at 0, p2 at i -> dx for player 0 is +i (before normalising)
            row += [0.0, 0.0, 1, 0, 0, 0, 0, 0]
            row += [float(i), 0.0, -1, 0, 0, 0, 0, 0]
            row += [0 if i in invalid else 1]
            w.writerow(row)


def test_decision_step_d_picks_up_source_frame_d_times_skip(tmp_path):
    from sokubot.data.state import STAGE_SPAN, read_state

    n, skip, T = 200, 4, 6
    p = tmp_path / "state.csv"
    write_state(p, n)
    arr, valid = read_state(p)

    # Emulate exactly the gather in soku.py's window loop.
    start = 3
    rows = [(start + k) * skip for k in range(T)]
    picked = arr[rows]

    assert picked.shape == (T, 2, len(STATE_CHANNELS))
    dx = STATE_CHANNELS.index("dx")
    expected = np.array([(start + k) * skip / STAGE_SPAN for k in range(T)])
    np.testing.assert_allclose(picked[:, 0, dx], expected, rtol=1e-6)
    # Player 1 sees the opponent on the other side.
    np.testing.assert_allclose(picked[:, 1, dx], -expected, rtol=1e-6)
    assert valid[rows].all()


def test_the_mask_travels_with_the_labels(tmp_path):
    from sokubot.data.state import read_state
    n, skip, T = 100, 4, 5
    p = tmp_path / "state.csv"
    # Frame 8 is decision step 2 at skip=4.
    write_state(p, n, invalid=(8,))
    arr, valid = read_state(p)
    rows = [k * skip for k in range(T)]
    assert valid[rows].tolist() == [True, True, False, True, True]


def test_a_capture_without_labels_is_not_advertised_as_having_them(tmp_path):
    """`discover_captures` must leave `state` as None rather than pointing at a
    path that does not exist, or the loader would fail per-capture at read time
    instead of skipping cleanly."""
    from sokubot.data.soku import Capture
    c = Capture(replay_id="x", video=tmp_path / "v.mp4",
                inputs=tmp_path / "inputs.csv", frames=10)
    assert c.state is None


def test_state_coef_defaults_to_off_so_the_loader_is_unchanged():
    """Every existing training command must behave exactly as before."""
    assert Config().state_coef == 0.0
