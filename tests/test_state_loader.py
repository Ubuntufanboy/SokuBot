"""Reading game-state ground truth from the extractor sidecar."""

from __future__ import annotations

import pytest

from sokubot.data.state import (STATE_CHANNELS, has_state_columns, read_state,
                                STAGE_SPAN)

_BASE = ("frame,game_frame,p1_input,p2_input,"
         "p1_up,p1_down,p1_left,p1_right,p1_a,p1_b,p1_c,p1_d,p1_change,p1_spell,"
         "p2_up,p2_down,p2_left,p2_right,p2_a,p2_b,p2_c,p2_d,p2_change,p2_spell")
_STATE = (",p1_x,p1_y,p1_dir,p1_action,p1_guarding,p1_wrongblock,p1_crushed,"
          "p1_knockdown,p2_x,p2_y,p2_dir,p2_action,p2_guarding,p2_wrongblock,"
          "p2_crushed,p2_knockdown")
# frame, game_frame, two masks, then 10 booleans per player.
_ZEROS = ",".join(["0"] * 24)


def _write(tmp_path, rows, with_state=True):
    p = tmp_path / "inputs.csv"
    head = _BASE + (_STATE if with_state else "")
    p.write_text(head + "\n" + "\n".join(rows) + "\n")
    return p


def test_dx_is_signed_and_opposite_for_the_two_players(tmp_path):
    """The one quantity the labels exist for. P1 at -100, P2 at +200: the
    opponent is to P1's right and to P2's left, so dx must flip sign."""
    row = _ZEROS + ",-100,0,1,0,0,0,0,0,200,0,-1,0,0,0,0,0"
    st, _ = read_state(_write(tmp_path, [row]))
    dx = STATE_CHANNELS.index("dx")
    assert st[0, 0, dx] == pytest.approx(300 / STAGE_SPAN)
    assert st[0, 1, dx] == pytest.approx(-300 / STAGE_SPAN)


def test_facing_is_normalised_to_plus_minus_one(tmp_path):
    row = _ZEROS + ",0,0,7,0,0,0,0,0,0,0,-3,0,0,0,0,0"
    st, _ = read_state(_write(tmp_path, [row]))
    f = STATE_CHANNELS.index("facing")
    assert st[0, 0, f] == 1.0 and st[0, 1, f] == -1.0


def test_guard_failure_modes_stay_distinct(tmp_path):
    """wrongblock and crushed are the two ways guarding fails and they are
    different mistakes -- a false negative in gap detection, and the crush that
    follows. Collapsing them into `guarding` would erase the mechanic."""
    row = _ZEROS + ",0,0,1,159,0,1,0,0,0,0,-1,143,0,0,1,0"
    st, _ = read_state(_write(tmp_path, [row]))
    g = STATE_CHANNELS.index("guarding")
    w = STATE_CHANNELS.index("wrongblock")
    c = STATE_CHANNELS.index("crushed")
    assert st[0, 0, w] == 1.0 and st[0, 0, g] == 0.0
    assert st[0, 1, c] == 1.0 and st[0, 1, g] == 0.0


def test_airborne_is_derived_from_y(tmp_path):
    row = _ZEROS + ",0,0,1,0,0,0,0,0,0,55,-1,0,0,0,0,0"
    st, _ = read_state(_write(tmp_path, [row]))
    a = STATE_CHANNELS.index("airborne")
    assert st[0, 0, a] == 0.0 and st[0, 1, a] == 1.0


def test_old_captures_are_detected_not_crashed_on(tmp_path):
    """A corpus mixing pre- and post-state captures must load. The columns were
    appended rather than inserted for exactly this."""
    p = _write(tmp_path, [_ZEROS], with_state=False)
    assert has_state_columns(p) is False
    with pytest.raises(ValueError, match="predates game-state logging"):
        read_state(p)


def test_a_wrong_offset_is_caught_rather_than_trained_on(tmp_path):
    """The real failure mode of a hand-picked struct offset is reading some
    other field and getting plausible-looking floats. A stage is ~1200 units,
    so anything far outside that is not a position."""
    row = _ZEROS + ",99999,0,1,0,0,0,0,0,-99999,0,-1,0,0,0,0,0"
    with pytest.raises(ValueError, match="not a position"):
        read_state(_write(tmp_path, [row]))


def test_a_truncated_row_is_named(tmp_path):
    """Captures are killed at MAX_FRAMES and on scene changes, so a partial
    final line happens. It should say so, not fail on a None several lines
    later."""
    good = _ZEROS + ",0,0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0"
    with pytest.raises(ValueError, match="is short"):
        read_state(_write(tmp_path, [good, _ZEROS + ",0,0,1"]))


# --- the label_valid mask -------------------------------------------------
# `pipeline/align_sidecar.py` fills frames the re-capture did not cover by
# repeating the nearest real row, and marks them. Those rows are invented, so
# a mask that quietly reads all-true would put fabricated supervision into
# training with nothing to show for it.

def _write_valid(tmp_path, rows, flags):
    p = tmp_path / "state.csv"
    head = _BASE + _STATE + ",label_valid"
    body = [f"{r},{f}" for r, f in zip(rows, flags)]
    p.write_text(head + "\n" + "\n".join(body) + "\n")
    return p


_ROW = _ZEROS + ",-100,0,1,0,0,0,0,0,200,0,-1,0,0,0,0,0"


def test_the_mask_marks_exactly_the_padded_rows(tmp_path):
    p = _write_valid(tmp_path, [_ROW] * 5, [0, 1, 1, 1, 0])
    st, valid = read_state(p)
    assert st.shape == (5, 2, len(STATE_CHANNELS))
    assert valid.dtype == bool
    assert valid.tolist() == [False, True, True, True, False]


def test_padded_rows_are_still_returned_so_row_i_is_frame_i(tmp_path):
    """Dropping them would break the one guarantee the aligned file makes."""
    p = _write_valid(tmp_path, [_ROW] * 4, [0, 1, 1, 0])
    st, valid = read_state(p)
    assert len(st) == 4 == len(valid)


def test_a_capture_without_the_column_reports_every_frame_valid(tmp_path):
    """A direct capture invented nothing, so all-true is the truth there --
    but it must come from the column being absent, not from a parse failure."""
    st, valid = read_state(_write(tmp_path, [_ROW, _ROW]))
    assert valid.tolist() == [True, True]


def test_a_gzipped_sidecar_reads_identically(tmp_path):
    """`align_sidecar` writes state.csv.gz by default -- 1.42 GB of plain CSV
    does not fit on the box that holds the corpus. If the loader could not open
    it the labels would simply be unreadable."""
    import gzip

    plain = _write_valid(tmp_path, [_ROW] * 3, [1, 0, 1])
    gz = tmp_path / "state.csv.gz"
    with gzip.open(gz, "wt", newline="") as fh:
        fh.write(plain.read_text())

    a, va = read_state(plain)
    b, vb = read_state(gz)
    assert (a == b).all() and va.tolist() == vb.tolist()
    assert has_state_columns(gz)
