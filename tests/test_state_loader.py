"""Reading game-state ground truth from the extractor sidecar."""

from __future__ import annotations

import pytest

from sokubot.data.state import (CH, PF, PROJ_FEATURES, STAGE_SPAN,
                                STATE_CHANNELS, VEL_SCALE, has_state_columns,
                                read_state)

_BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
# The per-player state block, in the extractor's own order
# (dll/src/video_encoder.cpp).
_FIELDS = ("x", "y", "vx", "vy", "ax", "ay", "dir", "action", "action_frame",
           "hitstop", "untech", "hitboxes", "hurtboxes", "hit_count", "hp",
           "spirit", "max_spirit", "spirit_delay", "timestop", "ground_dashes",
           "air_dashes", "correction", "combo_rate", "combo_hits",
           "combo_damage", "combo_limit", "guarding", "wrongblock", "crushed",
           "knockdown")
_PROJ_FIELDS = ("x", "y", "vx", "vy", "dir", "act", "hb")
SLOTS = 3          # fewer than the capture's 24; the loader reads the header

_BASE_COLS = (["frame", "game_frame", "p1_input", "p2_input"]
              + [f"p{p}_{b}" for p in (1, 2) for b in _BUTTONS])
_STATE_COLS = [f"p{p}_{c}" for p in (1, 2) for c in _FIELDS]
_PROJ_COLS = [c for p in (1, 2) for c in
              ([f"p{p}_proj_n", f"p{p}_proj_hb", f"p{p}_proj_raw"]
               + [f"p{p}_pr{k}_{f}" for k in range(SLOTS)
                  for f in _PROJ_FIELDS])]
HEADER = _BASE_COLS + _STATE_COLS + ["battle_frame"] + _PROJ_COLS


def _row(**over) -> dict:
    """One fully-populated row, so a test states only what it is about.

    Defaults are a legal neutral frame: both players on the floor at opposite
    ends, facing each other, full health, nothing in the air.
    """
    r = {c: "0" for c in HEADER}
    r.update({"p1_x": "400", "p2_x": "800", "p1_dir": "1", "p2_dir": "-1",
              "p1_hp": "10000", "p2_hp": "10000",
              "p1_max_spirit": "1000", "p2_max_spirit": "1000"})
    r.update({k: str(v) for k, v in over.items()})
    return r


def _write(tmp_path, rows, *, name="inputs.csv", cols=None) -> "object":
    cols = cols or HEADER
    p = tmp_path / name
    body = "\n".join(",".join(r[c] for c in cols) for r in rows)
    p.write_text(",".join(cols) + "\n" + body + "\n")
    return p


# --- the quantities the labels exist for ----------------------------------

def test_dx_is_signed_and_opposite_for_the_two_players(tmp_path):
    """P1 at 400, P2 at 700: the opponent is to P1's right and to P2's left,
    so dx must flip sign. This is the whole reason the labels exist -- blocking
    is holding away, and 'away' is the sign of this."""
    st, _, _, _ = read_state(_write(tmp_path, [_row(p1_x=400, p2_x=700)]))
    assert st[0, 0, CH["dx"]] == pytest.approx(300 / STAGE_SPAN)
    assert st[0, 1, CH["dx"]] == pytest.approx(-300 / STAGE_SPAN)


def test_facing_is_normalised_to_plus_minus_one(tmp_path):
    st, _, _, _ = read_state(_write(tmp_path, [_row(p1_dir=7, p2_dir=-3)]))
    assert st[0, 0, CH["facing"]] == 1.0
    assert st[0, 1, CH["facing"]] == -1.0


def test_speed_is_converted_out_of_the_facing_frame(tmp_path):
    """The game stores speed facing-relative: measured, world dx tracks
    vx*direction at err 1.59 where the raw column errs 6.20. A velocity whose
    sign flips with facing makes 'moving away' unlearnable, so the conversion
    happens once, here, where no consumer can forget it."""
    st, _, _, _ = read_state(_write(tmp_path, [
        _row(p1_dir=1, p1_vx=6, p2_dir=-1, p2_vx=6)]))
    # Both are moving forward at 6; forward is +x for P1 and -x for P2.
    assert st[0, 0, CH["vx"]] == pytest.approx(6 / VEL_SCALE)
    assert st[0, 1, CH["vx"]] == pytest.approx(-6 / VEL_SCALE)


def test_guard_failure_modes_stay_distinct(tmp_path):
    """wrongblock and crushed are the two ways guarding fails and they are
    different mistakes -- a false negative in gap detection, and the crush that
    follows. Collapsing them into `guarding` would erase the mechanic."""
    st, _, _, _ = read_state(_write(tmp_path, [
        _row(p1_wrongblock=1, p2_crushed=1)]))
    assert st[0, 0, CH["wrongblock"]] == 1.0 and st[0, 0, CH["guarding"]] == 0.0
    assert st[0, 1, CH["crushed"]] == 1.0 and st[0, 1, CH["guarding"]] == 0.0


def test_airborne_is_derived_from_y(tmp_path):
    st, _, _, _ = read_state(_write(tmp_path, [_row(p1_y=0, p2_y=55)]))
    assert st[0, 0, CH["airborne"]] == 0.0
    assert st[0, 1, CH["airborne"]] == 1.0


def test_spirit_is_a_fraction_because_the_maximum_varies(tmp_path):
    """max_spirit is per character, so the raw number is not comparable across
    a matchup and the fraction is."""
    st, _, _, _ = read_state(_write(tmp_path, [
        _row(p1_spirit=500, p1_max_spirit=1000,
             p2_spirit=500, p2_max_spirit=2000)]))
    assert st[0, 0, CH["spirit"]] == pytest.approx(0.5)
    assert st[0, 1, CH["spirit"]] == pytest.approx(0.25)


def test_a_zero_maximum_does_not_divide(tmp_path):
    """Between rounds max_spirit is genuinely 0, and a NaN there would poison
    every batch it landed in."""
    st, _, _, _ = read_state(_write(tmp_path, [_row(p1_max_spirit=0)]))
    assert st[0, 0, CH["spirit"]] == 0.0


def test_action_ids_come_back_as_integers(tmp_path):
    """Action ids are nominal -- 801 is not 'one more than' 800. Returning them
    as a float channel would ask a network to invent an ordering."""
    _, _, act, _ = read_state(_write(tmp_path, [_row(p1_action=150,
                                                     p2_action=805)]))
    assert act.dtype.kind == "i"
    assert act[0].tolist() == [150, 805]


# --- projectiles ----------------------------------------------------------

def test_projectiles_are_relative_to_who_they_fly_at(tmp_path):
    """`proj[:, p]` holds the objects player p OWNS, positioned against the
    player they are flying at -- so 'is this about to hit me' is a subtraction
    rather than an inference."""
    r = _row(p1_x=400, p2_x=800, p1_proj_n=1, p1_pr0_x=700, p1_pr0_y=0)
    _, pr, _, _ = read_state(_write(tmp_path, [r]))
    # P1's bullet at x=700 against P2 at x=800: 100 units short of the target.
    assert pr[0, 0, 0, PF["present"]] == 1.0
    assert pr[0, 0, 0, PF["dx"]] == pytest.approx(-100 / STAGE_SPAN)


def test_slots_past_the_live_count_are_empty(tmp_path):
    """The extractor writes zeros into unused slots so every row has the same
    width; `present` is what separates an empty slot from an object at 0,0."""
    r = _row(p1_proj_n=1, p1_pr0_x=500, p1_pr1_x=500)
    _, pr, _, _ = read_state(_write(tmp_path, [r]))
    assert pr[0, 0, 0, PF["present"]] == 1.0
    assert pr[0, 0, 1, PF["present"]] == 0.0
    assert (pr[0, 0, 1] == 0.0).all()


def test_closing_says_whether_it_is_coming_at_the_target(tmp_path):
    """Dodging is about the bullets that are arriving, and a bullet moving away
    is not a threat however close it is."""
    # P2 is the target at x=800. A bullet at 700 moving +x closes; at 900
    # moving +x it recedes. dir=1 so the facing conversion is the identity.
    r = _row(p2_x=800, p1_proj_n=2,
             p1_pr0_x=700, p1_pr0_vx=5, p1_pr0_dir=1,
             p1_pr1_x=900, p1_pr1_vx=5, p1_pr1_dir=1)
    _, pr, _, _ = read_state(_write(tmp_path, [r]))
    assert pr[0, 0, 0, PF["closing"]] > 0
    assert pr[0, 0, 1, PF["closing"]] < 0


def test_a_sentinel_coordinate_is_clamped_not_propagated(tmp_path):
    """Measured over 1.23M records, a rare few objects carry x = +-100 * 2^30 --
    real game values belonging to object types that use an extreme coordinate.
    One of them in a batch destroys the normalisation of every other feature in
    it, so they are truncated while honest off-screen bullets are kept."""
    r = _row(p1_proj_n=1, p1_pr0_x=107374182400.0)
    _, pr, _, _ = read_state(_write(tmp_path, [r]))
    assert abs(pr[0, 0, 0, PF["dx"]]) <= 4.0


def test_the_slot_count_is_read_from_the_header(tmp_path):
    """It has already changed once -- 8, then 24 -- and a reader with the number
    baked in would either crash or, far worse, silently read 8 of 24."""
    _, pr, _, _ = read_state(_write(tmp_path, [_row()]))
    assert pr.shape[2] == SLOTS
    assert pr.shape[3] == len(PROJ_FEATURES)


# --- failure modes --------------------------------------------------------

def test_old_captures_are_detected_not_crashed_on(tmp_path):
    """A corpus mixing pre- and post-state captures must be skippable rather
    than fatal, and the error has to name what is missing."""
    cols = [c for c in HEADER if "_pr" not in c and "proj" not in c]
    p = _write(tmp_path, [_row()], cols=cols)
    assert has_state_columns(p) is False
    with pytest.raises(ValueError, match="missing state columns"):
        read_state(p)


def test_a_wrong_offset_is_caught_rather_than_trained_on(tmp_path):
    """The real failure mode of a hand-picked struct offset is reading some
    other field and getting plausible-looking floats. A stage is ~1200 units,
    so anything far outside that is not a position. Characters are deliberately
    NOT clamped, precisely so that they can trip this."""
    with pytest.raises(ValueError, match="not a position"):
        read_state(_write(tmp_path, [_row(p1_x=99999, p2_x=-99999)]))


def test_a_truncated_row_is_dropped_not_stumbled_over(tmp_path):
    """Captures are killed at MAX_FRAMES and on scene changes, so a partial
    final line happens. It must not surface as a NoneType error far from the
    cause."""
    p = tmp_path / "inputs.csv"
    good = ",".join(_row()[c] for c in HEADER)
    p.write_text(",".join(HEADER) + "\n" + good + "\n" + "0,0,0\n")
    st, _, _, _ = read_state(p)
    assert len(st) == 1


# --- the label_valid mask -------------------------------------------------
# `pipeline/align_sidecar.py` fills frames the re-capture did not cover by
# repeating the nearest real row, and marks them. Those rows are invented, so
# a mask that quietly reads all-true would put fabricated supervision into
# training with nothing to show for it.

def _write_valid(tmp_path, n, flags):
    cols = HEADER + ["label_valid"]
    rows = [{**_row(), "label_valid": str(f)} for f in flags]
    return _write(tmp_path, rows[:n], name="state.csv", cols=cols)


def test_the_mask_marks_exactly_the_padded_rows(tmp_path):
    p = _write_valid(tmp_path, 5, [0, 1, 1, 1, 0])
    st, _, _, valid = read_state(p)
    assert st.shape == (5, 2, len(STATE_CHANNELS))
    assert valid.dtype == bool
    assert valid.tolist() == [False, True, True, True, False]


def test_padded_rows_are_still_returned_so_row_i_is_frame_i(tmp_path):
    """Dropping them would break the one guarantee the aligned file makes."""
    st, _, _, valid = read_state(_write_valid(tmp_path, 4, [0, 1, 1, 0]))
    assert len(st) == 4 == len(valid)


def test_a_capture_without_the_column_reports_every_frame_valid(tmp_path):
    """A direct capture invented nothing, so all-true is the truth there --
    but it must come from the column being absent, not a parse failure."""
    _, _, _, valid = read_state(_write(tmp_path, [_row(), _row()]))
    assert valid.tolist() == [True, True]


def test_a_gzipped_sidecar_reads_identically(tmp_path):
    """Both the collector and align_sidecar write .gz -- 11 GB of plain CSV
    does not fit on the box that holds the corpus. If the loader could not open
    it the labels would simply be unreadable."""
    import gzip

    plain = _write_valid(tmp_path, 3, [1, 0, 1])
    gz = tmp_path / "state.csv.gz"
    with gzip.open(gz, "wt", newline="") as fh:
        fh.write(plain.read_text())

    a, pa, aa, va = read_state(plain)
    b, pb, ab, vb = read_state(gz)
    assert (a == b).all() and (pa == pb).all() and (aa == ab).all()
    assert va.tolist() == vb.tolist()
