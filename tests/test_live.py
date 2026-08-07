"""Logic checks for the live control loop that need no game and no GPU.

The parts of `sokubot.live` that can be wrong *silently* are the ones with no
visible output: a scheduler that quietly plays the wrong tick, a gate that
chatters, a pad that emits an input the game has no defined response to. Those
are what this covers. Whether Wine sees the pad at all is a different kind of
question and is answered by `scripts/live_probe.py --probe-a` against the real
game.

    python -m pytest tests/test_live.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from sokubot.live.gate import ArmSwitch, BattleGate, may_act
from sokubot.live.schedule import NEUTRAL, ChunkScheduler, DelayPolicy


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------
def _chunk(vals):
    c = np.zeros((4, 10), dtype=np.float32)
    for t, idx in enumerate(vals):
        if idx is not None:
            c[t, idx] = 1.0
    return c


def test_plays_each_tick_in_order():
    s = ChunkScheduler(ticks=4)
    s.submit(_chunk([0, 1, 2, 3]), at_tick=100)
    for i, expect in enumerate((0, 1, 2, 3)):
        got = s.state_at(100 + i)
        assert got[expect] == 1.0 and got.sum() == 1.0
    assert s.stats()["applied"] == 4


def test_late_chunk_holds_the_last_tick_not_neutral_and_not_a_replay():
    """The distinction the whole module exists for."""
    s = ChunkScheduler(ticks=4)
    s.submit(_chunk([0, 1, 2, 3]), at_tick=100)
    for _ in range(4):
        pass
    # Three ticks past the end of the chunk, with nothing new submitted.
    for t in (104, 105, 106):
        got = s.state_at(t)
        assert got[3] == 1.0, "should hold the chunk's final tick"
        assert not np.array_equal(got, NEUTRAL), "must never fall to neutral"
        assert got[0] == 0.0, "must not replay the chunk from its start"
    assert s.stats()["held"] == 3


def test_idle_before_any_chunk_is_neutral():
    s = ChunkScheduler(ticks=4)
    assert np.array_equal(s.state_at(0), NEUTRAL)
    assert s.stats()["idle"] == 1


def test_tick_timing_is_absolute_not_accumulated():
    """A slow iteration must not shift every later tick."""
    s = ChunkScheduler(ticks=4, origin=1000.0)
    assert s.tick_index(1000.0) == 0
    assert s.tick_index(1000.0 + 10 * (1 / 60.0) + 0.001) == 10
    # Deadlines are exact multiples of the tick from the origin, forever.
    assert s.next_deadline(600) == pytest.approx(1000.0 + 10.0)


def test_delay_policy_targets_a_whole_decision_ahead():
    d = DelayPolicy(steps=2, frame_skip=4)
    assert d.target_tick(100) == 108
    assert d.seconds == pytest.approx(8 / 60.0)


def test_submit_rejects_a_mis_shaped_chunk():
    s = ChunkScheduler(ticks=4)
    with pytest.raises(ValueError):
        s.submit(np.zeros((3, 10), dtype=np.float32), at_tick=0)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def _frame(bar1: float, bar2: float) -> np.ndarray:
    """A 480x480 capture-orientation frame with health bars of a given fill.

    Built in screen space and flipped, because that is what the capture
    produces; if the gate forgets to flip it back, these fail.
    """
    from sokubot.data.hud import FILL_ROWS, P1_HP_X, P2_HP_X
    scr = np.zeros((480, 480, 3), dtype=np.uint8)
    fr, fc = FILL_ROWS
    for (x0, x1), frac in ((P1_HP_X, bar1), (P2_HP_X, bar2)):
        width = int((x1 - x0) * frac)
        # Yellow: bright red channel, low blue, green above the split.
        scr[fr:fc, x0:x0 + width] = (255, 210, 8)
    return scr[::-1]


def test_gate_arms_on_a_battle_frame_and_debounces():
    g = BattleGate()
    live = _frame(1.0, 1.0)
    assert not g.update(live), "must not arm on the first frame"
    assert not g.update(live)
    assert g.update(live), "arms after on_frames"


def test_gate_ignores_a_menu_frame():
    g = BattleGate()
    for _ in range(10):
        assert not g.update(_frame(0.0, 0.0))


def test_gate_needs_both_bars():
    g = BattleGate()
    for _ in range(10):
        assert not g.update(_frame(1.0, 0.0))


def test_gate_survives_a_brief_flash():
    """A screen wash blanks the bars for a few samples; the round is not over."""
    g = BattleGate()
    for _ in range(5):
        g.update(_frame(1.0, 1.0))
    assert g.in_battle
    for _ in range(g.off_frames - 1):
        g.update(_frame(0.0, 0.0))
    assert g.in_battle, "must not disarm mid-round on a transient blank"
    g.update(_frame(0.0, 0.0))
    assert not g.in_battle, "but a sustained blank does disarm"


def test_may_act_needs_both_the_human_and_the_battle():
    g, sw = BattleGate(), ArmSwitch()
    for _ in range(5):
        g.update(_frame(1.0, 1.0))
    assert g.in_battle and not may_act(g, sw), "disarmed human vetoes"
    sw.arm()
    assert may_act(g, sw)
    g.update(_frame(0.0, 0.0))
    sw.disarm()
    assert not may_act(g, sw)


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------
def test_pad_button_order_matches_the_corpus():
    """The 10-wide vector is shared with data/soku.py and rl/policy.py."""
    from sokubot.data.soku import BUTTONS as CORPUS_BUTTONS
    from sokubot.live.pad import BUTTONS as PAD_BUTTONS
    assert PAD_BUTTONS == CORPUS_BUTTONS


def test_pad_declares_no_pov_hat():
    """A centred hat is read as POV north, i.e. 'up' held forever.

    Wine maps an evdev hat to a DirectInput POV, whose neutral is a reserved
    value rather than the centre of its range, so a hat idling at (0, 0) reads
    as 0 degrees. Symptom: menus scroll upward and releasing the d-pad does
    nothing, because released *is* (0, 0). Caught during setup for the first
    live match.
    """
    import evdev
    from sokubot.live import pad as P
    ui = P.UInput
    caps_abs = []

    class _Spy:
        def __init__(self, caps, **kw):
            nonlocal caps_abs
            caps_abs = [c for c, _ in caps.get(evdev.ecodes.EV_ABS, [])]
            raise OSError("spy")

    P.UInput = _Spy
    try:
        try:
            P.VirtualPad().__enter__()
        except Exception:
            pass
    finally:
        P.UInput = ui
    hats = {evdev.ecodes.ABS_HAT0X, evdev.ecodes.ABS_HAT0Y}
    assert not hats & set(caps_abs), (
        "the pad declared a POV hat; a centred hat holds 'up' forever")
    assert evdev.ecodes.ABS_X in caps_abs and evdev.ecodes.ABS_Y in caps_abs, (
        "udev needs absolute X/Y to classify this as a joystick at all")


def test_game_can_never_be_launched_before_the_pad_exists():
    """The one ordering constraint that fails silently.

    Wine's dinput enumerates devices at init, so a pad created after the game
    starts is invisible for the whole session -- the agent presses buttons and
    the game ignores every one of them, which reads as a broken pad rather than
    a broken sequence. This has already been broken once, by moving four lines
    above a `with` block. Asserted on the syntax tree because the failure cannot
    be reproduced in a unit test without a real game.
    """
    import ast
    import pathlib
    src = pathlib.Path(__file__).parent.parent / "sokubot/live/agent.py"
    fn = next(n for n in ast.walk(ast.parse(src.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    all_launches = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", "") == "launch_game"]
    pad_block = next(w for w in ast.walk(fn) if isinstance(w, ast.With)
                     and "VirtualPad" in ast.dump(w.items[0]))
    inside = [n for n in ast.walk(pad_block) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "launch_game"]
    assert all_launches, "expected run() to call launch_game"
    assert len(inside) == len(all_launches), (
        "launch_game() escaped the VirtualPad block: the game would start "
        "before the pad exists and dinput would never see it")


def test_pad_indices_match_the_policy():
    from sokubot.live.pad import IDX
    from sokubot.rl import policy as P
    assert (IDX["up"], IDX["down"], IDX["left"], IDX["right"]) == (
        P.IDX_UP, P.IDX_DOWN, P.IDX_LEFT, P.IDX_RIGHT)
    assert tuple(IDX[b] for b in ("a", "b", "c", "d", "change", "spell")) == \
        P.FREE_BUTTONS
