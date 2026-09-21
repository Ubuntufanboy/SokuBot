"""The agent presses nothing unless a battle is on screen, and hands back when one ends.

Two independent conditions (gate.py): armed AND in a battle. These run the REAL session loop, once
with a gate the test controls (exact) and once with the real BattleGate on real-shaped frames
(menu -> battle -> menu), so the frame slicing and orientation are exercised too.

    python -m pytest tests/test_gate_session.py -q
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sokubot.live import status as st
from sokubot.live.gate import BattleGate as RealBattleGate

from test_live import _frame          # a 480x480 capture-orientation frame with health bars

from conftest import FakeGate


class Brain:
    """Identified already, so arming is not refused; counts decisions."""
    spec = {"history": 1, "ticks": 1}
    ok, last_error, rtt_ms, identified = True, "", 0.0, True

    def __init__(self):
        self.decides = 0

    def decide(self, pair):
        self.decides += 1
        time.sleep(0.002)
        return np.ones((1, 10), np.float32), None      # presses everything: easy to see

    def close(self):
        pass


@pytest.fixture
def battle(loop):
    """A vision session whose frames the test switches between menu and battle."""
    class Source:
        dead, still_for, fps, sink, vs = None, 0.0, 60, None, None

        def __init__(self):
            self.brain = Brain()
            self.in_battle = True
            self._battle = np.concatenate([_frame(1.0, 1.0), _frame(1.0, 1.0)], axis=2)
            self._menu = np.zeros((480, 480, 6), np.uint8)

        def _pair(self):
            return self._battle if self.in_battle else self._menu

    loop.pcm.VisionSource = Source
    src = Source()
    loop.src = src
    return loop, src


def start(lp, src):
    lp.start(ls=src)
    return lp.wait(lambda s: s.slot == 2)


# --- with a gate the test controls -------------------------------------------------
def test_armed_with_no_battle_presses_nothing_decides_nothing_and_says_waiting(battle):
    lp, src = battle
    FakeGate.open = False
    start(lp, src)
    lp.send("arm")
    s = lp.wait(lambda s: s.armed and s.gate_open is False)
    assert "waiting for a battle" in st.summarise(s)[0]
    time.sleep(0.6)
    assert lp.pad.calls == 0                               # not one button
    assert src.brain.decides == 0                          # and the pilot is held: no decisions


def test_when_the_battle_starts_it_plays(battle):
    lp, src = battle
    FakeGate.open = False
    start(lp, src)
    lp.send("arm"); lp.wait(lambda s: s.armed and s.gate_open is False)
    FakeGate.open = True
    s = lp.wait(lambda s: s.gate_open is True and s.latency_p50_ms is not None)
    assert "PLAYING" in st.summarise(s)[0]
    assert lp.pad.calls > 0 and src.brain.decides > 0


def test_a_battle_that_ends_hands_the_agent_back(battle):
    lp, src = battle
    FakeGate.open = True
    start(lp, src)
    lp.send("arm"); lp.wait(lambda s: s.armed and s.gate_open is True)
    lp.wait(lambda s: s.latency_p50_ms is not None)        # it really is playing
    FakeGate.open = False                                  # the set ended
    s = lp.wait(lambda s: (not s.armed) and "battle ended" in s.last_error, timeout=5)
    assert s.gate_open is False
    calls = lp.pad.calls
    time.sleep(0.4)
    assert lp.pad.calls == calls                           # it stopped pressing


def test_armed_before_any_battle_stays_armed_it_is_waiting_not_finished(battle):
    lp, src = battle
    FakeGate.open = False
    start(lp, src)
    lp.send("arm"); lp.wait(lambda s: s.armed)
    time.sleep(1.0)
    s = lp.status()
    assert s.armed is True and s.last_error == ""


def test_a_second_arm_after_a_hand_back_plays_the_next_battle(battle):
    lp, src = battle
    FakeGate.open = True
    start(lp, src)
    lp.send("arm"); lp.wait(lambda s: s.armed and s.latency_p50_ms is not None)
    FakeGate.open = False
    lp.wait(lambda s: (not s.armed) and "battle ended" in s.last_error, timeout=5)
    FakeGate.open = True                                   # the next set starts
    lp.send("arm")
    s = lp.wait(lambda s: s.armed and s.gate_open is True and s.last_error == "", timeout=5)
    assert s.armed


def test_a_run_with_no_frames_has_no_gate_and_claims_nothing(loop):
    loop.start()                                           # the plain non-vision session
    s = loop.wait(lambda s: s.slot == 2)
    assert s.gate_open is None
    loop.send("arm")
    assert "PLAYING" in st.summarise(loop.wait(lambda s: s.armed))[0]   # not "waiting for a battle"


def test_the_gate_is_built_to_hold_through_a_ko_flash(battle):
    """2 s to close at 10 Hz: longer than a KO flash or a super's screen wash (~0.33 s)."""
    lp, src = battle
    seen = {}
    orig = FakeGate.__init__

    def spy(self, *a, **k):
        seen.update(k); orig(self, *a, **k)
    FakeGate.__init__ = spy
    try:
        start(lp, src)
    finally:
        FakeGate.__init__ = orig
    import scripts.play_cheat_match as pcm
    assert seen.get("off_frames") == pcm.GATE_OFF_FRAMES == 20
    assert pcm.GATE_OFF_FRAMES * pcm.GATE_PERIOD_S > 0.33 * 3


# --- with the REAL gate on real-shaped frames --------------------------------------
def test_the_real_gate_sees_a_battle_frame_and_a_menu_frame_through_the_session(battle, monkeypatch):
    lp, src = battle
    monkeypatch.setattr(lp.pcm, "BattleGate", RealBattleGate)
    src.in_battle = False
    start(lp, src)
    lp.send("arm")
    s = lp.wait(lambda s: s.armed and s.gate_open is False)          # a menu is not a battle
    src.in_battle = True                                             # bars appear
    s = lp.wait(lambda s: s.gate_open is True and s.latency_p50_ms is not None, timeout=6)
    assert lp.pad.calls > 0
    src.in_battle = False                                            # back to a menu
    s = lp.wait(lambda s: (not s.armed) and "battle ended" in s.last_error, timeout=8)
    assert s.gate_open is False
