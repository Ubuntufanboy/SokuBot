"""The status channel, wired into the real control loop.

`fields_from` is pure and tested directly. The important test drives the REAL
`session()` -- real command FIFO, real Pilot thread, real StatusWriter -- with a fake
pad and a fake decide(), and reads the file the overlay would read. That is the only
place a mis-wired update (a stale armed flag, a missing close) would show up.

    python -m pytest tests/test_session_status.py -q
"""

from __future__ import annotations

import os
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from sokubot.live import status as st


# --- fields_from, pure ------------------------------------------------------
class FakePilot:
    def __init__(self, armed=False, stop_reason=None, lat=(), late=0, decides=0,
                 period=1 / 12):
        from collections import deque
        self.armed = threading.Event()
        if armed:
            self.armed.set()
        self.stop_reason = stop_reason
        self.lat_ms = deque(lat, maxlen=20000)
        self.late, self.decides, self.period = late, decides, period


class FakeBrain:
    def __init__(self, ok=True, err=""):
        self.ok, self.last_error = ok, err


def test_a_local_run_reports_the_slot_and_no_server():
    f = st.fields_from(FakePilot(), None, 2)
    assert f["slot"] == 2 and "server_ok" not in f       # not applicable, so not written


def test_armed_comes_from_the_pilot_not_from_a_copy():
    assert st.fields_from(FakePilot(armed=True), None, 1)["armed"] is True
    assert st.fields_from(FakePilot(armed=False), None, 1)["armed"] is False


def test_a_dead_server_is_an_error_naming_why():
    f = st.fields_from(FakePilot(armed=True), FakeBrain(False, "timeout: timed out"), 2)
    assert f["server_ok"] is False and "timed out" in f["last_error"]


def test_a_watchdog_reason_is_shown_only_while_disarmed():
    reason = "10 decisions in a row over budget"
    assert reason in st.fields_from(FakePilot(armed=False, stop_reason=reason), None, 2)["last_error"]
    # After the user re-arms, stop_reason is still set (nothing clears it); it must
    # not sit next to an armed agent as if it were current.
    assert st.fields_from(FakePilot(armed=True, stop_reason=reason), None, 2)["last_error"] == ""


def test_the_server_error_outranks_a_stale_watchdog_reason():
    f = st.fields_from(FakePilot(armed=False, stop_reason="old"), FakeBrain(False, "gone"), 2)
    assert "gone" in f["last_error"] and "old" not in f["last_error"]


def test_latency_and_missed_slots_come_from_the_recorded_decisions():
    f = st.fields_from(FakePilot(lat=[40.0] * 98 + [300.0, 300.0], late=5, decides=100), None, 2)
    assert f["latency_p50_ms"] == 40.0 and f["latency_p99_ms"] > 83.3
    assert f["missed_slot_pct"] == pytest.approx(5.0)


def test_no_decisions_yet_is_none_not_zero():
    f = st.fields_from(FakePilot(), None, 2)
    assert f["latency_p50_ms"] is None and f["missed_slot_pct"] is None


def test_fields_it_has_no_source_for_are_never_returned():
    f = st.fields_from(FakePilot(armed=True), FakeBrain(), 2)
    assert not {"scene", "rounds", "gate_open"} & set(f)


def test_a_local_run_with_no_server_is_not_shown_as_a_failure():
    head, level = st.summarise(st.Status(slot=2, armed=True, gate_open=True))   # server_ok None
    assert "NO SERVER" not in head and level == "ok"


def test_the_remote_brain_knows_when_its_server_stops_answering():
    from sokubot.live.nullbrain import NullBrain, NullServer
    from scripts.play_cheat_match import RemoteBrain
    srv = NullServer(NullBrain(drop_after=0)); srv.start(); assert srv.ready.wait(5)
    try:
        b = RemoteBrain("127.0.0.1", srv.port, size=224)
        b.info()
        assert b.ok is True and b.last_error == ""
        b.spec = b.info()
        with pytest.raises(OSError):
            b.decide(np.zeros((480, 640, 6), np.uint8))          # dropped by the server
        assert b.ok is False and b.last_error
    finally:
        srv.stop()


# --- the real session loop --------------------------------------------------
# The `loop` fixture (a real session() with a fake pad and decide()) lives in conftest.py.


def test_the_overlay_sees_the_slot_before_anything_is_armed(loop):
    loop.start()
    s = loop.wait(lambda s: s.slot == 2)
    assert s.armed is False and s.server_ok is None and s.last_error == ""


def test_arm_and_disarm_reach_the_status_file(loop):
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    loop.send("arm")
    s = loop.wait(lambda s: s.armed)
    assert s.armed
    loop.wait(lambda s: s.latency_p50_ms is not None)         # the pilot is really deciding
    loop.send("disarm")
    assert loop.wait(lambda s: not s.armed).armed is False


def test_a_watchdog_stop_reaches_the_status_file_with_its_reason(loop):
    # period is 1/60 s (ticks=1); every decision takes 40 ms; trip after 3 in a row
    loop.start(decide_ms=40, late_run_limit=3)
    loop.wait(lambda s: s.slot == 2)
    loop.send("arm")
    s = loop.wait(lambda s: (not s.armed) and "over budget" in s.last_error)
    assert "over budget" in s.last_error


def test_a_clean_stop_makes_the_status_offline_immediately(loop):
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    assert loop.stop() == 0
    assert loop.status() is None                               # closed: no waiting out staleness


def test_a_state_change_is_forced_out_not_left_to_the_rate_limit(loop):
    """The overlay must never lag an arm or a disarm by up to a whole interval."""
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    time.sleep(0.4)                       # several idle iterations before anything changes
    loop.send("arm"); loop.wait(lambda s: s.armed)
    loop.send("disarm"); loop.wait(lambda s: not s.armed)
    log = loop.writer_log
    first_armed = next(f for f, armed in log if armed is True)
    assert first_armed is True, "the update that first said armed=True was rate-limited"
    # ...and the disarm transition too: the first armed=False AFTER an armed=True.
    i = next(i for i, (f, armed) in enumerate(log) if armed is True)
    first_disarm = next(f for f, armed in log[i:] if armed is False)
    assert first_disarm is True
    # Steady state is NOT forced: idle iterations exist and they are rate-limited
    # heartbeats. (Forcing every update would defeat the limiter and hammer the disk.)
    unforced = [f for f, _ in log if not f]
    assert len(unforced) >= 3


# --- notices, the identity guard, hands-off ---------------------------------
def test_a_refusal_reason_is_shown_only_while_disarmed():
    f = st.fields_from(FakePilot(armed=False), None, 2, notice="NOT ARMED: identity unknown")
    assert f["last_error"] == "NOT ARMED: identity unknown"
    assert st.fields_from(FakePilot(armed=True), None, 2, notice="stale")["last_error"] == ""


def test_a_real_error_outranks_a_refusal_notice():
    f = st.fields_from(FakePilot(armed=False), FakeBrain(False, "gone"), 2, notice="n")
    assert "gone" in f["last_error"] and f["last_error"] != "n"


def test_the_remote_brain_tracks_whether_the_server_knows_who_it_is():
    from sokubot.live.nullbrain import NullBrain, NullServer
    from scripts.play_cheat_match import RemoteBrain
    srv = NullServer(NullBrain()); srv.start(); assert srv.ready.wait(5)
    try:
        b = RemoteBrain("127.0.0.1", srv.port, size=224)
        pair = np.zeros((480, 640, 6), np.uint8)
        assert b.identified is False
        b.calibrate(pair, pair)
        assert b.identified is True
        b.reset()
        assert b.identified is False
    finally:
        srv.stop()


class FakeVisionBrain:
    """What the session needs of a RemoteBrain, with identity under test control."""
    spec = {"history": 1, "ticks": 1}
    ok, last_error, rtt_ms = True, "", 0.0

    def __init__(self):
        self.identified = False

    def decide(self, pair):
        time.sleep(0.002)
        return np.zeros((1, 10), np.float32), None

    def close(self):
        pass


@pytest.fixture
def remote_ls(loop):
    """A stand-in VisionSource so session() takes its real vision + remote branch."""
    class FakeSource:
        dead, still_for, fps, sink, vs = None, 0.0, 60, None, None

        def __init__(self):
            self.brain = FakeVisionBrain()

        def _pair(self):
            return np.zeros((4, 4, 6), np.uint8)

        def latest(self):
            return np.zeros((4, 4, 3), np.uint8)
    loop.pcm.VisionSource = FakeSource
    return FakeSource()


def test_arming_before_the_server_knows_the_agent_is_refused_and_says_why(loop, remote_ls):
    loop.start(ls=remote_ls)
    loop.wait(lambda s: s.slot == 2)
    loop.send("arm")
    s = loop.wait(lambda s: "identity unknown" in s.last_error)
    assert s.armed is False                       # it did NOT arm and go silent
    remote_ls.brain.identified = True             # `whoami` succeeded
    loop.send("arm")
    s = loop.wait(lambda s: s.armed)
    assert s.armed and s.last_error == ""         # and the reason is gone once armed


def test_hands_off_disarms_like_disarm(loop):
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    loop.send("arm"); loop.wait(lambda s: s.armed)
    loop.send("hands-off")
    assert loop.wait(lambda s: not s.armed).armed is False


class FakeWatcher:
    """Stands in for KeyWatcher; the test presses the key by calling on_press."""
    instances = []

    def __init__(self, on_press, key, exclude):
        self.on_press, self.key, self.exclude = on_press, key, exclude
        FakeWatcher.instances.append(self)

    @classmethod
    def for_evdev(cls, on_press, key="KEY_F12", exclude=None):
        return cls(on_press, key, exclude)

    def start(self): pass
    def stop(self): pass


def test_the_hotkey_toggles_the_real_session(loop):
    FakeWatcher.instances.clear()
    loop.pcm.KeyWatcher = FakeWatcher
    loop.no_hotkey = False
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    (w,) = FakeWatcher.instances
    w.on_press()                                   # press 1: arm
    assert loop.wait(lambda s: s.armed).armed is True
    w.on_press()                                   # press 2: disarm
    assert loop.wait(lambda s: not s.armed).armed is False


def test_one_press_re_arms_after_a_watchdog_stop_and_it_stays_armed(loop):
    """The pilot is the source of truth. After a watchdog stop ONE press must arm."""
    FakeWatcher.instances.clear()
    loop.pcm.KeyWatcher = FakeWatcher
    loop.no_hotkey = False
    loop.start(decide_ms=40, late_run_limit=3)     # every decision over a 16.7 ms period
    loop.wait(lambda s: s.slot == 2)
    (w,) = FakeWatcher.instances
    w.on_press()
    loop.wait(lambda s: s.armed)
    loop.wait(lambda s: (not s.armed) and "over budget" in s.last_error)   # the watchdog stopped it
    loop.decide_ms = 2                             # the machine recovers
    w.on_press()                                   # ONE press
    s = loop.wait(lambda s: s.armed and s.last_error == "")
    time.sleep(0.4)
    assert loop.status().armed is True             # ...and it does not trip again


def test_the_hotkey_is_told_to_ignore_the_agents_own_pad(loop):
    FakeWatcher.instances.clear()
    loop.pcm.KeyWatcher = FakeWatcher
    loop.no_hotkey = False
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    assert FakeWatcher.instances[0].exclude == "/dev/input/event99"


def test_no_readable_keyboard_does_not_stop_the_session(loop, capsys):
    loop.pcm.KeyWatcher = types.SimpleNamespace(for_evdev=lambda *a, **k: None)
    loop.no_hotkey = False
    loop.start()
    loop.wait(lambda s: s.slot == 2)
    assert loop.stop() == 0
    assert "hotkey UNAVAILABLE" in capsys.readouterr().out
