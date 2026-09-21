"""One key press does the whole thing: calibrate, then arm.

Arming with an unknown identity is refused, and the fix used to be typing `whoami` in a
terminal -- which the game reads as game input. The hotkey now calibrates on a worker thread
and then arms. The property that is easy to break is the HEARTBEAT: calibration takes >= 3 s,
which is the status staleness limit, so a calibration run inline would flip the overlay to
OFFLINE in the middle of it.

    python -m pytest tests/test_calibrate.py -q
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from sokubot.live import status as st
from sokubot.live.nullbrain import NullBrain, NullServer

PAIR = np.zeros((64, 64, 6), np.uint8)


class FakeWatcher:
    instances = []

    def __init__(self, on_press, key, exclude):
        self.on_press = on_press
        FakeWatcher.instances.append(self)

    @classmethod
    def for_evdev(cls, on_press, key="KEY_F12", exclude=None):
        return cls(on_press, key, exclude)

    def start(self): pass
    def stop(self): pass


@pytest.fixture
def rig(loop, monkeypatch):
    """A real session with a real RemoteBrain (null server), identity NOT yet established."""
    made = []

    class FakeSource:
        dead, still_for, fps, sink, vs = None, 0.0, 60, None, None

        def __init__(self, brain):
            self.brain = brain

        def _pair(self):
            return PAIR

        def latest(self):
            return PAIR[:, :, 3:]

    def build(hold_s=0.05, **server_kw):
        srv = NullServer(NullBrain(**server_kw)); srv.start(); assert srv.ready.wait(5)
        made.append(srv)
        from scripts.play_cheat_match import RemoteBrain
        brain = RemoteBrain("127.0.0.1", srv.port, size=224)
        brain.spec = brain.info()
        calls = []
        orig = loop.pcm.remote_calibrate

        def short(src, b, pad, hold_s_=1.0):
            calls.append(1)
            return orig(src, b, pad, hold_s=hold_s)
        monkeypatch.setattr(loop.pcm, "remote_calibrate", short)
        monkeypatch.setattr(loop.pcm, "VisionSource", FakeSource)
        FakeWatcher.instances.clear()
        monkeypatch.setattr(loop.pcm, "KeyWatcher", FakeWatcher)
        loop.no_hotkey = False
        loop.brain, loop.calls, loop.srv = brain, calls, srv
        loop.start(ls=FakeSource(brain))
        loop.wait(lambda s: s.slot == 2)
        loop.key = FakeWatcher.instances[0].on_press
        return loop
    yield build
    for s in made:
        s.stop()


def test_one_press_calibrates_and_then_arms(rig):
    lp = rig()
    assert lp.brain.identified is False
    lp.key()
    lp.wait(lambda s: "CALIBRATING" in s.busy)                 # the player is told to keep hands off
    s = lp.wait(lambda s: s.armed and s.busy == "", timeout=8)
    assert lp.brain.identified is True
    assert lp.pad.pressed[:2] == ["left", "right"]             # it really swept the pad
    assert s.last_error == ""


def test_the_headline_says_calibrating_not_error(rig):
    lp = rig(hold_s=0.3)
    lp.key()
    s = lp.wait(lambda s: "CALIBRATING" in s.busy)
    head, level = st.summarise(s)
    assert "CALIBRATING" in head and "ERROR" not in head and level == "warn"
    lp.wait(lambda s: s.armed, timeout=10)


def test_a_second_press_during_calibration_does_not_start_another(rig):
    lp = rig(hold_s=0.3)
    lp.key(); lp.wait(lambda s: "CALIBRATING" in s.busy)
    lp.key(); lp.key()                                         # impatient
    lp.wait(lambda s: s.armed, timeout=10)
    assert len(lp.calls) == 1


def test_a_failed_calibration_says_so_and_does_not_arm(rig):
    lp = rig(calibration_fails=True)
    lp.key()
    s = lp.wait(lambda s: s.busy == "" and "calibration failed" in s.last_error, timeout=8)
    assert s.armed is False and "no movement" in s.last_error
    time.sleep(0.4)
    assert lp.status().armed is False                          # and it stays disarmed


def test_with_identity_already_known_one_press_arms_with_no_calibration(rig):
    lp = rig()
    lp.brain.calibrate(PAIR, PAIR)                             # already identified
    lp.key()
    lp.wait(lambda s: s.armed)
    assert lp.calls == []


def test_whoami_calibrates_without_arming(rig):
    lp = rig()
    lp.send("whoami")
    lp.wait(lambda s: "CALIBRATING" in s.busy)
    lp.wait(lambda s: s.busy == "")
    assert lp.brain.identified is True and lp.status().armed is False


def test_the_status_never_goes_offline_during_a_calibration_longer_than_the_staleness_limit(rig):
    """The whole reason it runs on a worker thread. 3 s of sweeping vs a 3.0 s staleness limit."""
    lp = rig(hold_s=1.3)                                       # sweep = 3.9 s > STALE_AFTER_S
    assert 3 * 1.3 > st.STALE_AFTER_S
    lp.send("whoami")
    lp.wait(lambda s: "CALIBRATING" in s.busy)
    t0 = time.time(); seen_offline = False
    while lp.status() is None or "CALIBRATING" in lp.status().busy:
        if lp.status() is None:
            seen_offline = True
        if time.time() - t0 > 12:
            break
        time.sleep(0.1)
    assert time.time() - t0 > st.STALE_AFTER_S                 # it really did outlast the limit
    assert not seen_offline
