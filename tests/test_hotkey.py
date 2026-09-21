"""The hotkey watcher: what counts as a press, and what must never count.

    python -m pytest tests/test_hotkey.py -q
"""

from __future__ import annotations

import threading
import time

import pytest

from sokubot.live.hotkey import DEFAULT_KEY, EV_KEY, KeyWatcher

F12 = 88            # KEY_F12
DOWN, UP, REPEAT = 1, 0, 2


class Feed:
    """A poll() that hands out queued events, then idles."""

    def __init__(self):
        self.q, self.lock = [], threading.Lock()

    def push(self, *events):
        with self.lock:
            self.q.extend(events)

    def __call__(self, timeout):
        with self.lock:
            out, self.q = self.q, []
        if not out:
            time.sleep(min(timeout, 0.01))
        return out


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def run(feed, clock=None, **kw):
    calls = []
    w = KeyWatcher(lambda: calls.append(1), F12, feed, clock=clock or Clock(), **kw)
    w.start()
    return w, calls


def settle(w, n):
    end = time.time() + 2
    while w.presses < n and time.time() < end:
        time.sleep(0.005)
    time.sleep(0.05)


def test_a_key_down_is_a_press():
    feed = Feed(); w, calls = run(feed)
    feed.push((EV_KEY, F12, DOWN)); settle(w, 1); w.stop()
    assert len(calls) == 1


@pytest.mark.parametrize("ev", [
    (EV_KEY, F12, UP),                 # release
    (EV_KEY, F12, REPEAT),             # autorepeat while held
    (EV_KEY, F12 + 1, DOWN),           # a different key
    (0, F12, DOWN),                    # not a key event at all (EV_SYN)
    (3, F12, DOWN),                    # an axis event
])
def test_only_a_key_down_of_the_right_key_counts(ev):
    feed = Feed(); w, calls = run(feed)
    feed.push(ev); time.sleep(0.15); w.stop()
    assert calls == []


def test_holding_the_key_is_one_press_not_many():
    feed = Feed(); w, calls = run(feed)
    feed.push((EV_KEY, F12, DOWN), (EV_KEY, F12, REPEAT), (EV_KEY, F12, REPEAT),
              (EV_KEY, F12, UP))
    settle(w, 1); w.stop()
    assert len(calls) == 1


def test_bounce_inside_the_debounce_window_is_one_press():
    feed, clock = Feed(), Clock()
    w, calls = run(feed, clock, debounce_s=0.3)
    feed.push((EV_KEY, F12, DOWN)); settle(w, 1)
    clock.t += 0.1
    feed.push((EV_KEY, F12, DOWN)); time.sleep(0.1)         # a bounce
    assert len(calls) == 1
    clock.t += 0.5
    feed.push((EV_KEY, F12, DOWN)); settle(w, 2); w.stop()  # a real second press
    assert len(calls) == 2


def test_a_callback_that_raises_does_not_end_the_hotkey(capsys):
    feed, clock = Feed(), Clock()
    n = {"c": 0}

    def cb():
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("boom")
    w = KeyWatcher(cb, F12, feed, clock=clock)
    w.start()
    feed.push((EV_KEY, F12, DOWN)); settle(w, 1)
    clock.t += 1.0
    feed.push((EV_KEY, F12, DOWN)); settle(w, 2); w.stop()
    assert n["c"] == 2
    assert "hotkey callback failed" in capsys.readouterr().out


def test_stop_ends_the_thread():
    w, _ = run(Feed()); w.stop()
    assert not w.is_alive()


def test_no_keyboards_is_none_not_a_silent_watcher(monkeypatch):
    import sys, types
    fake = types.SimpleNamespace(list_devices=lambda: [],
                                 ecodes=types.SimpleNamespace(ecodes={DEFAULT_KEY: F12}))
    monkeypatch.setitem(sys.modules, "evdev", fake)
    assert KeyWatcher.for_evdev(lambda: None) is None


def test_the_default_key_collides_with_no_pad_key_or_giuroll_key():
    from sokubot.live.pad import KEYPAD_CODES
    from sokubot.live.profiles import EVDEV_TO_DIK
    assert EVDEV_TO_DIK[DEFAULT_KEY] not in {0x09, 0x0A, 0x0B, 0x10, 0x13, 0x21, 0x22}
    assert DEFAULT_KEY not in {k for _, k in KEYPAD_CODES}


def test_the_default_key_is_bound_by_none_of_the_installed_profiles():
    """The key reaches the game too (dinput ignores focus), so it must be inert there."""
    from pathlib import Path
    from sokubot.live import profiles as pf
    game = Path("~/.wine-soku/drive_c/Games/Soku").expanduser()
    if not (game / "profile").is_dir():
        pytest.skip("no Soku install on this machine")
    bound = {name: [c for c, k in p.keys().items() if k == DEFAULT_KEY]
             for name, p in pf.load_all(game).items()}
    assert {n: c for n, c in bound.items() if c} == {}
