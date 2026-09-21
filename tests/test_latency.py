"""Latency accounting, and the property the Phase B gate depends on.

The gate arms the real pilot on the laptop and must be able to stop itself: a
sustained overrun has to disarm the pilot AND end the inference load, because the
alternative is a user frozen out of their own machine (2026-08-16, load 11 on four
cores). The second half of this file drives the real `Pilot` thread to prove that.

    python -m pytest tests/test_latency.py -q
"""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

from sokubot.live.latency import format_summary, summarise


# --- the arithmetic --------------------------------------------------------
def test_percentiles_of_a_known_series():
    s = summarise(range(1, 101), budget_ms=83.3)      # 1..100 ms
    assert s["n"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert s["p99"] == pytest.approx(99.01)
    assert s["max"] == 100.0
    assert s["over_budget"] == 17                     # 84..100
    assert s["over_budget_frac"] == pytest.approx(0.17)


def test_an_empty_series_is_none_not_zero():
    s = summarise([], 83.3)
    assert s["n"] == 0 and s["p50"] is None and s["p99"] is None
    assert "no decisions" in format_summary(s)


def test_a_comfortable_p50_does_not_hide_a_bad_p99():
    """The reason for percentiles: 98% fast decisions and two that stall."""
    s = summarise([40.0] * 98 + [400.0, 400.0], budget_ms=83.3)
    assert s["p50"] == 40.0
    assert s["p99"] > 83.3
    assert "OVER budget" in format_summary(s)


def test_a_loop_inside_its_budget_says_so():
    assert "WITHIN budget" in format_summary(summarise([40.0] * 50, 83.3))


# --- the real Pilot --------------------------------------------------------
@pytest.fixture
def pcm():
    import scripts.play_cheat_match as m
    return m


def _run_pilot(pcm, monkeypatch, decide_fn, seconds, *, limit=60, ticks=1):
    monkeypatch.setattr(pcm, "decide", decide_fn)
    src = types.SimpleNamespace()                     # no .dead, no .still_for
    p = pcm.Pilot(src, pol=object(), obs=object(), side=0, H=1, ticks=ticks)
    p.late_run_limit = limit
    p.start()
    p.armed.set()
    time.sleep(seconds)
    p.stopped.set()
    p.join(2.0)
    return p


def test_only_decisions_that_ran_the_policy_are_recorded(pcm, monkeypatch):
    calls = {"n": 0}

    def decide(ls, pol, obs, hist, side, H):
        calls["n"] += 1
        time.sleep(0.002)
        if calls["n"] <= 3:
            return None, object()             # history still filling: no policy
        return np.zeros((1, 10), np.float32), object()

    p = _run_pilot(pcm, monkeypatch, decide, 0.5)
    assert calls["n"] > 8
    assert len(p.lat_ms) == calls["n"] - 3    # the three early ones excluded
    assert min(p.lat_ms) >= 2.0               # each includes the 2 ms of "work"
    assert "latency:" in p.stats()


def test_a_sustained_overrun_disarms_and_ENDS_the_inference_load(pcm, monkeypatch):
    """The safety property. Period is 1/60 s; every decision takes 40 ms."""
    calls = {"n": 0}

    def slow(ls, pol, obs, hist, side, H):
        calls["n"] += 1
        time.sleep(0.040)
        return np.zeros((1, 10), np.float32), object()

    monkeypatch.setattr(pcm, "decide", slow)
    src = types.SimpleNamespace()
    p = pcm.Pilot(src, pol=object(), obs=object(), side=0, H=1, ticks=1)
    p.late_run_limit = 3
    p.start()
    p.armed.set()
    deadline = time.monotonic() + 3.0
    while p.armed.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not p.armed.is_set(), "the watchdog never disarmed a loop 2.4x over budget"
    assert "over budget" in p.stop_reason

    n_at_disarm = calls["n"]
    time.sleep(0.4)                           # ~10 more decisions if it were still running
    p.stopped.set()
    p.join(2.0)
    # At most the one decision that was already in flight when it disarmed.
    assert calls["n"] <= n_at_disarm + 1, "disarmed but still running inference"
    assert n_at_disarm <= 5                   # and it tripped promptly, not after 60


def test_a_loop_inside_its_budget_is_left_alone(pcm, monkeypatch):
    """No false trip: 2 ms decisions against a 16.7 ms period."""
    def fast(ls, pol, obs, hist, side, H):
        time.sleep(0.002)
        return np.zeros((1, 10), np.float32), object()

    p = _run_pilot(pcm, monkeypatch, fast, 0.5, limit=3)
    assert p.stop_reason is None
    assert p.late == 0
