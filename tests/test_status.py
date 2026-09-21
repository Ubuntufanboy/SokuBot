"""The status channel: what the overlay is allowed to believe.

The failure that matters is a display saying ARMED for a harness that died. These
tests are mostly about the reader refusing to trust a file.

    python -m pytest tests/test_status.py -q
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from sokubot.live import status as st


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def p(tmp_path):
    return tmp_path / "status.json"


def _read(p, clock, alive=lambda pid: True, **kw):
    return st.read_status(p, clock=clock, alive=alive, **kw)


# --- writing ---------------------------------------------------------------
def test_update_writes_a_file_the_reader_can_use(p):
    clk = Clock()
    w = st.StatusWriter(p, clock=clk)
    assert w.update(armed=True, slot=2, server_ok=True)
    s = _read(p, clk)
    assert (s.armed, s.slot, s.server_ok) == (True, 2, True)
    assert s.pid == os.getpid()


def test_a_rate_limited_update_is_merged_not_dropped(p):
    clk = Clock()
    w = st.StatusWriter(p, clock=clk, min_interval_s=0.25)
    assert w.update(slot=2)                      # written
    clk.t += 0.05
    assert not w.update(armed=True)              # inside the interval: held
    assert _read(p, clk).armed is False          # the file has not seen it...
    clk.t += 0.30
    assert w.update(server_ok=True)              # ...and the next write carries it
    s = _read(p, clk)
    assert (s.slot, s.armed, s.server_ok) == (2, True, True)


def test_force_writes_inside_the_interval(p):
    clk = Clock()
    w = st.StatusWriter(p, clock=clk)
    w.update(slot=1)
    assert w.update(force=True, armed=True)
    assert _read(p, clk).armed is True


def test_an_unknown_field_is_an_error_not_a_silent_no_op(p):
    w = st.StatusWriter(p, clock=Clock())
    with pytest.raises(TypeError, match="unknown status field"):
        w.update(armd=True)                      # a typo must not vanish


def test_no_temp_file_is_left_behind(p):
    w = st.StatusWriter(p, clock=Clock(), min_interval_s=0)
    for i in range(20):
        w.update(rounds=[i, 0])
    assert sorted(x.name for x in p.parent.iterdir()) == [p.name]


def test_concurrent_writers_lose_nothing(p):
    w = st.StatusWriter(p, clock=Clock(), min_interval_s=0)

    def hammer(name, val):
        for _ in range(200):
            w.update(**{name: val})

    ts = [threading.Thread(target=hammer, args=("armed", True)),
          threading.Thread(target=hammer, args=("server_ok", True)),
          threading.Thread(target=hammer, args=("slot", 2))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    w.update(force=True)
    d = json.loads(p.read_text())
    assert (d["armed"], d["server_ok"], d["slot"]) == (True, True, 2)


# --- the reader refuses to trust ------------------------------------------
def test_no_file_is_none(p):
    assert _read(p, Clock()) is None


def test_a_corrupt_file_is_none(p):
    p.write_text('{"schema": 1, "armed": tr')
    assert _read(p, Clock()) is None


def test_a_schema_it_does_not_know_is_none(p):
    p.write_text(json.dumps({"schema": 99, "armed": True, "pid": os.getpid(),
                             "updated": 1000.0}))
    assert _read(p, Clock()) is None


def test_a_dead_writer_is_none_even_if_the_file_is_fresh(p):
    clk = Clock()
    st.StatusWriter(p, clock=clk).update(armed=True, force=True)
    assert _read(p, clk, alive=lambda pid: False) is None


def test_a_silent_writer_goes_stale(p):
    clk = Clock()
    st.StatusWriter(p, clock=clk).update(armed=True, force=True)
    assert _read(p, clk) is not None
    clk.t += st.STALE_AFTER_S + 0.1
    assert _read(p, clk) is None


def test_a_heartbeat_keeps_an_idle_harness_alive(p):
    clk = Clock()
    w = st.StatusWriter(p, clock=clk)
    w.update(force=True)
    for _ in range(10):
        clk.t += 1.0
        assert w.heartbeat()
        assert _read(p, clk) is not None


def test_a_clean_close_is_offline_immediately(p):
    clk = Clock()
    w = st.StatusWriter(p, clock=clk)
    w.update(armed=True, server_ok=True, force=True)
    assert st.read_status(p, clock=clk) is not None      # real pid check: alive
    w.close()
    # Real liveness check, not the permissive fake: close() releases the pid, so
    # the reader must drop it without waiting out STALE_AFTER_S.
    assert st.read_status(p, clock=clk) is None


def test_pid_alive_knows_itself_and_a_reaped_child():
    import subprocess
    assert st.pid_alive(os.getpid())
    assert not st.pid_alive(0) and not st.pid_alive(-5)
    c = subprocess.Popen(["true"])
    c.wait()
    assert not st.pid_alive(c.pid)


# --- what the display says -------------------------------------------------
def _s(**kw):
    return st.Status(**kw)


def test_offline_is_not_off():
    assert st.summarise(None) == ("SokuBot: OFFLINE", "off")


def test_the_worst_true_thing_is_the_headline():
    # armed, in a battle, but the server is gone: that is not "PLAYING".
    head, level = st.summarise(_s(armed=True, gate_open=True, server_ok=False,
                                  slot=2))
    assert level == "warn" and "PLAYING" not in head and "NO SERVER" in head


def test_an_error_outranks_everything():
    head, level = st.summarise(_s(armed=True, gate_open=True, server_ok=True,
                                  last_error="capture lost"))
    assert level == "warn" and "capture lost" in head


@pytest.mark.parametrize("kw, word, level", [
    (dict(server_ok=True), "OFF", "idle"),
    (dict(server_ok=True, armed=True), "waiting for a battle", "idle"),
    (dict(server_ok=True, armed=True, gate_open=True), "PLAYING", "ok"),
])
def test_the_ordinary_states(kw, word, level):
    head, lv = st.summarise(_s(slot=2, **kw))
    assert word in head and lv == level and "P2" in head


# --- the default path is what actually runs -------------------------------------------
def test_the_writer_and_the_reader_agree_on_the_default_path(tmp_path, monkeypatch):
    """Every other test pins an explicit path. The harness and the overlay use NONE, and must
    land on the same file."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    w = st.StatusWriter()                                   # no path: what the harness does
    w.update(force=True, slot=2, armed=True, server_ok=True)
    assert w.path == st.default_path() == tmp_path / "sokubot.status.json"
    s = st.read_status()                                    # no path: what the overlay does
    assert s is not None and (s.slot, s.armed) == (2, True)


def test_the_overlay_process_reads_the_harnesss_default_file(tmp_path, monkeypatch):
    """End to end across a process boundary, with only the environment shared."""
    import subprocess
    import sys
    from pathlib import Path
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))     # restored afterwards, not deleted
    w = st.StatusWriter()                                    # this process is the live writer
    w.update(force=True, slot=2, armed=True, gate_open=True, server_ok=True)
    out = subprocess.run([sys.executable, "-m", "scripts.sokubot_overlay", "--once",
                          "--game-rect", "13,77,640,480"],
                         env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True,
                         text=True, timeout=60, cwd=str(Path(__file__).parent.parent))
    assert out.returncode == 0, out.stderr
    assert "PLAYING" in out.stdout and "P2" in out.stdout
