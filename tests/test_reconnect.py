"""A server that dies must not leave the agent looking armed, and must be found again.

Client level: `RemoteBrain` against the null server. Session level: the real control loop
with that client, so the whole chain -- drop, auto-disarm, probe, reconnect, re-identify,
re-arm -- is exercised as one.

    python -m pytest tests/test_reconnect.py -q
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from sokubot.live.nullbrain import NullBrain, NullServer

PAIR = np.zeros((64, 64, 6), np.uint8)


@pytest.fixture
def server():
    made = []

    def go(**kw):
        srv = NullServer(NullBrain(**kw)); srv.start(); assert srv.ready.wait(5)
        made.append(srv); return srv
    yield go
    for s in made:
        s.stop()


def client(srv):
    from scripts.play_cheat_match import RemoteBrain
    b = RemoteBrain("127.0.0.1", srv.port, size=224)
    b.spec = b.info()
    return b


# --- the client --------------------------------------------------------------
def test_the_next_call_after_a_drop_reconnects_by_itself(server):
    srv = server(drop_after=1)
    b = client(srv)
    b.calibrate(PAIR, PAIR); b.decide(PAIR)                # answered
    with pytest.raises(OSError):
        b.decide(PAIR)                                     # dropped
    assert b.ok is False
    srv.brain.drop_after = None                            # the server is healthy again
    assert b.ping().startswith("decides")                  # no manual reconnect needed
    assert b.ok is True and srv.connections == 2


def test_a_reconnect_forgets_identity_because_the_server_does(server):
    srv = server(drop_after=1)
    b = client(srv)
    b.calibrate(PAIR, PAIR)
    assert b.identified is True
    b.decide(PAIR)
    with pytest.raises(OSError):
        b.decide(PAIR)
    srv.brain.drop_after = None
    b.ping()
    assert b.identified is False
    assert b.decide(PAIR) == (None, None)                  # the server really did forget


def test_a_dead_server_is_retried_with_a_growing_backoff_not_hammered(server):
    srv = server()
    b = client(srv)
    srv.stop()
    with pytest.raises(OSError):
        b.ping()                                           # the connection is gone
    with pytest.raises(OSError):
        b.ping()                                           # first reconnect: refused
    assert b._backoff == 1.0
    t = time.perf_counter()
    with pytest.raises(ConnectionError, match="next reconnect in"):
        b.ping()                                           # inside the backoff: no attempt
    assert time.perf_counter() - t < 0.05
    for want in (2.0, 4.0, 4.0):                           # doubles, then caps
        b._next_try = 0.0
        with pytest.raises(OSError):
            b.ping()
        assert b._backoff == want


def test_a_server_that_came_back_with_a_different_cadence_is_refused(server):
    srv = server(drop_after=0)
    b = client(srv)
    b.calibrate(PAIR, PAIR)
    with pytest.raises(OSError):
        b.decide(PAIR)
    srv.brain.drop_after = None
    srv.brain.ticks = 6                                    # "restarted" with another checkpoint
    with pytest.raises(ConnectionError, match="changed its ticks"):
        b.ping()
    assert b.ok is False and "restart the client" in b.last_error


def test_concurrent_callers_do_not_read_each_others_replies(server):
    b = client(server(latency_ms=1))
    bad = []

    def hammer():
        for _ in range(60):
            try:
                r = b.ping()
                if not r.startswith("decides"):
                    bad.append(r)
            except Exception as e:                         # noqa: BLE001
                bad.append(repr(e))
    ts = [threading.Thread(target=hammer) for _ in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert bad == []


# --- the whole chain, through the real session loop -------------------------
@pytest.fixture
def chain(loop, server):
    class FakeSource:
        dead, still_for, fps, sink, vs = None, 0.0, 60, None, None

        def __init__(self, brain):
            self.brain = brain

        def _pair(self):
            return PAIR

        def latest(self):
            return PAIR[:, :, 3:]
    loop.pcm.VisionSource = FakeSource
    srv = server(drop_after=3)
    brain = client(srv)
    brain.calibrate(PAIR, PAIR)                            # identity established
    loop.srv, loop.brain = srv, brain
    loop.start(ls=FakeSource(brain))
    return loop


def _rearm_after_forgetting(lp):
    """The shared ending: it has forgotten the agent, so arm is refused with the reason,
    and works once identity is re-established."""
    lp.send("arm")
    s = lp.wait(lambda s: "identity unknown" in s.last_error)
    assert s.armed is False
    lp.brain.calibrate(PAIR, PAIR)                         # `whoami`
    lp.send("arm")
    assert lp.wait(lambda s: s.armed, timeout=5).armed


def test_decides_that_start_failing_disarm_the_agent_and_it_can_be_re_armed(chain):
    """The server stays reachable but drops every decide: found again at once."""
    lp = chain
    lp.wait(lambda s: s.slot == 2 and s.server_ok is True)
    lp.send("arm")
    lp.wait(lambda s: s.armed)
    # ~3 decisions later every decide is dropped. The loop must stop the agent and
    # say why, and once the probe reconnects the message must say the server is BACK.
    s = lp.wait(lambda s: (not s.armed) and "forgot the agent" in s.last_error, timeout=8)
    assert s.server_ok is True
    lp.srv.brain.drop_after = None
    _rearm_after_forgetting(lp)


def test_a_server_that_is_really_down_shows_no_server_then_recovers(chain):
    lp = chain
    lp.wait(lambda s: s.slot == 2 and s.server_ok is True)
    port = lp.srv.port
    lp.send("arm")
    lp.wait(lambda s: s.armed)
    lp.srv.stop()                                          # the process dies
    s = lp.wait(lambda s: (not s.armed) and s.server_ok is False, timeout=8)
    assert "server" in s.last_error
    time.sleep(1.5)                                        # still down: it keeps being retried
    assert lp.status().server_ok is False and lp.status().armed is False
    revived = NullServer(NullBrain(), port=port)           # restarted on the same port
    revived.start(); assert revived.ready.wait(5)
    try:
        lp.wait(lambda s: s.server_ok is True and not s.armed, timeout=10)
        _rearm_after_forgetting(lp)
    finally:
        revived.stop()
