"""The null server, and through it the real client's wire handling.

`RemoteBrain` is the production client; these tests point it at a server that runs
no model. What they pin is what the game host can OBSERVE of a vision server, so a
later change to either side that breaks the contract fails here, with no GPU.

    python -m pytest tests/test_nullbrain.py -q
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from sokubot.live import wire
from sokubot.live.nullbrain import BUTTONS, N_STATE, NullBrain, NullServer

PAIR = np.zeros((480, 640, 6), np.uint8)


@pytest.fixture
def serve():
    started = []

    def go(**kw):
        srv = NullServer(NullBrain(**kw))
        srv.start()
        assert srv.ready.wait(5)
        started.append(srv)
        return srv
    yield go
    for s in started:
        s.stop()


def _client(srv):
    from scripts.play_cheat_match import RemoteBrain
    b = RemoteBrain("127.0.0.1", srv.port, size=224)
    b.spec = b.info()
    return b


def test_the_constants_are_shared_not_copied():
    import scripts.serve_vision as sv
    assert sv.HDR is wire.HDR and sv.recv_exactly is wire.recv_exactly
    assert (sv.OP_DECIDE, sv.OP_CAL, sv.OP_RESET) == (wire.OP_DECIDE, wire.OP_CAL, wire.OP_RESET)


def test_the_state_width_matches_the_real_channel_set():
    from sokubot.data.state import CH
    assert N_STATE == len(CH)


def test_info_gives_the_cadence_the_client_needs(serve):
    b = _client(serve())
    assert b.spec["ticks"] == 5 and b.spec["history"] == 12 and b.spec["size"] == 224
    assert b.spec["hud_floats"] == 4


def test_it_says_nothing_until_calibrated_like_the_real_server(serve):
    b = _client(serve())
    assert b.decide(PAIR) == (None, None)


def test_calibrate_answers_with_the_sentence_the_client_looks_for(serve):
    b = _client(serve())
    assert "agent is" in b.calibrate(PAIR, PAIR)


def test_after_calibration_decide_returns_a_chunk_and_the_encoder_reading(serve):
    b = _client(serve())
    b.calibrate(PAIR, PAIR)
    act, enc = b.decide(PAIR)
    assert act.shape == (5, 10) and not act.any()          # neutral: presses nothing
    assert enc.shape == (2, N_STATE)


def test_reset_forgets_identity(serve):
    b = _client(serve())
    b.calibrate(PAIR, PAIR)
    assert b.decide(PAIR)[0] is not None
    assert b.reset() == "ok"
    assert b.decide(PAIR) == (None, None)


def test_wiggle_alternates_so_a_live_test_can_see_the_pad_move(serve):
    b = _client(serve(mode="wiggle"))
    b.calibrate(PAIR, PAIR)
    first, second = b.decide(PAIR)[0], b.decide(PAIR)[0]
    assert first[:, BUTTONS.index("left")].all() and not first[:, BUTTONS.index("right")].any()
    assert second[:, BUTTONS.index("right")].all() and not second[:, BUTTONS.index("left")].any()


def test_a_slow_server_shows_up_in_the_clients_round_trip(serve):
    b = _client(serve(latency_ms=60))
    b.calibrate(PAIR, PAIR)
    b.decide(PAIR)
    assert b.rtt_ms >= 50


def test_a_dropped_connection_is_an_error_at_the_client_not_silence(serve):
    srv = serve(drop_after=2)
    b = _client(srv)
    b.calibrate(PAIR, PAIR)
    b.decide(PAIR); b.decide(PAIR)
    with pytest.raises((ConnectionError, OSError)):
        b.decide(PAIR)                                       # the third is dropped


def test_the_server_keeps_listening_after_a_drop_so_a_reconnect_finds_it(serve):
    srv = serve(drop_after=1)
    b = _client(srv)
    b.calibrate(PAIR, PAIR)
    b.decide(PAIR)
    with pytest.raises((ConnectionError, OSError)):
        b.decide(PAIR)
    b2 = _client(srv)                                         # a fresh connection works
    assert b2.info()["ticks"] == 5
    assert srv.connections == 2


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        NullBrain(mode="mash")
