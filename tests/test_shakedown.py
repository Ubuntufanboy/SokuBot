"""The pure parts of the live shakedown driver (the rest needs the game and a human).

    python -m pytest tests/test_shakedown.py -q
"""

from __future__ import annotations

from scripts.shakedown import flow_complete, parse_window
from sokubot.live.overlay_layout import Rect

XWININFO = """
xwininfo: Window id: 0x6400001 "Touhou Hisoutensoku ver1.10a (eng v1.1a)"
  Absolute upper-left X:  13
  Absolute upper-left Y:  77
  Relative upper-left X:  0
  Relative upper-left Y:  0
  Width: 640
  Height: 480
  Map State: IsViewable
"""


def test_a_viewable_window_is_parsed():
    assert parse_window(XWININFO) == Rect(13, 77, 640, 480)


def test_an_unmapped_window_is_ignored():
    assert parse_window(XWININFO.replace("IsViewable", "IsUnMapped")) is None


def test_garbage_is_none_not_a_crash():
    assert parse_window("IsViewable but nothing else") is None


def test_the_flow_needs_play_a_hand_back_and_a_re_arm():
    seen = {"playing": 1, "handed_back": 1, "rearm_after_handback": 0, "recovered": 1}
    assert not flow_complete(seen, outage=True)
    seen["rearm_after_handback"] = 1
    assert flow_complete(seen, outage=True)


def test_the_outage_is_only_required_when_one_is_injected():
    seen = {"playing": 1, "handed_back": 1, "rearm_after_handback": 1, "recovered": 0}
    assert not flow_complete(seen, outage=True)
    assert flow_complete(seen, outage=False)
