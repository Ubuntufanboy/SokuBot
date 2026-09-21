"""The overlay must never sit on the game's picture.

Capture grabs the game window's rectangle from the screen, so anything drawn there is
fed to the agent as if it were the game. These tests are the guarantee.

    python -m pytest tests/test_overlay_layout.py -q
"""

from __future__ import annotations

import random

import pytest

from sokubot.live.overlay_layout import Rect, overlaps, place_outside

SCREEN = (1920, 1080)
BAR = (420, 44)


def test_the_real_setup_puts_it_just_below_the_game():
    # This laptop: a 640x480 game client at +13+77 on a 1920x1080 screen.
    x, y = place_outside(Rect(13, 77, 640, 480), SCREEN, BAR)
    assert (x, y) == (13, 77 + 480 + 12)


def test_no_game_window_yet_uses_a_corner():
    x, y = place_outside(None, SCREEN, BAR)
    assert 0 <= x and 0 <= y and x + BAR[0] <= SCREEN[0] and y + BAR[1] <= SCREEN[1]


def test_it_falls_back_above_then_to_the_side_when_below_has_no_room():
    game = Rect(100, 900, 640, 170)                        # hugging the bottom
    x, y = place_outside(game, SCREEN, BAR)
    assert y + BAR[1] <= game.y                            # went above


def test_a_game_that_fills_the_screen_leaves_no_place_so_it_is_hidden():
    assert place_outside(Rect(0, 0, 1920, 1080), SCREEN, BAR) is None


def test_a_bar_bigger_than_the_screen_is_hidden():
    assert place_outside(None, (300, 30), BAR) is None


def test_overlaps_counts_the_margin():
    a, b = Rect(0, 0, 100, 100), Rect(110, 0, 100, 100)
    assert not overlaps(a, b, 0) and not overlaps(a, b, 10)
    assert overlaps(a, b, 11)                              # 10 px apart is inside an 11 px margin


def test_touching_edges_do_not_count_as_overlap_with_no_margin():
    assert not overlaps(Rect(0, 0, 10, 10), Rect(10, 0, 10, 10))


def test_over_thousands_of_layouts_it_never_touches_the_game_and_never_leaves_the_screen():
    rng = random.Random(20260920)
    placed = hidden = 0
    for _ in range(4000):
        sw, sh = rng.choice([(1920, 1080), (1366, 768), (2560, 1440), (1024, 768), (800, 600)])
        gw, gh = rng.randint(200, min(sw, 1400)), rng.randint(150, min(sh, 900))
        game = Rect(rng.randint(-50, sw - 50), rng.randint(-50, sh - 50), gw, gh)
        size = (rng.randint(150, 500), rng.randint(24, 90))
        margin = rng.choice([0, 4, 12, 24])
        pos = place_outside(game, (sw, sh), size, margin)
        if pos is None:
            hidden += 1
            continue
        placed += 1
        r = Rect(*pos, *size)
        assert 0 <= r.x and 0 <= r.y and r.right <= sw and r.bottom <= sh, (game, size, pos)
        assert not overlaps(r, game, margin), (game, size, margin, pos)
    assert placed > 2000                                   # the test is not vacuous


# --- the text the strip shows ----------------------------------------------
def test_the_second_line_is_empty_until_there_are_decisions():
    from scripts.sokubot_overlay import detail
    from sokubot.live.status import Status
    assert detail(None) == "" and detail(Status()) == ""


def test_the_second_line_carries_latency_and_missed_slots():
    from scripts.sokubot_overlay import detail
    from sokubot.live.status import Status
    d = detail(Status(latency_p50_ms=52.4, latency_p99_ms=71.0, missed_slot_pct=1.2))
    assert "p50 52" in d and "p99 71" in d and "missed 1%" in d


def test_the_parser_for_a_forced_game_rect():
    from scripts.sokubot_overlay import parse_rect
    assert parse_rect("13,77,640,480") == Rect(13, 77, 640, 480)
