"""Which slot the agent plays, decided from configuration and never guessed.

`--side` used to default to 0. When the agent was actually player 2 that fed the
opponent's health bar to the policy as its own, with no error anywhere. These
tests pin the replacement: the slot is read from `config123.dat`, and the run is
refused when that reading is ambiguous or unsafe.

    python -m pytest tests/test_profiles.py -q

No game, no Wine: a throwaway game directory is built from the byte layout that
`profiles.py` documents.
"""

from __future__ import annotations

import struct

import pytest

from sokubot.live import profiles as pf
from sokubot.live.pad import KEYPAD_CODES


def _pf(path, device=pf.KEYBOARD_DEVICE, keys=()):
    """Write a profile. `keys` = eleven evdev names in `pf.CONTROLS` order."""
    codes = [pf.EVDEV_TO_DIK[k] for k in keys] or [0] * 11
    path.write_bytes(struct.pack("<12I", device, *codes)
                     + b"\0" * (pf.PROFILE_BYTES - 48))


def _config(game, p1, p2):
    blob = b"\0\0\0\0"
    for name in (p1, p2):
        raw = name.encode()
        blob += struct.pack("<I", len(raw)) + raw
    (game / "config123.dat").write_bytes(blob + b"\0" * 32)


AGENT_KEYS = [k for _, k in KEYPAD_CODES] + ["KEY_Q"]      # the eleventh: pause
HUMAN_KEYS = ["KEY_UP", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT", "KEY_P", "KEY_O",
              "KEY_I", "KEY_F", "KEY_A", "KEY_S", "KEY_Q"]
# Shares the agent's T and K, which the pad really sends.
CLASHING_KEYS = ["KEY_T", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT", "KEY_K", "KEY_O",
                 "KEY_I", "KEY_F", "KEY_A", "KEY_S", "KEY_Q"]


@pytest.fixture
def game(tmp_path):
    (tmp_path / "profile").mkdir()
    _pf(tmp_path / "profile" / "sokubot.pf", keys=AGENT_KEYS)
    _pf(tmp_path / "profile" / "anon.pf", keys=HUMAN_KEYS)
    _pf(tmp_path / "profile" / "clash.pf", keys=CLASHING_KEYS)
    return tmp_path


def test_player_two_is_side_one(game):
    _config(game, "anon", "sokubot")
    side, rep = pf.resolve_side(game)
    assert side == 1
    assert rep["agent_slot"] == 2


def test_player_one_is_side_zero(game):
    _config(game, "sokubot", "anon")
    assert pf.resolve_side(game)[0] == 0


def test_agent_in_neither_slot_is_refused_not_defaulted(game):
    _config(game, "anon", "clash")
    with pytest.raises(pf.SlotError, match="neither slot"):
        pf.resolve_side(game)


def test_a_loaded_profile_that_shares_a_key_is_refused(game):
    _config(game, "clash", "sokubot")
    with pytest.raises(pf.SlotError, match="clash"):
        pf.resolve_side(game)


def test_a_colliding_profile_that_nobody_loaded_does_not_stop_the_run(game):
    """`clash` is installed and collides, but the human is on `anon`."""
    _config(game, "anon", "sokubot")
    rep = pf.audit(game)
    assert "clash" in rep["collides_with_agent"]
    assert pf.live_collisions(rep) == []
    assert pf.resolve_side(game)[0] == 1


def test_a_pad_that_disagrees_with_the_profile_is_refused(game):
    """The agent would press keys the game is not listening for."""
    _pf(game / "profile" / "sokubot.pf", keys=HUMAN_KEYS)
    _config(game, "anon", "sokubot")
    with pytest.raises(pf.SlotError, match="disagree"):
        pf.resolve_side(game)


def test_a_joystick_profile_is_refused(game):
    """A DirectInput joystick makes Soku's menus scroll continuously."""
    _pf(game / "profile" / "sokubot.pf", device=0x00, keys=AGENT_KEYS)
    _config(game, "anon", "sokubot")
    with pytest.raises(pf.SlotError, match="not a keyboard"):
        pf.resolve_side(game)


def test_an_unreadable_game_directory_is_a_slot_error(tmp_path):
    with pytest.raises(pf.SlotError, match="cannot read profiles"):
        pf.resolve_side(tmp_path / "nowhere")


def test_a_shared_pause_key_is_not_a_collision(game):
    """Both profiles bind pause to Q, and the pad has no code for it.

    Crying wolf on the one profile that is always in use is how a real collision
    gets waved through -- `collisions` restricts itself to keys the pad can send.
    """
    _config(game, "anon", "sokubot")
    assert "anon" not in pf.audit(game)["collides_with_agent"]


# --- the probe and the launcher share one verdict ---------------------------
@pytest.fixture
def probe():
    from scripts.live_probe import profiles_probe
    return profiles_probe


@pytest.mark.parametrize("p1, p2", [
    ("anon", "sokubot"),        # good
    ("sokubot", "anon"),        # good, other chair
    ("anon", "clash"),          # agent in neither slot
    ("clash", "sokubot"),       # a loaded profile shares keys
])
def test_the_probe_passes_exactly_when_the_launcher_would_start(game, probe,
                                                                p1, p2, capsys):
    _config(game, p1, p2)
    try:
        pf.resolve_side(game)
        launcher_starts = True
    except pf.SlotError:
        launcher_starts = False
    assert (probe(game) == 0) is launcher_starts
    capsys.readouterr()


def test_the_probe_now_fails_an_agent_the_game_will_not_read(game, probe, capsys):
    """It used to print 'NOT SELECTED' and exit 0."""
    _config(game, "anon", "clash")
    assert probe(game) == 1
    assert "neither slot" in capsys.readouterr().err
