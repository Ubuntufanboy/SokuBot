"""`pause` must refuse under netplay, and must not be fooled by look-alike modules.

    python -m pytest tests/test_mods.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sokubot.live.mods import giuroll_enabled, pause_refusal


def settings(tmp_path, **modules):
    d = {"modules": {k.replace("__", "\\"): {"enabled": v} for k, v in modules.items()}}
    (tmp_path / "ModLoaderSettings.json").write_text(json.dumps(d))
    return tmp_path


def test_giuroll_on_is_netplay(tmp_path):
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": True})
    assert giuroll_enabled(g) is True


def test_giuroll_off_is_not(tmp_path):
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": False})
    assert giuroll_enabled(g) is False


def test_either_spelling_being_on_counts(tmp_path):
    """The file carries both `Modules\\` and `modules\\` keys; one on is enough."""
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": False,
                              "modules\\giuroll\\giuroll.dll": True})
    assert giuroll_enabled(g) is True


@pytest.mark.parametrize("key", [
    "Modules\\Giuroll-UI\\giuroll_ui.dll",            # the UI alone is not the netcode
    "Modules\\giuroll\\giuroll_loader_dll.dll",        # nor is its loader stub
    "Modules\\WindowResizer\\WindowResizer.dll",
])
def test_look_alike_modules_do_not_count(tmp_path, key):
    assert giuroll_enabled(settings(tmp_path, **{key: True})) is False


def test_no_settings_file_means_no_giuroll(tmp_path):
    assert giuroll_enabled(tmp_path) is False


@pytest.mark.parametrize("body", ["{not json", "[]", '{"other": 1}', ""])
def test_a_file_we_cannot_read_is_treated_as_netplay(tmp_path, body):
    """The wrong answer desyncs a match, so when we cannot tell we assume the worse case."""
    (tmp_path / "ModLoaderSettings.json").write_text(body)
    assert giuroll_enabled(tmp_path) is True


def test_pause_is_refused_under_netplay_and_says_what_to_use_instead(tmp_path):
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": True})
    msg = pause_refusal("pause", g)
    assert msg and "desync" in msg and "hands-off" in msg


def test_resume_is_never_refused(tmp_path):
    """A game that got stopped must always be startable again."""
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": True})
    assert pause_refusal("resume", g) is None


def test_pause_is_fine_locally(tmp_path):
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": False})
    assert pause_refusal("pause", g) is None


def test_other_commands_are_never_refused(tmp_path):
    g = settings(tmp_path, **{"Modules\\giuroll\\giuroll.dll": True})
    assert all(pause_refusal(c, g) is None for c in ("arm", "disarm", "stop", "shot"))


def test_the_guard_comes_before_the_pause_branch_in_the_real_handler():
    """The branch order IS the behaviour: an `elif` for pause placed first would win."""
    src = Path(__file__).parent.parent.joinpath("scripts", "play_match.py").read_text()
    guard = src.index('elif cmd in ("pause", "resume") and pause_refusal(cmd, GAME_DIR):')
    real = src.index('elif cmd in ("pause", "resume"):')
    assert guard < real


def test_this_machines_real_config_is_read_without_error():
    game = Path("~/.wine-soku/drive_c/Games/Soku").expanduser()
    if not (game / "ModLoaderSettings.json").is_file():
        pytest.skip("no Soku install")
    assert isinstance(giuroll_enabled(game), bool)
