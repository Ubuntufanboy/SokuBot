"""The GL mode the launcher gives the game.

Software GL hangs Wine's startup on this machine (2026-09-20) and hardware GL
does not, having been the other way round on 2026-08-16. The mode is therefore a
choice, and a wrong default costs a silent hang with no window, so it is pinned.

    python -m pytest tests/test_launch_env.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def launched_env(monkeypatch):
    import scripts.play_cheat_match as m
    seen = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        seen.update(kw)
        return object()

    monkeypatch.setattr(m.subprocess, "Popen", fake_popen)

    def go(**extra):
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
        m.launch_game(Path("/game"), Path("/prefix"), ":0")
        return seen["env"]
    return go


def test_the_default_is_hardware_gl_even_if_the_parent_forces_software(
        launched_env, monkeypatch):
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")     # inherited from a shell
    monkeypatch.delenv("SOKUBOT_GL", raising=False)
    assert "LIBGL_ALWAYS_SOFTWARE" not in launched_env()


def test_software_gl_is_available_on_request(launched_env, monkeypatch):
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    assert launched_env(SOKUBOT_GL="sw")["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_the_prefix_and_display_still_reach_the_game(launched_env, monkeypatch):
    monkeypatch.delenv("SOKUBOT_GL", raising=False)
    env = launched_env()
    assert env["WINEPREFIX"] == "/prefix" and env["DISPLAY"] == ":0"
    assert env["WINEDLLOVERRIDES"] == "d3d9=b"


# --- the mod loader --------------------------------------------------------------
# Measured 2026-09-20 by mapping the running game: with `d3d9=b` ZERO mod DLLs load (d3d9 comes
# from Wine's builtin); without it the loader runs and giuroll, WindowResizer, etc. are mapped.
# ModLoaderSettings.json is read only by the loader, so a module preset is a no-op unless the
# override is dropped.
def test_vanilla_is_the_default_and_bypasses_the_mod_loader(launched_env, monkeypatch):
    monkeypatch.delenv("SOKUBOT_GL", raising=False)
    assert launched_env()["WINEDLLOVERRIDES"] == "d3d9=b"


def test_mods_true_lets_the_loader_run_even_if_the_parent_shell_set_the_override(launched_env, monkeypatch):
    import scripts.play_cheat_match as m
    seen = {}
    monkeypatch.setenv("WINEDLLOVERRIDES", "d3d9=b")                 # inherited from a shell
    monkeypatch.setattr(m.subprocess, "Popen", lambda cmd, **kw: seen.update(kw) or object())
    m.launch_game(Path("/game"), Path("/prefix"), ":0", mods=True)
    assert "WINEDLLOVERRIDES" not in seen["env"]


def test_a_preset_means_the_loader_runs():
    """`_play` derives `mods` from the preset; pin the rule at the source."""
    import inspect
    import scripts.play_cheat_match as m
    src = inspect.getsource(m._play)
    assert 'mods = a.preset != "none"' in src and "mods=mods" in src
