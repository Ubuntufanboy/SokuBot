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
