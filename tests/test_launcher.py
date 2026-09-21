"""The launcher's guarantees: check before you launch, and clean up what you started.

`_play` (the real launch) needs a game, so it is replaced here; what is under test is the
ordering and the cleanup around it in `main()`.

    python -m pytest tests/test_launcher.py -q
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

import scripts.play_cheat_match as pcm
from sokubot.live.nullbrain import NullBrain, NullServer


@pytest.fixture
def null_server():
    srv = NullServer(NullBrain()); srv.start(); assert srv.ready.wait(5)
    yield srv
    srv.stop()


def ns(**kw):
    base = dict(server="127.0.0.1", port=1, side=1, display=":0", prefix=Path("/nope"),
                preset="none", sfe_root=Path("/nope"), no_overlay=False, close_game=False,
                verify=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- preflight ---------------------------------------------------------------
def test_preflight_passes_for_a_live_server(null_server):
    assert pcm.preflight_server(ns(port=null_server.port)) == 0


def test_preflight_fails_clearly_for_a_dead_server(capsys):
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()   # nothing there
    assert pcm.preflight_server(ns(port=port)) == 3
    out = capsys.readouterr().out
    assert "NOT STARTING" in out and "serve_null" in out


# --- presets -------------------------------------------------------------------
def test_no_preset_means_no_command():
    assert pcm.preset_command("none", Path("/x")) is None


def test_a_preset_runs_the_sfe_switcher_with_this_interpreter():
    cmd = pcm.preset_command("ablation", Path("/root/sfe"))
    assert cmd == [sys.executable, "/root/sfe/ops/soku_mods.py", "ablation"]


def test_the_default_preset_never_touches_the_users_module_set(monkeypatch):
    monkeypatch.setattr(pcm.subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    assert pcm.apply_preset(ns(preset="none")) == 0


def test_a_missing_switcher_refuses_to_start(capsys):
    assert pcm.apply_preset(ns(preset="netplay", sfe_root=Path("/does/not/exist"))) == 2
    assert "soku_mods.py" in capsys.readouterr().out


def test_a_failing_switcher_refuses_to_start(tmp_path, capsys):
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "soku_mods.py").write_text("import sys; sys.exit(1)\n")
    assert pcm.apply_preset(ns(preset="ablation", sfe_root=tmp_path)) == 2


def test_a_working_switcher_is_applied(tmp_path):
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "soku_mods.py").write_text("print('changed:')\n")
    assert pcm.apply_preset(ns(preset="ablation", sfe_root=tmp_path)) == 0


# --- overlay lifecycle -----------------------------------------------------------
class FakeProc:
    def __init__(self, hang=False):
        self.hang, self.terminated, self.killed = hang, False, False

    def terminate(self): self.terminated = True

    def wait(self, t):
        if self.hang:
            raise subprocess.TimeoutExpired("x", t)

    def kill(self): self.killed = True


def test_stopping_no_overlay_is_a_no_op():
    pcm.stop_overlay(None)


def test_the_overlay_is_terminated_and_a_hung_one_is_killed():
    a, b = FakeProc(), FakeProc(hang=True)
    pcm.stop_overlay(a); pcm.stop_overlay(b)
    assert a.terminated and not a.killed
    assert b.terminated and b.killed


# --- main(): the order things happen in ----------------------------------------
@pytest.fixture
def main_env(monkeypatch):
    calls = []
    monkeypatch.setattr(pcm, "resolve_side_arg", lambda a: 0)
    monkeypatch.setattr(pcm, "_play", lambda a: calls.append("play") or 0)
    monkeypatch.setattr(pcm, "start_overlay", lambda a: calls.append("overlay+") or FakeProc())
    monkeypatch.setattr(pcm, "stop_overlay", lambda p: calls.append("overlay-"))
    monkeypatch.setattr(pcm, "close_game", lambda a: calls.append("close"))
    monkeypatch.setattr(pcm, "apply_preset", lambda a: calls.append("preset") or 0)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["play_cheat_match", *argv])
        return pcm.main(), calls
    return run


def test_a_dead_server_never_launches_a_game_or_opens_an_overlay(main_env):
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    rc, calls = main_env("--server", "127.0.0.1", "--port", str(port), "--side", "1")
    assert rc == 3
    assert calls == []                                    # nothing was started at all


def test_a_live_server_runs_preset_then_overlay_then_the_game_then_cleans_up(main_env, null_server):
    rc, calls = main_env("--server", "127.0.0.1", "--port", str(null_server.port), "--side", "1")
    assert rc == 0
    assert calls == ["preset", "overlay+", "play", "overlay-"]


def test_the_overlay_is_stopped_even_if_the_run_raises(main_env, null_server, monkeypatch):
    stopped = []
    monkeypatch.setattr(pcm, "stop_overlay", lambda p: stopped.append(p))
    monkeypatch.setattr(pcm, "start_overlay", lambda a: "the-overlay")

    def boom(a):
        raise RuntimeError("the session blew up")
    monkeypatch.setattr(pcm, "_play", boom)
    monkeypatch.setattr(sys, "argv", ["p", "--server", "127.0.0.1", "--port",
                                      str(null_server.port), "--side", "1"])
    with pytest.raises(RuntimeError):
        pcm.main()
    assert stopped == ["the-overlay"]                     # cleaned up, and the right one


def test_the_game_is_closed_only_when_asked(main_env, null_server):
    _, calls = main_env("--server", "127.0.0.1", "--port", str(null_server.port), "--side", "1")
    assert "close" not in calls
    _, calls = main_env("--server", "127.0.0.1", "--port", str(null_server.port), "--side", "1", "--close-game")
    assert calls[-1] == "close"


def test_no_overlay_flag_skips_the_overlay(main_env, null_server):
    _, calls = main_env("--server", "127.0.0.1", "--port", str(null_server.port), "--side", "1", "--no-overlay")
    # never started; the cleanup still runs on `None`, which is a no-op
    assert calls == ["preset", "play", "overlay-"]
