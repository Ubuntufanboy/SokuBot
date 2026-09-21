"""The overlay must FOLLOW the game window, not just start beside it.

Found on the real game: an overlay started before the game window existed took the screen corner and
stayed there for the whole session, because Tk silently ignores a MOVE of an already-mapped window
until an idle pass. Every earlier test set the position before the first map, so none could see it.

This runs the real overlay on a private X display (Xvfb), starts it BEFORE any game window exists,
then creates one and checks that the overlay moves. Skipped where Xvfb/Tk/xdotool are missing.

    python -m pytest tests/test_overlay_live.py -q
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NEEDED = ("Xvfb", "xdotool", "xwininfo")

pytestmark = pytest.mark.skipif(any(shutil.which(t) is None for t in NEEDED),
                                reason="needs Xvfb, xdotool and xwininfo")

FAKE_GAME = ('import tkinter as tk\nr = tk.Tk(); r.title("Touhou Hisoutensoku fake"); '
             'r.geometry("640x480+13+77"); r.after(12000, r.destroy); r.mainloop()\n')


def free_display() -> str:
    for n in range(90, 120):
        if not Path(f"/tmp/.X11-unix/X{n}").exists():
            return f":{n}"
    pytest.skip("no free X display number")


def position(display: str, name: str):
    env = {**os.environ, "DISPLAY": display}
    for wid in subprocess.run(["xdotool", "search", "--name", name], capture_output=True,
                              text=True, env=env).stdout.split():
        g = subprocess.run(["xwininfo", "-id", wid], capture_output=True, text=True, env=env).stdout
        if "IsViewable" in g:
            return (int(g.split("Absolute upper-left X:")[1].split()[0]),
                    int(g.split("Absolute upper-left Y:")[1].split()[0]))
    return None


def wait_for(fn, want, timeout):
    end = time.time() + timeout
    got = None
    while time.time() < end:
        got = fn()
        if got == want:
            return got
        time.sleep(0.2)
    return got


@pytest.fixture
def xserver(tmp_path):
    d = free_display()
    x = subprocess.Popen(["Xvfb", d, "-screen", "0", "1920x1080x24"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    procs = [x]
    yield d, procs
    for p in reversed(procs):
        p.terminate()
        try:
            p.wait(3)
        except subprocess.TimeoutExpired:
            p.kill()


def test_an_overlay_started_before_the_game_moves_to_it_when_it_appears(xserver, tmp_path):
    display, procs = xserver
    env = {**os.environ, "DISPLAY": display, "XDG_RUNTIME_DIR": str(tmp_path), "OMP_NUM_THREADS": "1"}
    ov = subprocess.Popen([sys.executable, "-m", "scripts.sokubot_overlay", "--display", display,
                           "--seconds", "20"], cwd=str(ROOT), env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    procs.append(ov)
    # No game yet: it must take a corner, and must be there.
    corner = wait_for(lambda: position(display, "sokubot-overlay"), (12, 1022), 8)
    assert corner == (12, 1022), f"overlay never appeared in the corner: {corner}"
    # A game window appears: the overlay must MOVE to 12 px below it (13, 77+480+12).
    game = subprocess.Popen([sys.executable, "-c", FAKE_GAME], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(game)
    moved = wait_for(lambda: position(display, "sokubot-overlay"), (13, 569), 8)
    assert moved == (13, 569), f"the overlay stayed at {moved} after the game window appeared"
