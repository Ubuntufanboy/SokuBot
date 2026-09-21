"""Shared fixtures for the live-loop tests: a real session() with a fake pad and decide()."""
from __future__ import annotations

import os
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from sokubot.live import status as st


@pytest.fixture
def loop(monkeypatch, tmp_path):
    import scripts.play_cheat_match as pcm
    ctl = tmp_path / "ctl"
    monkeypatch.setattr(pcm, "CTL", ctl)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(pcm, "load_agent",
                        lambda p, device="cpu": (object(), object(), 1, 1, 8, {"step": 0}))

    writers = []

    class Rec(st.StatusWriter):
        """Records every update's `force` flag and whether it carried `armed`."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.log = []
            writers.append(self)

        def update(self, *, force=False, **kw):
            self.log.append((force, kw.get("armed")))
            return super().update(force=force, **kw)

    monkeypatch.setattr(pcm, "StatusWriter", Rec)

    class Pad:
        device_path = "/dev/input/event99"

        def __init__(self): self.calls = 0
        def neutral(self): pass
        def set_state(self, s): self.calls += 1

        def press_only(self, name):
            self.pressed = getattr(self, "pressed", []) + [name]

    class Loop:
        def __init__(self):
            self.rc = None
            self.no_hotkey = True
            self.status_path = tmp_path / "sokubot.status.json"

        def start(self, decide_ms=2, late_run_limit=60, ls=None):
            self.decide_ms = decide_ms

            def decide(ls, pol, obs, hist, side, H):
                time.sleep(self.decide_ms / 1000)
                return np.zeros((1, 10), np.float32), object()
            monkeypatch.setattr(pcm, "decide", decide)
            a = types.SimpleNamespace(side=1, policy=Path("x.pt"), truth=None, server=None,
                                      display=":0", record=Path("/dev/null"), port=5599,
                                      late_run_limit=late_run_limit,
                                      no_hotkey=self.no_hotkey, hotkey="KEY_F12")
            self.pad = Pad()
            self.t = threading.Thread(
                target=lambda: setattr(self, "rc", pcm.session(ls or types.SimpleNamespace(), self.pad, a)),
                daemon=True)
            self.t.start()
            deadline = time.time() + 5
            while not ctl.exists() and time.time() < deadline:
                time.sleep(0.02)
            return self

        def send(self, line):
            fd = os.open(ctl, os.O_WRONLY)
            os.write(fd, (line + "\n").encode()); os.close(fd)

        def status(self):
            return st.read_status(self.status_path)

        def wait(self, pred, timeout=5.0):
            end = time.time() + timeout
            while time.time() < end:
                s = self.status()
                if s is not None and pred(s):
                    return s
                time.sleep(0.03)
            raise AssertionError(f"status never satisfied the predicate; last: {self.status()}")

        def stop(self):
            self.send("stop"); self.t.join(5)
            return self.rc

        @property
        def writer_log(self):
            return writers[0].log

    lp = Loop()
    lp.pcm = pcm
    yield lp
    if lp.t.is_alive():
        lp.stop()

