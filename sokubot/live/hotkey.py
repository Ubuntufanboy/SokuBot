"""A global hotkey, read from the real keyboards.

    w = KeyWatcher.for_evdev(on_press=..., key="KEY_F12", exclude=pad.device_path)
    if w: w.start()

WHY GLOBAL, AND WHY evdev
-------------------------
During a match the GAME has focus, so a hotkey that needs a terminal focused cannot be
pressed when it is needed. Reading evdev needs the `input` group, which the uinput
setup already requires. Soku itself reads the keyboard from every window
(`soku-reads-the-keyboard-with-no-focus`), so the key reaches the game too: the default,
F12, is in nobody's profile (profiles bind 11 keys, none of them F-keys) and not one of
giuroll's (0x09/0x0A/0x0B/0x10/0x13/0x21/0x22), which `test_hotkey.py` pins.

THE AGENT'S OWN PAD IS EXCLUDED
-------------------------------
The virtual keypad is a real evdev keyboard as far as this is concerned. It never sends
F12, but excluding it means a future pad change cannot make the agent toggle itself.

A press is a key-DOWN only (value 1): autorepeat (2) and release (0) are ignored, and
presses inside `debounce_s` collapse to one, because one physical press on a bouncy
switch must not arm and disarm in the same instant.
"""

from __future__ import annotations

import select
import threading
import time
from typing import Callable, Optional

EV_KEY = 1
DEFAULT_KEY = "KEY_F12"

# (type, code, value)
Event = tuple[int, int, int]
Poll = Callable[[float], list[Event]]


class KeyWatcher(threading.Thread):
    def __init__(self, on_press: Callable[[], None], code: int, poll: Poll, *,
                 debounce_s: float = 0.3, clock: Callable[[], float] = time.monotonic):
        super().__init__(daemon=True)
        self._on_press, self.code, self._poll = on_press, code, poll
        self._debounce, self._clock = debounce_s, clock
        self._halt = threading.Event()
        self.presses = 0

    def run(self) -> None:
        last = float("-inf")
        while not self._halt.is_set():
            for etype, ecode, value in self._poll(0.25):
                if etype != EV_KEY or ecode != self.code or value != 1:
                    continue
                now = self._clock()
                if now - last < self._debounce:
                    continue
                last = now
                self.presses += 1
                try:
                    self._on_press()
                except Exception as e:                  # noqa: BLE001
                    # A bug in the callback must not silently end the hotkey: the
                    # user would find out when they press it and nothing happens.
                    print(f"  hotkey callback failed: {type(e).__name__}: {e}", flush=True)

    def stop(self) -> None:
        self._halt.set()
        self.join(1.0)

    @classmethod
    def for_evdev(cls, on_press: Callable[[], None], key: str = DEFAULT_KEY,
                  exclude: Optional[str] = None) -> "Optional[KeyWatcher]":
        """A watcher over every readable keyboard, or None if there is none.

        None -- not a watcher that never fires -- so the caller can tell the user that
        the hotkey is unavailable and why, instead of leaving them pressing a key that
        does nothing.
        """
        try:
            import evdev
            code = evdev.ecodes.ecodes[key]
            devs = []
            for path in evdev.list_devices():
                if path == exclude:
                    continue
                d = evdev.InputDevice(path)
                if EV_KEY in d.capabilities():
                    devs.append(d)
        except Exception:                                # noqa: BLE001
            return None
        if not devs:
            return None
        fds = {d.fd: d for d in devs}

        def poll(timeout: float) -> list[Event]:
            out: list[Event] = []
            ready, _, _ = select.select(list(fds), [], [], timeout)
            for fd in ready:
                try:
                    out.extend((e.type, e.code, e.value) for e in fds[fd].read())
                except OSError:
                    pass
            return out

        return cls(on_press, code, poll)
