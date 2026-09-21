"""A one-way status channel: the harness writes it, an overlay reads it.

    w = StatusWriter()                      # the harness, once
    w.update(armed=True, slot=2)            # from any thread, any time
    ...
    read_status()                           # the overlay, ~4 Hz

WHY A FILE, AND WHY IT IS ONE-WAY
---------------------------------
The overlay is a separate process with no business inside the control loop: it
must not be able to slow a decision, and a crash in it must not be able to stop
the pad. A small JSON file is the whole interface. Nothing flows back through it
-- commands still go through the FIFO -- so the display can never become a way to
drive the agent by accident.

The same schema is the contract a later in-game module would render, which is why
it carries only what a player needs to see and nothing about the game's state.

A STALE FILE MUST NOT READ AS LIVE
----------------------------------
The worst thing this could show is `ARMED` from a harness that died ten minutes
ago. A file cannot know its writer crashed, so the READER decides: the status is
bound to the writer's pid and to a heartbeat, and either failing turns it into
`OFFLINE`. That is the same failure that made a dead pid, an old log and a shared
log path all read as a healthy run elsewhere in this project.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

SCHEMA = 1

# A harness that has not written for this long is treated as gone. The writer
# heartbeats at its own rate limit, well inside it.
STALE_AFTER_S = 3.0
MIN_INTERVAL_S = 0.25


def default_path() -> Path:
    """`$XDG_RUNTIME_DIR` (per-user, tmpfs, cleared at logout) or `/tmp`."""
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "sokubot.status.json"


@dataclass
class Status:
    scene: str = "unknown"          # title | menu | select | battle | unknown
    slot: int | None = None         # player number the agent plays, 1 or 2
    armed: bool = False             # the agent is allowed to press buttons
    # True/False = a battle is / is not on screen. None = there is no gate in this
    # configuration, which must not read as "waiting for a battle".
    gate_open: bool | None = None
    rounds: list[int] = field(default_factory=lambda: [0, 0])   # [agent, foe]
    latency_p50_ms: float | None = None
    latency_p99_ms: float | None = None
    missed_slot_pct: float | None = None
    # True/False = there IS a server and it is/is not answering. None = no server in
    # this configuration (a local run), which is not a failure and must not read as one.
    server_ok: bool | None = None
    last_error: str = ""
    # Something the agent is doing that the player must know about but that is NOT a fault,
    # e.g. calibrating (hands off the keyboard). Kept apart from last_error so the overlay
    # never renders "CALIBRATING" as an ERROR.
    busy: str = ""
    # Filled by the writer, not by callers.
    schema: int = SCHEMA
    pid: int = 0
    updated: float = 0.0


_KNOWN = {f.name for f in fields(Status)}


class StatusWriter:
    """Merges updates and writes them atomically, rate-limited.

    Thread-safe: the decision thread and the console thread both report, and an
    unlocked merge would lose whichever wrote second.
    """

    def __init__(self, path: Path | None = None, *,
                 clock: Callable[[], float] = time.time,
                 min_interval_s: float = MIN_INTERVAL_S):
        self.path = Path(path) if path else default_path()
        self._clock = clock
        self._min = min_interval_s
        self._lock = threading.Lock()
        self._status = Status(pid=os.getpid())
        self._last_write = float("-inf")

    def update(self, *, force: bool = False, **fields_: object) -> bool:
        """Merge `fields_`; write if the interval has passed or `force`.

        Returns whether a write happened. A rate-limited call still MERGES, so
        the next write carries it -- nothing is dropped, only delayed. `force`
        is for transitions a viewer must not miss (arm, disarm, an error).
        """
        unknown = set(fields_) - _KNOWN
        if unknown:
            raise TypeError(f"unknown status field(s): {sorted(unknown)}")
        with self._lock:
            for k, v in fields_.items():
                setattr(self._status, k, v)
            now = self._clock()
            if not force and now - self._last_write < self._min:
                return False
            self._status.updated = now
            self._write(self._status)
            self._last_write = now
            return True

    def heartbeat(self) -> bool:
        """Refresh the timestamp with no other change, so an idle harness is not
        mistaken for a dead one."""
        return self.update()

    def close(self) -> None:
        """Say so, then go. `OFFLINE` would follow within `STALE_AFTER_S` anyway;
        this makes it immediate on a clean exit."""
        with self._lock:
            self._status.armed = False
            self._status.gate_open = False
            self._status.server_ok = False
            self._status.pid = 0        # no live owner
            self._status.updated = self._clock()
            self._write(self._status)

    def _write(self, s: Status) -> None:
        # Write beside the target and rename over it: a reader sees the old file
        # or the new one, never half of either.
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(s)))
        os.replace(tmp, self.path)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                     # exists, owned by someone else
    return True


def read_status(path: Path | None = None, *,
                clock: Callable[[], float] = time.time,
                alive: Callable[[int], bool] = pid_alive,
                stale_after_s: float = STALE_AFTER_S) -> Status | None:
    """The current status, or None when the harness is not there.

    None -- not a default `Status` -- for every way of not being there: no file,
    a file that will not parse, a schema this reader does not know, a writer whose
    pid is gone, or a heartbeat older than `stale_after_s`. A reader that
    returned a blank status would render "OFF", which is a claim.
    """
    p = Path(path) if path else default_path()
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        return None
    s = Status(**{k: v for k, v in raw.items() if k in _KNOWN})
    if not alive(s.pid):
        return None
    if clock() - s.updated > stale_after_s:
        return None
    return s


def summarise(s: Status | None) -> tuple[str, str]:
    """(headline, level) for a display. Pure, so it is testable without a screen.

    level is one of `off`, `warn`, `idle`, `ok`. Ordered so the WORST true thing
    is the headline: an armed agent with no server is not "ARMED".
    """
    if s is None:
        return "SokuBot: OFFLINE", "off"
    who = f"P{s.slot}" if s.slot else "?"
    if s.last_error:
        return f"SokuBot: ERROR - {s.last_error}", "warn"
    if s.busy:
        return f"SokuBot ({who}): {s.busy}", "warn"
    if s.server_ok is False:
        return "SokuBot: NO SERVER" + (" (still armed)" if s.armed else ""), "warn"
    if not s.armed:
        return f"SokuBot ({who}): OFF - hotkey to hand over", "idle"
    if s.gate_open is False:
        return f"SokuBot ({who}): ARMED, waiting for a battle", "idle"
    return f"SokuBot ({who}): PLAYING", "ok"


def fields_from(pilot, brain, slot: int, notice: str = "", busy: str = "",
                gate_open: bool | None = None) -> dict:
    """The status fields that have a REAL source today, read off the live objects.

    `gate_open` is whether a battle is on screen, or None when there is no gate (then the
    field is not written at all).

    `notice` is a one-line reason the last command was refused (an arm with no identity,
    say). It is shown only while disarmed, and only when there is no worse error.

    `pilot` and `brain` may be None (no policy loaded / no server). Fields with no
    source yet -- scene, rounds, gate_open -- are deliberately not returned, so they
    keep their defaults instead of a guess: a field that is always None is honest,
    and one rendered from a stale default is exactly what the reader was built to
    prevent.

    Pure and duck-typed so it is testable without a game: it touches only
    `pilot.armed.is_set()`, `.stop_reason`, `.lat_ms`, `.late`, `.decides`, `.period`
    and `brain.ok`, `.last_error`.
    """
    out: dict = {"slot": slot, "busy": busy}
    if gate_open is not None:               # no gate, no claim
        out["gate_open"] = gate_open
    armed = bool(pilot is not None and pilot.armed.is_set())
    out["armed"] = armed

    error = ""
    if brain is not None:
        out["server_ok"] = bool(brain.ok)
        if not brain.ok:
            error = f"server: {brain.last_error}" if brain.last_error else "server not answering"
    if not error and pilot is not None and not armed and pilot.stop_reason:
        # A watchdog stop is worth showing until the user re-arms, and NOT after:
        # `stop_reason` is never cleared, so an armed agent would otherwise sit next
        # to an old error.
        error = str(pilot.stop_reason)
    if not error and not armed and notice:
        # Why the LAST command did nothing -- e.g. an arm refused for want of an
        # identity. The user pressed a hotkey in a game window and cannot see the
        # terminal; without this, "nothing happened" is all they would ever learn.
        error = notice
    out["last_error"] = error

    if pilot is not None:
        from sokubot.live.latency import summarise
        # deque.copy() is one C call, so it cannot be mutated under us by the
        # decision thread the way iterating it can.
        s = summarise(list(pilot.lat_ms.copy())[-600:], 1000.0 * pilot.period)
        out["latency_p50_ms"], out["latency_p99_ms"] = s["p50"], s["p99"]
        out["missed_slot_pct"] = (100.0 * pilot.late / pilot.decides
                                  if pilot.decides else None)
    return out
