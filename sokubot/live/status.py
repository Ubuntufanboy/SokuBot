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
    gate_open: bool = False         # a battle is actually on screen
    rounds: list[int] = field(default_factory=lambda: [0, 0])   # [agent, foe]
    latency_p50_ms: float | None = None
    latency_p99_ms: float | None = None
    missed_slot_pct: float | None = None
    server_ok: bool = False
    last_error: str = ""
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
    if not s.server_ok:
        return "SokuBot: NO SERVER" + (" (still armed)" if s.armed else ""), "warn"
    if not s.armed:
        return f"SokuBot ({who}): OFF - hotkey to hand over", "idle"
    if not s.gate_open:
        return f"SokuBot ({who}): ARMED, waiting for a battle", "idle"
    return f"SokuBot ({who}): PLAYING", "ok"
