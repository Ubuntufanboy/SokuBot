"""A stand-in vision server that runs no model.

    python -m scripts.serve_null --mode neutral        # then point the client at it

It speaks the real wire format (`wire.py`) and behaves like `serve_vision.py` in
every way the CLIENT can observe -- an empty reply until calibrated, a sentence
containing "agent is" from calibrate, a chunk of `ticks` x 10 buttons plus the
encoder's 2 x 33 reading from decide -- but computes nothing. So everything on the
game host that is NOT inference can be built and tested with no GPU, no encoder,
no torch, and effectively zero CPU: launch, arm/disarm, the hotkey, the status
file, watchdogs, reconnect.

That matters because the game host cannot run the model (measured 2026-09-20: p50
138 ms against an 83 ms period). It is deliberately NOT a fake of the model's
quality; it never plays. `wiggle` exists only so a live test can see the pad move.

FAULTS
------
`latency_ms` delays every reply (a slow server), `drop_after` closes the connection
without replying after N decisions (a crashed or restarted server). The server
keeps listening after a drop so a client that reconnects finds it.
"""

from __future__ import annotations

import json
import select
import socket
import threading
import time

import numpy as np

from sokubot.live.wire import (HDR, OP_CAL, OP_DECIDE, OP_INFO, OP_PING,
                               OP_RESET, recv_exactly)

N_STATE = 33                    # sokubot.data.state.CH; asserted equal in the tests
BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
MODES = ("neutral", "wiggle")


class Drop(Exception):
    """Raised by `handle` to make the server close the connection unanswered."""


class NullBrain:
    def __init__(self, mode: str = "neutral", ticks: int = 5, history: int = 12,
                 latency_ms: float = 0.0, drop_after: int | None = None,
                 side_answer: str = "RIGHT"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.ticks, self.history = mode, ticks, history
        self.latency_ms = latency_ms
        self.drop_after = drop_after
        self.side_answer = side_answer
        self.calibrated = False
        self.decides = 0                # decisions that got an ANSWER
        self.requests = 0               # decide requests, answered or not
        self.dropped = 0

    def spec(self) -> dict:
        return {"size": 224, "delta": 2, "ticks": self.ticks,
                "history": self.history, "trusted": [], "hud_floats": 4,
                "dropped": []}

    def _chunk(self) -> bytes:
        act = np.zeros((self.ticks, len(BUTTONS)), np.uint8)
        if self.mode == "wiggle":
            act[:, BUTTONS.index("left" if self.decides % 2 == 0 else "right")] = 1
        return act.tobytes()

    def handle(self, op: bytes, payload: bytes) -> bytes:
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if op == OP_INFO:
            return json.dumps(self.spec()).encode()
        if op == OP_PING:
            return f"decides {self.decides} | hud 0 | 0.0 ms".encode()
        if op == OP_RESET:
            self.calibrated = False
            return b"ok"
        if op == OP_CAL:
            self.calibrated = True
            return (f"agent is the {self.side_answer} character "
                    f"(null server: nothing was measured)").encode()
        if op == OP_DECIDE:
            self.requests += 1
            if self.drop_after is not None and self.requests > self.drop_after:
                self.dropped += 1
                raise Drop()
            if not self.calibrated:
                return b""              # exactly what the real server does
            out = self._chunk() + np.zeros(2 * N_STATE, np.float32).tobytes()
            self.decides += 1
            return out
        return b""


class NullServer(threading.Thread):
    """Serves one client at a time on a background thread. `.port` is set once bound."""

    def __init__(self, brain: NullBrain, host: str = "127.0.0.1", port: int = 0):
        super().__init__(daemon=True)
        self.brain, self.host = brain, host
        self._want_port = port
        self.port = 0
        self.ready = threading.Event()
        self._halt = threading.Event()
        self._sock: socket.socket | None = None
        self.connections = 0

    def run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self._want_port))
        srv.listen(1)
        srv.settimeout(0.2)
        self._sock = srv
        self.port = srv.getsockname()[1]
        self.ready.set()
        while not self._halt.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connections += 1
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(2.0)        # a message that stalls mid-way is a dead client
            try:
                self._serve(conn)
            finally:
                conn.close()
        srv.close()

    def _serve(self, conn: socket.socket) -> None:
        while not self._halt.is_set():
            # Idle-wait with select so a timeout can only ever happen BETWEEN
            # messages. A timeout inside recv_exactly would discard the bytes it
            # had already read and leave the stream misaligned.
            if not select.select([conn], [], [], 0.2)[0]:
                continue
            try:
                op, n = HDR.unpack(recv_exactly(conn, HDR.size))
                payload = recv_exactly(conn, n) if n else b""
            except (ConnectionError, OSError):          # incl. socket.timeout
                return
            try:
                out = self.brain.handle(op, payload)
            except Drop:
                return                  # close without replying
            try:
                conn.sendall(HDR.pack(op, len(out)) + out)
            except OSError:
                return

    def stop(self) -> None:
        self._halt.set()
        self.join(2.0)
