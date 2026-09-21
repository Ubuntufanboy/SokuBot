"""The wire format between the game host and the vision inference server.

One request, one reply, over a single TCP connection: a 5-byte header (opcode,
payload length) then the payload. The reply reuses the request's opcode.

    D  decide     a frame pair (+ HUD floats) -> the next chunk of buttons + the
                  encoder's own reading, or an EMPTY reply when the server is not
                  ready to act (identity not established yet)
    C  calibrate  two frame pairs, before and after a held direction -> a sentence
                  that contains "agent is" on success
    R  reset      -> b"ok"; forgets identity and history
    P  ping       -> a status line
    I  info       -> JSON: the cadence and input shape the client must use

WHY THIS IS ITS OWN MODULE
--------------------------
It used to live in `scripts/serve_vision.py`, which imports torch and the whole
model stack at module level. The game host's client only needed four constants and
a read loop, and importing them dragged the server in with it -- and made it
impossible to stand up a stand-in server (`nullbrain.py`) that needs no model at
all. Nothing here imports anything heavy, and it must stay that way.

(`protocol.py` is the OLDER, Path A format with tick-stamped frames. They are not
interchangeable.)
"""

from __future__ import annotations

import socket
import struct

HDR = struct.Struct("!cI")          # opcode, payload length
OP_DECIDE, OP_CAL, OP_RESET, OP_PING, OP_INFO = b"D", b"C", b"R", b"P", b"I"


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)} of {n} bytes")
        buf += chunk
    return bytes(buf)
