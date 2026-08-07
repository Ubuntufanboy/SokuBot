"""The wire format between the game host and the inference host.

One request per decision, one reply per request, over a single TCP connection
with Nagle disabled. Deliberately not HTTP and not a framework: at 15 Hz the
only thing that matters is that a message is never delayed for batching, and
every layer added is another place to look when it is.

WHY THE CLIENT SENDS A TICK NUMBER
----------------------------------
The reply has to be applied at a *specific* game tick, not "as soon as it
arrives". The client stamps each frame with the absolute 60 Hz tick it was
captured at; the server echoes it back with the chunk. That makes a late reply
detectable by comparing ticks rather than by timing the round trip, and it
survives clock differences between the two machines because only the client's
clock is ever consulted.

WHY 480x480 JPEG AND NOT 224
----------------------------
Measured against the training chain (corpus frame -> 224 bilinear) in latent
cosine, on the encoder the policy actually consumes:

    480 JPEG q85 -> resize 224     0.9930      46 KB
    480 JPEG q75 -> resize 224     0.9899      36 KB
    resize 224 -> JPEG q85         0.9703      14 KB
    resize 224 -> JPEG q75         0.9152      11 KB

One ordinary 66.7 ms gameplay step moves the latent by cosine 0.9723. So
compressing at 224 costs as much as an entire step of real play, and at q75 it
costs three. Compressing at 480 costs a third of one. JPEG artifacts at 224 sit
at the same spatial scale as the encoder's 14 px patches; at 480 the bilinear
downscale averages them away.

The bandwidth this costs -- ~36 KB per decision, 4.3 Mbit/s -- was the reason to
want 224, and on a LAN it is irrelevant (measured 53 MB/s, RTT 2.2 ms).
"""

from __future__ import annotations

import socket
import struct

# frame_id, capture tick, side (0=P1, 1=P2), jpeg length
REQUEST = struct.Struct("!IIBI")
# frame_id, the absolute tick the chunk starts at, compensation steps used,
# then ticks*10 packed bits
REPLY_HEAD = struct.Struct("!IIB")

DEFAULT_PORT = 10800
JPEG_QUALITY = 85


class ProtocolError(RuntimeError):
    pass


def connect(host: str, port: int = DEFAULT_PORT,
            timeout: float = 5.0) -> socket.socket:
    s = socket.create_connection((host, port), timeout=timeout)
    # Without this a small reply can sit in the sender's buffer waiting for an
    # ack, which at 15 Hz is a whole extra round trip of delay for nothing.
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ProtocolError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def pack_request(frame_id: int, tick: int, side: int, jpeg: bytes) -> bytes:
    return REQUEST.pack(frame_id, tick, side, len(jpeg)) + jpeg


def read_request(sock: socket.socket) -> tuple[int, int, int, bytes]:
    frame_id, tick, side, n = REQUEST.unpack(recv_exactly(sock, REQUEST.size))
    return frame_id, tick, side, recv_exactly(sock, n)


def pack_reply(frame_id: int, start_tick: int, steps: int, bits: bytes) -> bytes:
    return REPLY_HEAD.pack(frame_id, start_tick, steps) + bits


def read_reply(sock: socket.socket, nbytes: int) -> tuple[int, int, int, bytes]:
    frame_id, start_tick, steps = REPLY_HEAD.unpack(
        recv_exactly(sock, REPLY_HEAD.size))
    return frame_id, start_tick, steps, recv_exactly(sock, nbytes)


def pack_chunk(chunk) -> bytes:
    """[ticks, 10] 0/1 -> one byte per tick. Ten buttons fit in ten bits."""
    out = bytearray()
    for tick in chunk:
        v = 0
        for i, b in enumerate(tick):
            if b > 0.5:
                v |= 1 << i
        out += struct.pack("!H", v)
    return bytes(out)


def unpack_chunk(bits: bytes, ticks: int):
    import numpy as np
    out = np.zeros((ticks, 10), dtype=np.float32)
    for t in range(ticks):
        (v,) = struct.unpack_from("!H", bits, t * 2)
        for i in range(10):
            if v & (1 << i):
                out[t, i] = 1.0
    return out


def chunk_bytes(ticks: int) -> int:
    return ticks * 2
