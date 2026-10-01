"""Can a compute node in one Slurm job reach a listener in another? How fast?

    python -m scripts.net_probe listen  --addr-file ~/sokubot-runs/netprobe.addr
    python -m scripts.net_probe connect --addr-file ~/sokubot-runs/netprobe.addr

The real-game PPO trainer puts the learner in one job (a GPU node) and the game-playing actors in
others (main-partition nodes); everything rests on this connection working. The listener writes
`host port nonce` to the address file; the connector checks the nonce comes back, then measures
small-message round trips and a bulk transfer each way.
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import time
from pathlib import Path


def recv_exact(c: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = c.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


def listen(a) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 0))
    srv.listen(4)
    host, port = socket.gethostname(), srv.getsockname()[1]
    nonce = os.urandom(8).hex()
    tmp = a.addr_file.with_suffix(".tmp")
    tmp.write_text(f"{host} {port} {nonce}\n")
    os.replace(tmp, a.addr_file)
    print(f"listening on {host}:{port} nonce {nonce}", flush=True)
    srv.settimeout(a.wait)
    c, peer = srv.accept()
    print(f"connection from {peer}", flush=True)
    c.sendall(nonce.encode() + b"\n")
    while True:                                   # echo framed messages until the peer closes
        try:
            n = struct.unpack("!I", recv_exact(c, 4))[0]
        except ConnectionError:
            break
        data = recv_exact(c, n)
        c.sendall(struct.pack("!I", len(data)) + data)
    print("peer closed; done", flush=True)
    return 0


def connect(a) -> int:
    t0 = time.monotonic()
    while not a.addr_file.exists():
        if time.monotonic() - t0 > a.wait:
            raise SystemExit(f"no address file {a.addr_file} after {a.wait}s")
        time.sleep(2)
    host, port, nonce = a.addr_file.read_text().split()
    print(f"connecting from {socket.gethostname()} to {host}:{port}", flush=True)
    c = socket.create_connection((host, int(port)), timeout=30)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    got = c.recv(64).decode().strip()
    print(f"nonce {'OK' if got == nonce else 'MISMATCH: ' + got}", flush=True)

    def rt(n: int) -> float:
        msg = os.urandom(n)
        t = time.perf_counter()
        c.sendall(struct.pack("!I", n) + msg)
        m = struct.unpack("!I", recv_exact(c, 4))[0]
        back = recv_exact(c, m)
        assert back == msg
        return time.perf_counter() - t

    small = sorted(rt(64) for _ in range(200))
    print(f"64 B round trip: median {1e3 * small[100]:.3f} ms, p99 {1e3 * small[197]:.3f} ms")
    for mb in (1, 8, 32):
        dt = min(rt(mb << 20) for _ in range(3))
        print(f"{mb} MB echoed in {dt * 1e3:.0f} ms = {2 * mb / dt:.0f} MB/s (both directions)")
    c.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("listen", "connect"))
    ap.add_argument("--addr-file", type=Path, required=True)
    ap.add_argument("--wait", type=float, default=1800.0)
    a = ap.parse_args(argv)
    return listen(a) if a.mode == "listen" else connect(a)


if __name__ == "__main__":
    raise SystemExit(main())
