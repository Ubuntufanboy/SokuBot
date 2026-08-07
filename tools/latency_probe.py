"""Measure a transport under the real control loop's workload, not a synthetic one.

    # on the GPU box
    python tools/latency_probe.py serve --port 8080

    # here
    python tools/latency_probe.py probe --host 1.2.3.4 --port 40080 --label direct
    python tools/latency_probe.py probe --host 100.x.y.z --port 8080 --label tailscale

**Why this exists rather than `ping`.** Planning measurements on this link were
actively misleading in two separate ways, and both would have produced a wrong
architecture:

* ICMP said avg 160-400 ms where a TCP handshake to the same host said 22-41 ms.
  Routers deprioritise ICMP and rate-limit SYN, so neither is what a steady
  small-packet flow experiences.
* A 512-byte echo measured *slower* than a 15,000-byte one (p50 177 vs 92 ms).
  That is not physically possible for one stationary path, so the path is
  non-stationary and any single short measurement of it is noise.

The consequence is that a transport has to be measured under the workload it
will actually carry, for long enough to see the tail, and compared against
alternatives measured the same way.

**The workload is asymmetric and that matters.** Upstream is one JPEG frame per
decision (kilobytes); downstream is one action chunk (4 ticks x 10 buttons, tens
of bytes). Measuring a symmetric echo overstates the return leg, which is the
leg the game is actually waiting on.

**What the numbers mean.** The control period is 66.7 ms (15 Hz). A reply later
than that missed its slot and the loop holds the previous chunk instead, so the
headline figure is not the mean -- it is the fraction of decisions that miss,
and how far the tail goes when they do.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

# frame_id (uint32) + payload length (uint32), then the payload.
HDR = struct.Struct("!II")
# The reply the real server sends: frame_id + 40 packed button bits + delay D.
REPLY = struct.Struct("!IQB")
CONTROL_PERIOD_S = 1.0 / 15.0


# ---------------------------------------------------------------------------
def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def serve(port: int, host: str = "0.0.0.0") -> int:
    """Echo the frame id back with a realistically tiny reply.

    Deliberately does no work: this measures the transport, and mixing in GPU
    time would make a slow link and a slow model indistinguishable. The real
    server's compute is measured separately and is single-digit milliseconds.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    print(f"latency probe listening on {host}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  connection from {addr}", flush=True)
        try:
            while True:
                fid, n = HDR.unpack(_recv_exactly(conn, HDR.size))
                _recv_exactly(conn, n)
                conn.sendall(REPLY.pack(fid, 0, 0))
        except (ConnectionError, OSError, struct.error):
            print("  connection closed", flush=True)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
def make_payload(size_kb: float) -> bytes:
    """A JPEG-shaped payload.

    Incompressible bytes, because a real JPEG is: a run of zeros would let any
    compressing hop on the path flatter the measurement.
    """
    import os
    return os.urandom(int(size_kb * 1024))


def probe(host: str, port: int, *, duration: float, size_kb: float,
          label: str, out: Path | None) -> int:
    """Send at the control rate for `duration` and report the tail."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    t_connect = time.perf_counter()
    sock.connect((host, port))
    connect_ms = (time.perf_counter() - t_connect) * 1000
    # Nagle batches small writes waiting for an ack, which on a 15 Hz loop means
    # the reply can sit in the sender's buffer for a whole extra RTT. Off.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    payload = make_payload(size_kb)
    samples: list[float] = []
    missed = 0
    t_end = time.perf_counter() + duration
    fid = 0
    try:
        while time.perf_counter() < t_end:
            slot = time.perf_counter()
            sock.sendall(HDR.pack(fid, len(payload)) + payload)
            try:
                back = _recv_exactly(sock, REPLY.size)
            except socket.timeout:
                missed += 1
                fid += 1
                continue
            rid, _, _ = REPLY.unpack(back)
            if rid != fid:
                missed += 1
            samples.append((time.perf_counter() - slot) * 1000)
            fid += 1
            # Pace to the control rate rather than flooding: a flood measures
            # throughput, and this loop is latency-bound, never throughput-bound.
            time.sleep(max(0.0, CONTROL_PERIOD_S - (time.perf_counter() - slot)))
    except (ConnectionError, OSError) as e:
        print(f"transport failed after {len(samples)} samples: {e}",
              file=sys.stderr)
    finally:
        sock.close()

    if not samples:
        print(f"{label}: no samples", file=sys.stderr)
        return 1

    samples.sort()
    q = lambda f: samples[min(len(samples) - 1, int(len(samples) * f))]
    over1 = sum(1 for s in samples if s > CONTROL_PERIOD_S * 1000)
    over2 = sum(1 for s in samples if s > CONTROL_PERIOD_S * 2000)
    rec = {
        "label": label, "host": host, "port": port,
        "size_kb": size_kb, "n": len(samples), "missed": missed,
        "connect_ms": round(connect_ms, 1),
        "min": round(samples[0], 1), "p50": round(statistics.median(samples), 1),
        "p90": round(q(.90), 1), "p99": round(q(.99), 1),
        "max": round(samples[-1], 1),
        "over_1_period_pct": round(100 * over1 / len(samples), 1),
        "over_2_periods_pct": round(100 * over2 / len(samples), 1),
    }
    print(f"\n{label}  ({len(samples)} decisions, {size_kb:.0f} KB up)")
    print(f"  min {rec['min']:6.1f}   p50 {rec['p50']:6.1f}   "
          f"p90 {rec['p90']:6.1f}   p99 {rec['p99']:6.1f}   "
          f"max {rec['max']:6.1f} ms")
    print(f"  missed a 66.7 ms slot: {rec['over_1_period_pct']:5.1f}%"
          f"      two slots: {rec['over_2_periods_pct']:5.1f}%")
    print(f"  => {verdict(rec['p90'])}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  appended to {out}")
    return 0


def sweep(host: str, port: int, *, sizes: list[float], block_s: float,
          rounds: int, out: Path | None) -> int:
    """Alternate payload sizes in short blocks on one connection.

    Sequential per-size probes cannot be compared on this link. A 512-byte echo
    once measured *slower* than a 15,000-byte one, which is impossible for a
    stationary path and therefore proves the path drifts faster than a
    back-to-back pair of measurements. Interleaving short blocks puts every size
    across the same stretch of network, so the comparison survives the drift.

    The question this answers is whether the queueing is **self-inflicted**: one
    480x480 JPEG per decision is ~4.3 Mbit/s against a ~25 Mbit/s uplink, which
    is exactly the sustained load that fills a bufferbloated cable queue. If the
    small payload is dramatically better, the fix is a smaller wire format. If
    it is not, the path is bad on its own account and no encoding will save it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    sock.connect((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payloads = {s: make_payload(s) for s in sizes}
    acc: dict[float, list[float]] = {s: [] for s in sizes}
    fid = 0
    try:
        for r in range(rounds):
            for s in sizes:
                pay = payloads[s]
                t_block = time.perf_counter() + block_s
                while time.perf_counter() < t_block:
                    slot = time.perf_counter()
                    sock.sendall(HDR.pack(fid, len(pay)) + pay)
                    try:
                        _recv_exactly(sock, REPLY.size)
                    except socket.timeout:
                        fid += 1
                        continue
                    acc[s].append((time.perf_counter() - slot) * 1000)
                    fid += 1
                    time.sleep(max(0.0, CONTROL_PERIOD_S -
                                   (time.perf_counter() - slot)))
            print(f"  round {r + 1}/{rounds} done", flush=True)
    except (ConnectionError, OSError) as e:
        print(f"transport failed: {e}", file=sys.stderr)
    finally:
        sock.close()

    print(f"\ninterleaved sweep, {rounds} rounds x {block_s:.0f}s per size")
    print(f"  {'KB up':>7}  {'n':>5}  {'min':>6} {'p50':>6} {'p90':>6} "
          f"{'p99':>6}   {'miss1':>6} {'miss2':>6}")
    rows = []
    for s in sizes:
        a = sorted(acc[s])
        if not a:
            continue
        q = lambda f: a[min(len(a) - 1, int(len(a) * f))]
        m1 = 100 * sum(1 for v in a if v > 66.7) / len(a)
        m2 = 100 * sum(1 for v in a if v > 133.4) / len(a)
        print(f"  {s:7.0f}  {len(a):5d}  {a[0]:6.1f} "
              f"{statistics.median(a):6.1f} {q(.9):6.1f} {q(.99):6.1f}   "
              f"{m1:5.1f}% {m2:5.1f}%")
        rows.append({"size_kb": s, "n": len(a), "min": round(a[0], 1),
                     "p50": round(statistics.median(a), 1), "p90": round(q(.9), 1),
                     "p99": round(q(.99), 1), "over_1_period_pct": round(m1, 1),
                     "over_2_periods_pct": round(m2, 1)})
    if out and rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps({"sweep": rows, "host": host, "port": port}) + "\n")
    return 0


def verdict(p90: float) -> str:
    """The rule, fixed before the measurement so it cannot be argued with after.

    The thresholds are the world model's, not a preference. `horizon_ablation`
    puts one-step rollout cosine at 0.9963 and four-step at 0.9717, and the
    action signal decays from r=0.55 at one step to r=0.09 at sixteen. Latency
    is compensated by rolling the latent forward D steps, so D is bounded by how
    far the predictor can be trusted -- two steps is comfortably inside it, four
    is not.
    """
    if p90 <= 60:
        return "usable: D=1 compensation step, ship the simple version"
    if p90 <= 120:
        return "usable: D=2 compensation steps plus the jitter buffer"
    return ("TOO SLOW: >2 steps of compensation leaves the trustworthy horizon. "
            "Move inference to a local machine instead of engineering the network")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the echo endpoint (on the GPU box)")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--host", default="0.0.0.0")

    p = sub.add_parser("probe", help="measure a transport from here")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--label", default="direct")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--size-kb", type=float, default=12.0,
                   help="upstream payload; a 480x480 JPEG is roughly 25 KB, "
                        "a 224x224 one roughly 8")
    p.add_argument("--out", type=Path, default=Path("results/latency.jsonl"))

    w = sub.add_parser("sweep", help="interleave payload sizes on one connection")
    w.add_argument("--host", required=True)
    w.add_argument("--port", type=int, required=True)
    w.add_argument("--sizes", type=float, nargs="+", default=[0.5, 8.0, 36.0])
    w.add_argument("--block", type=float, default=10.0)
    w.add_argument("--rounds", type=int, default=4)
    w.add_argument("--out", type=Path, default=Path("results/latency.jsonl"))

    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a.port, a.host)
    if a.cmd == "sweep":
        return sweep(a.host, a.port, sizes=a.sizes, block_s=a.block,
                     rounds=a.rounds, out=a.out)
    return probe(a.host, a.port, duration=a.duration, size_kb=a.size_kb,
                 label=a.label, out=a.out)


if __name__ == "__main__":
    raise SystemExit(main())
