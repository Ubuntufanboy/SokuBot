"""Inference server: frames in, buttons out. Runs on the box with the GPU.

    # on 192.168.1.130
    python -m scripts.serve_vision --encoder ~/rl/enc.pt --policy ~/rl/policy.pt

    # on the game host
    python -m scripts.play_cheat_match --attach --server 192.168.1.130 ...

WHY THIS EXISTS
---------------
Measured on the game host, an idle 4-core laptop with the game NOT running:

    encoder2  1.27M params   35 ms      (the old, weak encoder)
    P50 w64   2.43M params   82 ms
    P51 w128  3.50M params  153 ms

against an 83 ms decision period that also has to hold a 20 ms policy forward.
Every encoder worth deploying is unusable on that machine, and the one that
fits is the one that scores R^2 0.359 and cannot see velocity at all. On the
LAN box's GPU the largest of them is single-digit milliseconds.

This is also the standing rule for the project after 2026-08-16, when running
the model on the game host drove a 4-core machine to load 11, stretched the
decision loop from 83 ms to 284 ms, and locked the operator out of their own
keyboard. The game host captures, drives the pad, and nothing else.

WHAT GOES OVER THE WIRE, AND WHY IT IS 224px
---------------------------------------------
The client resizes to the encoder's input size before sending: 224x224x6 is
301 KB against 1.38 MB for the raw 480x480 pair, which at the 12 Hz decision
rate is 3.6 MB/s instead of 16.6 MB/s -- 6 ms of transfer instead of 26 ms on a
53 MB/s link. The resize uses `VisionState._resize`, the same PIL BOX filter
the server would have used, and `_raw` passes an already-correct size through
untouched, so the pixels the encoder sees are identical either way.

STATE LIVES HERE, NOT ON THE CLIENT
------------------------------------
The 12-step observation history, the identity calibration and the left/right
tracking all live server-side. Splitting them would mean two processes
agreeing about which character is the agent, and that is exactly the kind of
shared mutable belief that goes wrong silently.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sokubot.data.state import CH, STAGE_SPAN

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


class Session:
    """One match's worth of state: encoder, policy, history, identity."""

    def __init__(self, encoder: Path, policy: Path, device: str):
        from sokubot.live.visionstate import VisionState
        from scripts.play_cheat_match import load_agent
        self.vs = VisionState.load(encoder, device=device)
        self.pol, self.obs, self.H, self.ticks, _slots, ck = load_agent(
            policy, device=device)
        self.obs = self.obs.to(device)
        self.pol = self.pol.to(device)
        self.device = device
        self.hist: deque = deque(maxlen=max(self.H, 1))
        self.decides = 0
        self.hud_reads = 0
        self.dt_ms = 0.0
        self._last_cmd = 0.0
        print(f"encoder {encoder.name}: {len(self.vs.trusted)} channels "
              f"{', '.join(self.vs.trusted)}", flush=True)
        if self.vs.dropped:
            print(f"  dropped (below their corpus mean): "
                  f"{', '.join(self.vs.dropped)}", flush=True)
        print(f"policy {policy.name}: step {ck.get('step')} net "
              f"{ck.get('net', float('nan')):+.5f} | history {self.H} "
              f"ticks {self.ticks}", flush=True)

    def reset(self) -> bytes:
        self.hist.clear()
        self.vs.i_am_left = None
        self.vs.last_my_x = None
        self.vs._id_score = None
        self.vs._prev_x = None
        self._last_cmd = 0.0
        return b"ok"

    def calibrate(self, payload: bytes) -> bytes:
        """Two frame pairs, before and after the client held a direction."""
        n = len(payload) // 2
        before = self._decode(payload[:n])
        after = self._decode(payload[n:])
        s0, _ = self.vs._raw(before)
        s1, _ = self.vs._raw(after)
        move = np.abs(s1[:, CH["x"]] - s0[:, CH["x"]])
        # A real press moved 177 units; a pad that was reaching nothing still
        # "moved" 19 vs 14 through encoder noise and the old 1e-4 threshold
        # (0.12 game units) waved it through. An identity guessed from noise
        # is a bot playing the opponent's half of the observation.
        # BOTH CHARACTERS MOVING IS THE NORMAL CASE, NOT A FAILED PROBE.
        #
        # The operator hit this: "when calibrating, it walks forward and pushes
        # me. I didn't press move, I was pushed by Cirno." Soku characters have
        # solid pushboxes, so sweeping into the opponent shoves them along, and
        # a ratio test reads two genuinely different displacements as a tie --
        # 148 vs 232 units scored margin 0.36 and was rejected.
        #
        # A RATIO is the wrong statistic here. It is scale-free, so it gets
        # *stricter* as the real signal grows: the same 84-unit separation
        # passes at 20-vs-104 and fails at 148-vs-232. What matters is whether
        # the separation clears the noise, and the noise is an absolute number
        # of game units -- about 60 of live positional jitter against a
        # character roughly 50 wide. So test the SEPARATION IN UNITS.
        MIN_UNITS, MIN_SEP = 40.0, 80.0
        sep = float(abs(move[0] - move[1])) * STAGE_SPAN
        margin = float(abs(move[0] - move[1]) / max(float(move.max()), 1e-9))
        if float(move.max()) * STAGE_SPAN < MIN_UNITS:
            return (f"no movement detected ({move.max()*STAGE_SPAN:.0f} units, "
                    f"need {MIN_UNITS:.0f}) -- the pad is not reaching the "
                    f"game").encode()
        if sep < MIN_SEP:
            return (f"AMBIGUOUS: {move[0]*STAGE_SPAN:.0f} vs "
                    f"{move[1]*STAGE_SPAN:.0f} units, separated by {sep:.0f} "
                    f"(need {MIN_SEP:.0f}, the live noise floor) -- that was "
                    f"noise, not a character; retry hands-off").encode()
        left = bool(move[0] > move[1])
        self.vs.i_am_left = left
        self.vs.last_my_x = float(s1[0 if left else 1, CH["x"]])
        self.hist.clear()
        return (f"agent is the {'LEFT' if left else 'RIGHT'} character "
                f"(moved {move[0]*STAGE_SPAN:.0f} vs {move[1]*STAGE_SPAN:.0f} "
                f"units, separated by {sep:.0f}; ratio margin {margin:.2f} "
                f"is reported but not tested -- being pushed is normal)"
                ).encode()

    # hp and spirit for both players, ego-ordered, read off the HUD by the
    # client. See `HUD_FLOATS` in play_cheat_match.
    N_HUD = 4

    def _decode(self, payload: bytes) -> np.ndarray:
        size = self.vs.size
        n = size * size * 6
        return np.frombuffer(payload, np.uint8, n).reshape(size, size, 6)

    def _hud(self, payload: bytes) -> np.ndarray | None:
        """The client's HUD reading, or None from a client that sends none.

        Kept optional so an older client still plays -- badly, on the encoder's
        guess, but it plays, and the operator sees `hud none` in the status
        line rather than a handshake failure mid-match.
        """
        n = self.vs.size * self.vs.size * 6
        if len(payload) < n + self.N_HUD * 4:
            return None
        return np.frombuffer(payload, np.float32, self.N_HUD, n).copy()

    @torch.no_grad()
    def decide(self, payload: bytes) -> bytes:
        """Frame pair -> `ticks` x 10 buttons, or an empty reply if not ready."""
        t0 = time.perf_counter()
        if self.vs.i_am_left is None:
            return b""                      # not calibrated; client holds
        # Identity from our OWN commands, before the reordering that depends
        # on it. `_raw` is screen-ordered; note_command accumulates which row
        # actually moved the way we told it to.
        frame = self._decode(payload)
        st_raw, _ = self.vs._raw(frame)
        self.vs.note_command(st_raw, self._last_cmd)
        st, pr = self.vs.read(frame)
        # THE HEALTH BAR, READ WHERE IT IS.
        #
        # The encoder sees 224x224 downscaled from 640x480, at which size the
        # bar is a thin strip and the spirit hexagons are gone. Measured on two
        # real matches it was beaten by a constant predicting the MEAN of both
        # bars, and on the difference between the players -- the only part that
        # says who is winning -- it scored R2 0.086 for hp and 0.000 for
        # spirit. Read off the 480px HUD instead, those are 0.971 and 0.879
        # against 30272 corpus frames of ground truth.
        #
        # These arrive ALREADY EGO-ORDERED. The characters swap sides all
        # match, but the HUD does not -- P1's bar is top-left throughout -- so
        # the client maps them once from its own player slot and no side
        # inference touches them.
        hud = self._hud(payload)
        if hud is not None:
            self.hud_reads += 1
            st[:, CH["hp"]] = np.clip(hud[:2], 0.0, 1.0)
            st[:, CH["spirit"]] = np.clip(hud[2:], 0.0, 1.0)
        self.hist.append((st, pr))
        if len(self.hist) < self.H:
            return b""                      # still filling the window
        s = torch.from_numpy(np.stack([h[0] for h in self.hist]))[None].to(self.device)
        p = torch.from_numpy(np.stack([h[1] for h in self.hist]))[None].to(self.device)
        # side is 0 because the vision reader has ALREADY put the agent first;
        # letting StateObs re-order from a physical chair index would swap them
        # back and play the opponent's half of the observation.
        sd = torch.zeros(1, dtype=torch.long, device=self.device)
        act = self.pol(self.obs(s, p, sd), sd, sample=True).actions[0]
        # Remember the horizontal direction we just asked for, averaged over
        # the tick block: it is the evidence for the NEXT frame's identity.
        a = act.detach().cpu().numpy().reshape(-1, 10)
        self._last_cmd = float((a[:, 3] > 0.5).mean() - (a[:, 2] > 0.5).mean())
        self.decides += 1
        dt = (time.perf_counter() - t0) * 1000
        self.dt_ms = dt if not self.dt_ms else 0.9 * self.dt_ms + 0.1 * dt
        # The encoder's own reading rides along with the buttons. It costs 264
        # bytes and lets the client difference it against the game's real state,
        # which turns every match into a held-out accuracy measurement for the
        # encoder ON THE ACTUAL GAME rather than on corpus video.
        return (act.to(torch.uint8).cpu().numpy().tobytes()
                + st.astype(np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    sess = Session(a.encoder.expanduser(), a.policy.expanduser(), a.device)
    print(f"device {a.device}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    print(f"listening on {a.host}:{a.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        # Nagle batches small writes; a 50-byte button reply must not wait for
        # company when the client is blocked on it.
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"client {addr[0]} connected", flush=True)
        try:
            while True:
                op, n = HDR.unpack(recv_exactly(conn, HDR.size))
                payload = recv_exactly(conn, n) if n else b""
                # One malformed request must not end the match. A missing
                # Pillow on this box killed the whole server mid-handshake and
                # the client saw only "peer closed"; an error belongs in the
                # reply, where the operator can read it.
                try:
                    if op == OP_DECIDE:
                        out = sess.decide(payload)
                    elif op == OP_CAL:
                        out = sess.calibrate(payload)
                    elif op == OP_RESET:
                        out = sess.reset()
                    elif op == OP_INFO:
                        # The client builds the frame pair at the gap THIS
                        # encoder was trained on and resizes to its input size.
                        # Reporting them rather than duplicating the constants
                        # means a swapped encoder cannot silently mismatch.
                        out = json.dumps({"size": sess.vs.size,
                                          "delta": sess.vs.delta,
                                          "ticks": sess.ticks,
                                          "history": sess.H,
                                          "trusted": list(sess.vs.trusted),
                                          "hud_floats": Session.N_HUD,
                                          "dropped": list(sess.vs.dropped)}).encode()
                    elif op == OP_PING:
                        out = (f"decides {sess.decides} | "
                               f"hud {sess.hud_reads} | "
                               f"{sess.dt_ms:.1f} ms").encode()
                    else:
                        out = b""
                except Exception as exc:                    # noqa: BLE001
                    traceback.print_exc()
                    out = f"SERVER ERROR: {type(exc).__name__}: {exc}".encode()
                conn.sendall(HDR.pack(op, len(out)) + out)
        except (ConnectionError, OSError, struct.error) as e:
            print(f"client gone: {e}  ({sess.decides} decisions, "
                  f"{sess.dt_ms:.1f} ms mean)", flush=True)
        finally:
            conn.close()
            sess.reset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
