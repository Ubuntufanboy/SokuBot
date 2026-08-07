"""The game-side half of the live loop: capture -> send -> schedule -> pad.

    python -m sokubot.live.agent --server 192.168.1.130 --side 2

WHAT IT IS ALLOWED TO TOUCH
---------------------------
It opens exactly three things: an ffmpeg reading the game *window's pixels*, a
TCP socket to the inference host, and a `/dev/uinput` gamepad. It reads real
keyboards for one purpose only -- the arm/disarm hotkey -- and never as an
observation. **No game memory is read here or anywhere downstream of here**, and
that is the project's central constraint, not an implementation preference
(`docs/HANDOFF.md` section 8).

TWO THREADS, AND WHY
--------------------
The pad has to be written on the game's 60 Hz tick, steadily. Capture and the
network run at 15 Hz and can block. Sharing one thread means every network
hiccup becomes a missed pad tick, which the game reads as the agent releasing
its buttons -- a real, punishable action (see `schedule.py`).

So the **main thread owns the pad** and does nothing that can block: it wakes on
each tick deadline, asks the scheduler what to press, and presses it. A worker
thread does capture, encode, send, receive, and hands finished chunks over.
Their only shared state is the scheduler, whose `submit`/`state_at` are a single
reference assignment and a read.

FIXED DELAY, NOT BEST EFFORT
----------------------------
Every chunk is scheduled for a tick `steps` decisions after the frame it came
from, whether or not it arrived early. Applying replies as soon as they land
makes the agent's reaction time a function of instantaneous network luck, which
is neither learnable by the opponent nor stable for the policy. A constant delay
plays like a player with a constant reaction time. The world model compensates
for that constant on the server side.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .capture import (CaptureError, TrackedCapture, WindowCapture,
                      find_game_window)
from .gate import ArmSwitch, BattleGate, may_act
from .pad import BUTTONS, VirtualPad
from .protocol import (DEFAULT_PORT, JPEG_QUALITY, chunk_bytes, connect,
                       pack_request, read_reply, unpack_chunk)
from .schedule import TICK_S, ChunkScheduler, DelayPolicy

# Soku binds none of the function keys, so this reaches the agent without also
# doing something in the game.
HOTKEY = "KEY_F12"


@dataclass
class Decision:
    """One round trip, for the post-match log."""
    frame_id: int
    capture_tick: int
    target_tick: int
    sent_at: float
    got_at: float
    steps: int
    armed: bool
    in_battle: bool

    @property
    def rtt_ms(self) -> float:
        return (self.got_at - self.sent_at) * 1000


class Hotkey(threading.Thread):
    """Watches real keyboards for the arm/disarm key.

    Global rather than terminal-bound on purpose: the game window has focus
    during a match, so a hotkey that needs the terminal focused is a hotkey that
    cannot be pressed when it is needed. Reading evdev requires the `input`
    group, which the uinput setup already grants.
    """

    def __init__(self, switch: ArmSwitch, exclude: str, key: str = HOTKEY):
        super().__init__(daemon=True)
        self.switch = switch
        self.exclude = exclude
        self.key = key
        self.stop = threading.Event()
        self.available = False
        try:
            import evdev
            self.code = getattr(evdev.ecodes, key)
            self.devs = []
            for path in evdev.list_devices():
                if path == exclude:
                    continue
                d = evdev.InputDevice(path)
                if evdev.ecodes.EV_KEY in d.capabilities():
                    self.devs.append(d)
            self.available = bool(self.devs)
        except Exception:
            self.devs = []

    def run(self) -> None:
        import select
        import evdev
        if not self.devs:
            return
        fds = {d.fd: d for d in self.devs}
        while not self.stop.is_set():
            r, _, _ = select.select(list(fds), [], [], 0.25)
            for fd in r:
                try:
                    for ev in fds[fd].read():
                        if (ev.type == evdev.ecodes.EV_KEY
                                and ev.code == self.code and ev.value == 1):
                            state = self.switch.toggle()
                            print(f"\n[{'ARMED' if state else 'DISARMED'}]",
                                  flush=True)
                except OSError:
                    pass


class Worker(threading.Thread):
    """Capture -> gate -> encode -> send -> receive -> submit."""

    def __init__(self, cap: WindowCapture, sock: socket.socket,
                 sched: ChunkScheduler, delay: DelayPolicy, gate: BattleGate,
                 switch: ArmSwitch, side: int, ticks: int, quality: int):
        super().__init__(daemon=True)
        self.cap, self.sock, self.sched = cap, sock, sched
        self.delay, self.gate, self.switch = delay, gate, switch
        self.side, self.ticks, self.quality = side, ticks, quality
        self.stop = threading.Event()
        self.log: list[Decision] = []
        self.error: Exception | None = None
        # The most recent frame, for screenshots. Published here rather than
        # grabbed by a second X client: two grabbers on one window compete, and
        # a screenshot that is not the frame the agent acted on is misleading
        # exactly when it matters.
        self.last_frame: np.ndarray | None = None

    def run(self) -> None:
        import cv2
        nbytes = chunk_bytes(self.ticks)
        try:
            while not self.stop.is_set():
                frame_id, frame = self.cap.read(timeout_s=5.0)
                self.last_frame = frame
                tick = self.sched.tick_index()
                in_battle = self.gate.update(frame)
                if not may_act(self.gate, self.switch):
                    # Still capture, still keep the gate warm, but do not ask
                    # for an action and do not let a stale chunk keep playing.
                    self.sched.submit(np.zeros((self.ticks, 10), np.float32),
                                      at_tick=tick, chunk_id=-1)
                    continue

                ok, buf = cv2.imencode(
                    ".jpg", frame[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    continue
                sent = time.perf_counter()
                self.sock.sendall(pack_request(frame_id, tick, self.side,
                                               buf.tobytes()))
                rid, start_tick, steps, bits = read_reply(self.sock, nbytes)
                got = time.perf_counter()

                chunk = unpack_chunk(bits, self.ticks)
                # The server echoes the tick it wants the chunk applied at. If
                # that tick has already passed the reply is late; submitting it
                # anyway would play a stale plan, so it is dropped and the
                # scheduler keeps holding.
                now_tick = self.sched.tick_index()
                if start_tick + self.ticks > now_tick:
                    self.sched.submit(chunk, at_tick=start_tick, chunk_id=rid)
                self.log.append(Decision(rid, tick, start_tick, sent, got,
                                         steps, self.switch.armed, in_battle))
        except Exception as e:                       # noqa: BLE001
            self.error = e


def launch_game(prefix: Path, display: str, settle_s: float) -> subprocess.Popen:
    """Start Soku vanilla and windowed on `display`.

    ``d3d9=b`` forces Wine's builtin d3d9, bypassing the SWRSToys loader in the
    game folder. Deliberate: the capture path is X11 rather than the extractor
    DLL, and that DLL crashes at load on this host's new-WoW64 Wine anyway.
    """
    game_dir = prefix / "drive_c/Games/Soku"
    proc = subprocess.Popen(
        ["wine", "th123e.exe"], cwd=str(game_dir),
        env=dict(os.environ, WINEPREFIX=str(prefix), WINEDEBUG="-all",
                 WINEDLLOVERRIDES="d3d9=b", DISPLAY=display),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            find_game_window(display)
            break
        except CaptureError:
            time.sleep(1.0)
    time.sleep(settle_s)
    return proc


def seek_battle(pad: VirtualPad, cap: WindowCapture, gate: BattleGate,
                timeout_s: float = 120) -> bool:
    """Drive the pad through menus until the HUD says a battle is running.

    For unattended testing against the CPU. In a human match the person picks
    the characters, which is the honest arrangement anyway -- the agent has no
    character conditioning and no business choosing the matchup.
    """
    deadline = time.monotonic() + timeout_s
    i = 0
    while time.monotonic() < deadline:
        _, fr = cap.read(timeout_s=10)
        if gate.update(fr):
            return True
        pad.press_only("a" if i % 3 else "down")
        time.sleep(0.08)
        pad.neutral()
        i += 1
    return False


def kill_prefix(prefix: Path) -> None:
    """`wineserver -k` is the only teardown that reliably works.

    We launch ``th123e.exe`` but the game runs as ``th123.exe``, so waiting on
    the process we started does not mean the game is gone (runner/wine.py).
    """
    subprocess.run(["wineserver", "-k"],
                   env=dict(os.environ, WINEPREFIX=str(prefix), WINEDEBUG="-all"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=20)


def run(args) -> int:
    # THE PAD IS CREATED FIRST, ALWAYS. Wine's dinput enumerates devices when it
    # initialises, so a pad that does not exist before the game starts is
    # invisible to it for the whole session. This is the single most important
    # ordering constraint in the live loop and it has already been broken once:
    # an earlier version of this function launched the game above this line, and
    # the agent sat pressing buttons at the title screen while the game ignored
    # every one of them. The failure is silent and looks like a broken pad.
    #
    # Structurally, not by comment: everything that can start the game lives
    # inside this block.
    with VirtualPad() as pad:
        print(f"pad at {pad.device_path}", flush=True)

        launched = None
        if args.launch:
            print(f"launching the game from {args.prefix} ...", flush=True)
            launched = launch_game(args.prefix, args.display, args.settle)

        geom = find_game_window(args.display)
        print(f"game window {geom.w}x{geom.h} at +{geom.x}+{geom.y}", flush=True)
        if (geom.w, geom.h) != (640, 480):
            print(f"  WARNING: expected 640x480. The capture chain assumes the "
                  f"native render size; a scaled window changes the "
                  f"distribution the encoder sees.")

        switch = ArmSwitch()
        gate = BattleGate()
        sched = ChunkScheduler(ticks=args.ticks)
        delay = DelayPolicy(steps=args.steps, frame_skip=args.ticks)

        hot = Hotkey(switch, exclude=pad.device_path, key=args.hotkey)
        hot.start()
        print(f"arm/disarm: {args.hotkey}"
              if hot.available else
              "arm/disarm hotkey unavailable (no readable keyboards); "
              "starting ARMED")
        if not hot.available:
            switch.arm()

        sock = connect(args.server, args.port)
        print(f"connected to {args.server}:{args.port}", flush=True)

        # Re-locate the window immediately before capture: it can move while the
        # game settles, and a region sampled too early bakes a strip of desktop
        # into every frame for the whole session.
        geom2 = geom.refreshed()
        if geom2.moved_from(geom):
            print(f"  window moved to +{geom2.x}+{geom2.y}; using the new region", flush=True)

        with WindowCapture(geom2) as cap:
            if args.seek_battle:
                print("driving the menus to a battle ...", flush=True)
                if not seek_battle(pad, cap, gate):
                    print(f"never reached a battle (bars {gate.last[0]:.2f}/"
                          f"{gate.last[1]:.2f})")
                    if launched:
                        kill_prefix(args.prefix)
                    return 1
                print(f"  battle detected, bars {gate.last[0]:.2f}/"
                      f"{gate.last[1]:.2f}")
                switch.arm()
                print("[ARMED]", flush=True)

            worker = Worker(cap, sock, sched, delay, gate, switch, args.side,
                            args.ticks, args.quality)
            worker.start()
            print("running; Ctrl-C to stop\n", flush=True)
            tick = sched.tick_index()
            t_end = (time.perf_counter() + args.duration
                     if args.duration else float("inf"))
            try:
                while worker.is_alive() and time.perf_counter() < t_end:
                    target = sched.next_deadline(tick)
                    now = time.perf_counter()
                    if now < target:
                        time.sleep(target - now)
                    elif now - target > 0.5:
                        # Fell far behind (a suspend, a long GC). Re-sync rather
                        # than sprinting through hundreds of stale ticks.
                        tick = sched.tick_index()
                        continue
                    pad.set_state(sched.state_at(tick))
                    tick += 1
            except KeyboardInterrupt:
                print("\nstopping", flush=True)
            finally:
                worker.stop.set()
                hot.stop.set()
                pad.neutral()
                worker.join(timeout=3)
                sock.close()

        if worker.error:
            print(f"worker failed: {type(worker.error).__name__}: "
                  f"{worker.error}")
        report(worker.log, sched, args.out)
    if launched:
        kill_prefix(args.prefix)
    return 0


def report(log: list[Decision], sched: ChunkScheduler, out: Path | None) -> None:
    st = sched.stats()
    print(f"\nticks {st['ticks']}  applied {st['applied']}  "
          f"held {st['held']} ({st['held_pct']}%)  idle {st['idle']}")
    if not log:
        print("no decisions were made (never armed, or never in a battle)", flush=True)
        return
    rtts = sorted(d.rtt_ms for d in log)
    q = lambda f: rtts[min(len(rtts) - 1, int(len(rtts) * f))]
    late = sum(1 for d in log if d.target_tick + sched.ticks
               <= d.capture_tick + sched.ticks)
    print(f"decisions {len(log)}   round trip p50 {q(.5):.1f}  p90 {q(.9):.1f}  "
          f"p99 {q(.99):.1f} ms")
    print(f"held fraction is the honest measure of whether it was really "
          f"playing: {st['held_pct']}%")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump({"stats": st,
                       "decisions": [d.__dict__ for d in log]}, fh, indent=1)
        print(f"log -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", required=True, help="inference host")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--side", type=int, choices=(1, 2), default=2,
                    help="which player the agent is; 2 keeps you on P1")
    ap.add_argument("--steps", type=int, default=1,
                    help="latency compensation, must match the server")
    ap.add_argument("--ticks", type=int, default=4, help="chunk length")
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--hotkey", default=HOTKEY)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--out", type=Path, default=Path("results/match_log.json"))
    ap.add_argument("--launch", action="store_true",
                    help="start the game first (for unattended testing)")
    ap.add_argument("--seek-battle", action="store_true",
                    help="drive the menus to a battle and arm automatically")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds; 0 means run until Ctrl-C")
    ap.add_argument("--settle", type=float, default=25.0)
    ap.add_argument("--prefix", type=Path,
                    default=Path(os.environ.get("WINEPREFIX",
                                                Path.home() / ".wine-soku")))
    a = ap.parse_args()
    a.side -= 1                       # the policy's `side` is 0-based
    try:
        return run(a)
    except CaptureError as e:
        print(f"capture: {e}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
