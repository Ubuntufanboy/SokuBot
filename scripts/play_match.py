"""Run a human-vs-agent match: hold the pad, record, and take commands.

    # start it, then drive it from another shell / another turn
    python -m scripts.play_match --server 192.168.1.130 --side 2 \
        --record results/match.mp4

    echo 'shot /tmp/s.png'  > /tmp/sokubot.ctl   # look at the screen
    echo 'press right right down' > /tmp/sokubot.ctl
    echo 'arm'   > /tmp/sokubot.ctl
    echo 'stop'  > /tmp/sokubot.ctl

WHY A LONG-LIVED PROCESS WITH A CONTROL FIFO
--------------------------------------------
Two constraints collide. The pad **must exist before the game starts**, because
Wine's dinput enumerates devices at init and never looks again -- so the pad's
lifetime has to span the whole session including the human's controller setup.
But the menu work (picking a character) happens interactively, minutes later,
in response to a person saying they are ready.

A single process that owns the pad for the whole session and takes commands on a
FIFO satisfies both. The alternative -- a fresh process per action -- creates a
new pad each time, and every one after the first is invisible to the game.

Character selection is driven *by looking* rather than from a hard-coded grid
position: `shot` writes a PNG, the operator reads it and sends the next moves.
The roster layout, the cursor's starting cell and the two players' independent
cursors are all things that would have to be assumed otherwise, and a wrong
assumption here picks the wrong character silently.

THE AGENT ONLY PLAYS WHEN BOTH GATES OPEN
-----------------------------------------
`BattleGate` (HUD pixels say a battle is running) and `ArmSwitch` (a human said
so). Menu navigation happens with the switch closed, so the policy is never
asked what to do on a screen it has never seen.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from sokubot.live.agent import Worker, kill_prefix, launch_game, report
from sokubot.live.capture import (CaptureError, TrackedCapture, find_game_window)
from sokubot.live.gate import ArmSwitch, BattleGate
from sokubot.live.pad import BUTTONS, VirtualKeypad
from sokubot.live.protocol import DEFAULT_PORT, JPEG_QUALITY, connect
from sokubot.live.schedule import ChunkScheduler, DelayPolicy

CTL = Path("/tmp/sokubot.ctl")
HELP = """commands:
  key <KEY_X> [...]       tap raw keys by evdev name (setup menus only)\n  p1 <ctl> [ctl ...]      tap PLAYER ONE's keys (setup menus only)\n  press <btn> [btn ...]   tap each in turn (up down left right a b c d change spell)
  hold <btn> <seconds>    hold one control
  shot [path]             screenshot the game, right way up
  arm / disarm            let the agent play, or stop it
  pause / resume          SIGSTOP the game so typing cannot reach it
  rec [path] / rec stop   start or finish recording the match
  status                  gate, arm state, scheduler counters
  stop                    end the session
"""


class Recorder:
    """A second ffmpeg writing the match to disk.

    Separate from the capture that feeds the model, and deliberately so: the
    model's stream is decimated to 15 Hz, squashed to 480x480 and vertically
    flipped, which is right for the encoder and wrong for anything a person
    watches. This one keeps the game's own 640x480 at 60 fps.
    """

    def __init__(self, geom, out: Path, fps: int = 60, crf: int = 20):
        self.geom, self.out, self.fps, self.crf = geom, out, fps, crf
        self._p: subprocess.Popen | None = None

    def __enter__(self) -> "Recorder":
        self.out.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "x11grab", "-framerate", str(self.fps),
            "-video_size", f"{self.geom.w}x{self.geom.h}",
            "-draw_mouse", "0", "-i", self.geom.input_spec,
            # ultrafast: this shares two physical cores with the game and the
            # model's own capture, and a recorder that steals frames from the
            # game has changed the thing it was supposed to observe.
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-threads", "1",
            "-movflags", "+faststart", str(self.out),
        ]
        self._p = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc) -> None:
        if self._p and self._p.poll() is None:
            # 'q' lets ffmpeg finalise the moov atom; killing it leaves an
            # unplayable file, which would be a sad way to lose the match.
            try:
                self._p.stdin.write(b"q")
                self._p.stdin.flush()
            except OSError:
                pass
            try:
                self._p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._p.terminate()


class Control(threading.Thread):
    """Reads newline-delimited commands from a FIFO."""

    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path = path
        self.queue: list[str] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        if path.exists():
            path.unlink()
        os.mkfifo(path, 0o666)

    def run(self) -> None:
        while not self.stop.is_set():
            # Opening a FIFO for reading blocks until a writer appears, and
            # returns EOF when the last one leaves -- so this reopens in a loop
            # rather than treating EOF as the end of the session.
            try:
                with self.path.open("r") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            with self.lock:
                                self.queue.append(line)
            except OSError:
                time.sleep(0.2)

    def take(self) -> list[str]:
        with self.lock:
            out, self.queue = self.queue, []
        return out


def tap(pad, name: str, hold_s: float = 0.05,
        gap_s: float = 0.45) -> None:
    """One menu input, short enough not to trigger Soku's auto-repeat.

    Measured: a 0.09 s hold (about 5 frames at 60 fps) reliably produced the
    intended step *and then one extra step* seconds later -- the game's menu
    repeat, sitting right on its threshold. 0.05 s is ~3 frames, clearly below
    it, and the longer gap lets any pending repeat expire before the next input
    rather than compounding with it.

    Menu navigation is still done closed-loop -- press, screenshot, check --
    because a repeat threshold found by measurement on one screen is not a
    guarantee about every screen.
    """
    pad.press_only(name)
    time.sleep(hold_s)
    pad.neutral()
    time.sleep(gap_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--server", required=True)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--side", type=int, choices=(1, 2), default=2)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--record", type=Path, default=Path("results/match.mp4"))
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ctl", type=Path, default=CTL)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--prefix", type=Path,
                    default=Path(os.environ.get("WINEPREFIX",
                                                Path.home() / ".wine-soku")))
    ap.add_argument("--attach", action="store_true",
                    help="the game is already running (NOT usually right: the "
                         "pad must exist before the game starts)")
    ap.add_argument("--settle", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=Path("results/match_log.json"))
    a = ap.parse_args()
    side0 = a.side - 1

    import cv2

    # The pad comes first, always. See sokubot/live/pad.py.
    with VirtualKeypad() as pad:
        print(f"keypad at {pad.device_path} (P2 = numpad; select profile \"sokubot\")", flush=True)
        launched = None
        if not a.attach:
            print("launching the game ...", flush=True)
            launched = launch_game(a.prefix, a.display, a.settle)
        geom = find_game_window(a.display).refreshed()
        print(f"game window {geom.w}x{geom.h} at +{geom.x}+{geom.y}", flush=True)

        ctl = Control(a.ctl)
        ctl.start()
        switch, gate = ArmSwitch(), BattleGate()
        sched = ChunkScheduler(ticks=a.ticks)
        delay = DelayPolicy(steps=a.steps, frame_skip=a.ticks)
        sock = connect(a.server, a.port)
        print(f"connected to {a.server}:{a.port}", flush=True)

        # Recording starts on command, not at launch: the interesting part is
        # the match, and the setup that precedes it can be many minutes of
        # menus. Held in a one-element list so the command handler can swap it.
        rec: list[Recorder | None] = [None]

        def rec_start(path: Path) -> None:
            if rec[0] is not None:
                print("already recording", flush=True)
                return
            rec[0] = Recorder(geom, path)
            rec[0].__enter__()
            print(f"recording -> {path}", flush=True)

        def rec_stop() -> None:
            if rec[0] is None:
                print("not recording", flush=True)
                return
            rec[0].__exit__()
            print(f"recording saved", flush=True)
            rec[0] = None

        with TrackedCapture(geom) as cap:
            worker = Worker(cap, sock, sched, delay, gate, switch, side0,
                            a.ticks, a.quality)
            worker.start()
            print(f"\nready. control fifo: {a.ctl}\n{HELP}", flush=True)
            tick = sched.tick_index()
            last_frame = {"f": None}

            # The capture is owned by the worker, so screenshots come from
            # whatever it last saw rather than a second grabber competing for
            # the same X server.
            def latest():
                return worker.last_frame

            try:
                while worker.is_alive():
                    for line in ctl.take():
                        parts = line.split()
                        cmd, args = parts[0].lower(), parts[1:]
                        if cmd == "press":
                            for b in args:
                                if b in BUTTONS:
                                    tap(pad, b)
                            print(f"pressed {' '.join(args)}", flush=True)
                        elif cmd == "hold" and len(args) == 2:
                            pad.press_only(args[0])
                            time.sleep(float(args[1]))
                            pad.neutral()
                            print(f"held {args[0]}", flush=True)
                        elif cmd == "shot":
                            p = Path(args[0]) if args else Path("/tmp/soku.png")
                            fr = latest()
                            if fr is None:
                                print("no frame yet", flush=True)
                            else:
                                cv2.imwrite(str(p), fr[::-1, :, ::-1])
                                print(f"shot -> {p}", flush=True)
                        elif cmd == "arm":
                            switch.arm()
                            print("[ARMED]", flush=True)
                        elif cmd == "disarm":
                            switch.disarm()
                            pad.neutral()
                            print("[DISARMED]", flush=True)
                        elif cmd == "status":
                            print(f"armed={switch.armed} battle={gate.in_battle} "
                                  f"bars={[round(v,2) for v in gate.last]} "
                                  f"sched={sched.stats()}", flush=True)
                        elif cmd == "key":
                            for k in args:
                                try:
                                    pad.tap_key(k)
                                except ValueError as e:
                                    print(e, flush=True)
                                time.sleep(0.45)
                            print(f"key {' '.join(args)}", flush=True)
                        elif cmd == "p1":
                            for c in args:
                                try:
                                    pad.tap_p1(c)
                                except ValueError as e:
                                    print(e, flush=True)
                                time.sleep(0.45)
                            print(f"p1 {' '.join(args)}", flush=True)
                        elif cmd in ("pause", "resume"):
                            # Wine's dinput reads evdev directly and ignores
                            # window focus, so while the game runs it reads the
                            # real keyboard no matter what window is focused.
                            # With P1 bound to `bleh` (up=SPACE) that means
                            # typing a message scrolls the menu upward, once per
                            # space. SIGSTOP is the only reliable way to make
                            # the game stop listening without closing it -- and
                            # it must not be closed, because the pad is only
                            # visible to a game process that started after it.
                            sig = ("STOP" if cmd == "pause" else "CONT")
                            n = 0
                            for d in Path("/proc").iterdir():
                                if not d.name.isdigit():
                                    continue
                                try:
                                    if (d / "comm").read_text().strip().startswith("th123"):
                                        os.kill(int(d.name),
                                                getattr(__import__("signal"),
                                                        f"SIG{sig}"))
                                        n += 1
                                except OSError:
                                    pass
                            print(f"{cmd}d {n} game process(es)"
                                  f"{' - safe to type' if cmd == 'pause' else ''}",
                                  flush=True)
                        elif cmd == "rec":
                            if args and args[0] == "stop":
                                rec_stop()
                            else:
                                rec_start(Path(args[1]) if len(args) > 1
                                          else a.record)
                        elif cmd == "stop":
                            raise KeyboardInterrupt
                        else:
                            print(f"? {line}\n{HELP}", flush=True)

                    target = sched.next_deadline(tick)
                    now = time.perf_counter()
                    if now < target:
                        time.sleep(min(target - now, 0.05))
                        if time.perf_counter() < target:
                            continue
                    elif now - target > 0.5:
                        tick = sched.tick_index()
                        continue
                    # The pad is only driven by the scheduler while armed; menu
                    # commands above drive it directly.
                    if switch.armed:
                        pad.set_state(sched.state_at(tick))
                    tick += 1
            except KeyboardInterrupt:
                print("\nstopping", flush=True)
            finally:
                worker.stop.set()
                ctl.stop.set()
                pad.neutral()
                worker.join(timeout=3)
                sock.close()
                if rec[0] is not None:
                    rec_stop()

        if worker.error:
            print(f"worker failed: {type(worker.error).__name__}: {worker.error}",
                  flush=True)
        report(worker.log, sched, a.out)
        if launched:
            kill_prefix(a.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
