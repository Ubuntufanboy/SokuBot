"""Live shakedown of the play harness: null server, the REAL client and game, a human playing.

    python -m scripts.shakedown                        # records into runs/shakedown-<time>/
    python -m scripts.shakedown --mode neutral --no-outage

It is an INSTRUMENT, not a test with a verdict. It starts `serve_null` (no model) and the real
`play_cheat_match --server` client, which launches the game, creates the pad and starts the overlay,
then watches and records while a human plays:

    status_events.jsonl   every change of the status file the overlay reads, wall-clock stamped
    geometry.jsonl        overlay vs game window rectangles at each key event, + screenshots
    gate.csv              the battle gate's own bar readings and poll cost at 10 Hz
    cpu.jsonl             the client's CPU% and the load average every 5 s
    client.log            the client's own output; server.log likewise

Once the agent has been PLAYING for `--outage-after` seconds it kills the server for
`--outage-seconds`, then restarts it, so disarm-on-loss and reconnect are exercised live. It stops by
itself once it has seen the whole per-round flow -- calibrate, play, a hand-back after a KO, the outage
and its recovery, and a re-arm after the hand-back -- or at the time box, or when a `stop` file appears
in the output directory.

TALK TO THE HUMAN WITH DESKTOP POPUPS ONLY. The game reads the real keyboard from every window
(dinput ignores focus), so anything typed into a terminal during a run is also game input.

WHAT IT CANNOT TELL YOU. The null server answers "agent is ..." to every calibration whatever the pad
did, so a successful calibration proves nothing about the pad. Whether the pad reached the game has to
be read off the screenshots (the agent's sprite moves during a sweep). And the null server never plays,
so nothing here says anything about playing quality.

First run, 2026-09-20: the whole flow worked, and it found that the overlay did not follow the game
window when started before it (fixed in 821e023).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sokubot.live.overlay_layout import Rect, overlaps      # noqa: E402
from sokubot.live.status import read_status                  # noqa: E402

CTL = Path("/tmp/sokubot.ctl")


def flow_complete(seen: dict, outage: bool) -> bool:
    """Everything a run is for has been observed at least once."""
    need = ["playing", "handed_back", "rearm_after_handback"]
    if outage:
        need.append("recovered")
    return all(seen.get(k) for k in need)


def parse_window(xwininfo: str) -> Rect | None:
    """`xwininfo -id` output -> the window's absolute rectangle, if it is viewable."""
    if "IsViewable" not in xwininfo:
        return None
    try:
        return Rect(int(xwininfo.split("Absolute upper-left X:")[1].split()[0]),
                    int(xwininfo.split("Absolute upper-left Y:")[1].split()[0]),
                    int(xwininfo.split("Width:")[1].split()[0]),
                    int(xwininfo.split("Height:")[1].split()[0]))
    except (IndexError, ValueError):
        return None


def find_window(name: str, want_w: int | None = None) -> Rect | None:
    ids = subprocess.run(["xdotool", "search", "--name", name],
                         capture_output=True, text=True).stdout.split()
    for wid in ids:
        r = parse_window(subprocess.run(["xwininfo", "-id", wid],
                                        capture_output=True, text=True).stdout)
        if r is not None and (want_w is None or r.w == want_w):
            return r
    return None


def overlay_pids() -> list[int]:
    """Overlay processes, found by exact argv rather than `pgrep -f` (which matches itself)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"scripts.sokubot_overlay" in argv:
            out.append(int(d.name))
    return out


class Shakedown:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        self.t0 = time.time()
        self.seen = {"calibrating": 0, "playing": 0, "handed_back": 0, "outage": 0,
                     "recovered": 0, "rearm_after_handback": 0}
        self.snap_n = 0

    # ---- plumbing ------------------------------------------------------------------
    def now(self) -> float:
        return round(time.time() - self.t0, 2)

    def log(self, msg: str) -> None:
        print(f"[{self.now():7.1f}s] {msg}", flush=True)

    def notify(self, msg: str) -> None:
        if not self.a.no_popups:
            subprocess.run(["notify-send", "-u", "critical", "-t", "9000", "SokuBot shakedown", msg],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def append(self, name: str, obj: dict) -> None:
        with open(self.out / name, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def start_server(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.serve_null", "--mode", self.a.mode,
             "--port", str(self.a.port)],
            cwd=str(ROOT), stdout=open(self.out / "server.log", "a"), stderr=subprocess.STDOUT)

    def snapshot(self, event: str) -> dict:
        game, ov = find_window("Hisoutensoku", 640), find_window("sokubot-overlay")
        rec = {"t": self.now(), "event": event, "game": game and list(game),
               "overlay": ov and list(ov),
               "overlay_overlaps_game": overlaps(ov, game, 0) if game and ov else None,
               "gap_px": (ov.y - game.bottom) if game and ov else None}
        self.snap_n += 1
        if game:
            subprocess.run(["import", "-window", "root", "-crop",
                            f"{game.w + 80}x{game.h + 160}+{max(game.x - 30, 0)}+{max(game.y - 40, 0)}",
                            "+repage", str(self.out / f"snap{self.snap_n:02d}_{event}.png")],
                           capture_output=True)
        self.append("geometry.jsonl", rec)
        return rec

    @staticmethod
    def cpu_ticks(pid: int) -> int | None:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            return int(f[11]) + int(f[12])
        except OSError:
            return None

    # ---- the run -------------------------------------------------------------------
    def run(self) -> int:
        a = self.a
        server = self.start_server()
        time.sleep(1.0)
        env = {**os.environ, "SOKUBOT_GATE_LOG": str(self.out / "gate.csv"), "OMP_NUM_THREADS": "1"}
        client = subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.play_cheat_match", "--server", "127.0.0.1",
             "--port", str(a.port), "--close-game"],
            cwd=str(ROOT), env=env, stdout=open(self.out / "client.log", "w"), stderr=subprocess.STDOUT)
        self.log(f"server pid {server.pid}, client pid {client.pid} -> {self.out}")
        self.notify("Game is starting. Get into a Vs Player battle (you = P1), then press F12 after "
                    "FIGHT. Do NOT type in the terminal.")

        last_key = None
        playing_since = None
        stage, t_kill = 0, 0.0          # 0 before the outage, 1 server down, 2 restarted
        last_cpu = (time.time(), None)
        next_cpu = next_hud = 0.0
        done_at = None
        try:
            while True:
                t = time.time() - self.t0
                if client.poll() is not None:
                    self.log(f"client exited rc={client.returncode}")
                    break
                if t > a.max_seconds or (self.out / "stop").exists():
                    self.log("time box / stop file")
                    break
                if done_at and time.time() > done_at:
                    self.log("flow complete; stopping")
                    break

                s = read_status()
                key = None if s is None else (s.armed, s.gate_open, s.server_ok, s.busy,
                                              s.last_error, s.slot)
                if key != last_key:
                    self.append("status_events.jsonl", {
                        "t": self.now(), "offline": s is None,
                        **({} if s is None else {k: getattr(s, k) for k in (
                            "slot", "armed", "gate_open", "server_ok", "busy", "last_error",
                            "latency_p50_ms", "latency_p99_ms", "missed_slot_pct")})})
                    if s is not None:
                        self._on_change(s, stage)
                    last_key = key

                if s is not None and s.armed and s.gate_open and s.server_ok is not False:
                    playing_since = playing_since or time.time()
                else:
                    playing_since = None
                if (a.outage_after > 0 and stage == 0 and playing_since
                        and time.time() - playing_since >= a.outage_after):
                    server.terminate()
                    server.wait(3)
                    stage, t_kill = 1, time.time()
                    self.seen["outage"] = 1
                    self.log("SERVER KILLED")
                    self.append("status_events.jsonl", {"t": self.now(), "injected": "server_killed"})
                    self.notify(f"I killed the vision server on purpose. The overlay should say NO "
                                f"SERVER. It comes back in ~{a.outage_seconds:.0f} s.")
                if stage == 1 and time.time() - t_kill >= a.outage_seconds:
                    server = self.start_server()
                    stage = 2
                    self.log("SERVER RESTARTED")
                    self.append("status_events.jsonl", {"t": self.now(), "injected": "server_restarted"})
                    self.notify("Server restarted. Press F12 to re-calibrate and re-arm.")

                if t >= next_cpu:
                    next_cpu = t + 5
                    tk, tnow = self.cpu_ticks(client.pid), time.time()
                    if tk is not None and last_cpu[1] is not None:
                        pct = 100.0 * (tk - last_cpu[1]) / os.sysconf("SC_CLK_TCK") / (tnow - last_cpu[0])
                        self.append("cpu.jsonl", {"t": self.now(), "client_cpu_pct": round(pct, 1),
                                                  "load1": float(open("/proc/loadavg").read().split()[0])})
                    last_cpu = (tnow, tk)
                if t >= next_hud and s is not None and s.gate_open:
                    next_hud = t + 20
                    subprocess.run(["timeout", "2", "sh", "-c", f'echo hud > "{CTL}"'],
                                   capture_output=True)

                if not done_at and flow_complete(self.seen, a.outage_after > 0):
                    done_at = time.time() + 15
                    self.log("whole flow observed; stopping in 15 s")
                    self.notify("The shakedown has what it needs. It stops in ~15 s.")
                time.sleep(0.1)
        finally:
            self._shutdown(client, server)
        self.append("status_events.jsonl", {"t": self.now(), "summary": self.seen})
        self.log(f"summary {self.seen}")
        self.notify("Shakedown finished. The keyboard is yours again.")
        return 0 if flow_complete(self.seen, a.outage_after > 0) else 1

    def _on_change(self, s, stage: int) -> None:
        if s.busy and "CALIBRATING" in s.busy:
            self.seen["calibrating"] += 1
            self.snapshot("calibrating")
        if s.armed and s.gate_open:
            self.seen["playing"] += 1
            self.snapshot("playing")
            if self.seen["handed_back"]:
                self.seen["rearm_after_handback"] += 1
        if (not s.armed) and "round over" in (s.last_error or ""):
            self.seen["handed_back"] += 1
            self.snapshot("handed_back")
            self.log("HAND-BACK observed")
            self.notify("Round over: the agent handed back. Press F12 when the NEXT round starts "
                        "(hands off ~3 s).")
        if stage == 1 and s.server_ok is False:
            self.snapshot("no_server")
        if stage == 2 and s.server_ok is True and not self.seen["recovered"]:
            self.seen["recovered"] = 1
            self.log("server recovered in status")
            self.snapshot("recovered")

    def _shutdown(self, client, server) -> None:
        self.log("shutting down")
        if client.poll() is None:
            client.send_signal(signal.SIGINT)       # the client stops the overlay and, with
            try:                                    # --close-game, the game it launched
                client.wait(10)
            except subprocess.TimeoutExpired:
                client.kill()
        if server.poll() is None:
            server.terminate()
        for pid in overlay_pids():                  # only if the client died before its cleanup
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default runs/shakedown-<YYYYmmdd-HHMM>, which is gitignored)")
    ap.add_argument("--max-seconds", type=float, default=1500.0)
    ap.add_argument("--mode", choices=("wiggle", "neutral"), default="wiggle",
                    help="what the null server makes the agent do once armed")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--outage-after", type=float, default=20.0,
                    help="seconds of play before the server is killed (0 = no outage)")
    ap.add_argument("--no-outage", action="store_true")
    ap.add_argument("--outage-seconds", type=float, default=10.0)
    ap.add_argument("--no-popups", action="store_true")
    a = ap.parse_args()
    if a.no_outage:
        a.outage_after = 0.0
    a.out = a.out or ROOT / "runs" / time.strftime("shakedown-%Y%m%d-%H%M")
    a.out.mkdir(parents=True, exist_ok=True)
    return Shakedown(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
