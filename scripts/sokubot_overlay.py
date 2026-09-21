"""A small always-on-top strip showing what the agent is doing.

    python -m scripts.sokubot_overlay            # run beside the game
    python -m scripts.sokubot_overlay --once     # print what it WOULD show, no window

It reads the status file the harness writes (`sokubot/live/status.py`) and nothing else:
it cannot arm, disarm or steer the agent, and a crash in it cannot stop the pad.

IT NEVER COVERS THE GAME. Capture is a root-region grab of the game window's rectangle,
so anything drawn over the picture is fed to the agent as if it were the game
(`overlay_layout.py`). The strip is placed outside the game's rect, re-placed whenever the
window moves, and HIDDEN -- not placed anyway -- if there is nowhere to put it.

A stale or missing status reads OFFLINE, never the last thing it said: the reader binds the
file to the writer's pid and a heartbeat.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from sokubot.live.overlay_layout import Rect, place_outside
from sokubot.live.status import Status, read_status, summarise

BAR = (460, 46)
COLOURS = {                       # level -> (background, text)
    "ok":   ("#1f7a3a", "#ffffff"),
    "idle": ("#3a3f4b", "#e6e6e6"),
    "warn": ("#b36b00", "#ffffff"),
    "off":  ("#222222", "#8a8a8a"),
}


def detail(s: Status | None) -> str:
    """The second line: latency, when there is any."""
    if s is None or s.latency_p50_ms is None:
        return ""
    miss = f"  missed {s.missed_slot_pct:.0f}%" if s.missed_slot_pct is not None else ""
    return f"decide p50 {s.latency_p50_ms:.0f} / p99 {s.latency_p99_ms:.0f} ms{miss}"


def game_rect(display: str, override: Rect | None) -> Rect | None:
    if override is not None:
        return override
    try:
        from sokubot.live.capture import CaptureError, find_game_window
        g = find_game_window(display).refreshed()
        return Rect(g.x, g.y, g.w, g.h)
    except Exception:                                   # noqa: BLE001 -- no window yet
        return None


def parse_rect(text: str) -> Rect:
    x, y, w, h = (int(v) for v in text.split(","))
    return Rect(x, y, w, h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", type=Path, default=None, help="status file (default: the harness's)")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--game-rect", type=parse_rect, default=None, metavar="X,Y,W,H",
                    help="use this instead of looking for the window (testing)")
    ap.add_argument("--margin", type=int, default=12)
    ap.add_argument("--once", action="store_true", help="print the text and placement, open nothing")
    ap.add_argument("--seconds", type=float, default=0.0, help="exit after this long (testing)")
    a = ap.parse_args()

    if a.once:
        s = read_status(a.status)
        head, level = summarise(s)
        rect = game_rect(a.display, a.game_rect)
        pos = place_outside(rect, (1920, 1080), BAR, a.margin)
        print(f"{level:5s} | {head}\n      | {detail(s)}\n      | game {rect} -> overlay at {pos}")
        return 0

    import tkinter as tk
    root = tk.Tk()
    root.title("sokubot-overlay")
    root.overrideredirect(True)                 # no decorations, and never takes focus
    root.attributes("-topmost", True)
    label = tk.Label(root, text="", font=("monospace", 11, "bold"), anchor="w", justify="left",
                     padx=10, pady=4)
    label.pack(fill="both", expand=True)
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    state = {"pos": "unset", "rect": "unset", "t_place": 0.0, "hidden": False, "warned": False}
    t_end = time.time() + a.seconds if a.seconds else None

    def place() -> None:
        rect = game_rect(a.display, a.game_rect)
        pos = place_outside(rect, screen, BAR, a.margin)
        if pos == state["pos"] and rect == state["rect"]:
            return
        state["pos"], state["rect"] = pos, rect
        if pos is None:
            # Nowhere that is not on the picture. Hide rather than place it anyway.
            root.withdraw()
            state["hidden"] = True
            if not state["warned"]:
                print("overlay hidden: no room outside the game window", file=sys.stderr)
                state["warned"] = True
        else:
            root.geometry(f"{BAR[0]}x{BAR[1]}+{pos[0]}+{pos[1]}")
            if state["hidden"]:
                root.deiconify()
                state["hidden"] = False
            root.lift()

    def tick() -> None:
        s = read_status(a.status)
        head, level = summarise(s)
        bg, fg = COLOURS[level]
        second = detail(s)
        label.config(text=head + (f"\n{second}" if second else ""), bg=bg, fg=fg)
        root.configure(bg=bg)
        # The window can be moved or resized (WindowResizer), so re-check about once a second.
        if time.time() - state["t_place"] > 1.0:
            state["t_place"] = time.time()
            place()
        if t_end and time.time() > t_end:
            root.destroy()
            return
        root.after(250, tick)

    place()
    tick()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
