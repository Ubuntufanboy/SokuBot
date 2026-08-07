"""Live frames from the game window, in the exact shape the encoder was trained on.

WHY FFMPEG AND NOT AN IN-PROCESS GRABBER
----------------------------------------
The corpus went through a specific chain, and the model has only ever seen its
output. From ``SokuFrameExtractor/runner/encode.py`` and
``sokubot/data/soku.py``:

    game backbuffer 640x480 BGRA
      -> vflip, scale=480:480:flags=lanczos   -> h264 crf 26
      -> decode, scale=224:224:flags=bilinear, rgb24, /255

Reimplementing ``vflip`` and two rescales against a different resampler is a
silent way to hand the encoder frames it has never seen -- lanczos is not
bilinear, and neither is OpenCV's lanczos bit-for-bit swscale's. Driving the
same ffmpeg filters that built the corpus removes that whole class of bug, at
the cost of one subprocess. ``data/soku.py`` reads the corpus through an ffmpeg
pipe for the same reason, so this is the codebase's existing idiom rather than a
new one.

THE FLIP IS NOT OPTIONAL AND NOT COSMETIC
-----------------------------------------
``sokubot/data/hud.py`` records that ``encode.py``'s ``vflip`` over-corrects, so
**the corpus is stored upside down** -- confirmed here by eye against
``Smashlytics/soku-frames-a`` shard A-0001: health bars along the bottom, spirit
orbs along the top, stage name mirrored. An X11 grab is the right way up, so the
live path has to flip it to match. Skip this and the encoder sees an orientation
that does not occur anywhere in 200 hours of training data.

DECIMATION HAPPENS IN THE FILTER GRAPH
--------------------------------------
The decision rate is 15 Hz (``frame_skip=4`` at 60 fps), so three of every four
grabbed frames are discarded. Dropping them inside ffmpeg means they are never
scaled and never cross the pipe: at 480x480 rgb24 that is 41 MB/s of pipe
traffic avoided for 10 MB/s kept.

WHAT THIS IS ALLOWED TO SEE
---------------------------
Window pixels and the clock. Nothing else. There is no code path here that
reads game memory and there must not be -- see ``docs/HANDOFF.md`` section 8.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The game renders at a fixed 640x480 (SokuFrameExtractor's config.hpp), and the
# corpus squash is horizontal only: 640 -> 480 wide, 480 tall. Not a crop -- the
# left and right edges are exactly where the players are while zoning.
GAME_W, GAME_H = 640, 480
SQUARE = 480
FPS = 60
FRAME_SKIP = 4                       # 60 fps -> the 15 Hz decision rate
FRAME_BYTES = SQUARE * SQUARE * 3    # rgb24


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Geometry:
    """Where on the X display the game window is, and which window it is.

    Both are kept because they support two different capture modes, and the
    choice between them is a real trade-off rather than a detail:

    * **By root region** -- always returns what is actually on screen, and what
      this uses. The cost is that the region is absolute, so the window must not
      move after the region is sampled.
    * **By window id** -- would be immune to movement, but **does not work
      here**. Tried against the real game and ffmpeg's x11grab fails with
      ``Cannot get the image data ... error_code:8`` (BadMatch) and then reads
      nothing at all: XGetImage cannot read Soku's window drawable. Kept behind
      `by_window` because it is the right answer on a setup where it works, but
      it is off by default because on this one it is not a trade-off, it is a
      failure.

    The movement problem is therefore solved by *when* the geometry is sampled
    rather than by avoiding coordinates. Soku was observed at +13+77 on one
    launch and +13+88 on the next, and a run that sampled the region as soon as
    the window appeared captured a strip of desktop along one edge for the whole
    session. Call `find_game_window` again after the game has settled, directly
    before opening the capture -- `refreshed()` does exactly that.
    """
    x: int
    y: int
    w: int
    h: int
    display: str
    window_id: int = 0

    @property
    def input_spec(self) -> str:
        return f"{self.display}+{self.x},{self.y}"

    def refreshed(self) -> "Geometry":
        """Re-locate the window now. Call directly before opening a capture."""
        return find_game_window(self.display)

    def moved_from(self, other: "Geometry") -> bool:
        return (self.x, self.y, self.w, self.h) != (other.x, other.y,
                                                    other.w, other.h)


def find_game_window(display: str | None = None,
                     pattern: str = "Touhou") -> Geometry:
    """Locate the Soku window by title, via xdotool + xwininfo.

    Matching on the title rather than the process is deliberate: the English
    release is launched as ``th123e.exe`` but appears in the process table as
    ``th123.exe`` (see ``runner/wine.py``), so the process name is ambiguous
    while the title bar is not -- it reads
    "Touhou Hisoutensoku ver1.10a (eng v1.1a)".
    """
    display = display or os.environ.get("DISPLAY", ":0")
    env = dict(os.environ, DISPLAY=display)
    for tool in ("xdotool", "xwininfo"):
        if not shutil.which(tool):
            raise CaptureError(f"{tool} not found on PATH")

    res = subprocess.run(["xdotool", "search", "--name", pattern],
                         capture_output=True, text=True, env=env)
    ids = [w for w in res.stdout.split() if w.strip()]
    if not ids:
        raise CaptureError(
            f"no window matching {pattern!r} on {display}. Is the game running, "
            f"and is it on this display?")

    # Several windows can match (Wine creates hidden helpers); take the one that
    # is actually the size the game renders at.
    best: Geometry | None = None
    for wid in ids:
        info = subprocess.run(["xwininfo", "-id", wid], capture_output=True,
                              text=True, env=env).stdout
        try:
            x = int(re.search(r"Absolute upper-left X:\s+(-?\d+)", info)[1])
            y = int(re.search(r"Absolute upper-left Y:\s+(-?\d+)", info)[1])
            w = int(re.search(r"Width:\s+(\d+)", info)[1])
            h = int(re.search(r"Height:\s+(\d+)", info)[1])
        except (TypeError, IndexError):
            continue
        g = Geometry(x, y, w, h, display, int(wid, 0))
        if (w, h) == (GAME_W, GAME_H):
            return g
        if best is None or w * h > best.w * best.h:
            best = g
    if best is None:
        raise CaptureError(f"found windows matching {pattern!r} but none readable")
    return best


def build_command(geom: Geometry, *, fps: int = FPS,
                  frame_skip: int = FRAME_SKIP,
                  by_window: bool = False) -> list[str]:
    """ffmpeg argv: grab the game and emit 15 Hz 480x480 rgb24.

    ``select`` runs before the scales so discarded frames are never resampled.
    ``-vsync 0`` (passthrough) is required for the same reason it is in
    ``data/soku.py``: without it ffmpeg duplicates frames to honour the output
    rate and the decimation silently stops decimating.
    """
    vf = (f"select='not(mod(n\\,{frame_skip}))',"
          f"vflip,"
          f"scale={SQUARE}:{SQUARE}:flags=lanczos")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "x11grab",
        "-framerate", str(fps),
        "-video_size", f"{geom.w}x{geom.h}",
        "-draw_mouse", "0",
        # Do not let ffmpeg build a latency-hiding buffer; a late frame is worth
        # less than no frame, and the whole loop is a race against the clock.
        "-fflags", "nobuffer", "-flags", "low_delay",
    ]
    if by_window and geom.window_id:
        # Coordinates are relative to the window, so the window may move freely.
        cmd += ["-window_id", str(geom.window_id), "-i", geom.display]
    else:
        cmd += ["-i", geom.input_spec]
    cmd += ["-vf", vf, "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return cmd


class WindowCapture:
    """A live 15 Hz stream of 480x480 RGB frames, scoped to a `with` block.

    Frames come out already flipped and squashed, i.e. in *corpus* orientation
    and geometry. Everything downstream -- the 224 bilinear rescale, JPEG for
    transport -- is the caller's choice, because those trade fidelity against
    bytes on a link whose behaviour is measured rather than assumed.
    """

    def __init__(self, geom: Geometry, *, fps: int = FPS,
                 frame_skip: int = FRAME_SKIP, log_path: Path | None = None,
                 by_window: bool = False):
        self.geom = geom
        self.fps = fps
        self.frame_skip = frame_skip
        self.log_path = log_path
        self.by_window = by_window
        self._proc: subprocess.Popen | None = None
        self._log = None
        self.frame_id = -1

    def __enter__(self) -> "WindowCapture":
        if not shutil.which("ffmpeg"):
            raise CaptureError("ffmpeg not found on PATH")
        argv = build_command(self.geom, fps=self.fps, frame_skip=self.frame_skip,
                             by_window=self.by_window)
        self._log = open(self.log_path, "wb") if self.log_path else subprocess.PIPE
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=self._log,
            bufsize=0, env=dict(os.environ, DISPLAY=self.geom.display),
        )
        return self

    def read(self, timeout_s: float = 2.0) -> tuple[int, np.ndarray]:
        """Block for the next frame. Returns (frame_id, uint8 [480,480,3] RGB).

        Fixed-size framing rather than a marker-delimited codec: a rawvideo
        frame is exactly FRAME_BYTES, so a short read is unambiguously a dead
        producer instead of something to resynchronise from.
        """
        if self._proc is None or self._proc.stdout is None:
            raise CaptureError("capture is not open")
        buf = bytearray()
        deadline = time.monotonic() + timeout_s
        while len(buf) < FRAME_BYTES:
            chunk = self._proc.stdout.read(FRAME_BYTES - len(buf))
            if not chunk:
                raise CaptureError(
                    f"ffmpeg closed the pipe after {len(buf)} bytes "
                    f"({self._stderr_tail()})")
            buf += chunk
            if time.monotonic() > deadline:
                raise CaptureError(f"frame timed out after {timeout_s}s")
        self.frame_id += 1
        arr = np.frombuffer(bytes(buf), np.uint8).reshape(SQUARE, SQUARE, 3)
        return self.frame_id, arr

    def _stderr_tail(self) -> str:
        if self._log is subprocess.PIPE and self._proc and self._proc.stderr:
            try:
                return self._proc.stderr.read(400).decode(errors="replace").strip()
            except Exception:
                pass
        return "see ffmpeg log"

    def __exit__(self, *exc) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._log not in (None, subprocess.PIPE):
            self._log.close()


class TrackedCapture:
    """A `WindowCapture` that re-opens itself when the game window moves.

    Root-region capture reads screen coordinates, not a window, so anything that
    moves the window silently redirects the stream at whatever is now in that
    rectangle. This is not hypothetical: mid-session the game moved from
    +13+77 to +913+241 and the "game frames" became a picture of a terminal.
    Nothing downstream can detect that -- the frames are the right shape, they
    are not black, and the HUD gate simply reports no battle -- so it has to be
    caught here.

    Re-locating costs an `xdotool`/`xwininfo` pair, which is far too expensive
    per frame at 15 Hz, so it is done on a timer. Re-opening costs ~250 ms of
    lost frames, which is why it only happens when the geometry actually
    changed.
    """

    def __init__(self, geom: Geometry, *, check_every_s: float = 2.0, **kw):
        self.geom = geom
        self.check_every_s = check_every_s
        self.kw = kw
        self._cap: WindowCapture | None = None
        self._next_check = 0.0
        self.moves = 0

    def __enter__(self) -> "TrackedCapture":
        self._cap = WindowCapture(self.geom, **self.kw).__enter__()
        self._next_check = time.monotonic() + self.check_every_s
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.__exit__(*exc)
            self._cap = None

    @property
    def frame_id(self) -> int:
        return self._cap.frame_id if self._cap else -1

    def read(self, timeout_s: float = 2.0):
        now = time.monotonic()
        if now >= self._next_check:
            self._next_check = now + self.check_every_s
            try:
                fresh = find_game_window(self.geom.display)
            except CaptureError:
                fresh = None            # window gone; keep the old region and
                                        # let the caller's timeout speak
            if fresh is not None and fresh.moved_from(self.geom):
                self.moves += 1
                self.geom = fresh
                self._cap.__exit__()
                self._cap = WindowCapture(fresh, **self.kw).__enter__()
        return self._cap.read(timeout_s=timeout_s)


def to_model_input(frame480: np.ndarray, size: int = 224) -> np.ndarray:
    """480x480 corpus-orientation RGB -> the encoder's 224x224 input.

    Bilinear, because that is what ``data/soku.py`` used for every training
    sample. The resampler is part of the distribution, not an implementation
    detail: lanczos here would sharpen edges the model learned as soft.
    """
    import cv2
    return cv2.resize(frame480, (size, size), interpolation=cv2.INTER_LINEAR)
