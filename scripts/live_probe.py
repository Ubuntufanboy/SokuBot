"""Go/no-go probes for the live control loop. Run these before building on them.

    python -m scripts.live_probe --check          # environment readiness
    python -m scripts.live_probe --cycle          # Probe A: drive every control
    python -m scripts.live_probe --hold a --seconds 3
    python -m scripts.live_probe --demo           # canned gameplay sequence
    python -m scripts.live_probe --hold-open      # create the pad and wait
    python -m scripts.live_probe --capture        # Probe B: capture cost
    python -m scripts.live_probe --frames         # Probe C: live-vs-corpus fidelity

**Probe A is the gate for the whole milestone.** The live loop assumes Wine's
DirectInput enumerates a uinput-created gamepad, and nothing downstream is worth
writing until that is observed rather than assumed. The claim being tested is
narrow and falsifiable: *a virtual pad created before the game starts appears in
Soku's controller config and its inputs move a character.*

Order of operations matters and is not negotiable. Wine's dinput enumerates
devices when it initialises, so:

    1. start this script (it creates the pad and holds it open)
    2. *then* launch the game
    3. bind player 2 to the pad in Config -> Controller
    4. drive it from here

A pad created after the game started will not be found, and the failure looks
exactly like "Wine cannot see uinput devices" -- which is the wrong conclusion
and an expensive one. ``runner/vkbd.py`` records the same trap for keyboards.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sokubot.live.pad import BUTTONS, VirtualPad, VirtualPadError, available

UDEV_FIX = """\
  echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \\
    | sudo tee /etc/udev/rules.d/99-uinput.rules
  sudo groupadd -f input && sudo usermod -aG input $USER
  sudo udevadm control --reload-rules && sudo udevadm trigger
  # then log out and back in (group membership is only applied at login)"""

WIFI_FIX = """\
  sudo iw dev wlan0 set power_save off                      # now
  printf '[connection]\\nwifi.powersave = 2\\n' \\
    | sudo tee /etc/NetworkManager/conf.d/wifi-powersave-off.conf   # persistent
  sudo systemctl restart NetworkManager"""


def _wifi_power_save() -> bool | None:
    """True if a wifi interface has power save on, None if not applicable."""
    iw = shutil.which("iw")
    if not iw:
        return None
    for dev in Path("/sys/class/net").glob("*"):
        if not (dev / "wireless").exists():
            continue
        if (dev / "operstate").read_text().strip() != "up":
            continue
        out = subprocess.run([iw, "dev", dev.name, "get", "power_save"],
                             capture_output=True, text=True).stdout
        if "Power save:" in out:
            return "on" in out.split("Power save:")[1].lower()
    return None


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def check() -> int:
    """Report everything the live loop needs, and how to fix what is missing."""
    ok = True

    def line(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else 'XX'}] {label:<26} {detail}")

    print("environment")
    can, why = available()
    line("/dev/uinput writable", can, why)

    node = Path("/dev/uinput")
    line("uinput module loaded", node.exists(),
         "present" if node.exists() else "modprobe uinput")

    wine = shutil.which("wine")
    line("wine on PATH", bool(wine), wine or "not found")

    prefix = Path(os.environ.get("WINEPREFIX", Path.home() / ".wine-soku"))
    line("wine prefix", prefix.is_dir(), str(prefix))

    game = prefix / "drive_c/Games/Soku"
    exe = game / "th123e.exe"
    # th123e.exe is the English release and the one the corpus was collected on.
    # th123.exe sitting next to it is the original Japanese build -- launching
    # that one gives a game whose UI the encoder has never seen.
    line("game executable", exe.exists(), str(exe))

    disp = os.environ.get("DISPLAY", "")
    line("DISPLAY set", bool(disp), disp or "no X display")

    print("\njoystick stack (what Wine's dinput reads through)")
    for p in ("/usr/lib/wine/x86_64-unix/winebus.so",
              "/usr/lib/wine/i386-windows/dinput.dll"):
        line(Path(p).name, Path(p).exists(), p)
    sdl = list(Path("/usr/lib").glob("libSDL2-2.0.so*"))
    line("libSDL2", bool(sdl), str(sdl[0]) if sdl else "not found")

    print("\nnetwork (the control loop's latency lives or dies here)")
    ps = _wifi_power_save()
    # Measured on this laptop: with power save on, an interleaved payload sweep
    # to the GPU box reported p50 191 ms for a 0.5 KB payload against 90 ms for
    # a 36 KB one -- impossible for a stationary path, and the signature of a
    # radio that parks between sparse packets while the AP buffers until the
    # next beacon. Both had a 24 ms floor, so the path is fine and the NIC is
    # not. Dense traffic hides it, which is why a relay that keeps a constant
    # flow appears to "speed up" the connection.
    if ps is None:
        line("wifi power save", True, "not on wifi, or iw unavailable")
    else:
        line("wifi power save off", not ps,
             "on -- adds ~100 ms to sparse traffic" if ps else "off")

    if ps:
        print("\nto disable wifi power save (needs sudo):")
        print(WIFI_FIX)
    if not can:
        print("\nto make /dev/uinput writable (one time, needs sudo):")
        print(UDEV_FIX)
    print()
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# pad drivers
# ---------------------------------------------------------------------------
def _wine_env(prefix: Path) -> dict:
    """Environment for a windowed, GPU-accelerated, vanilla launch.

    ``d3d9=b`` forces Wine's builtin d3d9 so the SWRSToys loader in the game
    folder is bypassed. That is deliberate for live play: the capture path is
    X11, not the extractor DLL, and the DLL crashes at load on this host's
    new-WoW64 Wine anyway. A vanilla game is also what the netplay phase needs,
    since gameplay-affecting mods have to match on both sides.
    """
    return dict(
        os.environ,
        WINEPREFIX=str(prefix),
        WINEDEBUG="-all",
        WINEDLLOVERRIDES="d3d9=b",
    )


def joycpl(prefix: Path) -> int:
    """Open Wine's joystick control panel so the pad can be seen from outside Soku.

    Worth doing before touching the game: it separates "uinput did not produce a
    device Wine recognises" from "Soku did not accept the device Wine offered",
    and those have completely different fixes.
    """
    print(f"opening joy.cpl in {prefix} -- the pad should be listed, and its "
          f"axes should move while --cycle runs in another terminal")
    return subprocess.call(["wine", "control", "joy.cpl"], env=_wine_env(prefix))


def cycle(pad: VirtualPad, hold_s: float, rounds: int) -> None:
    """Assert each control in turn, announcing it. Probe A's main instrument."""
    for r in range(rounds):
        for name in BUTTONS:
            print(f"  [{r + 1}/{rounds}] {name}", flush=True)
            pad.press_only(name)
            time.sleep(hold_s)
            pad.neutral()
            time.sleep(hold_s / 4)


def hold(pad: VirtualPad, name: str, seconds: float) -> None:
    """Assert one control for a while -- for binding it in Soku's config."""
    print(f"  holding {name} for {seconds:.1f}s", flush=True)
    pad.press_only(name)
    time.sleep(seconds)
    pad.neutral()


def demo(pad: VirtualPad) -> None:
    """A canned gameplay sequence, driven at the game's own 60 Hz tick."""
    TICK = 1.0 / 60.0

    def do(label: str, ticks: int, **pressed) -> None:
        print(f"  {label:<16} {ticks:>3} ticks  {sorted(pressed) or 'neutral'}",
              flush=True)
        v = [1 if b in pressed else 0 for b in BUTTONS]
        for _ in range(ticks):
            pad.set_state(v)
            time.sleep(TICK)

    do("walk right", 60, right=1)
    do("neutral", 15)
    do("walk left", 60, left=1)
    do("neutral", 15)
    do("jump", 20, up=1)
    do("neutral", 30)
    do("5A", 6, a=1)
    do("neutral", 20)
    do("2B", 8, down=1, b=1)
    do("neutral", 20)
    do("dash right", 30, right=1, d=1)
    do("neutral", 15)
    pad.neutral()


# ---------------------------------------------------------------------------
# --milestone1  (Phase 1: a character moves with no human and no model)
# ---------------------------------------------------------------------------
def milestone1(prefix: Path, settle_s: float, work: Path) -> int:
    """Launch, reach a battle using only the pad, then play a scripted round.

    This is the end of the actuation work: `BattleGate` decides when the game is
    in a battle from HUD pixels, and the pad plays a sequence into it. Nothing
    here involves the model, so a failure is unambiguously an actuation failure.

    The success test is *directional*, not just "the screen changed". Walking
    right and walking left move the character opposite ways, so the mean column
    of frame-to-frame change shifts in opposite directions. A game ignoring the
    pad, or a pad wired to the wrong axis, cannot produce that.
    """
    import numpy as np
    from sokubot.live.capture import (CaptureError, WindowCapture,
                                      find_game_window)
    from sokubot.live.gate import BattleGate

    game_dir = prefix / "drive_c/Games/Soku"
    with VirtualPad() as pad:
        print(f"pad at {pad.device_path}")
        proc = subprocess.Popen(["wine", "th123e.exe"], cwd=str(game_dir),
                                env=_wine_env(prefix), stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            geom = None
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and geom is None:
                try:
                    geom = find_game_window()
                except CaptureError:
                    time.sleep(1.0)
            if geom is None:
                print("game window never appeared", file=sys.stderr)
                return 1
            print(f"window {geom.w}x{geom.h} at +{geom.x}+{geom.y}; "
                  f"settling {settle_s:.0f}s")
            time.sleep(settle_s)
            # Re-locate after settling: the window can still move while the game
            # finishes starting, and a region sampled too early captures a strip
            # of desktop for the entire session.
            settled = geom.refreshed()
            if settled.moved_from(geom):
                print(f"  window moved to +{settled.x}+{settled.y} while "
                      f"settling; using the new region")
            geom = settled

            watch = HumanWatch(exclude=pad.device_path)
            gate = BattleGate()
            with WindowCapture(geom) as cap:
                print("navigating to a battle with the pad alone ...")
                deadline = time.monotonic() + 120
                i = 0
                while time.monotonic() < deadline:
                    _, fr = cap.read(timeout_s=10)
                    if gate.update(fr):
                        break
                    # Confirm, with the occasional direction so a menu that
                    # needs a selection does not sit forever on the same entry.
                    pad.press_only("a" if i % 3 else "down")
                    time.sleep(0.08)
                    pad.neutral()
                    i += 1
                if not gate.in_battle:
                    print(f"never reached a battle (bars {gate.last[0]:.2f}/"
                          f"{gate.last[1]:.2f})", file=sys.stderr)
                    return 1
                print(f"  battle detected, bars {gate.last[0]:.2f}/"
                      f"{gate.last[1]:.2f}")

                def motion(label: str, n: int, **pressed) -> float:
                    """Column centroid of frame-to-frame change while holding."""
                    v = [1 if b in pressed else 0 for b in BUTTONS]
                    pad.set_state(v)
                    _, prev = cap.read()
                    cols = []
                    for _ in range(n):
                        _, cur = cap.read()
                        d = np.abs(cur.astype(np.int16) -
                                   prev.astype(np.int16)).mean(axis=(0, 2))
                        if d.sum() > 1e-6:
                            cols.append(float((d * np.arange(len(d))).sum() / d.sum()))
                        prev = cur
                    pad.neutral()
                    c = float(np.mean(cols)) if cols else float("nan")
                    print(f"  {label:<14} change centroid x = {c:6.1f}")
                    return c

                right = motion("hold right", 12, right=1)
                left = motion("hold left", 12, left=1)
                import cv2
                work.mkdir(parents=True, exist_ok=True)
                _, shot = cap.read()
                cv2.imwrite(str(work / "milestone1.png"), shot[::-1, :, ::-1])
                print(f"  screenshot -> {work / 'milestone1.png'}")

            human = watch.drain()
            watch.close()
            if human:
                print(f"\n  => CONTAMINATED: {human} real key presses", file=sys.stderr)
                return 2
            print(f"\n  right - left = {right - left:+.1f} px")
            if right > left:
                print("  => PASS: the pad moves the character, in the right "
                      "direction on both axes.")
                return 0
            print("  => FAIL: activity did not shift the way holding right and "
                  "left should.")
            return 1
        finally:
            _kill_prefix(prefix)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


# ---------------------------------------------------------------------------
# --probe-a  (THE GATE)
# ---------------------------------------------------------------------------
def probe_a(prefix: Path, work: Path, settle_s: float) -> int:
    """Does a uinput pad created before launch actually reach the game?

    Everything downstream assumes it does, so this asks the question in the one
    form that cannot be argued with: launch the game, drive the pad, and see
    whether the screen responds differently than it does when nothing is driven.

    **The control is the whole point.** Soku's title and menu screens animate on
    their own, so "the pixels changed after I pressed something" proves nothing.
    This measures frame-to-frame change over a quiet window first, then over an
    identical window while the pad is driven, and compares. Without the quiet
    baseline the probe would pass on a game that ignores the pad completely.

    **The second control exists because the first run of this probe was
    contaminated.** A human at the keyboard during the driven window and not
    during the quiet one reproduces a large ratio perfectly, and that is exactly
    what happened -- the operator was navigating menus by hand. So the probe now
    watches every *real* keyboard and gamepad on the machine for the duration
    and refuses to report a result if any of them produced an event. Asking the
    operator to sit still is not a control; measuring whether they did is.
    """
    import numpy as np
    from sokubot.live.capture import (CaptureError, WindowCapture,
                                      find_game_window)

    work.mkdir(parents=True, exist_ok=True)
    game_dir = prefix / "drive_c/Games/Soku"
    if not (game_dir / "th123e.exe").exists():
        print(f"game not found at {game_dir}", file=sys.stderr)
        return 1

    with VirtualPad() as pad:
        print(f"pad at {pad.device_path} -- created BEFORE the game, which is "
              f"the part that matters: dinput enumerates at init.")
        proc = subprocess.Popen(["wine", "th123e.exe"], cwd=str(game_dir),
                                env=_wine_env(prefix),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            geom = None
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    geom = find_game_window()
                    break
                except CaptureError:
                    time.sleep(1.0)
            if geom is None:
                print("game window never appeared", file=sys.stderr)
                return 1
            print(f"game window {geom.w}x{geom.h} at +{geom.x}+{geom.y}")
            print(f"letting it settle for {settle_s:.0f}s ...")
            time.sleep(settle_s)

            watch = HumanWatch(exclude=pad.device_path)
            print(f"watching {len(watch.devs)} real input devices for "
                  f"contamination -- do not touch the keyboard or mouse")

            def activity(label: str, drive: bool, n: int = 30) -> tuple[float, int]:
                """Mean frame-to-frame change over n frames, and human events seen."""
                watch.drain()                       # discard anything pending
                with WindowCapture(geom) as cap:
                    _, prev = cap.read(timeout_s=10)
                    diffs, human = [], 0
                    for i in range(n):
                        if drive:
                            # Alternate a direction and confirm; on any Soku
                            # screen at least one of them does something.
                            pad.press_only(("down", "a", "up", "a")[i % 4])
                        _, cur = cap.read()
                        human += watch.drain()
                        diffs.append(np.abs(cur.astype(np.int16) -
                                            prev.astype(np.int16)).mean())
                        prev = cur
                    pad.neutral()
                m = float(np.mean(diffs))
                print(f"  {label:<28} mean |frame delta| {m:7.3f}"
                      f"   human events {human}")
                return m, human

            quiet, h1 = activity("no input (control)", False)
            driven, h2 = activity("driving the pad", True)
            watch.close()
            if h1 or h2:
                who = ", ".join(f"{k} x{v}" for k, v in watch.per_device.items())
                print(f"\n  => CONTAMINATED: {h1 + h2} real key/button presses "
                      f"during the run ({who}). Re-run without touching the "
                      f"machine; this result proves nothing either way.")
                return 2
            frames_dir = work / "probe_a"
            frames_dir.mkdir(exist_ok=True)
            with WindowCapture(geom) as cap:
                _, shot = cap.read(timeout_s=10)
            # Saved the right way up, since the capture is in corpus (flipped)
            # orientation and a human is going to look at this.
            import cv2
            cv2.imwrite(str(frames_dir / "screen.png"), shot[::-1, :, ::-1])
            print(f"  screenshot -> {frames_dir / 'screen.png'}")

            ratio = driven / max(quiet, 1e-6)
            print(f"\n  driven / quiet = {ratio:.2f}x")
            if ratio > 1.5:
                print("  => PASS: the pad reaches the game.")
                return 0
            print("  => INCONCLUSIVE or FAIL: driving the pad did not change "
                  "what the game does.\n     Before concluding Wine cannot see "
                  "the pad, check that the game is on a screen where these "
                  "buttons do something, and that P1 is bound to the pad in "
                  "Config -> Controller.")
            return 1
        finally:
            _kill_prefix(prefix)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


class HumanWatch:
    """Counts key/button presses from real input devices, so contamination shows.

    Opened non-blocking and drained rather than polled on a thread: the probe
    only needs to know *whether* a human touched anything between two points in
    time, and a counter read at the boundaries answers that without adding a
    thread that could itself perturb the timing.

    **Only EV_KEY counts.** The first version counted absolute-axis events too
    and flagged every run as contaminated, because this laptop has an
    `ST LIS3LV02DL Accelerometer` that reports EV_ABS continuously -- 17 events
    in a window where nobody touched anything. Motion is not a keypress. Keys
    and buttons are the only things a person can press, and they are what menu
    navigation produces.
    """

    def __init__(self, exclude: str):
        import evdev
        self.EV_KEY = evdev.ecodes.EV_KEY
        self.devs = []
        for path in evdev.list_devices():
            if path == exclude:
                continue
            try:
                d = evdev.InputDevice(path)
            except OSError:
                continue
            if self.EV_KEY in d.capabilities():
                self.devs.append(d)
        self.per_device: dict[str, int] = {}

    def drain(self) -> int:
        """Consume pending events; count only key/button presses."""
        n = 0
        for d in self.devs:
            try:
                while (ev := d.read_one()) is not None:
                    if ev.type == self.EV_KEY:
                        n += 1
                        self.per_device[d.name] = self.per_device.get(d.name, 0) + 1
            except (BlockingIOError, OSError):
                pass
        return n

    def close(self) -> None:
        for d in self.devs:
            try:
                d.close()
            except OSError:
                pass


def _kill_prefix(prefix: Path) -> None:
    """`wineserver -k` is the only teardown that reliably works.

    The launcher is ``th123e.exe`` but the game appears in the process table as
    ``th123.exe``, so waiting on the process we started does not mean the game
    is gone -- see ``SokuFrameExtractor/runner/wine.py``.
    """
    subprocess.run(["wineserver", "-k"],
                   env=dict(os.environ, WINEPREFIX=str(prefix),
                            WINEDEBUG="-all"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=20)


# ---------------------------------------------------------------------------
# --capture  (Probe B)
# ---------------------------------------------------------------------------
def capture_probe(seconds: float, geom_arg: str | None) -> int:
    """Measure what the capture path costs and prove it is not producing black.

    Two independent failure modes are being ruled out. Cost: the game and the
    grabber share two physical cores, so a grabber that wants a whole one is not
    affordable. Content: Wine's GL window can hand back undefined pixels under a
    compositor, and a black stream would sail through every shape check
    downstream while telling the encoder nothing.
    """
    from sokubot.live.capture import (CaptureError, Geometry, WindowCapture,
                                      find_game_window, to_model_input)
    import numpy as np

    if geom_arg:
        x, y, w, h = (int(v) for v in geom_arg.split(","))
        geom = Geometry(x, y, w, h, os.environ.get("DISPLAY", ":0"))
        print(f"using explicit geometry {geom}")
    else:
        try:
            geom = find_game_window()
            print(f"found game window at {geom}")
            if (geom.w, geom.h) != (640, 480):
                print(f"  WARNING: window is {geom.w}x{geom.h}, not 640x480. The "
                      f"capture geometry assumes the native render size; a "
                      f"scaled window silently changes the distribution.")
        except CaptureError as e:
            print(f"  {e}\n  falling back to the root region for a cost-only "
                  f"measurement")
            geom = Geometry(0, 0, 640, 480, os.environ.get("DISPLAY", ":0"))

    hz = os.sysconf("SC_CLK_TCK")
    with WindowCapture(geom) as cap:
        t0 = time.perf_counter()
        _, first = cap.read(timeout_s=10)
        print(f"  first frame after {(time.perf_counter() - t0) * 1000:.0f} ms")

        pid = cap._proc.pid

        def cpu() -> float:
            f = open(f"/proc/{pid}/stat").read().split()
            return (int(f[13]) + int(f[14])) / hz

        reads, resizes = [], []
        c0, t0 = cpu(), time.perf_counter()
        n, dark = 0, 0
        while time.perf_counter() - t0 < seconds:
            t = time.perf_counter()
            _, fr = cap.read()
            reads.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            m = to_model_input(fr)
            resizes.append((time.perf_counter() - t) * 1000)
            if fr.std() < 1.0:
                dark += 1
            n += 1
        wall, used = time.perf_counter() - t0, cpu() - c0

    reads.sort(); resizes.sort()
    p = lambda a, f: a[min(len(a) - 1, int(len(a) * f))]
    print(f"  rate        {n / wall:5.1f} Hz         (target 15.0)")
    print(f"  read()      p50 {p(reads, .5):5.1f}  p90 {p(reads, .9):5.1f} ms "
          f"(paced by the stream; near 66.7 means no backlog)")
    print(f"  resize 224  p50 {p(resizes, .5):5.2f}  p90 {p(resizes, .9):5.2f} ms")
    print(f"  ffmpeg cpu  {used / wall * 100:4.0f}% of one core")
    print(f"  black frames {dark}/{n}")
    ok = dark == 0 and n / wall > 13.0
    print(f"  => {'ok' if ok else 'PROBLEM'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# --frames  (Probe C)
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://huggingface.co/datasets/Smashlytics/soku-frames-a/"
              "resolve/main/shards/A-0001.tar")


def fetch_corpus_sample(out: Path, head_mb: int = 48) -> Path:
    """Pull one capture out of the public corpus without downloading a shard.

    A shard is ~1 GB and this needs one video. Tar is sequential and the repo is
    public, so a ranged GET of the first slice contains the first capture whole.
    """
    video = out / "video.mp4"
    if video.exists():
        return video
    import io, tarfile, requests
    out.mkdir(parents=True, exist_ok=True)
    n = head_mb * 1024 * 1024
    r = requests.get(CORPUS_URL, headers={"Range": f"bytes=0-{n - 1}"}, timeout=300)
    r.raise_for_status()
    tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r|")
    try:
        for m in tf:
            if not m.isfile() or not m.name.endswith("video.mp4"):
                continue
            data = tf.extractfile(m).read()
            if len(data) != m.size:        # truncated by the range; unusable
                break
            video.write_bytes(data)
            break
    except (tarfile.TarError, EOFError):
        pass
    if not video.exists():
        raise RuntimeError(f"no complete video.mp4 in the first {head_mb} MB")
    return video


def frames_probe(ckpt: Path, work: Path) -> int:
    """How far does the replay-mode overlay move the latent the policy consumes?

    The corpus is 100% *replay* footage and carries a `N HIT / Damage / Rate /
    Limit` block in the upper corner, following whichever player is being
    combo'd. A live Versus match may not draw it. Rather than guess whether that
    matters, measure it in latent space against two calibrations: one real
    gameplay step (the smallest change the model is ever asked to resolve) and
    unrelated frames (the scale of the space).
    """
    import numpy as np
    import torch
    from sokubot.config import Config
    from sokubot.model.world_model import LeWorldModel

    video = fetch_corpus_sample(work)
    print(f"  corpus sample {video} ({video.stat().st_size / 1e6:.0f} MB)")

    def decode(expr: str, count: int) -> np.ndarray:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
               "-vf", f"{expr},scale=224:224:flags=bilinear",
               "-vsync", "0", "-frames:v", str(count),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, 224, 224, 3).copy()

    START = 1200
    fr = decode(f"select='gte(n\\,{START})*not(mod(n-{START}\\,60))'", 24)
    adj = decode(f"select='gte(n\\,{START})*not(mod(n-{START}\\,4))'", 16)
    print(f"  frames {fr.shape[0]} sampled, {adj.shape[0]} adjacent")

    # The overlay sits in the upper corners in screen space. The corpus is
    # stored vertically flipped, which puts screen y 100..190 of 480 at 290..380
    # -> 135..178 at 224. x 0..110 and 370..480 -> 0..52 and 173..224.
    blank = fr.copy()
    blank[:, 135:178, 0:52] = 0
    blank[:, 135:178, 173:224] = 0

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: Config = blob["cfg"]
    model = LeWorldModel(cfg)
    model.load_state_dict(blob["model"])
    model.eval()
    torch.set_num_threads(4)

    def encode(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.inference_mode():
            return model.encoder(t)

    cs = torch.nn.functional.cosine_similarity
    z0, z1, za = encode(fr), encode(blank), encode(adj)
    over = cs(z0, z1, dim=-1)
    step = cs(za[:-1], za[1:], dim=-1)
    rand = cs(z0, z0[torch.randperm(len(z0))], dim=-1)

    print(f"  overlay blanked vs original  cosine {over.mean():.4f} "
          f"+- {over.std():.4f}")
    print(f"  one 66.7 ms gameplay step    cosine {step.mean():.4f} "
          f"+- {step.std():.4f}")
    print(f"  unrelated frames             cosine {rand.mean():.4f}")
    ratio = (1 - over.mean().item()) / max(1 - step.mean().item(), 1e-9)
    print(f"  => the overlay is {ratio:.2f}x one gameplay step")
    if ratio < 0.5:
        print("     small enough to ignore; no masking needed in the live path")
    else:
        print("     large enough to matter; blank the same region live so both "
              "sides are missing it equally")
    return 0


# ---------------------------------------------------------------------------
def profiles_probe(game: Path, agent: str = "sokubot") -> int:
    """Print the key-binding audit `pad.py` has advertised for a month.

    Wine's dinput ignores window focus, so a key the agent's device sends is
    delivered to whichever player's profile holds it -- both, if both do. This
    is the pre-flight for local Vs Player. Online it does not apply: the
    opponent is on their own machine.
    """
    from sokubot.live import profiles as pf
    try:
        r = pf.audit(game, agent)
    except (OSError, ValueError) as e:
        print(f"cannot read profiles under {game}: {e}", file=sys.stderr)
        return 1

    sel = r["selected"]
    slot = r["agent_slot"]
    print(f"game       {game}")
    print(f"selected   P1={sel['p1']}  P2={sel['p2']}")
    print(f"agent      {agent} -> "
          + (f"player {slot}" if slot else
             "NOT SELECTED for either slot -- the game will not read the pad"))
    print()
    print("bindings")
    for name, keys in sorted(r["keys"].items()):
        mark = "*" if name == agent else " "
        row = " ".join(f"{c}={k.removeprefix('KEY_')}" for c, k in keys.items())
        print(f" {mark}{name:11s} {row}")
    print()
    for line in r["pad_complaints"]:
        print(f"PAD MISMATCH  {line}")
    if not r["pad_complaints"]:
        print("pad agrees with the agent's profile on all ten controls")
    print()
    for name, c in sorted(r["collides_with_agent"].items()):
        detail = ", ".join(f"{k.removeprefix('KEY_')} ({a} vs {b})"
                           for k, (a, b) in sorted(c.items()))
        print(f"COLLIDES      {name}: {detail}")
    for name, c in sorted(r["benign_overlap"].items()):
        detail = ", ".join(k.removeprefix("KEY_") for k in sorted(c))
        print(f"  (benign)    {name}: {detail} -- the pad has no code for these")
    print()
    print("safe to play against locally: "
          + (", ".join(r["safe_opponent_profiles"]) or "NONE"))
    # A collision with the profile that is actually loaded is the only one that
    # can bite tonight, so it is the only one that fails the probe.
    bad = pf.live_collisions(r)
    if bad or r["pad_complaints"]:
        print(f"\nFAIL: {bad or 'pad mismatch'}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report environment readiness and exit")
    ap.add_argument("--joycpl", action="store_true",
                    help="open Wine's joystick control panel")
    ap.add_argument("--cycle", action="store_true",
                    help="drive every control in turn")
    ap.add_argument("--demo", action="store_true",
                    help="canned gameplay sequence at 60 Hz")
    ap.add_argument("--hold", metavar="CONTROL",
                    help=f"assert one control; one of {' '.join(BUTTONS)}")
    ap.add_argument("--hold-open", action="store_true",
                    help="create the pad and wait, so the game can start after it")
    ap.add_argument("--probe-a", action="store_true",
                    help="THE GATE: launch the game and prove the pad reaches it")
    ap.add_argument("--milestone1", action="store_true",
                    help="reach a battle with the pad alone and move a character")
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds to let the game reach its title screen")
    ap.add_argument("--capture", action="store_true",
                    help="measure capture cost and check for black frames")
    ap.add_argument("--geometry", metavar="X,Y,W,H",
                    help="capture region, instead of locating the game window")
    ap.add_argument("--frames", action="store_true",
                    help="corpus-vs-live frame fidelity in latent space")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("/home/anon/K0NTR0L-2/artifacts/ckpt_cf/"
                                 "best_bnfix.pt"))
    ap.add_argument("--work", type=Path,
                    default=Path.home() / ".cache/sokubot/corpus_sample")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--profiles", action="store_true",
                    help="audit profile/*.pf key bindings for collisions with "
                         "the agent, and say which slot loads which profile")
    ap.add_argument("--game", type=Path,
                    default=Path.home() / ".wine-soku/drive_c/Games/Soku",
                    help="game directory holding profile/ and config123.dat")
    ap.add_argument("--prefix", type=Path,
                    default=Path(os.environ.get("WINEPREFIX",
                                                Path.home() / ".wine-soku")))
    a = ap.parse_args()

    if a.profiles:
        return profiles_probe(a.game)
    if a.check:
        return check()
    if a.joycpl:
        return joycpl(a.prefix)
    if a.probe_a:
        return probe_a(a.prefix, a.work.parent, a.settle)
    if a.milestone1:
        return milestone1(a.prefix, a.settle, a.work.parent / "milestone1")
    if a.capture:
        return capture_probe(max(a.seconds, 8.0), a.geometry)
    if a.frames:
        return frames_probe(a.ckpt, a.work)

    if not any((a.cycle, a.demo, a.hold, a.hold_open)):
        # Everything below needs the pad; everything above does not.
        ap.print_help()
        return 2

    try:
        with VirtualPad() as pad:
            print(f"pad up at {pad.device_path}")
            print("start the game NOW if it is not already running -- dinput "
                  "enumerates at init and will not find a pad created later.\n")
            if a.hold_open:
                print("holding the pad open; Ctrl-C to release")
                while True:
                    time.sleep(1.0)
            if a.cycle:
                cycle(pad, a.seconds, a.rounds)
            if a.hold:
                hold(pad, a.hold, a.seconds)
            if a.demo:
                demo(pad)
    except VirtualPadError as e:
        print(f"pad unavailable: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nreleased")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
