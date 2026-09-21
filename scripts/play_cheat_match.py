"""THE CHEATING DIAGNOSTIC: the policy reads game memory and plays a human.

    # step 1, always: prove the reader agrees with the game
    python -m scripts.play_cheat_match --verify

    # step 2: play
    python -m scripts.play_cheat_match --policy ~/rl/grpo_sym/policy_best.pt

THIS VIOLATES THE INFERENCE CONSTRAINT ON PURPOSE. The shipped agent must
consume pixels and its own inputs; this one reads `hp`, positions and hitboxes
straight out of the process. It is a probe, not a product, and every artefact it
writes is named so nobody can mistake it for a real evaluation.

WHAT IT BUYS
------------
The policy scores +8 HP/step inside a world model that over-predicts damage
1.4-2x off the human action manifold. Whether that transfers is the single most
important open question and is unanswerable until it plays something real. The
pixels-to-state encoder does not exist yet and is a day of work. Feeding the
policy perfect state separates "the policy is weak" from "the perception is
weak" BEFORE paying for the encoder: play well here and the encoder is the whole
remaining problem; play badly and the simulator's number does not transfer,
which is worth knowing first.

THREE THINGS THAT ARE LOAD-BEARING
-----------------------------------
**The pad must exist before the game starts.** Wine's dinput enumerates devices
at init and never looks again. A pad created afterwards is invisible and the
failure looks exactly like "uinput does not work under Wine".

**The game must be a DESCENDANT of this process.** `kernel.yama.ptrace_scope`
is 1, so `/proc/<pid>/mem` is readable only for descendants. Wine complicates
this: a pre-existing `wineserver` adopts the game, and then it is nobody's
child. So any stale wineserver for this prefix is stopped first, and the fresh
one spawns underneath us. Skipping that step is what produced `PermissionError`
the first time this was attempted.

**Audio off, verified rather than assumed.** The documented overrides did not
mute the game under wine-11.6 and it played music on a shared laptop. The driver
is now disabled in the prefix registry AND the sink list is checked after
launch, because "I set the env var" is not evidence.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from sokubot.data.state import CH, FULL_HP, STAGE_SPAN
from sokubot.live.capture import find_game_window
from sokubot.live.hotkey import DEFAULT_KEY, KeyWatcher
from sokubot.live.latency import format_summary, summarise
from sokubot.live.memstate import Frame, LiveState, find_game_pid
from sokubot.live.pad import BUTTONS, VirtualKeypad
from sokubot.live.status import StatusWriter, fields_from
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.state_arena import StateObs
# Reused rather than reimplemented: the recorder finalises the mp4 properly on
# exit and the FIFO reader survives writers coming and going, both of which
# were learned the hard way in the pixel loop.
from scripts.play_match import Control, Recorder, tap

CTL = Path("/tmp/sokubot.ctl")


def mute_game_stream(tries: int = 20) -> bool:
    """Mute the game's audio STREAM, leaving its audio device working.

    Two wrong approaches came first. `WINEDLLOVERRIDES=winepulse=d;...` is what
    the project notes prescribe and it silently does nothing under wine-11.6 --
    the game came up playing music on a shared laptop. Disabling the driver in
    the prefix registry does mute it, and also breaks the game: Soku pops
    "Failed to initialize DirectSound object" and then renders a black window,
    because it wants a working sound device even if nothing is audible.

    So the device stays real and the STREAM gets muted, which is the layer the
    problem actually lives at. Polled because the sink-input does not exist
    until the game opens it, which is a second or two after the window appears.
    """
    for _ in range(tries):
        try:
            out = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"],
                                 capture_output=True, text=True, timeout=10).stdout
            import json as _json
            for si in _json.loads(out or "[]"):
                blob = _json.dumps(si).lower()
                if "th123" in blob or "wine" in blob:
                    subprocess.run(["pactl", "set-sink-input-mute",
                                    str(si["index"]), "1"],
                                   capture_output=True, timeout=10)
                    return True
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        time.sleep(1)
    return False


def stop_stale_wine(prefix: Path) -> None:
    """Stop any wineserver holding this prefix, so the game becomes our child.

    Kill by explicit lookup rather than `pkill -f wineserver`: that pattern
    appears in this very command line on some shells, and a self-match makes a
    live server look dead. /proc scanning cannot do that.
    """
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            cmd = Path(entry.path, "cmdline").read_bytes().decode("utf-8", "replace")
            env = Path(entry.path, "environ").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "wineserver" in cmd and str(prefix) in env:
            try:
                os.kill(int(entry.name), 15)
            except OSError:
                pass
    time.sleep(2)


def launch_game(game_dir: Path, prefix: Path, display: str,
                mods: bool = False) -> subprocess.Popen:
    env = {**os.environ, "DISPLAY": display, "WINEPREFIX": str(prefix),
           "WINEDEBUG": "-all",
           # Audio deliberately NOT disabled. This used to set
           # PULSE_SERVER=/nonexistent, which contradicts the comment thirty
           # lines below in main() -- "muting it broke the game once
           # (disabling the Wine driver made Soku fail DirectSound init and
           # render black)". Measured again here: the process came up, slept in
           # poll_schedule_timeout on 27 ticks of CPU over two minutes, and
           # never created a window. The operator wants the music on anyway.
           "WINEDLLOVERRIDES": "d3d9=b"}
    # THE MOD LOADER IS OFF BY DEFAULT, and that is why --preset exists as a switch.
    #
    # `d3d9=b` makes Wine use its own builtin d3d9 instead of the game folder's
    # d3d9.dll, which IS SokuModLoader, and the loader is the only thing that ever
    # reads ModLoaderSettings.json. Measured 2026-09-20 by mapping the running process:
    #
    #     d3d9=b        mod DLLs mapped: 0   d3d9 from /usr/lib/wine/.../d3d9.dll
    #     no override   mod DLLs mapped: 7   d3d9 from Games/Soku/d3d9.dll
    #                   (ReplayInputView+, SokuLobbiesMod, WindowResizer)
    #
    # So with the override, no module preset can have ANY effect -- not `ablation`, and
    # above all not `netplay`, whose entire point is loading giuroll. `mods=True` drops
    # the override so the loader runs and the preset means something.
    #
    # The override was added because the loader "rendered a pure black window" and popped
    # a SokuFrameExtractor dialog. The dialog is a module-set problem (turn the extractor
    # off, which `ablation` and `netplay` both do); the black window was very likely the
    # forced-software-GL hang described below.
    if mods:
        env.pop("WINEDLLOVERRIDES", None)
    # GL MODE, MEASURED TWICE AND THE ANSWER CHANGED.
    #
    # 2026-08-16: the host's Intel Haswell stack emitted "DRI3 error: Could not
    # get DRI3 device" and the game rendered a pure black window, so this
    # forced llvmpipe (software GL), which the capture containers always use.
    #
    # 2026-09-20: the opposite, same Wine 11.6 and Mesa 25.3.5. With
    # LIBGL_ALWAYS_SOFTWARE=1 th123.exe parks in ntsync_schedule and NEVER
    # creates a window (Mesa: "libEGL warning: Not allowed to force software
    # rendering when API explicitly selects a hardware device"); even a
    # brand-new prefix's wineboot hangs the same way. Hardware GL opens the
    # window in 8 s and renders the title screen correctly (frame checked: mean
    # RGB ~ (101, 84, 81), 1.4% near-black).
    #
    # So hardware is the default and SOKUBOT_GL=sw forces software. Whichever
    # you pick, LOOK at a frame before believing a run: the failure this
    # replaced was a black window that produced plausible-looking numbers.
    if os.environ.get("SOKUBOT_GL", "hw").lower() == "sw":
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    else:
        env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    # No setsid, no shell: the game has to stay in our process tree or
    # /proc/pid/mem is unreadable under ptrace_scope=1.
    return subprocess.Popen(["wine", "th123e.exe"], cwd=str(game_dir), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def audio_is_playing() -> bool:
    try:
        out = subprocess.run(["pactl", "list", "sink-inputs"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "th123" in out.lower()


# ---------------------------------------------------------------------------
# verification: does the reader agree with the game?
# ---------------------------------------------------------------------------
def verify(ls: LiveState, pad: VirtualKeypad, seconds: float = 20.0) -> int:
    """Causal checks, not a checksum.

    The gold-standard check is a frame-aligned join against the extractor's own
    CSV, which needs the capture container; that container is not available on
    this laptop (no rootfs, and `sfe-bwrap` runs `--unshare-pid` so a
    containerised game is not readable from the host anyway). So the reader is
    checked the way `pipeline/verify_extended.py` checks the extractor: by
    relationships that cannot hold by accident.

    This exists because the first draft of `memstate.py` GUESSED all four
    action-id ranges and every one was wrong. A wrong range does not crash --
    it produces a flag pinned at false, and `guarding` pinned at false is
    indistinguishable from an agent that never blocks, which is the exact
    question the whole experiment is about.
    """
    print("\nwatching the game for a battle ...", flush=True)
    t0 = time.time()
    frames = []
    while time.time() - t0 < seconds:
        f = ls.read()
        if f is not None:
            frames.append(f)
        time.sleep(1 / 60)
    if not frames:
        print("  no battle detected -- the BattleManager pointer stayed null.\n"
              "  Get into a match first, then re-run --verify.")
        return 1

    bf = np.array([f.battle_frame for f in frames])
    S = np.stack([f.state for f in frames])
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    d = np.diff(bf)
    check("battle_frame advances", (d >= 0).all() and (d > 0).any(),
          f"{bf[0]} -> {bf[-1]} over {len(bf)} reads, no backwards steps")
    hp = S[:, :, CH["hp"]]
    check("health in range", (hp >= -0.01).all() and (hp <= 1.01).all(),
          f"p1 {hp[:,0].min():.3f}..{hp[:,0].max():.3f}  "
          f"p2 {hp[:,1].min():.3f}..{hp[:,1].max():.3f} (fraction of {FULL_HP:.0f})")
    dx = S[:, 0, CH["dx"]]
    check("separation is a position", np.abs(dx).max() < 2.0,
          f"|dx| max {np.abs(dx).max():.3f} stage widths")
    # dx is defined as opponent-minus-me, so the two players' readings must be
    # exact negatives of each other. Nothing but a correct read produces this.
    mirror = np.abs(S[:, 0, CH["dx"]] + S[:, 1, CH["dx"]]).max()
    check("dx is antisymmetric", mirror < 1e-5,
          f"max |dx(p1) + dx(p2)| = {mirror:.2e}")
    face = S[:, :, CH["facing"]]
    check("facing is +-1", np.isin(face, (-1.0, 0.0, 1.0)).all(),
          f"values seen {sorted(set(face.ravel().tolist()))}")
    # Almost all the time the two face each other, which is a fact about the
    # game and not about the reader -- but a reader that mixed the two players
    # up would break it.
    opposed = float((face[:, 0] * face[:, 1] < 0).mean())
    check("players face each other", opposed > 0.8,
          f"{opposed:.1%} of frames")
    for name in ("guarding", "wrongblock", "crushed", "knockdown", "airborne"):
        v = S[:, :, CH[name]]
        check(f"{name} is a flag", np.isin(v, (0.0, 1.0)).all(),
              f"rate {v.mean():.4f}")
    print("\n  NOTE: a flag at rate 0.0000 is not proof of correctness -- it may\n"
          "  simply not have happened. Make it happen (block something, get\n"
          "  knocked down) and re-run before trusting that channel.")
    return 0 if ok else 2


HELP = """commands (write a line to /tmp/sokubot.ctl):
  shot [path]             screenshot the game window (default /tmp/soku.png)
  press <btn> [btn ...]   tap the AGENT's buttons: up down left right a b c d
                          change spell
  key <KEY_X> [...]       tap raw evdev keys
  whoami                  press a direction and report WHICH player moved
  state                   print the live reading for both players
  hud                     print both health bars as READ FROM THE SCREEN, and
                          which one --side says is ours. Check it against the
                          screen before arming: get this bit wrong and the
                          agent plays to the opponent's health.
  arm | fight             let the agent play. Refused until identity is known
                          (`whoami`), for a local run AND a --server run; the reason
                          is also shown in the status overlay.
  disarm | hands-off      stop playing and neutralise the pad. The game keeps
                          running, so this is safe where a pause is not.
  toggle                  arm if disarmed, disarm if armed (what the hotkey sends)
  latency                 per-decision p50/p90/p99 against the period, over
                          decisions that ran the policy
  rec [path] / rec stop   start or stop recording
  stop                    end the session
"""


def whoami(ls: LiveState, pad: VirtualKeypad) -> str:
    """Which player does the pad drive? Measured, not assumed.

    The agent's side depends on which player slot has the sokubot profile
    loaded, and that is set in-game -- the operator reports it was P1 last
    session and the notes assume P2. Rather than trust either, hold a direction
    and see whose x actually moves. This is the calibration the pixel path will
    need anyway; with memory it is exact.
    """
    before = ls.read()
    if before is None:
        return "no battle running -- get into a match first"
    pad.press_only("left")
    time.sleep(0.35)
    mid = ls.read()
    pad.neutral()
    time.sleep(0.2)
    after = ls.read()
    if mid is None or after is None:
        return "lost the battle state mid-probe"
    dx = [abs(float(mid.state[i, CH["x"]] - before.state[i, CH["x"]]))
          for i in (0, 1)]
    moved = int(np.argmax(dx))
    sep = abs(dx[moved] - dx[1 - moved])
    if max(dx) < 1e-6:
        return ("neither player moved. The pad is not reaching the game, or "
                "no slot has the sokubot profile.")
    if sep < 0.5 * max(dx):
        return (f"AMBIGUOUS: both moved ({dx[0]:.5f} vs {dx[1]:.5f}). Someone "
                f"else was pressing a key -- try again with hands off.")
    return (f"the pad drives PLAYER {moved + 1} "
            f"(moved {dx[moved]*STAGE_SPAN:.1f} units against "
            f"{dx[1-moved]*STAGE_SPAN:.1f} for the other)")


def describe(ls: LiveState) -> str:
    f = ls.read()
    if f is None:
        return "no battle (BattleManager is null)"
    out = [f"battle_frame {f.battle_frame}"]
    for i in (0, 1):
        s = f.state[i]
        out.append(
            f"  P{i+1} hp {s[CH['hp']]*FULL_HP:6.0f} spirit {s[CH['spirit']]:.2f} "
            f"x {s[CH['x']]*STAGE_SPAN:7.1f} dx {s[CH['dx']]*STAGE_SPAN:+7.1f} "
            f"face {s[CH['facing']]:+.0f} act {f.action[i]:4d} "
            f"guard {s[CH['guarding']]:.0f} wblk {s[CH['wrongblock']]:.0f} "
            f"kd {s[CH['knockdown']]:.0f} air {s[CH['airborne']]:.0f} "
            f"hb {s[CH['hitboxes']]*10:.0f} proj {s[CH['proj_n']]*10:.0f}")
    return "\n".join(out)


def load_agent(path: Path, device="cpu"):
    """Policy + its OWN observation normalisation, from the checkpoint.

    The normalisation travels with the weights rather than being recomputed:
    the policy's input space is defined by the statistics it trained under, and
    a live reader feeding it differently-scaled numbers is asking it to play a
    game it has never seen.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    H, ticks, slots = int(ck["history"]), int(ck["ticks"]), int(ck["slots"])
    obs = StateObs(np.zeros(33, np.float32), np.ones(33, np.float32),
                   np.zeros(7, np.float32), np.ones(7, np.float32), slots)
    obs.load_state_dict(ck["obs"])
    pol = SokuPolicy(obs.dim, H, ticks)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, obs, H, ticks, slots, ck


def decide(ls, pol, obs, hist, side, H):
    """One decision: read, append, and return `ticks` frames of buttons."""
    f = ls.read()
    if f is None:
        return None, None
    hist.append((f.state, f.proj))
    if len(hist) < H:
        return None, f
    s = torch.from_numpy(np.stack([h[0] for h in hist]))[None]
    p = torch.from_numpy(np.stack([h[1] for h in hist]))[None]
    sd = torch.tensor([side], dtype=torch.long)
    with torch.no_grad():
        o = obs(s, p, sd)
        act = pol(o, sd, sample=True).actions[0]
    return act, f


class TruthObserver:
    """The game's own state, read alongside the match and fed to NOBODY.

    THIS DOES NOT TOUCH THE POLICY. The inference constraint is that the agent
    consumes pixels and its own inputs; it says nothing about what an
    instrument may watch. The policy's observation comes from the encoder over
    the socket and this object is never in that path -- it has no route to the
    server and the pilot never passes it anything. Two jobs:

    KNOWING WHEN THE MATCH ENDS. `LiveState.read()` returns None whenever the
    scene is not a live battle, which is exactly the signal that was missing
    when the operator had to ask for the run to be killed by hand. Rounds
    inside a set stay in the battle scene, so this fires on the SET ending
    rather than between rounds.

    MEASURING THE ENCODER ON THE REAL GAME. Every decision records what the
    encoder said beside what was actually true, which is a held-out accuracy
    measurement on the thing we actually care about instead of on corpus video
    the encoder was fitted near.
    """

    def __init__(self, log_path: Path, side: int = 0):
        # Which PLAYER the agent holds. Not inferred: it is which profile slot
        # the operator put sokubot in, and it is what makes the comparison
        # below line up row for row.
        self.side = int(side)
        self.ls = None
        self.log_path = Path(log_path)
        self.truth: list = []
        self.enc: list = []
        self.acts: list = []
        self.blind_run = 0
        self.reads = 0
        self.last = None
        try:
            pid = find_game_pid()
            if pid is not None:
                self.ls = LiveState.attach(pid)
                self.pid = pid
        except (PermissionError, OSError) as e:
            print(f"  truth reader unavailable ({e}); match-end detection will "
                  f"fall back to the static-screen watchdog", flush=True)

    @property
    def ok(self) -> bool:
        return self.ls is not None

    def sample(self, enc_state, act):
        """-> True while a battle is live, False when it is not."""
        if self.ls is None:
            return True
        f = self.ls.read()
        self.reads += 1
        if f is None:
            self.blind_run += 1
            return False
        self.blind_run = 0
        self.last = f
        if enc_state is not None:
            self.truth.append(f.state.copy())
            self.enc.append(np.asarray(enc_state, np.float32))
            self.acts.append(np.asarray(act, np.float32))
        return True

    def summary(self) -> str:
        """Score the observation the policy actually received, two ways.

        TWO BUGS LIVED IN THE OLD VERSION OF THIS METHOD, and both of them
        produced confident wrong conclusions about the encoder.

        ORDERING. `E` is ego-ordered, me first; `T` is player-major. The old
        code differenced them directly and called the result R2, which for an
        agent holding player 2 compares every channel against the wrong
        character. It reported x at -1.865 and dx at -2.342 -- "worse than a
        constant" -- for an encoder whose x was fine. Truth is reordered by
        `side` here, which is a KNOWN constant (which slot holds the agent's
        profile), not the inferred side. Any residual mismatch on the position
        channels is then a real measurement of the identity error rather than
        an artifact of the instrument.

        DENOMINATOR. Per-row R2 is flattered by anything the two players share.
        The two health bars correlate at +0.73, so a model that predicts their
        AVERAGE for both rows scores about 0.87 per row while knowing nothing
        about either player -- which is exactly what the encoder was doing, and
        why it passed its gate. `R2 diff` scores the p1-p2 difference: the part
        that says who is winning, and the only part a policy can act on.
        """
        if not self.truth:
            return "no ground-truth samples"
        T = np.stack(self.truth); E = np.stack(self.enc)
        order = [self.side, 1 - self.side]
        Tp = T[:, order]
        out = [f"{len(T)} paired samples written to {self.log_path.name}",
               f"  truth ordered as [me=P{self.side + 1}, foe], to match the "
               f"ego-ordered observation",
               f"    {'channel':<9}{'R2 row':>8}{'R2 diff':>9}"
               f"{'R2 if avg':>11}"]
        np.savez_compressed(self.log_path, truth=T, encoder=E,
                            actions=np.stack(self.acts),
                            side=np.int32(self.side))
        # y and dy are ON WATCH, not yet dropped. One match put them at
        # live-diff R2 -0.019 and -0.020 -- no information about the height
        # DIFFERENCE -- against a corpus R2 of 0.707 and 0.534. That is one
        # clean sample on one stage, and the corpus fit is strong enough that
        # dropping them on it would be the same mistake in the other
        # direction. They are reported every match so the second sample
        # settles it.
        for name in ("x", "dx", "airborne", "y", "dy", "vx", "hp", "spirit"):
            c = CH[name]
            t, e = Tp[:, :, c], E[:, :, c]
            ss = max(float(((t - t.mean()) ** 2).sum()), 1e-9)
            r2 = 1 - float(((e - t) ** 2).sum()) / ss
            avg = np.tile(t.mean(1, keepdims=True), (1, 2))
            r2a = 1 - float(((avg - t) ** 2).sum()) / ss
            dt, de = t[:, 0] - t[:, 1], e[:, 0] - e[:, 1]
            sd = max(float(((dt - dt.mean()) ** 2).sum()), 1e-9)
            r2d = 1 - float(((de - dt) ** 2).sum()) / sd
            flag = "  <-- beaten by the average" if r2a > r2 else ""
            out.append(f"    {name:<9}{r2:>8.3f}{r2d:>9.3f}{r2a:>11.3f}{flag}")
        return "\n".join(out)


class RemoteBrain:
    """The encoder and policy, on the other end of a socket.

    The game host measured 82-153 ms for one encoder forward alone against an
    83 ms decision period, on an idle machine with the game not running. It
    cannot run the model, and the project rule after 2026-08-16 is that it must
    not try: it captures, it drives the pad, nothing else.

    Resizing happens HERE rather than on the server, because 224x224x6 is
    301 KB against the raw pair's 1.38 MB -- 6 ms on the wire instead of 26 ms
    at the 12 Hz decision rate. `VisionState._resize` is a static method, so
    this is bit-for-bit the same PIL BOX filter the server would have applied,
    and the server's `_raw` passes an already-correct size through unchanged.
    """

    # hp and spirit for both players, ego-ordered: [my_hp, foe_hp, my_sp, foe_sp]
    HUD_FLOATS = 4

    def __init__(self, host: str, port: int, size: int, timeout: float = 2.0,
                 my_player: int = 1):
        import socket
        # Is the server answering? The last call decides. `last_error` is what the
        # status channel shows, because "NO SERVER" alone does not say whether the
        # process died, the network dropped or it merely stopped replying.
        self.ok = True
        self.last_error = ""
        # Has the server established which character is the agent? Until it has, it
        # answers every decide with nothing and the pad sits idle -- which looks
        # exactly like a working, armed agent that is choosing to do nothing.
        self.identified = False
        # WHICH HEALTH BAR IS MINE.
        #
        # 0 = the agent holds player 1, whose bar is top-left. This is the one
        # thing about the HUD that has to be told rather than seen, and unlike
        # the SIDE it never changes during a match -- the characters swap sides
        # freely, the bars do not move. `hud` in the console prints both bars
        # so the operator can check it against the screen in one glance.
        self.my_player = int(my_player)
        self.hud_ok = 0
        self.hud_fail = 0
        self.host, self.port, self.size = host, port, size
        self.timeout = timeout
        # One request in flight at a time. The pilot thread and the console thread
        # both call the server, and two interleaved request/reply pairs on one
        # socket read each other's replies.
        self._lock = threading.Lock()
        self._backoff, self._next_try = 0.5, 0.0
        self.sock = self._open()
        self.rtt_ms = 0.0

    def _open(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def _reconnect(self) -> None:
        """Open a fresh connection, at most once per backoff interval. Raises if it cannot.

        The server FORGETS who the agent is when a connection closes
        (serve_vision resets its session in a `finally`), and a restart may be a
        different build. So a reconnect clears `identified`, and refuses a server
        whose cadence or input size changed under us: the pilot's pacing and the
        frames it ships were set from the old one.
        """
        import json
        now = time.monotonic()
        if now < self._next_try:
            raise ConnectionError(f"server down; next reconnect in {self._next_try - now:.1f}s")
        try:
            sock = self._open()
        except OSError as e:
            self._backoff = min(self._backoff * 2, 4.0)
            self._next_try = now + self._backoff
            self.last_error = f"reconnect failed: {type(e).__name__}: {e}"
            raise
        old, self.sock = self.sock, sock
        try:
            old.close()
        except OSError:
            pass
        self.identified = False
        spec = getattr(self, "spec", None)
        if spec is not None:
            try:
                got = json.loads(self._exchange(b"I", b"").decode())
            except (OSError, ValueError) as e:
                self._next_try = now + 4.0
                self.last_error = f"reconnected but could not read the server's spec: {e}"
                raise ConnectionError(self.last_error) from e
            for k in ("ticks", "history", "size"):
                if got.get(k) != spec.get(k):
                    self._next_try = now + 4.0
                    self.last_error = (f"server changed its {k} ({spec.get(k)} -> "
                                       f"{got.get(k)}); restart the client")
                    raise ConnectionError(self.last_error)
        self._backoff, self._next_try = 0.5, 0.0

    def _exchange(self, op: bytes, payload: bytes) -> bytes:
        from sokubot.live.wire import HDR, recv_exactly
        self.sock.sendall(HDR.pack(op, len(payload)) + payload)
        rop, n = HDR.unpack(recv_exactly(self.sock, HDR.size))
        return recv_exactly(self.sock, n) if n else b""

    def _call(self, op: bytes, payload: bytes = b"") -> bytes:
        with self._lock:
            if not self.ok:
                self._reconnect()
            t0 = time.perf_counter()
            try:
                out = self._exchange(op, payload)
            except OSError as e:        # ConnectionError and socket.timeout are both OSError
                self.ok, self.last_error = False, f"{type(e).__name__}: {e}"
                raise
            self.ok, self.last_error = True, ""
            dt = (time.perf_counter() - t0) * 1000
            self.rtt_ms = dt if not self.rtt_ms else 0.9 * self.rtt_ms + 0.1 * dt
            return out

    def _shrink(self, pair: np.ndarray) -> bytes:
        from sokubot.live.visionstate import VisionState
        a = VisionState._resize(pair[:, :, :3], self.size)
        b = VisionState._resize(pair[:, :, 3:], self.size)
        return np.ascontiguousarray(np.concatenate([a, b], axis=2)).tobytes()

    def read_hud(self, pair: np.ndarray) -> np.ndarray | None:
        """The health and spirit bars off the FULL-RESOLUTION frame.

        This runs on the game host rather than the server because the server
        only ever receives the 224px shrink, and 224px is exactly the size at
        which these two channels stop being readable. It is a few hundred
        microseconds of numpy on two fixed slices, against an 83 ms budget.

        Returned EGO-ORDERED: me first. The mapping is by player slot, not by
        side, because the bars do not move when the characters cross.
        """
        from sokubot.data.hud import read_frame
        try:
            hp, sp = read_frame(np.ascontiguousarray(pair[:, :, 3:]))
        except (ValueError, IndexError):
            self.hud_fail += 1
            return None
        self.hud_ok += 1
        me, foe = self.my_player, 1 - self.my_player
        return np.array([hp[me], hp[foe], sp[me], sp[foe]], dtype=np.float32)

    def decide(self, pair: np.ndarray):
        """-> (buttons [ticks,10], encoder_state [2,33]) or (None, None)."""
        body = self._shrink(pair)
        hud = self.read_hud(pair)
        if hud is not None:
            body += hud.tobytes()
        out = self._call(b"D", body)
        if not out:
            return None, None
        n_act = int(self.spec["ticks"]) * 10
        act = np.frombuffer(out[:n_act], np.uint8).reshape(-1, 10).astype(np.float32)
        enc = np.frombuffer(out[n_act:], np.float32).reshape(2, -1).copy()
        return act, enc

    def calibrate(self, before: np.ndarray, after: np.ndarray) -> str:
        msg = self._call(b"C", self._shrink(before) + self._shrink(after)).decode()
        self.identified = "agent is" in msg
        return msg

    def reset(self) -> str:
        out = self._call(b"R").decode()
        self.identified = False
        return out

    def ping(self) -> str:
        return self._call(b"P").decode()

    def info(self) -> dict:
        import json
        return json.loads(self._call(b"I").decode())

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class PipeRecorder:
    """Records the match from frames the capture ALREADY has.

    `play_match.Recorder` starts a second ffmpeg with its own `x11grab`, which
    on a 4-core laptop measured 51% of a core and, worse, doubled the X
    framebuffer readback that was already pinning Xorg and the window manager.
    Every one of those frames had just been grabbed by the capture thread a
    millisecond earlier. So this one encodes from stdin and the screen is read
    once.

    The frames arrive in CORPUS orientation -- vertically flipped and squashed
    to 480x480 -- because that is what the encoder needs. A person watching
    needs neither, so ffmpeg flips it back and restores the 640x480 aspect on
    the way out. The recording is what a human would have seen; the model's
    input is not.

    DROPPING IS THE POINT
    ---------------------
    `offer` never blocks. If the encoder cannot keep up, frames are dropped
    from the RECORDING rather than stalling the capture thread, because a
    stalled capture thread stalls the agent. A recording with a few dropped
    frames is a recording; an agent that missed its decision window is a lost
    match.
    """

    def __init__(self, out: Path, fps: int, size: int = 480,
                 out_w: int = 640, out_h: int = 480, crf: int = 23,
                 queue_max: int = 8):
        self.out, self.fps, self.size = Path(out), fps, size
        self.out_w, self.out_h, self.crf = out_w, out_h, crf
        self.q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=queue_max)
        self._p = None
        self._t = None
        self.dropped = self.written = 0

    def __enter__(self) -> "PipeRecorder":
        self.out.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.size}x{self.size}", "-framerate", str(self.fps),
            "-i", "-",
            # Undo the two transforms the encoder's pipeline applied, so the
            # file is the game as it looked rather than as the model saw it.
            "-vf", f"vflip,scale={self.out_w}:{self.out_h}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-threads", "1",
            "-movflags", "+faststart", str(self.out),
        ]
        self._p = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()
        return self

    def offer(self, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _pump(self) -> None:
        while True:
            fr = self.q.get()
            if fr is None:
                break
            try:
                self._p.stdin.write(fr.tobytes())
                self.written += 1
            except (BrokenPipeError, OSError, ValueError):
                break

    def __exit__(self, *exc) -> None:
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        if self._t:
            self._t.join(timeout=5)
        if self._p and self._p.poll() is None:
            # Closing stdin is this pipeline's equivalent of 'q': ffmpeg sees
            # EOF, finalises the moov atom and exits. Killing it instead leaves
            # an unplayable file, which is a sad way to lose a match.
            try:
                self._p.stdin.close()
            except OSError:
                pass
            try:
                self._p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._p.terminate()

    def stats(self) -> str:
        tot = self.written + self.dropped
        return (f"recorded {self.written} frames, dropped {self.dropped}"
                f" ({100*self.dropped/max(tot,1):.1f}%)")


class VisionSource:
    """Reads the policy's observation off the SCREEN. No game memory at all.

    Same `.read() -> Frame | None` interface as `LiveState`, so the session does
    not care which is plugged in -- which is the point: the ONLY difference
    between the cheating run and the honest one should be where the numbers
    come from.

    THE CAPTURE HAS TO BE THE CORPUS CAPTURE, NOT A SCREENSHOT
    -----------------------------------------------------------
    This used to grab the root window with ImageMagick and crop it. That is
    wrong in two ways that a screenshot will never tell you about, because the
    image looks perfectly fine to a human either way:

    * **Orientation.** Corpus video is stored VERTICALLY FLIPPED -- checked, not
      assumed: in a decoded corpus frame the spell-card icons that belong at the
      bottom sit at the top, the portraits are upside down at the bottom, and
      the timer digits read mirrored. `build_command` puts a `vflip` in the
      filter chain for exactly that reason. An unflipped screenshot is an
      orientation the encoder has never once seen.
    * **Geometry.** The game renders 640x480 and the corpus squashes it
      horizontally to 480x480 with lanczos. Feeding 640x480 straight to a
      224x224 area-resize gets the aspect right by accident and the resampling
      wrong.

    So the frames come from `WindowCapture`, which is the same ffmpeg command
    that built the corpus. Free correctness beats a reimplementation here.

    WHY A THREAD AND A SHORT RING
    ------------------------------
    The encoder eats a PAIR of frames `delta` apart because velocity is a
    derivative and one image does not carry it. `delta` is in 60 fps video
    frames -- 2 of them, 33 ms -- while decisions happen every 5 frames, 83 ms.
    Sampling the pair from consecutive decisions would space it 83 ms apart:
    2.5x the training gap, which is a different input distribution presented to
    a network that cannot tell it moved. So the capture runs free in a thread
    and the ring hands out frames the right gap apart whenever a decision asks.

    WHAT THE GRAB RATE COSTS, AND WHY IT IS 30 AND NOT 60
    ------------------------------------------------------
    The pair needs a TIME gap of `delta/60` s. Grabbing at 60 fps and taking
    every delta-th frame gives that; so does grabbing at 30 fps and taking
    every (delta/2)-th, for exactly half the x11grab, half the lanczos squash,
    and half the framebuffer readback that X and the window manager have to
    service. The encoder cannot tell the difference -- the two frames are the
    same 33 ms apart either way.

    That is not a micro-optimisation. On 2026-08-16 two 60 fps grabs on a
    4-core laptop drove Xorg to 44% and openbox to 49% while the game rendered
    on software GL, and the decision loop that had measured 85 ms on an idle
    machine took 284 ms under that load -- the agent played at 3.5 Hz having
    trained at 15 Hz. 30 fps costs one third of a frame period in staleness and
    buys back a core.
    """

    def __init__(self, vs, geom, fps: int | None = None, sink=None,
                 delta: int | None = None):
        from sokubot.live.capture import WindowCapture
        self.vs = vs
        # With a remote encoder there is no local VisionState, so the frame gap
        # is handed in from the server's own checkpoint rather than guessed.
        self.delta = int(delta if delta is not None
                         else getattr(vs, "delta", 2))
        # Halve the grab rate whenever the gap divides evenly, so the pair keeps
        # the same wall-clock spacing. An odd delta cannot be halved without
        # changing the gap, so it grabs at 60 and takes every delta-th.
        if fps is None:
            fps, step = (30, self.delta // 2) if self.delta % 2 == 0 \
                else (60, self.delta)
        else:
            step = max(1, round(self.delta * fps / 60))
        self.fps, self.step = fps, max(1, step)
        self.cap = WindowCapture(geom, fps=fps, frame_skip=1)
        self.ring = deque(maxlen=self.step + 1)
        self.sink = sink
        self.brain = None          # set when inference is remote
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.dead = None
        self.reads = self.flips = self.low_conf = self.stalls = 0
        # Motion watchdog. 0.5 of a grey level averaged over a subsampled
        # channel is well under any real animation and well over encoder noise
        # on a frozen image.
        self.still_threshold = 0.5
        self.last_motion = time.monotonic()
        self.still_for = 0.0

    # -- capture lifecycle ---------------------------------------------------
    def __enter__(self):
        self.cap.__enter__()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.cap.__exit__(*exc)

    def _pump(self):
        from sokubot.live.capture import CaptureError
        while not self._stop.is_set():
            try:
                _fid, fr = self.cap.read(timeout_s=2.0)
            except (CaptureError, ValueError) as e:
                # Record it rather than dying silently in a daemon thread; a
                # stalled ring otherwise looks exactly like a paused game.
                self.dead = str(e)
                return
            with self._lock:
                prev = self.ring[-1] if self.ring else None
                self.ring.append(fr)
            # Is anything happening on screen at all? A menu, a paused game, a
            # finished match and a crashed game all look the same to an encoder
            # -- it will happily read two characters out of a static image and
            # the policy will keep pressing buttons at it. Cheap frame-to-frame
            # difference on a subsample; the full 480x480 abs-diff at 30 fps is
            # not worth a core.
            if prev is not None:
                d = float(np.abs(fr[::8, ::8, 0].astype(np.int16)
                                 - prev[::8, ::8, 0].astype(np.int16)).mean())
                now = time.monotonic()
                if d > self.still_threshold:
                    self.last_motion = now
                self.still_for = now - self.last_motion
            if self.sink is not None:
                self.sink.offer(fr)

    def _pair(self):
        """The current frame stacked on the one `delta/60` s before it."""
        with self._lock:
            if len(self.ring) < self.ring.maxlen:
                return None
            older, now = self.ring[0], self.ring[-1]
        return np.concatenate([older, now], axis=2)

    # -- the LiveState interface --------------------------------------------
    def read(self):
        pair = self._pair()
        if pair is None or self.vs.i_am_left is None:
            self.stalls += 1
            return None
        was = self.vs.i_am_left
        st, pr = self.vs.read(pair)
        self.reads += 1
        if was is not None and was != self.vs.i_am_left:
            self.flips += 1
        # A character is ~50 units wide, so a margin under that is an
        # assignment that could as easily have gone the other way.
        if self.vs.confidence < 50.0:
            self.low_conf += 1
        return Frame(st, pr, np.zeros(2, np.int32), self.reads)

    # -- identity, the one bit the pixels do not carry -----------------------
    def calibrate(self, pad: VirtualKeypad, settle_s: float = 0.35) -> str:
        if self.dead:
            return f"capture is dead: {self.dead}"

        def press_and_sample(hold, secs):
            before = self._pair()
            pad.press_only(hold)
            time.sleep(secs)
            after = self._pair()
            pad.neutral()
            return before, after

        if self._pair() is None:
            return "capture has not filled the frame ring yet -- try again"
        return self.vs.calibrate(press_and_sample, "left", settle_s)

    def describe(self) -> str:
        pair = self._pair()
        if pair is None:
            return f"no frames yet{' -- ' + self.dead if self.dead else ''}"
        if self.vs.i_am_left is None:
            return "not calibrated: run `whoami` first"
        # `_raw` rather than `read`: read() advances the association tracker,
        # and the pilot thread is the only thing that may do that. A status
        # command that perturbs the state it reports is a bad instrument.
        st, _pr = self.vs._raw(pair)
        if not self.vs.i_am_left:
            st = st[::-1]
        out = [f"vision reads {self.reads} | side flips {self.flips} | "
               f"low-confidence {self.low_conf} | stalls {self.stalls} | "
               f"margin {self.vs.confidence:.0f}u | agent is the "
               f"{'LEFT' if self.vs.i_am_left else 'RIGHT'} character"]
        for i, who in enumerate(("ME ", "FOE")):
            s = st[i]
            out.append(
                f"  {who} hp {s[CH['hp']]*FULL_HP:6.0f} "
                f"spirit {s[CH['spirit']]:.2f} "
                f"x {s[CH['x']]*STAGE_SPAN:7.1f} "
                f"dx {s[CH['dx']]*STAGE_SPAN:+7.1f} "
                f"air {s[CH['airborne']]:.2f}   (23 channels are the corpus "
                f"mean -- the encoder cannot see them)")
        return "\n".join(out)


class Pilot(threading.Thread):
    """Decides in the background at the training cadence.

    The obvious loop -- read, decide, then play the chunk -- adds the decision
    cost to the decision period. With memory that cost is a millisecond and the
    error is invisible. With the encoder it is 44 ms measured on this machine
    (38 ms of convolution, 8 ms of resampling), so an 83 ms cycle becomes 127 ms
    and the agent silently acts at 7.9 Hz having trained at 15 Hz. Every
    reaction it learned would arrive half a beat late, and nothing in the
    output would say so.

    So deciding and actuating run in separate threads. This one publishes a
    chunk every `ticks/60` s and the caller plays whatever is newest, which
    puts the encoder's cost in the same wall-clock window as the playback it
    overlaps instead of in series with it. The pacing is explicit rather than
    free-running because the history window has to be spaced the way it was in
    training too -- running as fast as the encoder allows would feed the policy
    a 44 ms history stride instead of an 83 ms one.

    Used for the memory path as well, though it does not need it, so the two
    runs differ only in where the numbers come from.
    """

    def __init__(self, src, pol, obs, side: int, H: int, ticks: int,
                 brain=None, truth=None):
        super().__init__(daemon=True)
        self.src, self.pol, self.obs = src, pol, obs
        # When set, the encoder and policy live on another machine and this
        # thread only ships frames and receives buttons.
        self.brain = brain
        # A passive instrument. Never consulted for an action -- see
        # TruthObserver's docstring.
        self.truth = truth
        self.ended = False
        self.side, self.H, self.ticks = side, H, ticks
        self.period = ticks / 60.0
        self.armed = threading.Event()
        self.stopped = threading.Event()
        self.cv = threading.Condition()
        self.chunk = None
        self.seq = 0
        self.decides = self.late = self.blind = 0
        self.late_run = 0
        self.blind_run = 0
        self.dt_ms = 0.0
        # Every decision that produced an action, in ms. Kept apart from the
        # EMA above, which cannot show a p99 twice the period.
        self.lat_ms = deque(maxlen=20000)
        self.stop_reason = None
        # 12 s of a frozen screen is far longer than any hitstop, KO freeze or
        # round transition, and far shorter than a human's patience.
        self.still_limit_s = 12.0
        # 60 consecutive overruns is ~8 s of the agent playing at the wrong
        # rate -- long enough not to trip on a transient, short enough that
        # nobody has to watch it happen.
        self.late_run_limit = 60
        # 20 decisions (~1.7 s) of the game reporting no live battle. Rounds
        # inside a set stay in the battle scene, so this fires when the SET
        # ends rather than between rounds.
        self.end_run_limit = 20
        # RE-ANCHORING.
        #
        # Which on-screen character the agent believes it is drifts, and the
        # drift is absorbing: measured against the game's memory over a real
        # match, the continuity tracker was right 64% of the time with PERFECT
        # inputs and ~51% live, in wrong runs up to 7.1 s long. For that share
        # of the match the policy read the opponent's row as its own.
        #
        # Passive identification from the agent's own inputs was tried and is
        # chance (43-51%): it commands a direction on only 28% of decisions and
        # brief taps do not separate the characters. What does work is what
        # calibration does -- hold a direction for about a second and watch,
        # which measured 647 units against 127.
        #
        # So it is re-run periodically, at the cost of ~1.2 s of walking every
        # `reanchor_s`. That is real lost offence and it is the cheaper error:
        # the alternative is stretches of the match spent fighting from the
        # wrong side of the board.
        # RE-ANCHORING IS OFF.
        #
        # It re-runs calibration mid-match, and calibration needs the opponent
        # to hold still: live, every one of them came back AMBIGUOUS or "no
        # movement" while the human was moving, and each still OVERWROTE the
        # identity with whatever it had guessed. It also costs ~1.2 s of walking
        # each time. With identity pinned at the start (see visionstate.read),
        # a periodic re-guess can only make it worse.
        #
        # Set to a positive number of seconds to bring it back.
        self.reanchor_s = float(os.environ.get("SOKUBOT_REANCHOR_S", "0") or 0)
        self.last_anchor = time.monotonic()
        self.reanchors = 0
        self.reanchor_fn = None
        # ~13 s of reading nothing. The ring needs a moment to refill after a
        # hiccup, so this is not tripped by a single stall.
        self.blind_run_limit = 160

    def run(self) -> None:
        hist = deque(maxlen=max(self.H, 1))
        nxt = time.monotonic()
        while not self.stopped.is_set():
            if not self.armed.is_set():
                hist.clear()
                self._publish(None)
                time.sleep(0.05)
                nxt = time.monotonic()
                continue
            t0 = time.monotonic()
            try:
                if self.brain is not None:
                    pair = self.src._pair()
                    # `f` only distinguishes "no frames at all" (hands off the
                    # pad) from "frames but no action yet" (hold and wait while
                    # the server fills its history or waits for calibration).
                    f = None if pair is None else True
                    enc = None
                    act = None
                    if pair is not None:
                        act, enc = self.brain.decide(pair)
                    if self.truth is not None and self.truth.ok:
                        if not self.truth.sample(enc, act):
                            if self.truth.blind_run >= self.end_run_limit:
                                self.ended = True
                else:
                    act, f = decide(self.src, self.pol, self.obs, hist,
                                    self.side, self.H)
            except Exception as e:                  # a dead read must not
                self.blind += 1                     # silently freeze the pad
                print(f"  decide failed: {e}", flush=True)
                act, f = None, None
            dt = (time.monotonic() - t0) * 1000.0
            self.dt_ms = dt if not self.dt_ms else 0.9 * self.dt_ms + 0.1 * dt
            # Only decisions that ran the policy: while the history window
            # fills, decide() returns early and would flatter the numbers.
            if act is not None:
                self.lat_ms.append(dt)
            if f is None:
                hist.clear()
                self.blind += 1
                self.blind_run += 1
            else:
                self.decides += 1
                self.blind_run = 0
            self._publish(act if f is not None else None)
            # Periodic re-anchor, between decisions so it never interleaves
            # with a half-played action chunk.
            if (self.reanchor_s > 0 and self.reanchor_fn is not None
                    and self.armed.is_set()
                    and time.monotonic() - self.last_anchor > self.reanchor_s):
                try:
                    msg = self.reanchor_fn()
                    self.reanchors += 1
                    if "agent is" not in msg:
                        print(f"  re-anchor inconclusive: {msg}", flush=True)
                except Exception as e:                      # noqa: BLE001
                    print(f"  re-anchor failed: {e}", flush=True)
                self.last_anchor = time.monotonic()
                hist.clear()          # the identity may have changed under us
                nxt = time.monotonic()
            nxt += self.period
            slack = nxt - time.monotonic()
            if slack <= 0:
                self.late += 1
                self.late_run += 1
                nxt = time.monotonic()      # drop the debt rather than sprint
            else:
                self.late_run = 0
                time.sleep(slack)
            self._watchdogs()

    def _watchdogs(self) -> None:
        """Stop on our own, rather than making a human ask us to.

        Two conditions, both learned on 2026-08-16. The screen going still
        means the match is over, the game is paused or it has crashed -- an
        encoder cannot tell any of those from play and will keep feeding the
        policy a frozen image forever. And a decision loop running far over
        budget is not "a bit slow": at 284 ms against 83 ms the agent is
        playing at a third of its trained rate, which is not the agent under
        test. Both used to end with the operator typing `help` into a machine
        too busy to accept the keystrokes.
        """
        # A dead capture is not a still screen -- no frames arrive at all, so
        # the stillness clock never advances and that watchdog would wait
        # forever. Found by pointing the test at an off-screen region: ffmpeg
        # refused to start and the pilot went blind with nothing reporting it.
        dead = getattr(self.src, "dead", None)
        if dead:
            self.armed.clear()
            self.stop_reason = f"capture died: {dead}"
            print(f"\n*** AUTO-DISARM: {self.stop_reason} ***\n", flush=True)
            return
        if self.blind_run >= self.blind_run_limit:
            self.armed.clear()
            self.stop_reason = (f"{self.blind_run} consecutive reads returned "
                                f"nothing -- no usable frames")
            print(f"\n*** AUTO-DISARM: {self.stop_reason} ***\n", flush=True)
            return
        if self.ended:
            self.armed.clear()
            self.stop_reason = ("the game reports no live battle -- the match "
                                "is over")
            print(f"\n*** AUTO-DISARM: {self.stop_reason} ***\n", flush=True)
            return
        still = getattr(self.src, "still_for", 0.0)
        if still > self.still_limit_s:
            self.armed.clear()
            self.stop_reason = (f"screen has been static for {still:.0f}s -- "
                                f"match over, paused, or the game is gone")
            print(f"\n*** AUTO-DISARM: {self.stop_reason} ***\n", flush=True)
            return
        if self.late_run >= self.late_run_limit:
            self.armed.clear()
            self.stop_reason = (
                f"{self.late_run} decisions in a row over budget "
                f"({self.dt_ms:.0f} ms vs {1000*self.period:.0f} ms) -- the "
                f"machine cannot run this loop and the agent is not playing at "
                f"the rate it trained at")
            print(f"\n*** AUTO-DISARM: {self.stop_reason} ***\n", flush=True)
            return
        # Warn long before the hard stop, so a human sees it coming.
        if self.late_run and self.late_run % 25 == 0:
            print(f"  WARNING: {self.late_run} late decisions in a row, "
                  f"{self.dt_ms:.0f} ms of a {1000*self.period:.0f} ms budget",
                  flush=True)

    def _publish(self, act) -> None:
        with self.cv:
            self.chunk, self.seq = act, self.seq + 1
            self.cv.notify_all()

    def take(self, last_seq: int, timeout: float = 0.2):
        """The newest chunk, waiting briefly for one we have not played yet."""
        with self.cv:
            if self.seq == last_seq:
                self.cv.wait(timeout)
            return self.chunk, self.seq

    def stats(self) -> str:
        extra = ""
        if self.brain is not None:
            extra = f" | round trip {self.brain.rtt_ms:.0f} ms"
        if self.reanchors:
            extra += f" | re-anchored {self.reanchors}x"
        return (f"decisions {self.decides} | blind {self.blind} | late "
                f"{self.late} | decide {self.dt_ms:.0f} ms of a "
                f"{1000*self.period:.0f} ms budget{extra}\n  latency: "
                + self.latency_report())

    def latency_report(self) -> str:
        return format_summary(summarise(self.lat_ms, 1000 * self.period))


def _pad_path(pad):
    """The pad's evdev path, or None. Excluded from the hotkey so the agent's own
    device can never toggle it."""
    try:
        return pad.device_path
    except Exception:                                    # noqa: BLE001
        return None


def session(ls, pad: VirtualKeypad, a) -> int:
    """Long-lived: holds the pad, takes commands, records."""
    pol = obs = None
    H = ticks = 0
    side = a.side
    vision = isinstance(ls, VisionSource)
    if vision:
        # `StateObs` ego-orders from `side`, but the vision reader has ALREADY
        # put the agent first -- that is what `calibrate` is for. Applying both
        # would swap the two back and hand the policy the opponent's half of the
        # observation, which trains and runs and plays the wrong character. The
        # physical chair is carried by the calibration, not by this index.
        side = 0
    if a.policy:
        pol, obs, H, ticks, _slots, ck = load_agent(a.policy)
        print(f"policy {a.policy} step {ck.get('step')} net "
              f"{ck.get('net', float('nan')):+.5f} | history {H} ticks {ticks} "
              f"| {'vision: ego-ordered by calibration' if vision else f'agent is PLAYER {side + 1}'}",
              flush=True)
    brain = getattr(ls, "brain", None) if vision else None
    if brain is not None:
        # No local weights, so the cadence has to come from the SERVER's
        # checkpoint. Left at the 0 default, `period = ticks/60` is zero and
        # the pilot spins flat out, which is not a subtle failure but is a
        # silent one -- the loop looks alive and the agent acts at random times.
        pol = pol or True
        H = int(brain.spec["history"])
        ticks = int(brain.spec["ticks"])
        print(f"cadence from server: history {H}, ticks {ticks} "
              f"-> {1000*ticks/60:.1f} ms per decision", flush=True)
    truth = None
    if brain is not None and a.truth:
        truth = TruthObserver(Path(a.truth), side=a.side)
        if truth.ok:
            print(f"truth observer attached to pid {truth.pid} "
                  f"(OBSERVER ONLY -- never reaches the policy); match end "
                  f"detected from the game's own scene state", flush=True)
    pilot = (Pilot(ls, pol, obs, side, H, ticks, brain=brain, truth=truth)
             if pol is not None else None)
    if pilot is not None:
        pilot.late_run_limit = a.late_run_limit
    if pilot is not None and brain is not None:
        pilot.reanchor_fn = lambda: remote_calibrate(ls, brain, pad, hold_s=0.6)
        print(f"re-anchoring identity every {pilot.reanchor_s:.0f}s "
              f"(~1.2s of walking each time; the tracker alone measured 51-64% "
              f"correct)", flush=True)
    if pilot is not None:
        pilot.start()
    last_seq = 0
    armed = False
    period = 1.0 / 60.0
    ctl = Control(CTL)
    ctl.start()
    print(HELP, flush=True)
    print(f"control FIFO: {CTL}\n", flush=True)
    rec = None
    # One-way status for the overlay. Only fields with a real source are written
    # (see status.fields_from); the rest keep their honest defaults.
    slot = a.side + 1
    notice = ""
    next_probe = 0.0
    lost_server = False
    status = StatusWriter()
    status.update(force=True, **fields_from(pilot, brain, slot))
    status_armed = False
    hk = None
    if not a.no_hotkey:
        hk = KeyWatcher.for_evdev(lambda: ctl.push("toggle"), key=a.hotkey,
                                  exclude=_pad_path(pad))
        if hk is None:
            print("hotkey UNAVAILABLE (no readable keyboard -- is your user in the "
                  "`input` group?). Use the FIFO instead.", flush=True)
        else:
            hk.start()
            print(f"hotkey {a.hotkey}: arm/disarm from any window", flush=True)
    try:
        while True:
            for line in ctl.take():
                parts = line.split()
                cmd, args = parts[0], parts[1:]
                if cmd == "toggle":
                    # From the hotkey. The PILOT is the source of truth: the local
                    # `armed` only catches up with a watchdog stop later in the same
                    # loop iteration, so it can lag it by one pass.
                    cmd = ("disarm" if pilot is not None and pilot.armed.is_set()
                           else "arm")
                    print(f"[hotkey] {cmd}", flush=True)
                if cmd == "stop":
                    return 0
                elif cmd == "shot":
                    path = args[0] if args else "/tmp/soku.png"
                    subprocess.run(["import", "-window", "root", path],
                                   env={**os.environ, "DISPLAY": a.display},
                                   capture_output=True, timeout=30)
                    print(f"-> {path}", flush=True)
                elif cmd == "press":
                    for b in args:
                        if b in BUTTONS:
                            tap(pad, b)
                        else:
                            print(f"  unknown button {b!r}; want {BUTTONS}")
                    print(f"pressed {args}", flush=True)
                elif cmd == "key":
                    for k in args:
                        pad.tap_key(k)
                    print(f"keyed {args}", flush=True)
                elif cmd in ("arm", "fight"):
                    if pol is None:
                        notice = "no policy loaded"
                        print("no --policy loaded; restart with one", flush=True)
                    elif (vision and brain is None
                          and ls.vs.i_am_left is None
                          and not (ls.vs.n_char and ls.vs.my_char is not None
                                   and not ls.vs.mirror_match)):
                        # Without identity the agent does not know which of the
                        # two characters it is, and the observation is
                        # ego-ordered. Arming here would play the opponent's
                        # side of the match. Refuse rather than coin-toss it.
                        #
                        # A character head IS identity, so it satisfies this --
                        # except in a mirror match, where the two rows are the
                        # same class and the head is not merely unreliable but
                        # meaningless. There the probe is still required.
                        how = ("`character <id>`" if ls.vs.n_char
                               else "`whoami`")
                        extra = (" (mirror match: the character head cannot "
                                 "separate two of the same character, so the "
                                 "probe is still needed)"
                                 if ls.vs.mirror_match else "")
                        notice = f"NOT ARMED: identity unknown ({how} first)"
                        print(f"NOT ARMED: run {how} first -- the encoder "
                              f"reports the scene left-to-right and cannot say "
                              f"which character is the agent.{extra}",
                              flush=True)
                    elif brain is not None and not brain.identified:
                        # The server holds every decision until it knows which
                        # character is the agent, so arming now would look armed and
                        # do nothing. Refuse, and say so where the user can see it.
                        notice = "NOT ARMED: identity unknown (run `whoami` first)"
                        print(f"{notice} -- the server answers nothing until it "
                              f"knows which character is the agent", flush=True)
                    else:
                        armed = True
                        notice = ""
                        pilot.armed.set()
                        print("ARMED -- the agent is now playing", flush=True)
                elif cmd in ("disarm", "hands-off"):
                    armed = False
                    notice = ""
                    if pilot is not None:
                        pilot.armed.clear()
                    pad.neutral()
                    print(f"disarmed, pad neutral"
                          + (f"\n  {pilot.stats()}" if pilot else ""),
                          flush=True)
                elif cmd in ("character", "char"):
                    # Configuration, not game state: the agent knows what it
                    # picked at the select screen. With a character head this
                    # replaces `whoami` outright -- no 3 s of walking, and the
                    # answer is recomputed every frame so a crossup cannot
                    # strand the policy on the opponent's row.
                    if not vision:
                        print("character is a vision-path command", flush=True)
                    elif not args:
                        print("usage: character <my_id> [opponent_id]   "
                              "(0=Reimu 1=Marisa 2=Sakuya 3=Alice 4=Patchouli "
                              "5=Youmu 6=Remilia 7=Yuyuko 8=Yukari 9=Suika "
                              "10=Reisen 11=Aya 12=Komachi 13=Iku 14=Tenshi "
                              "15=Sanae 16=Cirno 17=Meiling 18=Utsuho "
                              "19=Suwako)", flush=True)
                    else:
                        try:
                            mine = int(args[0])
                            opp = int(args[1]) if len(args) > 1 else None
                        except ValueError:
                            print("character ids are integers", flush=True)
                        else:
                            print(ls.vs.set_character(mine, opp), flush=True)
                elif cmd == "whoami":
                    if armed:
                        # Calibration holds a direction and watches; the pilot
                        # is also writing to the pad. Two writers make the
                        # measurement meaningless in a way that still prints a
                        # confident answer.
                        print("disarm first -- calibration needs the pad to "
                              "itself", flush=True)
                    elif brain is not None:
                        print(remote_calibrate(ls, brain, pad), flush=True)
                    else:
                        print(ls.calibrate(pad) if vision else whoami(ls, pad),
                              flush=True)
                elif cmd == "hud":
                    if brain is None:
                        print("hud is a vision-path command (needs --server)",
                              flush=True)
                    else:
                        # `ls` IS the VisionSource on this path -- see
                        # vision_session's `session(src, pad, a)`.
                        pair = ls._pair() if hasattr(ls, "_pair") else None
                        v = brain.read_hud(pair) if pair is not None else None
                        if v is None:
                            print("no frame pair yet", flush=True)
                        else:
                            print(f"  MY hp {v[0]:.3f}  spirit {v[2]:.3f}   "
                                  f"(player {brain.my_player + 1}, "
                                  f"{'top-left' if brain.my_player == 0 else 'top-right'}"
                                  f" bar)\n"
                                  f"  FOE   hp {v[1]:.3f}  spirit {v[3]:.3f}\n"
                                  f"  reads ok {brain.hud_ok} failed "
                                  f"{brain.hud_fail}", flush=True)
                elif cmd == "state":
                    if brain is not None:
                        print(f"{brain.ping()} | round trip "
                              f"{brain.rtt_ms:.0f} ms | still "
                              f"{ls.still_for:.1f}s", flush=True)
                    else:
                        print(ls.describe() if vision else describe(ls),
                              flush=True)
                elif cmd == "latency":
                    print(pilot.latency_report() if pilot else "no pilot",
                          flush=True)
                elif cmd == "rec":
                    if args and args[0] == "stop":
                        if rec:
                            if vision:
                                ls.sink = None
                            rec.__exit__()
                            print(f"recording finalised"
                                  + (f" -- {rec.stats()}" if vision else ""),
                                  flush=True)
                        rec = None
                    else:
                        out = Path(args[0]) if args else a.record
                        if vision:
                            # Encode the frames the capture already grabbed. A
                            # second x11grab would read the same screen twice
                            # and cost a core the decision loop needs.
                            rec = PipeRecorder(out, ls.fps).__enter__()
                            ls.sink = rec
                        else:
                            rec = Recorder(find_game_window(a.display),
                                           out).__enter__()
                        print(f"recording -> {out}", flush=True)
                else:
                    print(f"unknown command {cmd!r}\n{HELP}", flush=True)
            if armed and pilot is not None and not pilot.armed.is_set():
                # The pilot stopped itself. Without this the loop would keep
                # replaying the last chunk into a match that is already over.
                armed = False
                pad.neutral()
                print(f"disarmed by watchdog: {pilot.stop_reason}\n"
                      f"  {pilot.stats()}\n"
                      f"  `arm` to resume once the situation is fixed",
                      flush=True)
            if brain is not None:
                if armed and (not brain.ok or not brain.identified):
                    # The server went away, or came back having forgotten who the
                    # agent is. Either way the agent is not playing, however armed it
                    # looks -- so say so and stop, rather than sit idle for the ~13 s
                    # the blind-read watchdog would take.
                    armed = False
                    pilot.armed.clear()
                    pad.neutral()
                    lost_server = not brain.ok
                    notice = ("server lost: reconnecting. `whoami`, then re-arm"
                              if lost_server else
                              "identity lost (server restarted): `whoami`, then re-arm")
                    print(f"\n*** AUTO-DISARM: {notice} ***\n", flush=True)
                if not brain.ok and time.monotonic() >= next_probe:
                    # A disarmed pilot makes no calls, so nothing else would ever notice
                    # the server come back. A ping goes through the reconnect path.
                    next_probe = time.monotonic() + 1.0
                    try:
                        brain.ping()
                    except OSError:
                        pass
                if lost_server and brain.ok and not armed:
                    # Found again. Do not leave "server lost" on screen: it is stale
                    # and would send the user looking for a problem that is gone.
                    lost_server = False
                    notice = "server is back but forgot the agent: `whoami`, then re-arm"
            # A transition (arm, disarm, a watchdog stop) is written at once so the
            # overlay never lags a state change; everything else is rate-limited.
            now_armed = bool(pilot is not None and pilot.armed.is_set())
            status.update(force=now_armed != status_armed,
                          **fields_from(pilot, brain, slot, notice))
            status_armed = now_armed
            if armed:
                act, seq = pilot.take(last_seq)
                if seq == last_seq:
                    time.sleep(period)      # the decider is late; hold
                else:
                    last_seq = seq
                    if act is None:
                        # Between rounds, in a menu, or still filling the
                        # history window: hands off the sticks rather than
                        # holding whatever the last decision was.
                        pad.neutral()
                        time.sleep(period)
                    else:
                        for t in range(ticks):
                            pad.set_state(act[t].tolist())
                            time.sleep(period)
            else:
                time.sleep(0.05)
    finally:
        if pilot is not None:
            pilot.armed.clear()
            pilot.stopped.set()
            print(pilot.stats(), flush=True)
            if pilot.truth is not None and pilot.truth.ok:
                print(pilot.truth.summary(), flush=True)
        pad.neutral()
        if rec:
            rec.__exit__()
        if hk is not None:
            hk.stop()
        status.close()
        ctl.stop.set()


def remote_calibrate(src, brain, pad: VirtualKeypad,
                     hold_s: float = 1.0) -> str:
    """Sweep LEFT then RIGHT and let the server see which character swings.

    The first version held one direction for 0.35 s and compared before/after.
    Measured against the game's own memory, that moves a character about 14
    units while the encoder's live positional noise is about 60 -- the signal
    was six times under the noise floor, and the old threshold duly "found" an
    identity in it (19 vs 14 units, then 55 vs 64).

    Two changes, both aimed at signal-to-noise rather than at the threshold:
    hold four times as long, and sample at the two ENDS of a left-then-right
    sweep rather than either side of one press. The agent's x swings by roughly
    twice the one-way distance -- 200+ units against that 60-unit noise -- while
    a character nobody is driving stays put. Same server call, far better
    separated inputs.

    Still only pixels and the agent's own inputs: the sweep is our buttons and
    the reading is the encoder's.
    """
    if src._pair() is None:
        return "capture has not filled the frame ring yet -- try again"
    pad.press_only("left")
    time.sleep(hold_s)
    left_end = src._pair()
    pad.press_only("right")
    time.sleep(hold_s * 2)          # back past the start, to the far end
    right_end = src._pair()
    pad.neutral()
    if left_end is None or right_end is None:
        return "lost the capture mid-probe"
    return brain.calibrate(left_end, right_end)


def vision_session(pad: VirtualKeypad, a) -> int:
    """The honest run: pixels and the agent's own inputs, nothing else.

    Deliberately does not call `find_game_pid` or touch /proc at all. The
    memory path waits on the game by finding its process; this one waits on the
    WINDOW, so there is no code path in this branch that could read the game's
    state even by accident. That is worth a few lines of duplication -- "we
    didn't read memory" should be visible in the control flow rather than
    argued from the fact that nobody called the reader.
    """
    from sokubot.live.capture import CaptureError

    if a.server:
        return remote_vision_session(pad, a)

    from sokubot.live.visionstate import VisionState

    # Pinned, not left to chance. The encoder forward measured 30 ms at two
    # threads and 68 ms at four on this machine -- oversubscription is not a
    # small loss here, it is worse than single-threaded, and the game is
    # rendering on this same CPU through software GL. Whatever OMP would have
    # guessed from the core count is the wrong answer.
    torch.set_num_threads(2)
    vs = VisionState.load(a.encoder, device="cpu")
    order = np.argsort(-vs.r2_per_channel)
    sup = ", ".join(f"{vs.supervised[i]} {vs.r2_per_channel[i]:+.2f}"
                    for i in order)
    n_known = len(vs.trusted)
    print(f"encoder {a.encoder}\n  pair delta {vs.delta} frames, input "
          f"{vs.size}px\n  held-out R2: {sup}\n  USING {n_known} channels: "
          f"{', '.join(vs.trusted)}", flush=True)
    if vs.dropped:
        print(f"  DROPPED {len(vs.dropped)} that lost to their own corpus "
              f"mean: {', '.join(vs.dropped)}", flush=True)
    print(f"  {33 - n_known} of 33 channels and ALL projectiles carry the "
          f"corpus mean -- the policy is told 'unknown', not a guess.",
          flush=True)
    if "x" not in vs.trusted:
        # Everything rests on x: calibration presses a direction and watches x
        # move, and the frame-to-frame association tracks x. A constant x makes
        # both silently meaningless.
        print("REFUSING TO PLAY: `x` did not beat its corpus mean, so identity "
              "calibration and side tracking have nothing to stand on.")
        return 7

    geom = None
    print("waiting for the game window ...", flush=True)
    for _ in range(600):
        try:
            geom = find_game_window(a.display)
            break
        except CaptureError:
            time.sleep(1)
    if geom is None:
        print("no game window appeared")
        return 4
    # Sampled late on purpose: Soku lands at a different offset on each launch
    # and a region grabbed too early captures a strip of desktop for the whole
    # session.
    time.sleep(1.0)
    geom = geom.refreshed()
    print(f"game window {geom.w}x{geom.h} at +{geom.x}+{geom.y}", flush=True)

    with VisionSource(vs, geom) as src:
        for _ in range(100):                 # let the frame ring fill
            if src._pair() is not None:
                break
            time.sleep(0.1)
        if src.dead:
            print(f"capture died immediately: {src.dead}")
            return 6
        if src._pair() is None:
            print("capture produced no frames")
            return 6
        print("capture is live. Run `whoami` to calibrate, then `arm`.",
              flush=True)
        return session(src, pad, a)


def remote_vision_session(pad: VirtualKeypad, a) -> int:
    """Capture here, think on the LAN box. See scripts/serve_vision.py."""
    from sokubot.live.capture import CaptureError

    brain = RemoteBrain(a.server, a.port, size=224, my_player=a.side)
    info = brain.info()
    brain.size = int(info["size"])
    brain.spec = info
    print(f"server {a.server}:{a.port} -> input {info['size']}px, pair delta "
          f"{info['delta']} frames, history {info['history']}, ticks "
          f"{info['ticks']}\n  USING {len(info['trusted'])} channels: "
          f"{', '.join(info['trusted'])}", flush=True)
    if info["dropped"]:
        print(f"  DROPPED (below their corpus mean): "
              f"{', '.join(info['dropped'])}", flush=True)
    print(f"  ping: {brain.ping()} | round trip {brain.rtt_ms:.1f} ms",
          flush=True)

    geom = None
    print("waiting for the game window ...", flush=True)
    for _ in range(600):
        try:
            geom = find_game_window(a.display)
            break
        except CaptureError:
            time.sleep(1)
    if geom is None:
        print("no game window appeared")
        return 4
    time.sleep(1.0)
    geom = geom.refreshed()
    print(f"game window {geom.w}x{geom.h} at +{geom.x}+{geom.y}", flush=True)

    with VisionSource(None, geom, delta=int(info["delta"])) as src:
        src.brain = brain
        for _ in range(100):
            if src._pair() is not None:
                break
            time.sleep(0.1)
        if src.dead or src._pair() is None:
            print(f"capture failed: {src.dead}")
            return 6
        brain.reset()
        print("capture is live. Run `whoami` to calibrate, then `arm`.",
              flush=True)
        try:
            return session(src, pad, a)
        finally:
            brain.close()


def resolve_side_arg(a) -> int:
    """Fill `a.side` from the profiles, or say why not. 0 = go, else an exit code.

    Runs before the pad or the game exist, so a refusal costs nothing. An
    explicit --side is the operator overriding the profiles: it is checked and
    warned about, never refused.
    """
    from sokubot.live import profiles as pf
    try:
        side, _ = pf.resolve_side(a.game)
    except pf.SlotError as e:
        if a.side is not None:
            print(f"WARNING: could not check --side {a.side} against the "
                  f"profiles: {e}", flush=True)
            return 0
        print(f"NOT STARTING: {e}", flush=True)
        return 2
    if a.side is None:
        a.side = side
        # Read from config123.dat NOW. With --attach the operator starts the
        # game afterwards, and a profile they change in the game's own menu is
        # not seen here -- the same wrong-bar failure this exists to prevent.
        print(f"side: the agent is player {side + 1} (its profile is selected "
              f"for slot {side + 1} in config123.dat, read just now). If you "
              f"change profiles in the game, this is stale: run `hud` and check "
              f"the bar it calls yours against the screen.", flush=True)
    elif a.side != side:
        print(f"WARNING: --side {a.side} says player {a.side + 1}, but the "
              f"profiles put the agent in slot {side + 1}. If --side is wrong "
              f"the policy is fed the OPPONENT's health as its own.", flush=True)
    return 0


def preflight_server(a) -> int:
    """Can we reach the vision server? Checked BEFORE the game is launched.

    The client used to launch the game and only then connect, so a server that was not
    running produced a traceback with a game window left open behind it.
    """
    try:
        b = RemoteBrain(a.server, a.port, size=224, my_player=a.side)
        b.info()
        b.close()
    except OSError as e:
        print(f"NOT STARTING: no vision server at {a.server}:{a.port} "
              f"({type(e).__name__}: {e}).\n  Start one: `python -m scripts.serve_null` "
              f"(no model, for testing the harness) or `scripts.serve_vision`.",
              flush=True)
        return 3
    return 0


def preset_command(name: str, sfe_root: Path):
    """The command that switches the SWRSToys module set, or None for `none`."""
    if name == "none":
        return None
    return [sys.executable, str(Path(sfe_root).expanduser() / "ops" / "soku_mods.py"), name]


def apply_preset(a) -> int:
    """Apply --preset, or leave the user's module set exactly as it is (the default)."""
    cmd = preset_command(a.preset, a.sfe_root)
    if cmd is None:
        return 0
    if not Path(cmd[1]).is_file():
        print(f"NOT STARTING: --preset {a.preset} needs {cmd[1]} (set --sfe-root).", flush=True)
        return 2
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"NOT STARTING: module preset failed:\n{r.stdout}{r.stderr}", flush=True)
        return 2
    print(f"module preset {a.preset!r} applied", flush=True)
    return 0


def start_overlay(a):
    """The status strip, as a separate process so it cannot slow or stop the pad."""
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "scripts.sokubot_overlay", "--display", a.display],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"overlay not started: {e}", flush=True)
        return None


def stop_overlay(proc) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(2)
    except subprocess.TimeoutExpired:
        proc.kill()


def close_game(a) -> None:
    """Close the game we launched (opt-in: a crash of THIS process should not, by
    default, take a match the human is playing down with it)."""
    stop_stale_wine(a.prefix)


def _play(a) -> int:
    """Everything after argument parsing: stop stale wine, create the pad, launch, run."""
    proc = None
    if not a.attach:
        stop_stale_wine(a.prefix)
    pad = VirtualKeypad()
    with pad:                      # the pad must exist BEFORE the game starts
        if not a.attach:
            print(f"launching {a.game}/th123e.exe on {a.display} ...", flush=True)
            mods = a.preset != "none"
            print(f"mod loader: {'ON (preset ' + a.preset + ')' if mods else 'OFF -- vanilla game, ModLoaderSettings.json is not read'}", flush=True)
            proc = launch_game(a.game, a.prefix, a.display, mods=mods)
            if not (a.encoder or a.server):
                # Only the MEMORY path needs the process to exist by now. The vision
                # paths poll for the game WINDOW themselves (up to 10 minutes), so a
                # blind sleep here just delayed every start by 12 s and still was not
                # a guarantee.
                time.sleep(12)
            # Audio is left alone on purpose. Muting it broke the game once
            # (disabling the Wine driver made Soku fail DirectSound init and
            # render black) and the operator prefers the music on anyway.
        if a.encoder or a.server:
            return vision_session(pad, a)

        # With --attach the operator launches the game themselves, which they
        # may well do AFTER this starts -- and that ordering is the correct one,
        # because Wine's dinput enumerates input devices once at init and the
        # pad has to exist first. So wait rather than fail.
        pid = find_game_pid()
        if pid is None and a.attach:
            print("pad is up. START THE GAME NOW -- waiting for it ...",
                  flush=True)
            for _ in range(600):
                pid = find_game_pid()
                if pid is not None:
                    break
                time.sleep(1)
        if pid is None:
            print("game process not found")
            return 4
        try:
            ls = LiveState.attach(pid)
        except PermissionError:
            print(f"cannot read /proc/{pid}/mem. The game is not a descendant "
                  f"of this process -- a stale wineserver probably adopted it. "
                  f"Kill every wineserver for this prefix and retry, or set "
                  f"kernel.yama.ptrace_scope=0.")
            return 5
        print(f"attached to pid {pid}", flush=True)

        if a.verify:
            return verify(ls, pad, a.seconds)
        return session(ls, pad, a)



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", type=Path,
                    default=Path("~/.wine-soku/drive_c/Games/Soku").expanduser())
    ap.add_argument("--prefix", type=Path,
                    default=Path("~/.wine-soku").expanduser())
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--hotkey", default=DEFAULT_KEY,
                    help="evdev key that arms/disarms from ANY window (default F12). "
                         "The game has focus during a match, so a key that needs the "
                         "terminal is a key you cannot press when you need it.")
    ap.add_argument("--preset", choices=("none", "ablation", "netplay"), default="none",
                    help="switch the SWRSToys module set via SokuFrameExtractor's "
                         "ops/soku_mods.py AND let the mod loader run. `none` (default) "
                         "launches a VANILLA game with the loader bypassed, so no module "
                         "(giuroll included) is loaded and the settings file is not read.")
    ap.add_argument("--sfe-root", type=Path,
                    default=Path("~/K0NTR0L-2/SokuFrameExtractor").expanduser())
    ap.add_argument("--no-overlay", action="store_true",
                    help="do not start the status strip (scripts.sokubot_overlay)")
    ap.add_argument("--close-game", action="store_true",
                    help="close the game this run launched when it exits (default: leave "
                         "it, so a crash here does not end the human's match)")
    ap.add_argument("--no-hotkey", action="store_true",
                    help="do not watch the keyboards (also what to use without "
                         "`input` group access)")
    ap.add_argument("--late-run-limit", type=int, default=60,
                    help="auto-disarm after this many decisions IN A ROW over "
                         "the period. 60 (~5 s) suits play; a latency "
                         "measurement wants ~10 so an overloaded machine "
                         "stops itself within a second or two.")
    ap.add_argument("--verify", action="store_true",
                    help="run the reader checks instead of playing. ALWAYS "
                         "do this first on a fresh machine or after any "
                         "change to memstate.py.")
    ap.add_argument("--attach", action="store_true",
                    help="use an already-running game instead of launching "
                         "one. Needs kernel.yama.ptrace_scope=0, because an "
                         "existing game is not our descendant.")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--encoder", type=Path, default=None,
                    help="a pixels-to-state encoder from scripts.train_encoder. "
                         "With this the agent reads the SCREEN and its own "
                         "inputs -- no game memory -- which is the honest "
                         "configuration. Without it the state comes from "
                         "/proc/pid/mem, which is cheating and is what the "
                         "filename says.")
    ap.add_argument("--server", default=None,
                    help="run the encoder and policy on this host instead of "
                         "locally. REQUIRED for any encoder worth deploying: "
                         "the game host measured 82-153 ms for one encoder "
                         "forward against an 83 ms decision period.")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--truth", type=Path,
                    default=Path("results/match_truth.npz"),
                    help="read the game's own state as a PASSIVE INSTRUMENT: "
                         "it detects when the match ends and records what the "
                         "encoder said beside what was true. It never reaches "
                         "the policy, which still sees only pixels and its own "
                         "inputs. Pass an empty string to disable.")
    ap.add_argument("--side", type=int, default=None, choices=(0, 1),
                    help="0 = the agent is PLAYER 1, 1 = PLAYER 2. Default: read "
                         "it from which slot the agent's profile is selected in "
                         "(config123.dat), and refuse to start if that is "
                         "ambiguous or unsafe. This is load-bearing in a second "
                         "place: it picks WHICH HEALTH BAR IS MINE, because the "
                         "HUD is indexed by player and never moves when the "
                         "characters swap sides. `whoami` settles the SIDE, "
                         "which is a different bit; run `hud` to check this one "
                         "against the screen.")
    # The filename says what it is. This run reads the game's memory, which is
    # cheating, and a file called match.mp4 sitting beside honest ones is how a
    # probe gets quoted as a result six months later.
    ap.add_argument("--character", type=int, default=None,
                    help="the character id the agent is playing (0-19). With a "
                         "character-head encoder this replaces the `whoami` "
                         "probe entirely, so an unattended run needs no "
                         "console interaction to establish identity.")
    ap.add_argument("--opponent-character", type=int, default=None,
                    help="the opponent's character id, when known. Only used "
                         "to detect a MIRROR match, where the character head "
                         "cannot separate the rows and the probe is required.")
    ap.add_argument("--record", type=Path,
                    default=Path("results/CHEATING_memory-fed_diagnostic.mp4"))
    a = ap.parse_args()
    if (a.encoder or a.server) and a.record == ap.get_default("record"):
        # The default filename shouts CHEATING because the default run does.
        # This one does not read a byte of game memory, and calling its
        # recording a cheating diagnostic would be its own kind of wrong.
        a.record = Path("results/VISION_no-memory_diagnostic.mp4")

    if a.verify:
        if a.side is None:
            a.side = 0                 # the reader checks do not use it
    else:
        rc = resolve_side_arg(a)
        if rc:
            return rc

    if a.server and not a.verify:
        rc = preflight_server(a)
        if rc:
            return rc
    rc = apply_preset(a)
    if rc:
        return rc
    overlay = None if (a.no_overlay or a.verify) else start_overlay(a)
    try:
        return _play(a)
    finally:
        stop_overlay(overlay)
        if a.close_game:
            close_game(a)


if __name__ == "__main__":
    raise SystemExit(main())
