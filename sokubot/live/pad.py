"""A virtual gamepad, created through /dev/uinput.

WHY A GAMEPAD AND NOT A KEYBOARD
--------------------------------
``SokuFrameExtractor/runner/vkbd.py`` establishes the hard fact this module is
built on: Soku reads input through DirectInput, and **Wine's dinput reads real
evdev devices**, not Wine's synthetic Win32 queue and not the X server's
synthetic events. Six injection mechanisms above the kernel were tried and all
six were swallowed. uinput is the one layer below all of them, because writes to
it become genuine kernel input events.

That module makes a virtual *keyboard*. For a live human-vs-agent match a
keyboard is the wrong device, and for a reason that follows from the same root
cause: **dinput reads the device directly, so it ignores window focus.** A
virtual keyboard would therefore be read by the human's player as well as the
agent's, and its keystrokes would also land in whatever window happens to be
focused when the operator alt-tabs.

dinput enumerates joysticks as *separate devices*. A virtual gamepad is
therefore invisible to a keyboard-driven player: Soku's controller config binds
player 2 to this pad, player 1 stays on the keyboard, and neither can see the
other's input. That isolation is the entire reason a QEMU guest looked necessary
and the entire reason it turns out not to be.

WHAT THE DEVICE HAS TO DECLARE
------------------------------
Two independent consumers have to accept this device, and they want different
things:

* **udev** decides whether the device is a joystick at all. Its ``input_id``
  builtin tags ``ID_INPUT_JOYSTICK`` when it sees absolute X/Y axes together
  with keys in the ``BTN_GAMEPAD`` range. Miss either half and the device is
  classified as something else, and Wine's joystick backend never looks at it.
* **dinput** will not report a control the device never claimed. Anything the
  game might poll has to be advertised at creation time, which is why the
  capability lists below are explicit rather than minimal.

**The d-pad hat is deliberately NOT declared, and this is not an optimisation.**
An earlier version advertised ``ABS_HAT0X``/``ABS_HAT0Y`` alongside the stick on
the theory that offering both was safer than guessing which one Soku reads. The
result was a menu that scrolled upward forever: Wine maps an evdev hat onto a
DirectInput **POV**, whose neutral is a reserved value (-1), *not* the centre of
its range. A hat resting at (0, 0) is therefore read as an angle of 0 degrees --
due north -- so the pad held "up" permanently and no amount of releasing it
helped, because releasing it *is* (0, 0).

The stick alone has no such ambiguity: its centre is its neutral. Anything
needing a hat should add one only with a POV-centred idle value, and should
prove it by leaving the pad untouched on a menu and watching nothing happen.

ORDERING MATTERS
----------------
Wine's dinput enumerates devices when it initialises, so the pad has to exist
*before* the game starts or it will not be found. Hold it open for the whole
session: the ``with`` block should outlive the game process, not the other way
round.

REQUIREMENTS
------------
``/dev/uinput`` must be writable. It is root-only on a stock Arch install; see
``scripts/live_probe.py --check`` for the one-time udev rule that fixes it.
"""

from __future__ import annotations

import os
from typing import Sequence

try:
    from evdev import AbsInfo, UInput, ecodes
    HAVE_EVDEV = True
except ImportError:                                   # pragma: no cover
    HAVE_EVDEV = False


# The corpus button order, from data/soku.py. Every array crossing this module's
# boundary is in this order, so it is stated once and never re-derived.
BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")
IDX = {name: i for i, name in enumerate(BUTTONS)}

# Stick range. Overridable from the environment because the correct choice was
# settled by measurement rather than by reading a specification -- see
# `scripts/live_probe.py --drift`, which counts how far a menu cursor moves
# while nothing is pressed.
#
# The signed range every Linux gamepad uses (-32767..32767, neutral 0) does NOT
# work here: with it the menu scrolls continuously, so something between the
# kernel and Soku is not treating 0 as the centre. An unsigned range whose
# neutral is its own midpoint removes the question entirely, because there is no
# sign convention left to disagree about.
AXIS_MIN = int(os.environ.get("SOKUBOT_AXIS_MIN", "0"))
AXIS_MAX = int(os.environ.get("SOKUBOT_AXIS_MAX", "65535"))
AXIS_MID = (AXIS_MIN + AXIS_MAX) // 2

# The six action buttons, mapped onto the standard Linux gamepad codes. Ordering
# here is not arbitrary: Wine assigns DirectInput button *indices* by ascending
# evdev code, so listing them in code order makes the index each button gets
# stable across runs. It still does not need to match any particular pad, since
# the human binds them by pressing each one in Soku's controller config -- but a
# binding made once has to keep working, and that requires the indices not to
# move.
BUTTON_CODES = (
    ("a",      ecodes.BTN_SOUTH),    # 0x130
    ("b",      ecodes.BTN_EAST),     # 0x131
    ("c",      ecodes.BTN_NORTH),    # 0x133
    ("d",      ecodes.BTN_WEST),     # 0x134
    ("change", ecodes.BTN_TL),       # 0x136
    ("spell",  ecodes.BTN_TR),       # 0x137
) if HAVE_EVDEV else ()

# Soku binds "change card" and "spell card" as their own buttons in the
# controller config, even though the default *keyboard* layout derives them from
# A+B and B+C. The extractor records them as their own channels (bits 8 and 9 in
# config.hpp), so the agent emits them as their own buttons and the two
# representations agree.


# ---------------------------------------------------------------------------
# The keypad device, which is what actually works
# ---------------------------------------------------------------------------
# **Soku's joystick handling under Wine makes the menus unusable.** Measured
# with continuous sampling of the main-menu cursor (`scratchpad/drift2.py`):
# once the game receives *any* input from a DirectInput joystick it begins
# scrolling continuously, ~60 row transitions in 25 s. This is independent of
# everything about the pad that was tried:
#
#     axes 0..65535 (centre 32767)      DRIFTS
#     axes -32767..32767 (centre 0)     DRIFTS
#     no absolute axes at all           DRIFTS
#     Xbox 360 VID/PID vs generic       DRIFTS either way
#     confirm with a keyboard instead   STABLE (0 transitions)
#
# The user reports the same behaviour with *physical* controllers, which is the
# clue that settles it: this is the game, not the virtual device.
#
# So the agent drives a **keyboard** after all, and the objection that made a
# gamepad look necessary -- that dinput ignores focus, so the agent's keys would
# also reach the human's player -- is answered by choosing keys the human cannot
# press. This laptop has no numeric keypad, so the keypad scancodes are
# unreachable by the person and unambiguous for the agent.
#
# `profile/sokubot.pf` binds player 2 to exactly these keys.
# The numeric keypad was the first choice -- this laptop has none, so the human
# could not possibly press those keys. It does not work: the device demonstrably
# emits KEY_KP6/KEY_KP7 (verified by reading its own event node) and Soku
# ignores them, with NumLock on or off. Something between evdev and the game's
# DirectInput scancodes drops the keypad block.
#
# Arrows plus ZXCVBN instead, which is the scheme `profile1p.pf` uses and which
# demonstrably drives the game. Collision with the human is avoided by choosing
# against their actual profile rather than by hoping: P1 (`anon`) holds
# W/A/S/D + P/O/SPACE/F/I/U, so none of these overlap. Check this again if the
# human's profile changes -- `scripts/live_probe.py --profiles` prints them.
KEYPAD_CODES = (
    ("up",     "KEY_T"),  ("down",  "KEY_G"),
    ("left",   "KEY_H"),  ("right", "KEY_J"),
    ("a",      "KEY_K"),  ("b",     "KEY_L"),
    ("c",      "KEY_M"),  ("d",     "KEY_N"),
    ("change", "KEY_B"),  ("spell", "KEY_V"),
)

# Declared in addition to the bound keys, so the binding can be changed without
# recreating the device -- and the device cannot be recreated without restarting
# the game, because dinput enumerates once at init.
SPARE_KEYS = tuple(
    # Everything the agent might ever need to *send*, declared once. A uinput
    # device's capabilities are fixed at creation and the device cannot be
    # recreated without restarting the game (dinput enumerates at init), so a
    # missing key here costs a full session restart. It cost three.
    ["KEY_%s" % c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    + ["KEY_KP%d" % i for i in range(10)]
    + ["KEY_UP", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT",
       "KEY_ENTER", "KEY_ESC", "KEY_SPACE", "KEY_LEFTSHIFT",
       "KEY_F13", "KEY_F14", "KEY_F15", "KEY_COMMA", "KEY_DOT"]
)


# P1's controls, for setup only. Read off `profile/anon.pf`.
P1_KEYS = {"up": "KEY_W", "down": "KEY_S", "left": "KEY_A", "right": "KEY_D",
           "confirm": "KEY_P", "cancel": "KEY_O"}


class VirtualPadError(RuntimeError):
    pass


class VirtualKeypad:
    """A uinput keyboard restricted to the numeric keypad, in `BUTTONS` order.

    Same whole-state API as :class:`VirtualPad` so the rest of the live loop
    does not care which one it is holding. Left/right and up/down are still
    resolved against each other, because the game cannot represent both and
    ``rl/policy.py`` guarantees it never asks for both.
    """

    def __init__(self, name: str = "SokuBot Virtual Keypad",
                 settle_s: float = 1.0):
        self.name = name
        self.settle_s = settle_s
        self._ui: "UInput | None" = None
        self._last: tuple[int, ...] | None = None
        self._codes = ({n: getattr(ecodes, k) for n, k in KEYPAD_CODES}
                       if HAVE_EVDEV else {})

    def __enter__(self) -> "VirtualKeypad":
        if not HAVE_EVDEV:
            raise VirtualPadError("python-evdev is not installed")
        try:
            spare = [getattr(ecodes, k) for k in SPARE_KEYS
                     if hasattr(ecodes, k)]
            keys = sorted(set(list(self._codes.values()) + spare))
            self._ui = UInput({ecodes.EV_KEY: keys}, name=self.name)
        except (PermissionError, FileNotFoundError, OSError) as e:
            raise VirtualPadError(
                f"cannot open /dev/uinput ({e}). Run "
                f"`python -m scripts.live_probe --check` for the udev rule.") from e
        import time
        time.sleep(self.settle_s)   # dinput enumerates at init; let udev settle
        self.neutral()
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._ui is not None:
                self.neutral()
        finally:
            if self._ui is not None:
                self._ui.close()
                self._ui = None

    def neutral(self) -> None:
        self.set_state([0] * len(BUTTONS))

    def set_state(self, buttons: Sequence[float]) -> None:
        if self._ui is None:
            raise VirtualPadError("keypad is not open")
        if len(buttons) != len(BUTTONS):
            raise ValueError(f"expected {len(BUTTONS)} buttons, got {len(buttons)}")
        state = tuple(1 if float(b) > 0.5 else 0 for b in buttons)
        if state[IDX["left"]] and state[IDX["right"]]:
            state = list(state); state[IDX["left"]] = state[IDX["right"]] = 0
            state = tuple(state)
        if state[IDX["up"]] and state[IDX["down"]]:
            state = list(state); state[IDX["up"]] = state[IDX["down"]] = 0
            state = tuple(state)
        if state == self._last:
            return
        self._last = state
        for name, code in self._codes.items():
            self._ui.write(ecodes.EV_KEY, code, state[IDX[name]])
        self._ui.syn()          # one report per tick, never half-applied

    def press_only(self, name: str) -> None:
        if name not in IDX:
            raise ValueError(f"unknown control {name!r}")
        v = [0] * len(BUTTONS)
        v[IDX[name]] = 1
        self.set_state(v)

    def tap_key(self, key: str, hold_s: float = 0.05) -> None:
        """Tap one raw key by evdev name, e.g. "KEY_Z". Setup menus only."""
        import time
        if self._ui is None:
            raise VirtualPadError("keypad is not open")
        if not key.startswith("KEY_"):
            key = "KEY_" + key.upper()
        if not hasattr(ecodes, key):
            raise ValueError(f"no such key {key!r}")
        code = getattr(ecodes, key)
        self._ui.write(ecodes.EV_KEY, code, 1); self._ui.syn()
        time.sleep(hold_s)
        self._ui.write(ecodes.EV_KEY, code, 0); self._ui.syn()

    def tap_p1(self, control: str, hold_s: float = 0.05) -> None:
        """Tap one of PLAYER ONE's keys. Setup menus only.

        Kept deliberately separate from `set_state`, which is the only path the
        policy's output can take and which can only ever emit P2's ten buttons.
        Nothing in the agent loop calls this.
        """
        import time
        if self._ui is None:
            raise VirtualPadError("keypad is not open")
        key = P1_KEYS.get(control)
        if key is None or not hasattr(ecodes, key):
            raise ValueError(f"unknown P1 control {control!r}")
        code = getattr(ecodes, key)
        self._ui.write(ecodes.EV_KEY, code, 1); self._ui.syn()
        time.sleep(hold_s)
        self._ui.write(ecodes.EV_KEY, code, 0); self._ui.syn()

    @property
    def device_path(self) -> str:
        if self._ui is None:
            raise VirtualPadError("keypad is not open")
        return self._ui.device.path


class VirtualPad:
    """A uinput gamepad, scoped to a `with` block.

    The state is set wholesale rather than pressed and released: the policy
    emits a complete 10-wide button vector per tick, so a whole-state API cannot
    drift out of sync with it the way an incremental one can.
    """

    def __init__(self, name: str = "SokuBot Virtual Pad", settle_s: float = 1.0):
        self.name = name
        self.settle_s = settle_s
        self._ui: "UInput | None" = None
        self._last: tuple[int, ...] | None = None

    # ------------------------------------------------------------------ setup
    def __enter__(self) -> "VirtualPad":
        if not HAVE_EVDEV:
            raise VirtualPadError(
                "python-evdev is not installed; cannot create a virtual pad"
            )
        axis = AbsInfo(value=AXIS_MID, min=AXIS_MIN, max=AXIS_MAX,
                       fuzz=0, flat=0, resolution=0)
        caps = {
            ecodes.EV_KEY: [code for _, code in BUTTON_CODES],
            # No ABS_HAT0X/Y. See the module docstring: a centred hat reads as
            # POV north and holds "up" forever.
            ecodes.EV_ABS: [(ecodes.ABS_X, axis), (ecodes.ABS_Y, axis)],
        }
        try:
            # A plausible vendor/product helps SDL and udev treat this as a real
            # gamepad rather than something to be filtered; bustype USB matches
            # what every physical pad reports.
            self._ui = UInput(caps, name=self.name, vendor=0x045e,
                              product=0x028e, version=0x0110,
                              bustype=ecodes.BUS_USB)
        except (PermissionError, FileNotFoundError, OSError) as e:
            raise VirtualPadError(
                f"cannot open /dev/uinput ({e}). Run "
                f"`python -m scripts.live_probe --check` for the one-time udev "
                f"rule, then log out and back in."
            ) from e

        # Give udev time to run its rules and Wine's backend time to notice.
        # vkbd.py established the same wait for the keyboard case: without it a
        # game started immediately after can begin enumerating before the device
        # is fully registered, and a device dinput missed once is missed for the
        # whole run.
        import time
        time.sleep(self.settle_s)
        self.neutral()
        return self

    def __exit__(self, *exc) -> None:
        # Release everything before closing. A held direction or button that
        # outlives the process leaves the character walking into a corner with
        # nothing left to stop it, and the device node disappearing does not
        # imply a release event was delivered.
        try:
            if self._ui is not None:
                self.neutral()
        finally:
            if self._ui is not None:
                self._ui.close()
                self._ui = None

    # ------------------------------------------------------------------ state
    def neutral(self) -> None:
        """Centre the stick and release every button."""
        self.set_state([0] * len(BUTTONS))

    def set_state(self, buttons: Sequence[float]) -> None:
        """Apply one tick of the 10-wide button vector, in ``BUTTONS`` order.

        Values are thresholded at 0.5 so a float tensor row can be passed
        straight through without the caller casting it first.
        """
        if self._ui is None:
            raise VirtualPadError("pad is not open")
        if len(buttons) != len(BUTTONS):
            raise ValueError(
                f"expected {len(BUTTONS)} buttons in the order {BUTTONS}, "
                f"got {len(buttons)}"
            )
        state = tuple(1 if float(b) > 0.5 else 0 for b in buttons)

        # Opposing directions are unrepresentable in the game: SWRCHARINPUT
        # holds lr and ud as signed integers and the extractor derives the
        # booleans from one number's sign, so no frame in 200 h of corpus has
        # both. rl/policy.py enforces that structurally by factoring each axis
        # as a 3-way categorical. This is the belt to that braces -- a hand-made
        # test vector or a future non-categorical policy would otherwise emit a
        # direction the game has no defined response to, and a silent
        # cancellation is easier to debug than whichever one the game latches.
        lr = state[IDX["right"]] - state[IDX["left"]]
        ud = state[IDX["down"]] - state[IDX["up"]]

        if state == self._last:
            return                      # nothing changed; do not spam the queue
        self._last = state

        ui = self._ui
        # Neutral is the midpoint of the declared range, not zero.
        ui.write(ecodes.EV_ABS, ecodes.ABS_X, AXIS_MID + lr * (AXIS_MAX - AXIS_MID))
        ui.write(ecodes.EV_ABS, ecodes.ABS_Y, AXIS_MID + ud * (AXIS_MAX - AXIS_MID))
        for name, code in BUTTON_CODES:
            ui.write(ecodes.EV_KEY, code, state[IDX[name]])
        # One SYN_REPORT for the whole tick, so the game never observes a
        # half-applied frame -- e.g. the new direction with the old buttons.
        ui.syn()

    def press_only(self, name: str) -> None:
        """Assert exactly one control, everything else neutral.

        Used when binding buttons in Soku's controller config, which asks for
        one input at a time and takes the first thing it sees.
        """
        if name not in IDX:
            raise ValueError(f"unknown control {name!r}; expected one of {BUTTONS}")
        v = [0] * len(BUTTONS)
        v[IDX[name]] = 1
        self.set_state(v)

    # ------------------------------------------------------------------ introspection
    @property
    def device_path(self) -> str:
        """The /dev/input/eventN node the kernel gave this device."""
        if self._ui is None:
            raise VirtualPadError("pad is not open")
        return self._ui.device.path


def available() -> tuple[bool, str]:
    """Whether a virtual pad can be created here, and why not if it cannot."""
    if not HAVE_EVDEV:
        return False, "python-evdev not installed"
    try:
        ui = UInput(
            {ecodes.EV_KEY: [ecodes.BTN_SOUTH],
             ecodes.EV_ABS: [(ecodes.ABS_X,
                              AbsInfo(0, -1, 1, 0, 0, 0))]},
            name="sokubot-probe",
        )
        ui.close()
        return True, "ok"
    except Exception as e:                            # pragma: no cover
        return False, str(e)
