"""Soku input profiles (`profile/*.pf`) and which slot uses which.

    from sokubot.live import profiles
    profiles.audit(Path.home() / ".wine-soku/drive_c/Games/Soku")

WHY THIS EXISTS
---------------
Wine's DirectInput reads evdev devices directly and ignores window focus, so
the agent's virtual keyboard is read by the HUMAN's player too if the two
profiles share a key. That is not a hypothetical: of the eight profiles
installed, `noob`, `bleh` and `apollo` each collide with the agent's bindings.
Until now nothing in either repo could read a `.pf` at all -- `pad.py:153`
points at `scripts/live_probe.py --profiles`, a flag that was never written,
and its own comment about the human's bindings has since gone stale (it says
`anon` holds W/A/S/D; `anon.pf` now holds the arrow keys).

So the collision check is a measurement rather than a comment.

THE FORMAT, DECODED FROM THE BYTES
----------------------------------
`profile/*.pf`, 3386 bytes. Twelve little-endian uint32 at offset 0:

    [0]      input device; 0xFF = keyboard
    [1..10]  DirectInput scancodes for up, down, left, right,
             a, b, c, d, change-card, spell-card
    [11]     one more binding (Q or RETURN across the installed set)

then 14 dwords and 80 41-byte deck records, none of which this touches.

`config123.dat` says which profile each SLOT loads, as two length-prefixed
strings at offset 4:

    uint32 len, char[len] p1_profile, uint32 len, char[len] p2_profile

Writing that file is how the agent's profile gets selected without navigating
the Key Config menu.

SCANCODES ARE NOT EVDEV KEYS, AND THE DIFFERENCE HAS BITTEN
-----------------------------------------------------------
`pad.py` names keys the way evdev does (`KEY_T`); a profile stores the
DirectInput scancode (0x14). They are different numbering schemes that happen
to agree often enough to look identical. The numeric keypad is where they come
apart: the uinput device demonstrably emits KEY_KP6/KEY_KP7 and Soku ignores
them with NumLock either way, which cost a rewrite of the whole binding.
`check_pad_agrees` compares the two directly rather than trusting that they
match.
"""

from __future__ import annotations

import struct
from pathlib import Path

# DirectInput scancode -> evdev key name, for every code any installed profile
# uses plus the ordinary typing keys an operator might bind. Codes absent here
# are reported as `DIK_xx` rather than guessed at.
DIK_TO_EVDEV: dict[int, str] = {
    0x01: "KEY_ESC",
    **{0x02 + i: f"KEY_{d}" for i, d in enumerate("1234567890")},
    0x0C: "KEY_MINUS", 0x0D: "KEY_EQUAL", 0x0E: "KEY_BACKSPACE", 0x0F: "KEY_TAB",
    **{c: f"KEY_{k}" for c, k in zip(range(0x10, 0x1A), "QWERTYUIOP")},
    0x1A: "KEY_LEFTBRACE", 0x1B: "KEY_RIGHTBRACE", 0x1C: "KEY_ENTER",
    0x1D: "KEY_LEFTCTRL",
    **{c: f"KEY_{k}" for c, k in zip(range(0x1E, 0x27), "ASDFGHJKL")},
    0x27: "KEY_SEMICOLON", 0x28: "KEY_APOSTROPHE", 0x29: "KEY_GRAVE",
    0x2A: "KEY_LEFTSHIFT", 0x2B: "KEY_BACKSLASH",
    **{c: f"KEY_{k}" for c, k in zip(range(0x2C, 0x33), "ZXCVBNM")},
    0x33: "KEY_COMMA", 0x34: "KEY_DOT", 0x35: "KEY_SLASH",
    0x36: "KEY_RIGHTSHIFT", 0x38: "KEY_LEFTALT", 0x39: "KEY_SPACE",
    0x3A: "KEY_CAPSLOCK",
    **{0x3B + i: f"KEY_F{i + 1}" for i in range(10)},
    0x45: "KEY_NUMLOCK", 0x46: "KEY_SCROLLLOCK",
    0x47: "KEY_KP7", 0x48: "KEY_KP8", 0x49: "KEY_KP9", 0x4A: "KEY_KPMINUS",
    0x4B: "KEY_KP4", 0x4C: "KEY_KP5", 0x4D: "KEY_KP6", 0x4E: "KEY_KPPLUS",
    0x4F: "KEY_KP1", 0x50: "KEY_KP2", 0x51: "KEY_KP3", 0x52: "KEY_KP0",
    0x53: "KEY_KPDOT", 0x57: "KEY_F11", 0x58: "KEY_F12",
    0x9C: "KEY_KPENTER", 0x9D: "KEY_RIGHTCTRL", 0xB5: "KEY_KPSLASH",
    0xB8: "KEY_RIGHTALT",
    0xC7: "KEY_HOME", 0xC8: "KEY_UP", 0xC9: "KEY_PAGEUP",
    0xCB: "KEY_LEFT", 0xCD: "KEY_RIGHT", 0xCF: "KEY_END",
    0xD0: "KEY_DOWN", 0xD1: "KEY_PAGEDOWN", 0xD2: "KEY_INSERT",
    0xD3: "KEY_DELETE",
}
EVDEV_TO_DIK = {v: k for k, v in DIK_TO_EVDEV.items()}

# Same order as pad.BUTTONS, which is the corpus order, plus the trailing slot.
CONTROLS = ("up", "down", "left", "right",
            "a", "b", "c", "d", "change", "spell", "extra")

PROFILE_BYTES = 3386
KEYBOARD_DEVICE = 0xFF


def key_name(dik: int) -> str:
    return DIK_TO_EVDEV.get(dik, f"DIK_{dik:02X}")


class Profile:
    """One `.pf` file: the device, the eleven bindings, and the raw bytes."""

    def __init__(self, path: Path, raw: bytes):
        self.path, self.raw = path, raw
        head = struct.unpack("<12I", raw[:48])
        self.device = head[0]
        self.codes: dict[str, int] = dict(zip(CONTROLS, head[1:12]))

    @classmethod
    def load(cls, path: Path) -> "Profile":
        raw = Path(path).read_bytes()
        if len(raw) != PROFILE_BYTES:
            raise ValueError(f"{path}: {len(raw)} bytes, expected {PROFILE_BYTES}")
        return cls(Path(path), raw)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def is_keyboard(self) -> bool:
        return self.device == KEYBOARD_DEVICE

    def keys(self) -> dict[str, str]:
        """control -> evdev key name."""
        return {c: key_name(d) for c, d in self.codes.items()}

    def key_set(self) -> set[str]:
        """Every evdev key this profile holds, for collision arithmetic."""
        return set(self.keys().values())

    def rebind(self, mapping: dict[str, str]) -> bytes:
        """New file bytes with `control -> evdev key` applied. Not written.

        Only the twelve header dwords change; decks and the rest of the file
        are carried through byte for byte, because nothing here understands
        them and a rewrite that dropped them would lose the user's decks.
        """
        codes = dict(self.codes)
        for control, key in mapping.items():
            if control not in CONTROLS:
                raise KeyError(f"unknown control {control!r}")
            if key not in EVDEV_TO_DIK:
                raise KeyError(f"no DirectInput scancode known for {key!r}")
            codes[control] = EVDEV_TO_DIK[key]
        head = struct.pack("<12I", KEYBOARD_DEVICE,
                           *(codes[c] for c in CONTROLS))
        return head + self.raw[48:]


def load_all(game_dir: Path) -> dict[str, Profile]:
    d = Path(game_dir) / "profile"
    return {p.stem: Profile.load(p) for p in sorted(d.glob("*.pf"))}


def selected_profiles(game_dir: Path) -> tuple[str, str]:
    """(p1, p2) profile names the game will load, from `config123.dat`."""
    raw = (Path(game_dir) / "config123.dat").read_bytes()
    off = 4
    out = []
    for _ in range(2):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        if not 0 < n <= 64 or off + n > len(raw):
            raise ValueError(f"config123.dat: implausible name length {n} at {off - 4}")
        out.append(raw[off:off + n].decode("ascii", "replace"))
        off += n
    return out[0].removesuffix(".pf"), out[1].removesuffix(".pf")


def collisions(a: Profile, b: Profile,
               emitted: set[str] | None = None) -> dict[str, tuple[str, str]]:
    """evdev key -> (a's control, b's control) for every shared key.

    Only meaningful between two KEYBOARD profiles: dinput reads the device, so
    two players on one keyboard genuinely share a key press.

    `emitted` restricts the answer to keys the agent's device can ACTUALLY
    send. That distinction is not pedantry. The eleventh binding is `pause`,
    and both `anon` and `sokubot` bind it to Q -- so a naive overlap test
    reports the human's live profile as colliding, every session, on a key the
    pad has no code for. Crying wolf on the one profile that is always in use
    is how a real collision gets waved through.
    """
    if not (a.is_keyboard and b.is_keyboard):
        return {}
    out = {}
    for ca, ka in a.keys().items():
        if emitted is not None and ka not in emitted:
            continue
        for cb, kb in b.keys().items():
            if ka == kb:
                out[ka] = (ca, cb)
    return out


def check_pad_agrees(agent: Profile, pad_codes) -> list[str]:
    """Complaints where the profile disagrees with the uinput device's keys.

    `pad_codes` is `sokubot.live.pad.KEYPAD_CODES`. A disagreement means the
    agent presses a key the game is not listening for, which looks exactly like
    "uinput does not reach the game".
    """
    want = dict(pad_codes)
    have = agent.keys()
    out = []
    if not agent.is_keyboard:
        out.append(f"{agent.name}: device is 0x{agent.device:02X}, not a keyboard "
                   "(0xFF) -- Soku will read a joystick, and its menus drift")
    for control, key in want.items():
        got = have.get(control)
        if got != key:
            out.append(f"{agent.name}: {control} is {got}, pad sends {key}")
    return out


def audit(game_dir: Path, agent: str = "sokubot") -> dict:
    """Everything the operator needs before arming, as data."""
    profs = load_all(Path(game_dir))
    p1, p2 = selected_profiles(Path(game_dir))
    a = profs.get(agent)
    report = {
        "selected": {"p1": p1, "p2": p2},
        "agent": agent,
        "agent_slot": 1 if p1 == agent else 2 if p2 == agent else None,
        "keys": {n: p.keys() for n, p in profs.items()},
        "collides_with_agent": {},
        "benign_overlap": {},
        "safe_opponent_profiles": [],
        "pad_complaints": [],
    }
    if a is None:
        report["pad_complaints"].append(f"no profile named {agent!r} installed")
        return report
    from .pad import KEYPAD_CODES
    emitted = {k for _, k in KEYPAD_CODES}
    for name, p in profs.items():
        if name == agent:
            continue
        c = collisions(a, p, emitted)
        if c:
            report["collides_with_agent"][name] = c
        # Overlap on a key the pad cannot send is worth printing once and not
        # worth blocking on.
        benign = {k: v for k, v in collisions(a, p).items() if k not in c}
        if benign:
            report["benign_overlap"][name] = benign
    report["pad_complaints"] = check_pad_agrees(a, KEYPAD_CODES)
    report["safe_opponent_profiles"] = sorted(
        n for n in profs if n != agent and n not in report["collides_with_agent"])
    return report
