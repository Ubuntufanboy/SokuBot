"""Read the running game's state straight out of its memory. THIS IS CHEATING.

    from sokubot.live.memstate import LiveState
    ls = LiveState.attach()
    state, proj, ok = ls.read()          # [2, 33], [2, slots, 7]

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
-----------------------------------------
The standing constraint is that at PLAY time the agent consumes pixels and its
own inputs, nothing else. This module violates that deliberately and its output
must never reach a shipped agent. It exists for one experiment: the trained
policy reads the game's state directly, plays a human, and the result separates
"the policy is weak" from "the perception is weak" BEFORE the pixels-to-state
encoder is built. If the policy plays well on perfect state, the encoder is the
whole remaining problem and worth the day it costs. If it plays badly, then the
+8 HP/step measured inside the simulator does not transfer, which is the more
important thing to find out and is currently unanswerable.

Every artefact produced through this path is named so that nobody can mistake it
for a real evaluation.

WHY /proc/pid/mem AND NOT THE DLL
----------------------------------
The extractor already does this pointer walk in-process, but it writes a CSV at
end of capture -- there is no live stream, and adding one means a C++ change and
an MSVC cross-compile in the loop. The addresses are ABSOLUTE (`0x008985E4` and
friends in dll/src/session.cpp), so Soku is a fixed-base 32-bit image and the
same walk works from outside through `/proc/<pid>/mem`. No rebuild, no
injection, and the offsets are the ones `pipeline/verify_extended.py` already
checked against the running game.

`ptrace_scope` is 1 on this machine, which permits reading a DESCENDANT's
memory. The harness launches the game, so that holds. Attaching to a game
someone else started needs `kernel.yama.ptrace_scope=0`.

THE NORMALISATION MUST MATCH data/state.py EXACTLY
---------------------------------------------------
The policy was trained on `read_state`'s output, so anything different here is a
different input space and the agent is being asked to play blind. The
derivations are therefore duplicated rather than approximated -- facing-relative
speed multiplied out to world frame, spirit as a fraction of `max_spirit`,
health over the exact int16 rather than the HUD bar -- and
`scripts/verify_memstate.py` checks this reader against the DLL's own CSV on a
replay before it is allowed to drive anything.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.state import (CH, COUNT_SCALE, FLOOR_EPS, FRAME_SCALE, FULL_HP,
                          PF, POS_CLAMP, PROJ_FEATURES, STAGE_SPAN,
                          STATE_CHANNELS, VEL_SCALE)

# --- absolute addresses, from dll/src/session.cpp ---------------------------
ADDR_BATTLE_MANAGER = 0x008985E4
BM_FRAME_COUNT      = 0x04
BM_PLAYER1          = 0x0C
BM_PLAYER2          = 0x10
# A NULL BattleManager is NOT how you tell there is no battle. Read at the
# stage-select screen the pointer is non-null and stale, and the values behind
# it are last match's or nobody's: measured live, `battle_frame 25101048`,
# `hp -15059`, `action 50456`, `hitboxes 223`. Handing that to a policy is
# feeding it noise shaped like a game.
#
# So liveness comes from the scene id, which is what `session.cpp` itself gates
# on (`if (currentScene() == SCENE_BATTLE)`), plus a sanity test on the values.
ADDR_SCENE_ID = 0x008A0044
SCENE_BATTLE = 5
SCENE_BATTLEWATCH = 15

# --- character offsets, from dll/include/sfe/player_state.hpp ---------------
OFF = {
    "x": (0x0EC, "f"), "y": (0x0F0, "f"),
    "vx": (0x0F4, "f"), "vy": (0x0F8, "f"),
    "ax": (0x0FC, "f"), "ay": (0x100, "f"),
    "dir": (0x104, "b"),
    "action": (0x13C, "H"), "action_frame": (0x144, "I"),
    "hp": (0x184, "h"), "hit_count": (0x194, "b"), "hitstop": (0x196, "H"),
    "hitboxes": (0x1CB, "B"), "hurtboxes": (0x1CC, "B"),
    "ground_dashes": (0x49A, "B"), "air_dashes": (0x49B, "B"),
    "spirit": (0x49E, "H"), "max_spirit": (0x4A0, "H"),
    "spirit_delay": (0x4A2, "H"), "timestop": (0x4A8, "H"),
    "correction": (0x4AD, "b"),
    "combo_rate": (0x4B0, "f"), "combo_hits": (0x4B4, "H"),
    "combo_damage": (0x4B6, "H"), "combo_limit": (0x4B8, "H"),
    "untech": (0x4BA, "H"),
}
CHAR_OBJLIST   = 0x6F8
OBJLIST_LIST   = 0x58
LIST_HEAD      = 0x04
LIST_SIZE      = 0x08
NODE_NEXT      = 0x00
NODE_VAL       = 0x08
PROJ_LIST_SANITY = 512

_SIZE = {"f": 4, "b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4}


def find_game_pid(image: str = "th123.exe") -> int | None:
    """PID of the process that has the GAME IMAGE mapped, not merely a matching
    name.

    Two things make the obvious version wrong. `th123e.exe` is the English
    launcher and `th123.exe` is the game it brings up; both are alive at once
    and both match a name test, but only one has the code the offsets refer to.
    Picking by name and `os.scandir` order attached to the launcher, whose
    address space contains none of it. So selection is by the definitive
    property -- `/proc/<pid>/maps` contains the image mapped at its fixed base.

    Deliberately not `pgrep -f` either: that matches its own command line, so a
    pattern naming the game inside a command that also names the game reports a
    dead game as alive.
    """
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            maps = Path(entry.path, "maps").read_text()
        except OSError:
            continue
        if image in maps and "00400000-" in maps:
            return int(entry.name)
    return None


class GameMemory:
    """Random access into another process's address space."""

    def __init__(self, pid: int):
        self.pid = pid
        self.fh = open(f"/proc/{pid}/mem", "rb", buffering=0)

    def close(self) -> None:
        try:
            self.fh.close()
        except OSError:
            pass

    def raw(self, addr: int, n: int) -> bytes | None:
        # A null or unmapped pointer is a normal reading between matches, not
        # an error: the BattleManager does not exist outside a battle.
        if addr <= 0 or addr > 0xFFFFFFFF:
            return None
        try:
            self.fh.seek(addr)
            b = self.fh.read(n)
        except (OSError, ValueError, OverflowError):
            return None
        return b if b and len(b) == n else None

    def val(self, addr: int, code: str):
        b = self.raw(addr, _SIZE[code])
        return None if b is None else struct.unpack("<" + code, b)[0]

    def ptr(self, addr: int) -> int:
        v = self.val(addr, "I")
        return 0 if v is None else v


@dataclass
class Frame:
    state: np.ndarray            # [2, len(STATE_CHANNELS)]
    proj: np.ndarray             # [2, slots, len(PROJ_FEATURES)]
    action: np.ndarray           # [2] int32, nominal action ids
    battle_frame: int


class LiveState:
    """The extractor's read, done from outside, at whatever rate it is called."""

    def __init__(self, mem: GameMemory | None, slots: int = 8):
        self.mem = mem
        self.slots = slots

    def _ensure(self) -> bool:
        """Re-find the game if it went away. THE GAME RESTARTS; WE MUST NOT.

        The pad has to be created before the game so Wine's dinput enumerates
        it, which makes the session outlive any number of game launches. But
        the reader was binding a pid ONCE at startup, so the first time the
        operator relaunched the game the session carried on reading a dead
        process and reported "no battle" through an entire live round.

        Rebinding here means a game restart costs nothing, where previously it
        cost a session restart -- which killed the pad, which forced another
        game restart, which is a loop with no exit.
        """
        if self.mem is not None:
            try:
                if Path(f"/proc/{self.mem.pid}").exists():
                    st = Path(f"/proc/{self.mem.pid}/stat").read_text()
                    if ") Z" not in st:          # not a zombie
                        return True
            except OSError:
                pass
            self.mem.close()
            self.mem = None
        pid = find_game_pid()
        if pid is None:
            return False
        try:
            self.mem = GameMemory(pid)
        except OSError:
            return False
        return True

    @classmethod
    def attach(cls, pid: int | None = None, slots: int = 8) -> "LiveState":
        pid = pid or find_game_pid()
        # A missing game is not fatal any more: the session holds the pad and
        # waits, and `_ensure` binds whenever the game appears.
        return cls(GameMemory(pid) if pid else None, slots)

    # ------------------------------------------------------------------
    def in_battle(self) -> bool:
        """Is a battle actually running? Scene id, not a pointer null-check."""
        sid = self.mem.val(ADDR_SCENE_ID, "I")
        return sid in (SCENE_BATTLE, SCENE_BATTLEWATCH)

    def _chars(self):
        if not self.in_battle():
            return None
        bm = self.mem.ptr(ADDR_BATTLE_MANAGER)
        if not bm:
            return None
        p1 = self.mem.ptr(bm + BM_PLAYER1)
        p2 = self.mem.ptr(bm + BM_PLAYER2)
        if not p1 or not p2:
            return None
        bf = self.mem.val(bm + BM_FRAME_COUNT, "I") or 0
        return p1, p2, bf

    def _fields(self, base: int) -> dict | None:
        out = {}
        for name, (off, code) in OFF.items():
            v = self.mem.val(base + off, code)
            if v is None:
                return None
            out[name] = v
        return out

    def _projectiles(self, base: int, tx: float, ty: float) -> np.ndarray:
        """This player's live objects, expressed against the target they fly at.

        Mirrors `dll/src/player_state.cpp:walkProjectiles`, including the
        danger-first ordering -- live hitbox first, then nearest -- because slot
        k is a meaningful question ("the k-th most threatening object") only if
        it is filled the same way it was during training.
        """
        out = np.zeros((self.slots, len(PROJ_FEATURES)), dtype=np.float32)
        mgr = self.mem.ptr(base + CHAR_OBJLIST)
        if not mgr:
            return out
        lst = mgr + OBJLIST_LIST
        size = self.mem.val(lst + LIST_SIZE, "I")
        head = self.mem.ptr(lst + LIST_HEAD)
        if not size or not head or size > PROJ_LIST_SANITY:
            return out
        found = []
        node = self.mem.ptr(head + NODE_NEXT)
        for _ in range(min(int(size), PROJ_LIST_SANITY)):
            if not node or node == head:
                break
            obj = self.mem.ptr(node + NODE_VAL)
            if obj:
                px = self.mem.val(obj + OFF["x"][0], "f")
                py = self.mem.val(obj + OFF["y"][0], "f")
                pvx = self.mem.val(obj + OFF["vx"][0], "f")
                pvy = self.mem.val(obj + OFF["vy"][0], "f")
                pdir = self.mem.val(obj + OFF["dir"][0], "b")
                phb = self.mem.val(obj + OFF["hitboxes"][0], "B")
                if None not in (px, py, pvx, pvy, pdir, phb):
                    d2 = (px - tx) ** 2 + (py - ty) ** 2
                    found.append((0 if phb else 1, d2, px, py, pvx, pvy,
                                  pdir, phb))
            node = self.mem.ptr(node + NODE_NEXT)
        found.sort(key=lambda r: (r[0], r[1]))
        for k, (_t, _d, px, py, pvx, pvy, pdir, phb) in enumerate(found[:self.slots]):
            d = 1.0 if pdir > 0 else (-1.0 if pdir < 0 else 0.0)
            dx = float(np.clip((px - tx) / STAGE_SPAN, -POS_CLAMP, POS_CLAMP))
            dy = float(np.clip((py - ty) / STAGE_SPAN, -POS_CLAMP, POS_CLAMP))
            vx = float(np.clip(pvx * d / VEL_SCALE, -POS_CLAMP, POS_CLAMP))
            vy = float(np.clip(pvy / VEL_SCALE, -POS_CLAMP, POS_CLAMP))
            f = out[k]
            f[PF["present"]] = 1.0
            f[PF["dx"]] = dx
            f[PF["dy"]] = dy
            f[PF["vx"]] = vx
            f[PF["vy"]] = vy
            f[PF["hb"]] = 1.0 if phb > 0 else 0.0
            f[PF["closing"]] = -np.sign(dx) * np.sign(vx)
        return out

    def read(self) -> Frame | None:
        """One frame of state, or None if no battle is running."""
        if not self._ensure():
            return None
        got = self._chars()
        if got is None:
            return None
        p1, p2, bf = got
        raw = [self._fields(p1), self._fields(p2)]
        if raw[0] is None or raw[1] is None:
            return None
        # Second gate: the scene can be right while the objects are still being
        # built during the load transition. Health outside the int16 the game
        # keeps, or an action id far past the enum, means "not yet" -- and it is
        # cheaper to skip a frame than to act on one that is wrong.
        for r in raw:
            if not (0 <= r["hp"] <= 10000) or not (0 <= r["action"] < 1000):
                return None
        # `guarding`, `wrongblock`, `crushed` and `knockdown` are derived from
        # the action id in the extractor; the ranges come from SokuLib's own
        # ACTION_* names. Kept in one place so a wrong range shows up as a flag
        # that never fires rather than as a channel that quietly lies.
        state = np.zeros((2, len(STATE_CHANNELS)), dtype=np.float32)
        proj = np.zeros((2, self.slots, len(PROJ_FEATURES)), dtype=np.float32)
        action = np.array([raw[0]["action"], raw[1]["action"]], dtype=np.int32)
        xs = [float(raw[0]["x"]), float(raw[1]["x"])]
        ys = [float(raw[0]["y"]), float(raw[1]["y"])]

        for me in (0, 1):
            them = 1 - me
            r, s = raw[me], state[me]
            s[CH["dx"]] = (xs[them] - xs[me]) / STAGE_SPAN
            s[CH["dy"]] = (ys[them] - ys[me]) / STAGE_SPAN
            s[CH["x"]] = xs[me] / STAGE_SPAN
            s[CH["y"]] = ys[me] / STAGE_SPAN
            d = r["dir"]
            facing = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
            s[CH["facing"]] = facing
            # Facing-relative -> world, exactly as data/state.py does it once.
            s[CH["vx"]] = r["vx"] * facing / VEL_SCALE
            s[CH["vy"]] = r["vy"] / VEL_SCALE
            s[CH["ax"]] = r["ax"] * facing / VEL_SCALE
            s[CH["ay"]] = r["ay"] / VEL_SCALE
            s[CH["hitboxes"]] = r["hitboxes"] / COUNT_SCALE
            s[CH["hurtboxes"]] = r["hurtboxes"] / COUNT_SCALE
            s[CH["hitstop"]] = r["hitstop"] / FRAME_SCALE
            s[CH["untech"]] = r["untech"] / FRAME_SCALE
            s[CH["action_frame"]] = r["action_frame"] / FRAME_SCALE
            s[CH["hit_count"]] = r["hit_count"] / COUNT_SCALE
            s[CH["hp"]] = r["hp"] / FULL_HP
            msp = r["max_spirit"]
            s[CH["spirit"]] = (r["spirit"] / max(msp, 1)) if msp > 0 else 0.0
            s[CH["spirit_delay"]] = r["spirit_delay"] / FRAME_SCALE
            s[CH["timestop"]] = r["timestop"] / FRAME_SCALE
            s[CH["ground_dashes"]] = r["ground_dashes"]
            s[CH["air_dashes"]] = r["air_dashes"]
            s[CH["correction"]] = r["correction"] / 100.0
            s[CH["combo_rate"]] = r["combo_rate"]
            s[CH["combo_hits"]] = r["combo_hits"] / COUNT_SCALE
            s[CH["combo_damage"]] = r["combo_damage"] / FULL_HP
            s[CH["combo_limit"]] = r["combo_limit"] / COUNT_SCALE
            s[CH["airborne"]] = 1.0 if ys[me] > FLOOR_EPS else 0.0
            act = r["action"]
            s[CH["guarding"]] = 1.0 if _is_guard(act) else 0.0
            s[CH["wrongblock"]] = 1.0 if _is_wrongblock(act) else 0.0
            s[CH["crushed"]] = 1.0 if _is_crushed(act) else 0.0
            s[CH["knockdown"]] = 1.0 if _is_knockdown(act) else 0.0

        for me in (0, 1):
            them = 1 - me
            proj[me] = self._projectiles(
                [p1, p2][me], xs[them], ys[them])
            n = float((proj[me][:, PF["present"]] > 0).sum())
            state[me][CH["proj_n"]] = n / COUNT_SCALE
            state[me][CH["proj_hb"]] = float(
                (proj[me][:, PF["hb"]] > 0).sum()) / COUNT_SCALE
        return Frame(state, proj, action, int(bf))


# --- action-id ranges -------------------------------------------------------
# COPIED from dll/include/sfe/player_state.hpp, not inferred. The first version
# of this file guessed all four ranges and every one was wrong -- guard as
# 150-155/160-165 against the real 150-157 plus 158, crushed as 170/171 against
# the real 143/145, knockdown as 180-199 against the real 97/98/100. A guessed
# range does not fail loudly; it produces a flag that is silently always false,
# and `guarding` always false is indistinguishable from an agent that never
# blocks -- which is the exact question this whole experiment is about.
ACT_RIGHTBLOCK_FIRST, ACT_RIGHTBLOCK_LAST = 150, 157
ACT_AIR_GUARD = 158
ACT_WRONGBLOCK_FIRST, ACT_WRONGBLOCK_LAST = 159, 166
ACT_GROUND_CRUSHED, ACT_AIR_CRUSHED = 143, 145
ACT_KNOCKED_DOWN, ACT_KNOCKED_DOWN_STATIC, ACT_GRABBED = 97, 98, 100


def _is_guard(a: int) -> bool:
    # AIR_GUARD sits BETWEEN the two blockstun ranges, which is why this is not
    # one contiguous test.
    return (ACT_RIGHTBLOCK_FIRST <= a <= ACT_RIGHTBLOCK_LAST) or a == ACT_AIR_GUARD


def _is_wrongblock(a: int) -> bool:
    return ACT_WRONGBLOCK_FIRST <= a <= ACT_WRONGBLOCK_LAST


def _is_crushed(a: int) -> bool:
    return a in (ACT_GROUND_CRUSHED, ACT_AIR_CRUSHED)


def _is_knockdown(a: int) -> bool:
    return a in (ACT_KNOCKED_DOWN, ACT_KNOCKED_DOWN_STATIC, ACT_GRABBED)
