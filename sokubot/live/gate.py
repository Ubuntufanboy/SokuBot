"""When is the agent allowed to press anything?

Two independent conditions, and both must hold. They are independent on
purpose, because they fail in different directions:

* **Are we in a battle?** Read off the HUD pixels, reusing the constants in
  ``sokubot/data/hud.py`` that were calibrated against the corpus. Outside a
  battle the policy's output is meaningless -- the encoder has only ever seen
  battle frames -- and letting it press buttons in a menu means it navigates the
  game somewhere nobody asked for.
* **Has a human armed it?** A hardware-style cutoff the model cannot influence.
  A gate derived purely from what the agent observes can be wrong in exactly the
  situation where being wrong is worst, and there has to be one control whose
  behaviour does not depend on the thing it is controlling.

Reading the HUD is pixels, not memory, so it is inside the project's constraint
(``docs/HANDOFF.md`` section 8). It is also used only to decide *whether to act*,
never as an observation the policy consumes or a reward it learns from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.hud import FILL_ROWS, P1_HP_X, P2_HP_X, _split_bar

# A battle frame has two health bars of the right geometry, both substantially
# lit. `_split_bar` returns (yellow, red) as fractions of the 189 px bar; their
# sum is health remaining plus the damage of any combo in progress, which stays
# near 1.0 at round start and falls from there.
#
# The threshold is deliberately low. A false negative costs one held chunk; a
# false positive lets the agent mash in a menu. Both bars must pass, which is
# what makes a menu screen that happens to have bright pixels in one band fail.
MIN_BAR = 0.15

# Both conditions have to persist. Round transitions, KO flashes and the
# whole-screen washes documented in hud.py all move the bars briefly, and a gate
# that reacts to a single frame chatters between armed and disarmed at exactly
# the moments the game is most confusing to read.
ON_FRAMES = 3
OFF_FRAMES = 8          # slower to disarm than to arm; see below


@dataclass
class BattleGate:
    """Debounced 'is a battle happening' detector over live 480x480 frames."""

    min_bar: float = MIN_BAR
    on_frames: int = ON_FRAMES
    off_frames: int = OFF_FRAMES
    in_battle: bool = False
    _on: int = field(default=0, repr=False)
    _off: int = field(default=0, repr=False)
    last: tuple[float, float] = (0.0, 0.0)

    def update(self, frame480: np.ndarray) -> bool:
        """Feed one capture-orientation frame; returns whether a battle is on.

        The frame arrives vertically flipped, because that is the orientation
        the corpus is stored in and therefore the one `capture.py` produces.
        hud.py's constants are screen-space, so it is flipped back here -- the
        same `flip=True` that `read_trace` applies by default.
        """
        f = frame480[::-1][None]                     # -> screen space, [1,H,W,3]
        fr, fc = FILL_ROWS
        y1, r1 = _split_bar(f[:, fr:fc, P1_HP_X[0]:P1_HP_X[1]])
        y2, r2 = _split_bar(f[:, fr:fc, P2_HP_X[0]:P2_HP_X[1]])
        b1, b2 = float(y1[0] + r1[0]), float(y2[0] + r2[0])
        self.last = (b1, b2)

        looks_like_battle = b1 >= self.min_bar and b2 >= self.min_bar
        if looks_like_battle:
            self._on += 1
            self._off = 0
            if self._on >= self.on_frames:
                self.in_battle = True
        else:
            self._off += 1
            self._on = 0
            # Disarming is slower than arming because the expensive mistake is
            # asymmetric. A KO flash or a super's screen wash blanks the bars for
            # up to ~20 frames at 60 fps (hud.py, DIP_WINDOW) -- at the 15 Hz
            # decision rate that is a handful of samples -- and dropping the
            # agent to neutral mid-round because the screen went white is a free
            # punish for the opponent.
            if self._off >= self.off_frames:
                self.in_battle = False
        return self.in_battle


@dataclass
class ArmSwitch:
    """The human's cutoff. Toggled from outside; defaults to disarmed.

    Deliberately trivial and deliberately separate from `BattleGate`: this is
    the control that has to keep working when the other one is wrong.
    """

    armed: bool = False

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def toggle(self) -> bool:
        self.armed = not self.armed
        return self.armed


def may_act(gate: BattleGate, switch: ArmSwitch) -> bool:
    return switch.armed and gate.in_battle
