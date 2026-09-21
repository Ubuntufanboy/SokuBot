"""Where the status overlay may sit: anywhere the game's picture is not.

THE RULE, AND WHY IT IS NOT NEGOTIABLE
--------------------------------------
Capture is a root-region X11 grab of the game window's rectangle. Whatever is painted
over that rectangle is captured INSTEAD of the game -- a terminal, a notification, and
this overlay. An overlay drawn over the picture is not a display, it is a corruption of
the agent's only input, and it fails silently: the encoder reads a strip of UI as
scenery and the numbers still look plausible.

So the overlay is placed OUTSIDE the game's client rect, with a margin, and if there is
nowhere it can go it is hidden rather than placed anyway. `place_outside` returns None
in that case and the caller must not show the window.

Pure geometry, no display needed.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


def overlaps(a: Rect, b: Rect, margin: int = 0) -> bool:
    """True if `a` and `b` are closer than `margin` pixels (or intersect)."""
    return not (a.right + margin <= b.x or b.right + margin <= a.x
                or a.bottom + margin <= b.y or b.bottom + margin <= a.y)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def place_outside(game: Optional[Rect], screen: tuple[int, int],
                  size: tuple[int, int], margin: int = 12) -> Optional[tuple[int, int]]:
    """Top-left for a `size` window that stays on `screen` and clear of `game`.

    Tries below, above, right and left of the game window, then the four screen
    corners, and returns the first that fits. With no game window yet it takes the
    bottom-left corner. None means there is no room anywhere: do not show it.
    """
    sw, sh = screen
    w, h = size
    if w > sw or h > sh:
        return None

    def ok(x: int, y: int) -> bool:
        r = Rect(x, y, w, h)
        inside = 0 <= x and 0 <= y and x + w <= sw and y + h <= sh
        return inside and (game is None or not overlaps(r, game, margin))

    corners = [(margin, sh - h - margin), (sw - w - margin, sh - h - margin),
               (margin, margin), (sw - w - margin, margin)]
    cands: list[tuple[int, int]] = []
    if game is not None:
        gx = _clamp(game.x, 0, sw - w)
        gy = _clamp(game.y, 0, sh - h)
        cands += [(gx, game.bottom + margin),          # below
                  (gx, game.y - margin - h),           # above
                  (game.right + margin, gy),           # right
                  (game.x - margin - w, gy)]           # left
    cands += corners
    for x, y in cands:
        if ok(x, y):
            return x, y
    return None
