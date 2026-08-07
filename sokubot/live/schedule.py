"""Turn 15 Hz decisions into 60 Hz button state.

The policy emits one *chunk* per decision: `frame_skip` ticks x 10 buttons
(``Config.soku()`` -> ticks 4, 60 fps / 4 = 15 Hz). The game reads input every
tick, so something has to hold the chunk and play it out one tick at a time.
That is all this does, plus the one decision that actually matters:

WHAT TO DO WHEN THE NEXT CHUNK IS LATE
--------------------------------------
It will be late often. Inference is not free and, wherever it runs, the reply
does not arrive on a schedule the game respects. Three options:

* **Neutral.** Wrong. Releasing everything is not the absence of a decision, it
  is the decision to stop blocking, stop holding a charge, and drop out of a
  dash. In a fighting game neutral is a specific, punishable state.
* **Re-run the last chunk from its start.** Turns a late reply into a repeated
  input -- a second 5A, a second jump. Invents actions nobody chose.
* **Hold the final tick of the last chunk.** What this does. A held direction
  keeps walking, a held button keeps charging, and a released one stays
  released. It is the continuation of the last decision rather than a new one,
  which is also what a human does when they have not decided yet.

The distinction only exists because the chunk has internal structure. Holding
tick 3 of the previous chunk is *not* the same as replaying the chunk, and
conflating them was worth writing down.

TIMING IS ABSOLUTE, NOT RELATIVE
--------------------------------
Ticks are scheduled against a fixed origin rather than by sleeping a tick's
worth each time. Accumulating `sleep(1/60)` drifts by however long each
iteration's work took, and at 60 Hz that drift becomes a whole tick within
seconds. `next_deadline` recomputes from the origin, so a slow iteration is
absorbed instead of compounding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

TICK_HZ = 60.0
TICK_S = 1.0 / TICK_HZ

NEUTRAL = np.zeros(10, dtype=np.float32)


@dataclass
class ChunkScheduler:
    """Holds the current action chunk and says what to press on each tick.

    `ticks` is the chunk length in game frames. `submit` replaces the chunk;
    `state_at` reads out the tick due at a given time.
    """

    ticks: int = 4
    origin: float = field(default_factory=time.perf_counter)
    _chunk: np.ndarray | None = field(default=None, repr=False)
    _chunk_tick: int = field(default=0, repr=False)
    _chunk_id: int = -1

    # Counters worth reporting after a match: a loop that held 60% of its ticks
    # was not really playing, and that has to be visible rather than inferred.
    applied: int = 0
    held: int = 0
    idle: int = 0

    def submit(self, chunk: np.ndarray, at_tick: int, chunk_id: int = -1) -> None:
        """Install a chunk to begin at absolute tick `at_tick`."""
        if chunk.shape != (self.ticks, 10):
            raise ValueError(f"expected chunk {(self.ticks, 10)}, got {chunk.shape}")
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self._chunk_tick = at_tick
        self._chunk_id = chunk_id

    def tick_index(self, now: float | None = None) -> int:
        """Which absolute 60 Hz tick we are in."""
        now = time.perf_counter() if now is None else now
        return int((now - self.origin) / TICK_S)

    def next_deadline(self, tick: int) -> float:
        """Wall-clock time of the *start* of absolute tick `tick`."""
        return self.origin + tick * TICK_S

    def state_at(self, tick: int) -> np.ndarray:
        """The 10-wide button vector due at absolute tick `tick`."""
        if self._chunk is None:
            self.idle += 1
            return NEUTRAL
        offset = tick - self._chunk_tick
        if offset < 0:
            # The chunk is for a tick that has not arrived; keep holding what
            # came before it rather than pre-empting it.
            self.held += 1
            return self._chunk[0]
        if offset < self.ticks:
            self.applied += 1
            return self._chunk[offset]
        # Past the end of the chunk: the next one is late. Hold its last tick.
        self.held += 1
        return self._chunk[-1]

    def stats(self) -> dict:
        total = self.applied + self.held + self.idle
        return {
            "ticks": total,
            "applied": self.applied,
            "held": self.held,
            "idle": self.idle,
            "held_pct": round(100 * self.held / total, 1) if total else 0.0,
        }


@dataclass
class DelayPolicy:
    """How many decision steps ahead a chunk is scheduled for.

    Fixed rather than adaptive, and that is the point. A constant delay plays
    like a player with a fixed reaction time; a delay that tracks the network
    plays like a different player every second, and neither the policy nor the
    person it is playing can adapt to that. The world model compensates for the
    constant part by rolling the latent forward `steps` decisions, which is only
    trustworthy for one or two -- one-step rollout cosine is 0.9963 against
    0.9717 at four (`scripts/horizon_ablation.py`).
    """

    steps: int = 1
    frame_skip: int = 4

    def target_tick(self, captured_at_tick: int) -> int:
        return captured_at_tick + self.steps * self.frame_skip

    @property
    def seconds(self) -> float:
        return self.steps * self.frame_skip * TICK_S
