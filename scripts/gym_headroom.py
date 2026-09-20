"""How much damage is actually available in each gym's window, in real data.

    python -m scripts.gym_headroom --corpus ~/corpus --cache ~/rl/bank_state.npz

A gym that a policy cannot improve on may be a hard mechanic, or it may be a
window in which nothing happens. Those need opposite responses -- more training
versus a longer horizon -- and the training curve cannot tell them apart, so
this asks the corpus instead. No model is involved: it measures what real
players did in the same windows the gym samples.

`okizeme` is the case that motivated it. It is flat at +0.5 HP/step after 7600
GRPO steps while `neutral` reaches +20. The mechanic is meeting an opponent as
they get off the floor, and the payoff of doing it well arrives when they are
up -- which may simply be past the eight decision steps the rollout covers. If
the real corpus also shows near-zero damage in that window, the gym is not
teaching a hard thing badly; it is asking a question the horizon cannot contain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sokubot.data import state_bank
from sokubot.data.state import CH, FULL_HP
from scripts.build_gyms import build as build_gyms
from scripts.train_state_grpo import valid_starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--ticks", type=int, default=5)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--history", type=int, default=12)
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32])
    a = ap.parse_args()

    S, P, A, E, V, names = state_bank.load(a.corpus, a.ticks, a.slots, a.cache)
    hp = S[:, :, CH["hp"]]
    big = max(a.horizons)
    gyms = build_gyms(S, P, V, E, big, a.history)
    starts = valid_starts(E, V, a.history, big)

    def headroom(idx, side, h):
        """Damage EXCHANGED over h steps from these starts, in game HP.

        Reported as what the agent could win (their drop) and what it stands to
        lose (its own), separately -- a window where both sides take 300 HP is a
        very different drill from one where neither takes any, and their
        difference hides that.
        """
        me, them = side, 1 - side
        d_them = np.clip(hp[idx + h, them] - hp[idx, them], None, 0)
        d_me = np.clip(hp[idx + h, me] - hp[idx, me], None, 0)
        return -d_them.mean() * FULL_HP, -d_me.mean() * FULL_HP

    rows = []
    for name in sorted(gyms):
        st, sd = gyms[name]
        if len(st) == 0:
            continue
        take = np.random.default_rng(0).choice(len(st), min(200_000, len(st)),
                                               replace=False)
        rows.append((name, st[take], sd[take]))
    rows.append(("(corpus)", starts, np.zeros(len(starts), np.int64)))

    print(f"\nreal damage in each gym's window, game HP, corpus not model")
    hdr = "  ".join(f"h{h}: deal/take" for h in a.horizons)
    print(f"  {'gym':<20} {'pairs':>8}  {hdr}")
    for name, st, sd in rows:
        cells = []
        for h in a.horizons:
            w, l = headroom(st, sd, h)
            cells.append(f"{w:6.1f}/{l:5.1f}")
        print(f"  {name:<20} {len(st):>8}  {'  '.join(cells)}")
    print("\n'deal' is the opponent's mean health drop over the window and "
          "'take' is the agent's own.\nA gym whose deal column is small at h8 "
          "cannot be improved much at h8, however\nwell the policy learns: "
          "there is no damage in the window to move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
