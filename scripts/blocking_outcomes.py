"""What does holding away do when an attack actually arrives?

Pooling over all near frames was wrong. Most of them have nothing incoming, so
"away" there is just walking backwards -- a valid action that produces no guard
flag and no hit -- and averaging it in drives the apparent effect toward zero
and muddles what is being asked.

Condition instead on an attack going live next step: the opponent's hitbox is
inactive at t and active at t+1, and the defender is not already guarding. That
is pre-treatment with respect to the defender's stick, because the opponent's
animation was committed frames earlier and this collector's attacker never
reacts to the defender.

Three outcomes, because holding away has two distinct good effects and lumping
them hides one:
    GUARD    the attack connected and was blocked
    HIT      the attack connected and was not
    AVOID    no contact -- often because walking away moved out of range
"""
import sys
import numpy as np

sys.path.insert(0, "/home/anon/SokuBot")
from sokubot.data.state import CH

NEAR = 250.0 / 1200.0
d = np.load(sys.argv[1])
S, A, E = d["S"], d["A"], d["E"]
nxt = np.zeros(len(S), bool); nxt[:-1] = E[1:] == E[:-1]

for p in (0, 1):
    dx = S[:, p, CH["dx"]]
    left = A[:, :, p * 10 + 2].mean(1) > 0.5
    right = A[:, :, p * 10 + 3].mean(1) > 0.5
    one = left ^ right
    away = np.where(dx > 0, left, right)

    hb = S[:, 1 - p, CH["hitboxes"]] > 0.01
    hb_next = np.zeros(len(S), bool); hb_next[:-1] = hb[1:]
    arrives = (~hb) & hb_next                      # goes live next step

    guard_now = S[:, p, CH["guarding"]] > 0.5
    g_next = np.zeros(len(S), bool); g_next[:-1] = S[1:, p, CH["guarding"]] > 0.5
    w_next = np.zeros(len(S), bool); w_next[:-1] = S[1:, p, CH["wrongblock"]] > 0.5
    hp = S[:, p, CH["hp"]]
    hit_next = np.zeros(len(S), bool); hit_next[:-1] = hp[1:] < hp[:-1] - 1e-6

    m = nxt & one & (np.abs(dx) < NEAR) & arrives & ~guard_now
    blocked = g_next | w_next
    for lab, sel in (("away", m & away), ("toward", m & ~away)):
        n = int(sel.sum())
        if n < 50:
            print("p%d %-7s too few (n=%d)" % (p + 1, lab, n)); continue
        b = blocked[sel].mean(); h = hit_next[sel].mean()
        print("p%d %-7s n=%6d   GUARD %.4f   HIT %.4f   AVOID %.4f"
              % (p + 1, lab, n, b, h, 1 - b - h))
