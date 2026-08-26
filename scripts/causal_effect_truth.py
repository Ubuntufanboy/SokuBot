"""Empirical away-vs-toward effect on guarding, per corpus.

In the v3 corpus the defender's stance is drawn from a mode schedule that no
game state touches; the only world input to the action is distance. So within a
distance stratum there is no back-door path from stance to next state, and the
plain empirical difference IS the causal effect. In the human corpus the same
number is a confounded association: people hold away BECAUSE they expect an
attack, so "away" is partly a proxy for "attack incoming".

Both are in probability units, directly comparable to the models' away-toward
numbers. Stratified by distance and pooled with equal weight per stratum, so
the two corpora's different distance distributions cannot drive the contrast.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/anon/SokuBot")
from sokubot.data.state import CH

NEAR = 250.0 / 1200.0
BINS = np.linspace(0.0, NEAR, 6)


def effect(cache, name):
    d = np.load(cache)
    S, A, E = d["S"], d["A"], d["E"]
    nxt = np.zeros(len(S), bool)
    nxt[:-1] = E[1:] == E[:-1]

    rows = []
    for p in (0, 1):
        dx = S[:, p, CH["dx"]]
        left = A[:, :, p * 10 + 2].mean(1) > 0.5
        right = A[:, :, p * 10 + 3].mean(1) > 0.5
        one = left ^ right
        away = np.where(dx > 0, left, right)
        g = np.zeros(len(S), bool)
        g[:-1] = S[1:, p, CH["guarding"]] > 0.5
        ok = nxt & one & (np.abs(dx) < NEAR)
        rows.append((np.abs(dx)[ok], away[ok], g[ok]))

    dxa = np.concatenate([r[0] for r in rows])
    aw = np.concatenate([r[1] for r in rows])
    gu = np.concatenate([r[2] for r in rows])

    diffs, ns = [], []
    for i in range(len(BINS) - 1):
        m = (dxa >= BINS[i]) & (dxa < BINS[i + 1])
        a, t = m & aw, m & ~aw
        if a.sum() < 200 or t.sum() < 200:
            continue
        diffs.append(gu[a].mean() - gu[t].mean())
        ns.append(int(m.sum()))
    pooled_raw = gu[aw].mean() - gu[~aw].mean()
    print("%-6s n=%d  P(guard|away)=%.4f  P(guard|toward)=%.4f" %
          (name, len(gu), gu[aw].mean(), gu[~aw].mean()))
    print("        raw away-toward %+.5f   distance-stratified %+.5f"
          % (pooled_raw, float(np.mean(diffs)) if diffs else float("nan")))
    print("        per-bin: %s" % " ".join("%+.4f" % x for x in diffs))


for name, cache in (("human", sys.argv[1]), ("v3", sys.argv[2]), ("v1", sys.argv[3])):
    effect(cache, name)
