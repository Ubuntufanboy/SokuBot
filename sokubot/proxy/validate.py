"""Does the proxy actually suffer from Soku's problems? Run before trusting it.

Each check is stated with the real number it has to reproduce. A toy that is
easy where Soku is hard proves nothing, so this is allowed to FAIL -- and if it
does, the proxy gets fixed or abandoned rather than reported.
"""
from __future__ import annotations

import numpy as np

from .engine import (ATTACK, CHANNELS, IDLE, LEFT, REACH, RIGHT, initial,
                     observe, step)
from .policy import corpus, counterfactual_pairs, rollout

GI = CHANNELS.index("guarding")
AI = CHANNELS.index("actionable")
DX = CHANNELS.index("dx")


def _away_action(dx):
    return LEFT if dx > 0 else RIGHT


def report():
    O, A, E = corpus(200)
    n = len(O)
    out = {}

    # 1 & 2: rarity and non-actionability -----------------------------------
    out["guard_rate"] = float(O[:, 0, GI].mean())
    out["actionable"] = float(O[:, 0, AI].mean())

    # 3: OBSERVATIONAL association between holding away and guarding next step
    # Guard WITHIN K frames, the same window the causal probe forces, because
    # a 1-frame association against a 3-frame intervention compares two
    # different quantities -- which is the error that made the first run report
    # the association as SMALLER than the effect it is supposed to overstate.
    K = 3
    nxt_guard = np.zeros(n, bool)
    for k in range(1, K + 1):
        nxt_guard[:-k] |= O[k:, 0, GI] > 0.5
    same = np.ones(n, bool)
    for k in range(1, K + 1):
        same[:n - k] &= (E[k:] == E[:n - k])
    dx = O[:, 0, DX]
    held_away = np.array([A[i, 0] == _away_action(dx[i]) for i in range(n)])
    held_tow = np.array([A[i, 0] == (RIGHT if dx[i] > 0 else LEFT)
                         for i in range(n)])
    can = (O[:, 0, AI] > 0.5) & same & (np.abs(dx) * 200 < REACH * 2)
    ra = nxt_guard[can & held_away].mean() if (can & held_away).any() else np.nan
    rt = nxt_guard[can & held_tow].mean() if (can & held_tow).any() else np.nan
    out["assoc_effect"] = float(ra - rt)
    out["assoc_n"] = int((can & held_away).sum())

    # 4: THE CAUSAL EFFECT, which the real game could not supply -------------
    # Same state, away vs toward, opponent held fixed. No conditioning needed.
    rng = np.random.default_rng(0)
    hits_a = hits_b = trials = 0
    m = 0
    while trials < 4000:
        o, a, S = rollout(50_000 + m); m += 1
        for _ in range(30):
            if len(S) < 10:
                break
            t = int(rng.integers(1, len(S) - 4))
            s0 = S[t]
            if not s0.p[0].actionable:
                continue
            d = s0.p[1].x - s0.p[0].x
            if abs(d) > REACH * 2:
                continue
            opp = [int(a[min(t + k, len(a) - 1)][1]) for k in range(3)]
            res = []
            for mine in (_away_action(d), RIGHT if d > 0 else LEFT):
                s, got = s0, 0
                for k in range(3):
                    s = step(s, mine, opp[k])
                    got |= s.p[0].guarding
                res.append(got)
            hits_a += res[0]; hits_b += res[1]; trials += 1
    out["causal_effect"] = float((hits_a - hits_b) / max(trials, 1))
    out["causal_n"] = trials
    return out


CHECKS = [
    ("guard is rare (Soku 4.55%)", "guard_rate", 0.02, 0.09),
    ("most frames non-actionable (Soku 34.8% actionable)", "actionable", 0.15, 0.50),
    ("observational association is substantial (Soku +0.069)", "assoc_effect", 0.02, 1.0),
    ("causal effect is SMALLER than association (the confound)", None, None, None),
]


def main():
    r = report()
    print(f"{'guard base rate':<44} {r['guard_rate']*100:6.2f}%")
    print(f"{'actionable frames':<44} {r['actionable']*100:6.2f}%")
    print(f"{'observational assoc (away - toward)':<44} {r['assoc_effect']:+7.4f}"
          f"  (n={r['assoc_n']})")
    print(f"{'TRUE causal effect (same state, forced)':<44} "
          f"{r['causal_effect']:+7.4f}  (n={r['causal_n']})")
    ratio = r["assoc_effect"] / r["causal_effect"] if r["causal_effect"] else float("nan")
    print(f"{'association / causal':<44} {ratio:7.2f}x")
    print()
    ok = True
    for name, key, lo, hi in CHECKS[:3]:
        v = r[key]
        good = lo <= v <= hi
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {v:.4f}")
    conf = r["assoc_effect"] > r["causal_effect"] * 1.2
    ok &= conf
    print(f"  [{'PASS' if conf else 'FAIL'}] confounding present "
          f"(assoc > 1.2x causal)")
    print("\nPROXY VALID" if ok else "\nPROXY REJECTED -- do not draw conclusions from it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
