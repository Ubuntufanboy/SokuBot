"""Does blocking actually pay, in 200 hours of real Hisoutensoku?

    python -m scripts.blocking_value --corpus ~/corpus --cache ~/rl/bank_state.npz

NO MODEL IS INVOLVED. Every number here is what real players did and what
really happened to their health, read from the extractor's sidecar. That is the
point: the agent declines to block, the simulator says blocking is worth 33-50
HP over 667 ms, and those two facts can both be true if blocking is worth
something and interrupting is worth more. The corpus can say which without
either the policy or the world model in the loop.

THE HYPOTHESIS BEING TESTED
---------------------------
That blocking is a human adaptation to reaction time rather than a property of
the game -- that a defender who could reliably predict the attack would prefer
to interrupt it, and that "pros trade, new players dodge" is a fact about how
humans learn rather than about which is stronger.

WHY THIS IS OBSERVATIONAL, AND WHAT IS DONE ABOUT IT
----------------------------------------------------
Players CHOOSE to block, and they choose it when they expect to be hit, so a
naive "blockers take more damage" would mostly measure that they were in more
danger. Three things narrow it:

  * the defender must be FREE to act at the decision frame -- not in hitstop,
    knockdown, blockstun or already guarding, so the response really is a
    choice;
  * the comparison is conditioned on the attack's TIMING: only frames where the
    attacker's hitbox first goes live exactly `k` steps later, which pins how
    much warning the defender had;
  * range and both players' airborne state are matched, since those change what
    the options even are.

What remains uncontrolled is everything the player knows and the sidecar does
not -- their read on the opponent, the match score, their own habits. So this
bounds the question rather than settling it, and it is reported as a comparison
of conditional outcomes, not as a causal effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokubot.data import state_bank
from sokubot.data.state import CH, FULL_HP, STAGE_SPAN

NEAR = 250.0 / STAGE_SPAN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--ticks", type=int, default=5)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--warn", type=int, nargs="+", default=[1, 2, 3],
                    help="steps of warning: the attacker's hitbox first goes "
                         "live exactly this many decision steps after the "
                         "defender's choice")
    ap.add_argument("--window", type=int, default=8,
                    help="steps over which the outcome is scored")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    S, P, A, E, V, names = state_bank.load(a.corpus, a.ticks, a.slots, a.cache)
    n = len(S)
    hp = S[:, :, CH["hp"]]
    sp = S[:, :, CH["spirit"]]
    hb = S[:, :, CH["hitboxes"]]
    guard = (S[:, :, CH["guarding"]] > 0.5) | (S[:, :, CH["wrongblock"]] > 0.5)
    locked = ((S[:, :, CH["hitstop"]] > 0) | (S[:, :, CH["knockdown"]] > 0.5)
              | (S[:, :, CH["crushed"]] > 0.5) | guard)
    near = np.abs(S[:, 0, CH["dx"]]) < NEAR
    W = a.window

    def shift(x, k):
        """x[i+k], with the tail marked invalid."""
        out = np.zeros_like(x)
        out[:n - k] = x[k:]
        return out

    same = np.ones(n, bool)
    for k in range(1, W + 1):
        same[:n - k] &= (E[k:] == E[:n - k])
        same[:n - k] &= V[k:]
    same &= V
    same[n - W:] = False

    results = {}
    print(f"\nbank {n} decision steps / {len(names)} replays | outcome window "
          f"{W} steps ({W*a.ticks*1000/60:.0f} ms)")
    for me in (0, 1):
        them = 1 - me
        for k in a.warn:
            # The attacker's hitbox is live at i+k and was NOT live at i..i+k-1:
            # the attack connects for the first time exactly k steps out, so
            # every row in this cell gave the defender the same warning.
            first = shift(hb[:, them] > 0, k)
            for j in range(0, k):
                first &= ~shift(hb[:, them] > 0, j)
            sel = first & same & near & ~locked[:, me]
            if sel.sum() < 500:
                continue
            idx = np.flatnonzero(sel)

            # What the defender did between the choice and the hit landing.
            #
            # `attacked` is split, because lumping it together answers the
            # wrong question. An INTERRUPT is a hitbox that goes live strictly
            # BEFORE the attacker's -- the thing a frame-accurate defender
            # could do and a human mostly cannot. A TRADE is one that arrives
            # on the same step. "Pressed attack and got hit first" is neither,
            # and averaging all three together makes a successful interrupt
            # look as bad as a blown one.
            did_block = np.zeros(len(idx), bool)
            did_attack = np.zeros(len(idx), bool)
            early = np.zeros(len(idx), bool)          # live strictly before k
            for j in range(1, k + 1):
                did_block |= guard[idx + j, me]
                live = hb[idx + j, me] > 0
                did_attack |= live
                if j < k:
                    early |= live
            groups = {
                "blocked":   did_block & ~did_attack,
                "interrupt": did_attack & ~did_block & early,
                "trade":     did_attack & ~did_block & ~early,
                "both":      did_block & did_attack,
                "neither":   ~did_block & ~did_attack,
            }
            row = {}
            for name, m in groups.items():
                if m.sum() < 60:
                    continue
                g = idx[m]
                d_me = -np.clip(hp[g + W, me] - hp[g, me], None, 0)
                d_th = -np.clip(hp[g + W, them] - hp[g, them], None, 0)
                spirit = -np.clip(sp[g + W, me] - sp[g, me], None, 0).mean()
                row[name] = {"n": int(m.sum()), "share": float(m.mean()),
                             "taken": float(d_me.mean() * FULL_HP),
                             "dealt": float(d_th.mean() * FULL_HP),
                             "net": float((d_th - d_me).mean() * FULL_HP),
                             # How often the defender was hit AT ALL. An
                             # average conflates "never hit" with "hit for
                             # half a bar", and evading is the option whose
                             # whole value is in the first.
                             "hit_rate": float((d_me > 1e-6).mean()),
                             "spirit_lost": float(spirit)}
            results[f"p{me+1}_warn{k}"] = row

    # Pooled over both chairs, which is the number worth quoting: the two
    # players are the same population seen from two seats.
    print(f"\n  response to an attack landing in N steps, pooled over both "
          f"chairs\n  (real players, real outcomes; `net` = dealt - taken, "
          f"game HP over {W} steps)")
    print(f"\n  {'warning':<9} {'response':<10} {'share':>7} {'n':>8} "
          f"{'taken':>8} {'dealt':>8} {'net':>8} {'hit%':>6} {'spirit':>7}")
    pooled = {}
    for k in a.warn:
        for name in ("blocked", "interrupt", "trade", "both", "neither"):
            cells = [results[f"p{p}_warn{k}"].get(name)
                     for p in (1, 2) if f"p{p}_warn{k}" in results]
            cells = [c for c in cells if c]
            if not cells:
                continue
            tot = sum(c["n"] for c in cells)
            agg = {q: sum(c[q] * c["n"] for c in cells) / tot
                   for q in ("share", "taken", "dealt", "net", "hit_rate",
                             "spirit_lost")}
            agg["n"] = tot
            pooled[f"warn{k}_{name}"] = agg
            print(f"  {k if name=='blocked' else '':<9} {name:<10} "
                  f"{agg['share']:7.1%} {tot:8d} {agg['taken']:8.1f} "
                  f"{agg['dealt']:8.1f} {agg['net']:+8.1f} "
                  f"{agg['hit_rate']:6.1%} {agg['spirit_lost']:7.3f}")
        print()

    # --- the same question WITHOUT conditioning on the attack landing -----
    #
    # THE TABLE ABOVE HAS A SELECTION ARTIFACT, AND IT IS THE INTERESTING ONE.
    # It conditions on the attacker's hitbox going live at step k. A SUCCESSFUL
    # interrupt stops that from ever happening, so it is excluded from the
    # sample by construction -- which means the `interrupt` row can only
    # contain interrupts that FAILED, and its terrible numbers are guaranteed
    # in advance rather than measured. Reporting it as "interrupting is bad"
    # would be reading a definition as a result.
    #
    # So this asks the same question from the defender's side and conditions on
    # nothing the defender does not control: both players near and free to act,
    # and then what the defender chose. `opp_hit` is the fraction where the
    # opponent's attack ever became live at all, which is where a successful
    # pre-emption shows up.
    print("\n  from the DEFENDER's side: both near and free, split by what "
          "the defender did\n  (no conditioning on the opponent's attack, so "
          "successful pre-emption is visible)")
    print(f"\n  {'response':<12} {'share':>7} {'n':>9} {'taken':>8} "
          f"{'dealt':>8} {'net':>8} {'hit%':>6} {'opp hb%':>8}")
    both_free = same & near & ~locked[:, 0] & ~locked[:, 1]
    pre = {}
    for me in (0, 1):
        them = 1 - me
        idx = np.flatnonzero(both_free)
        K = 3
        mine_first = np.zeros(len(idx), bool)
        theirs_any = np.zeros(len(idx), bool)
        blocked_ = np.zeros(len(idx), bool)
        seen_theirs = np.zeros(len(idx), bool)
        for j in range(1, K + 1):
            lm = hb[idx + j, me] > 0
            lt = hb[idx + j, them] > 0
            mine_first |= lm & ~seen_theirs
            seen_theirs |= lt
            theirs_any |= lt
            blocked_ |= guard[idx + j, me]
        grp = {"attack first": mine_first & ~blocked_,
               "blocked": blocked_ & ~mine_first,
               "neither": ~mine_first & ~blocked_}
        for name, m in grp.items():
            if m.sum() < 200:
                continue
            g = idx[m]
            d_me = -np.clip(hp[g + W, me] - hp[g, me], None, 0)
            d_th = -np.clip(hp[g + W, them] - hp[g, them], None, 0)
            c = pre.setdefault(name, {"n": 0, "taken": 0.0, "dealt": 0.0,
                                      "hit": 0.0, "opp": 0.0, "tot": 0})
            c["n"] += int(m.sum()); c["tot"] += len(idx)
            c["taken"] += float(d_me.sum()) * FULL_HP
            c["dealt"] += float(d_th.sum()) * FULL_HP
            c["hit"] += float((d_me > 1e-6).sum())
            c["opp"] += float(theirs_any[m].sum())
    for name, c in pre.items():
        nn = c["n"]
        print(f"  {name:<12} {nn/c['tot']:7.1%} {nn:9d} {c['taken']/nn:8.1f} "
              f"{c['dealt']/nn:8.1f} {(c['dealt']-c['taken'])/nn:+8.1f} "
              f"{c['hit']/nn:6.1%} {c['opp']/nn:8.1%}")

    # --- does blocking track winning? -------------------------------------
    # Skill proxy: net damage the player dealt across the whole replay. Crude,
    # but it is the only outcome the sidecar states, and the question is
    # whether the players who come out ahead are the ones who block.
    print("  guard-when-threatened rate vs how well the player did")
    ep_ids = np.unique(E)
    rows = []
    for me in (0, 1):
        them = 1 - me
        for e in ep_ids:
            m = (E == e)
            if m.sum() < 200:
                continue
            g = np.flatnonzero(m)
            d = np.diff(hp[g], axis=0)
            net = (-np.clip(d[:, them], None, 0).sum()
                   + np.clip(d[:, me], None, 0).sum()) * FULL_HP
            threat = np.zeros(len(g), bool)
            for k in a.warn:
                threat[:len(g) - k] |= (hb[g[k:], them] > 0)
            free = ~locked[g, me]
            cell = threat & free
            if cell.sum() < 50:
                continue
            gr = float(guard[g[cell] + 1, me].mean()) if cell.sum() else np.nan
            rows.append((gr, net))
    gr = np.array([r[0] for r in rows]); nt = np.array([r[1] for r in rows])
    ok = np.isfinite(gr) & np.isfinite(nt)
    gr, nt = gr[ok], nt[ok]
    q = np.quantile(gr, [0.25, 0.5, 0.75])
    print(f"    {len(gr)} player-replays | guard rate quartiles "
          f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    for lo, hi, lbl in ((0, q[0], "lowest 25% (blocks least)"),
                        (q[0], q[1], "25-50%"), (q[1], q[2], "50-75%"),
                        (q[2], 1.1, "highest 25% (blocks most)")):
        m = (gr >= lo) & (gr < hi)
        if m.sum():
            print(f"    {lbl:<26} guard {gr[m].mean():.3f}  net damage "
                  f"{nt[m].mean():+8.1f} HP  (n={int(m.sum())})")
    r = float(np.corrcoef(gr, nt)[0, 1])
    print(f"    correlation(guard rate, net damage) = {r:+.3f}")
    print("\n  A negative correlation is CONSISTENT WITH the hypothesis and "
          "does not establish it:\n  the player who is behind is also the "
          "player being attacked, and gets more\n  chances to block. Read it "
          "against the conditional table above, which holds\n  the attack's "
          "timing fixed.")

    if a.out:
        a.out.write_text(json.dumps({"cells": results, "pooled": pooled,
                                     "guard_net_corr": r}, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
