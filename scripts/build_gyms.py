"""Carve the corpus into named situations to drill specific mechanics.

    python -m scripts.build_gyms --sidecars ~/state-out0 ~/state-out1 \
        --out ~/gyms.npz

WHAT A GYM IS HERE
------------------
Training rolls forward from real corpus start states, so a gym is not a new
environment: it is a **filtered start-state distribution**. That makes it
cheap, and it is why drilling a mechanic at a four-step horizon works at all --
the same situation can be presented thousands of times a minute, and the
*decision* the mechanic needs is per-step even when the behaviour lasts
seconds. A block held for five seconds is seventy-five consecutive decisions to
keep holding, each one inside the horizon.

PAIRS, NOT INDICES
------------------
Every gym yields `(start, side)`, never a start alone. "Being combo'd" is not a
property of a state, it is a property of a state *and which chair you sit in* --
the same frame is `escape_pressure` for one player and `okizeme` for the other.
A gym that returned bare indices and let the trainer sample a side at random
would hand the agent the attacker's seat half the time and teach the opposite
mechanic in the same breath.

THE EIGHT MECHANICS
-------------------
The list is an expert's, not mine: master these and beating Keema is likely.
Where a mechanic decomposes into two decisions the gyms follow the decision,
not the label, because the reward has to land on a choice the agent is actually
making at that frame.

    1 blocking            -> `block_enter`, `block_gap`
    2 dodging / grazing   -> `projectile_dodge`
    3 spellcards          -> `spell_incoming`, `spirit_starved`   (PARTIAL)
    4 okizeme             -> `okizeme`
    5 getting advantage   -> `make_them_block`
    6 combos              -> `combo_extend`
    7 escaping disadvantage -> `escape_pressure`, `cornered`
    8 reading             -> `neutral`                            (NOT A GYM)

TWO OF THEM ARE NOT HONESTLY BUILDABLE YET, AND SAY SO
------------------------------------------------------
**Spellcards are partial.** The sidecar carries `spirit`, `max_spirit` and
`timestop`, so "a spell is being declared at me" and "I am resource-starved"
are exact. What it does NOT carry is which cards are in hand: `cardCount` at
0x5E6 and `hand` at 0x5E8 are documented in SokuLib and simply were not logged.
Knowing *what a spellcard does* is the half of the mechanic that needs them, so
`spell_incoming` drills reacting to a declaration and nothing here drills
choosing between cards. Adding those columns is a capture change, not an
analysis change.

**Reading is not a start-state distribution at all.** Predicting what an
opponent tends to do is a property of the opponent across many frames, so no
filter over single frames can select for it -- `neutral` is included because it
is the situation in which reading operates, not because sampling it teaches
reading. That mechanic wants an opponent model (an IDM over the observed
transition, fitted per opponent), and calling a gym after it would be naming a
file for a hope. See `sokubot/rl/counterfactual.py` for the machinery that
would host it.

THE LABELS ARE EXACT, AND ONE OF THEM IS NOT USED
-------------------------------------------------
Every selector below reads the game's own state, verified against the running
game by `pipeline/verify_extended.py` -- 14 checks over five replays, each one
a relationship the field could not satisfy by accident. None of the reward
probe's 0.116 residual is involved, and nothing here is inferred from pixels.

`untech` is the exception, and nothing here selects on it. SokuLib names the
field and the offset reads it, but its behaviour is not the countdown the name
suggests: measured over 337 000 frames it HOLDS its value on 90.6% of the ticks
where it is non-zero, decrements on 9.1%, reaches 26 507, and is non-zero on
64.2% of frames where the player is not knocked down, not blocking and not in
hitstop. Whatever it counts, "frames until I can act again" is not it, and a
64% base rate is also why the verification check that watched it fire after
every landed hit was nearly vacuous.

So the situations that mean "I cannot act" are built from `hitstop`,
`knockdown`, `crushed` and the two guard flags, all four of which were verified
by relationships that a base rate cannot fake -- hitstop against the opponent's
hitbox onsets, the guard flags against the stick holding away on 89-98% of
frames. First versions of `escape_pressure` and the free-to-act precondition
used `untech` and selected 83% of the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokubot.data.state import (CH, PF, STAGE_SPAN, has_state_columns,
                                read_state)

# ---------------------------------------------------------------------------
# Thresholds, in the normalised units of data/state.py. Each is a claim about
# the game, so each says what it is rather than being a bare number.
# ---------------------------------------------------------------------------
# "Close enough that the opponent can reach me." Blocking is only a decision at
# a range where an attack can land; at full screen the same frame is a neutral
# one and drilling it as a blocking rep teaches nothing.
NEAR = 250.0 / STAGE_SPAN
# How near a bullet has to be before dodging is the live question. Wider than
# NEAR because a projectile crosses the gap on its own.
THREAT = 450.0 / STAGE_SPAN
# The stage runs about x in [40, 1240]. Within this much of either edge there
# is no room left to retreat, which is what makes cornered a distinct mechanic
# rather than a worse version of blocking.
CORNER = 220.0 / STAGE_SPAN
# Spirit low enough that the next blocked string threatens a crush.
LOW_SPIRIT = 0.35


def _episode_ok(ep: np.ndarray, horizon: int, history: int) -> np.ndarray:
    """Frames whose whole window stays inside one replay."""
    n = len(ep)
    ok = np.ones(n, dtype=bool)
    ok[: history - 1] = False
    ok[n - horizon - 1:] = False
    for k in range(1, horizon + 1):
        ok[: n - k] &= (ep[k:] == ep[: n - k])
    for k in range(1, history):
        ok[k:] &= (ep[: n - k] == ep[k:])
    return ok


def build(S: np.ndarray, P: np.ndarray, V: np.ndarray, ep: np.ndarray,
          horizon: int, history: int) -> dict:
    """-> {name: (starts [N], sides [N])}, side 0 meaning P1 is the agent."""
    idx = np.flatnonzero(_episode_ok(ep, horizon, history))

    # Windows containing an invented label are dropped outright. `label_valid`
    # is 0 on frames the re-capture did not cover, filled by repeating a
    # neighbour; selecting a gym on an invented frame drills a situation that
    # never happened.
    keep = np.ones(len(idx), dtype=bool)
    for k in range(0, horizon + 1):
        keep &= V[idx + k]

    gyms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, m1, m2):
        m1, m2 = m1 & keep, m2 & keep
        st = np.concatenate([idx[m1], idx[m2]])
        sd = np.concatenate([np.zeros(int(m1.sum()), np.int64),
                             np.ones(int(m2.sum()), np.int64)])
        order = np.argsort(st, kind="mergesort")
        gyms[name] = (st[order], sd[order])

    def now(col: str, side: int) -> np.ndarray:
        return S[idx, side, CH[col]] > 0.5

    def soon(col: str, side: int) -> np.ndarray:
        """Does `col` turn on within the horizon, having been off now?

        The frame BEFORE the state begins is the rep, because that is where the
        input is still a decision. Once in blockstun the stick is committed.
        """
        later = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            later |= S[idx + k, side, CH[col]] > 0.5
        return (~now(col, side)) & later

    def within(col: str, side: int) -> np.ndarray:
        """Is `col` on at any point across the window, including now?"""
        out = now(col, side)
        for k in range(1, horizon + 1):
            out |= S[idx + k, side, CH[col]] > 0.5
        return out

    def val(col: str, side: int, k: int = 0) -> np.ndarray:
        return S[idx + k, side, CH[col]]

    near = np.abs(val("dx", 0)) < NEAR

    # "I cannot act": in hitstop, on the floor, guard broken, or committed to
    # blockstun. Built from the four verified flags and NOT from `untech` --
    # see the module docstring for why that field selects two thirds of the
    # corpus.
    locked = [(val("hitstop", s) > 0) | now("knockdown", s) | now("crushed", s)
              | now("guarding", s) | now("wrongblock", s) for s in (0, 1)]
    # `free` is the precondition for every gym whose mechanic is an ACTION.
    # Drilling "dodge this" on a frame where the agent is in hitstop rewards it
    # for something it could not have chosen.
    free = [~locked[s] for s in (0, 1)]

    # --- 1. BLOCKING -------------------------------------------------------
    # Entering the block: the frame before blockstun, in range, able to act.
    add("block_enter",
        (soon("guarding", 0) | soon("wrongblock", 0)) & near & free[0],
        (soon("guarding", 1) | soon("wrongblock", 1)) & near & free[1])

    # The gap. "When there is a gap in the attack, you must quickly punish."
    # I am in blockstun now, and within the horizon the attacker's hitboxes are
    # gone AND they have nothing live in the air -- the moment holding becomes
    # the wrong answer. This is the half of blocking that is about stopping.
    def gap(me: int) -> np.ndarray:
        them = 1 - me
        blocking_now = now("guarding", me) | now("wrongblock", me)
        opening = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            opening |= ((S[idx + k, them, CH["hitboxes"]] == 0)
                        & (S[idx + k, them, CH["proj_hb"]] == 0)
                        & (S[idx + k, me, CH["guarding"]] < 0.5)
                        & (S[idx + k, me, CH["wrongblock"]] < 0.5))
        return blocking_now & opening & near
    add("block_gap", gap(0), gap(1))

    # --- 2. DODGING / GRAZING ---------------------------------------------
    # A live bullet of theirs, closing on me, inside threat range, while I can
    # still act. `proj[:, p]` are the objects player p OWNS, already expressed
    # relative to the player they fly at and sorted danger-first, so the ones
    # threatening `me` are `proj[:, 1 - me]` and slot 0 is the most urgent.
    def incoming(me: int) -> np.ndarray:
        them = 1 - me
        q = P[idx][:, them]                       # [n, slots, features]
        live = (q[..., PF["present"]] > 0.5) & (q[..., PF["hb"]] > 0.5)
        close = np.hypot(q[..., PF["dx"]], q[..., PF["dy"]]) < THREAT
        closing = q[..., PF["closing"]] > 0
        return (live & close & closing).any(axis=1)
    add("projectile_dodge",
        incoming(0) & free[0] & ~now("guarding", 0),
        incoming(1) & free[1] & ~now("guarding", 1))

    # --- 3. SPELLCARDS (partial -- see the module docstring) --------------
    # A declaration freezes the game, so timestop rising on THEM is the cue.
    def declares(them: int) -> np.ndarray:
        base = S[idx, them, CH["timestop"]]
        out = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            out |= S[idx + k, them, CH["timestop"]] > base
        return out
    add("spell_incoming", declares(1) & near, declares(0) & near)

    # Resource-starved with an opponent in range: the road to a guard crush.
    add("spirit_starved",
        (val("spirit", 0) < LOW_SPIRIT) & near & free[0],
        (val("spirit", 1) < LOW_SPIRIT) & near & free[1])

    # --- 4. OKIZEME -------------------------------------------------------
    # They are on the floor and getting up inside the window, and I am close
    # enough to meet them. Requiring the knockdown flag to CLEAR is what makes
    # this the wake-up timing drill rather than "stand near a body" -- and the
    # flag is the verified one; the first version used `untech` reaching zero,
    # which is a counter that mostly does not count.
    def waking(them: int) -> np.ndarray:
        down = S[idx, them, CH["knockdown"]] > 0.5
        up = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            up |= S[idx + k, them, CH["knockdown"]] < 0.5
        return down & up
    add("okizeme", waking(1) & near & free[0], waking(0) & near & free[1])

    # --- 5. GETTING ADVANTAGE ---------------------------------------------
    # Making them block: my hitbox is live and their guard comes on. The reward
    # is frame advantage, which is why this is separate from landing a hit.
    def pressuring(me: int) -> np.ndarray:
        them = 1 - me
        out = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            out |= ((S[idx + k, me, CH["hitboxes"]] > 0)
                    & ((S[idx + k, them, CH["guarding"]] > 0.5)
                       | (S[idx + k, them, CH["wrongblock"]] > 0.5)))
        return out
    add("make_them_block", pressuring(0) & near, pressuring(1) & near)

    # --- 6. COMBOS --------------------------------------------------------
    # A combo of mine is LIVE and has not hit its limit -- the frames where the
    # next button decides whether the string continues.
    #
    # "Live" is the opponent being in hitstop, not `combo_hits > 0`. The combo
    # counters persist after a string ends and hold the last one's values, so
    # `combo_hits > 0` is true on 64% of the corpus and selects most of the
    # match. Opponent hitstop takes that to 6.0%, and hitstop is the field
    # verified against hitbox onsets.
    def extending(me: int) -> np.ndarray:
        them = 1 - me
        return ((val("combo_hits", me) > 0)
                & (val("combo_limit", me) > val("combo_hits", me))
                & (S[idx, them, CH["hitstop"]] > 0))
    add("combo_extend", extending(0), extending(1))

    # --- 7. ESCAPING DISADVANTAGE -----------------------------------------
    # Locked down with them in range. The mechanic is not "keep holding" -- it
    # is recovering position and finding the punish, which is why this is a
    # different gym from `block_enter` and not a synonym.
    add("escape_pressure", locked[0] & near, locked[1] & near)

    # Cornered: my BACK is to the wall, with them close.
    #
    # Being near an edge is not enough and selects 46% of frames -- the stage
    # clamps at x 40 and 1240 and players sit against those clamps constantly.
    # What makes it the mechanic is having no room to retreat, which means the
    # opponent is on the side away from my wall. Selected on absolute x, which
    # is exactly why raw position is a channel and not just `dx`.
    def cornered(me: int) -> np.ndarray:
        x, dx = val("x", me), val("dx", me)
        lo, hi = 40.0 / STAGE_SPAN, 1240.0 / STAGE_SPAN
        back_to_left = (x - lo < CORNER) & (dx > 0)    # wall left, them right
        back_to_right = (hi - x < CORNER) & (dx < 0)   # wall right, them left
        return (back_to_left | back_to_right) & near
    add("cornered", cornered(0), cornered(1))

    # --- 8. READING (not a gym -- see the module docstring) ---------------
    # Both free, in range, nothing in flight: the situation reading operates
    # in. Included so the distribution exists; it does not teach reading.
    calm = [(val("combo_hits", s) == 0) & free[s] for s in (0, 1)]
    add("neutral", calm[0] & calm[1] & near, calm[1] & calm[0] & near)
    return gyms


def load_bank(dirs: list[Path], limit: int = 0,
              name: str = "") -> tuple[np.ndarray, ...]:
    """Concatenate per-replay sidecars into one bank with an episode column."""
    paths: list[Path] = []
    for root in dirs:
        root = root.expanduser()
        if root.is_file():
            paths.append(root)
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for cand in ([child / name] if name else
                         [child / "state.csv.gz", child / "inputs.csv.gz",
                          child / "inputs.csv"]):
                if cand.exists():
                    paths.append(cand)
                    break
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no sidecars found under {[str(d) for d in dirs]}")

    S, P, V, E = [], [], [], []
    skipped = 0
    for e, p in enumerate(paths):
        try:
            if not has_state_columns(p):
                skipped += 1
                continue
            s, pr, _act, v = read_state(p)
        except (ValueError, OSError) as exc:
            # One bad replay must not end the build; a corpus of 2003 will
            # always contain a capture that was killed mid-write.
            print(f"  skip {p.parent.name}: {exc}")
            skipped += 1
            continue
        S.append(s)
        P.append(pr)
        V.append(v)
        E.append(np.full(len(s), e, dtype=np.int32))
    if not S:
        raise SystemExit("every sidecar was skipped; none carry full state")
    if skipped:
        print(f"  ({skipped} of {len(paths)} sidecars skipped)")
    return (np.concatenate(S), np.concatenate(P), np.concatenate(V),
            np.concatenate(E))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sidecars", type=Path, nargs="+", required=True,
                    help="capture roots (dirs of per-replay dirs), or files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="",
                    help="sidecar filename inside each replay dir "
                         "(default: state.csv.gz, else inputs.csv[.gz])")
    ap.add_argument("--limit", type=int, default=0, help="first N replays only")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--history", type=int, default=3)
    a = ap.parse_args()

    print(f"loading sidecars from {[str(d) for d in a.sidecars]} ...")
    S, P, V, E = load_bank(a.sidecars, a.limit, a.name)
    n_reps = int(E.max()) + 1
    print(f"bank {len(S)} frames, {n_reps} replays, {P.shape[2]} projectile "
          f"slots\n")

    gyms = build(S, P, V, E, a.horizon, a.history)
    total = sum(len(v[0]) for v in gyms.values())

    print(f"horizon {a.horizon}, history {a.history}\n")
    print("  gym                 pairs    per-1k   replays   mean hp(me)  "
          "mean dx")
    meta = {}
    for name, (st, sd) in gyms.items():
        if len(st) == 0:
            print(f"  {name:<18} {0:7d}   SELECTS NOTHING")
            meta[name] = {"pairs": 0}
            continue
        mine_hp = S[st, sd, CH["hp"]]
        dx = np.abs(S[st, sd, CH["dx"]])
        reps = len(np.unique(E[st]))
        meta[name] = {"pairs": int(len(st)), "replays": int(reps),
                      "per_1k_frames": round(1000 * len(st) / len(S), 2),
                      "mean_hp_mine": float(mine_hp.mean()),
                      "mean_abs_dx": float(dx.mean())}
        print(f"  {name:<18} {len(st):7d}  {1000*len(st)/len(S):7.1f}  "
              f"{reps:7d}   {mine_hp.mean():10.3f}  {dx.mean():8.3f}")
    print(f"  {'(total)':<18} {total:7d}")

    np.savez(a.out.expanduser(),
             **{f"{k}_starts": v[0] for k, v in gyms.items()},
             **{f"{k}_sides": v[1] for k, v in gyms.items()},
             names=np.array(list(gyms)), horizon=a.horizon,
             history=a.history, frames=len(S), replays=n_reps)
    Path(str(a.out.expanduser()) + ".json").write_text(json.dumps(
        {"sidecars": [str(d) for d in a.sidecars], "horizon": a.horizon,
         "frames": int(len(S)), "replays": n_reps, "gyms": meta}, indent=1))
    print(f"\n-> {a.out}")

    empty = [k for k, v in meta.items() if v["pairs"] == 0]
    if empty:
        print(f"\nSELECTS NOTHING: {', '.join(empty)}. A gym with no pairs is a "
              f"selector\nbug, not a rare situation -- check its thresholds "
              f"against the channel units.")
    print("\nNOT drilled here, and not because it was forgotten:\n"
          "  * choosing between spellcards -- needs cardCount (0x5E6) and hand\n"
          "    (0x5E8) logged at capture. `spell_incoming` drills reacting to a\n"
          "    declaration, which is the half the sidecar can see.\n"
          "  * reading -- not expressible as a filter over single frames. It\n"
          "    wants an opponent model, and `neutral` is the situation it\n"
          "    operates in rather than a drill for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
