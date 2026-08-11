"""Carve the bank into named situations to drill specific mechanics.

    python -m scripts.build_gyms --bank ~/bank_hud.npz --out ~/gyms.npz

WHAT A GYM IS HERE
------------------
Training already rolls forward from real corpus start states, so a gym is not a
new environment: it is a **filtered start-state distribution**. That makes it
cheap, and it is why drilling a mechanic at H=4 works at all -- the same
situation can be presented thousands of times per minute, and the *decision* the
mechanic needs is per-step even when the behaviour lasts seconds. A block held
for five seconds is seventy-five consecutive decisions to keep holding, each one
inside the horizon.

PAIRS, NOT INDICES
------------------
Every gym yields `(start, side)`, never a start alone. "Being combo'd" is not a
property of a state, it is a property of a state *and which chair you sit in* --
the same frame is `under_pressure` for one player and `pressuring` for the other.
A gym that returned bare indices and let the trainer sample a side at random
would hand the agent the attacker's seat half the time and teach the opposite
mechanic in the same breath.

THE LABELS ARE EXACT, NOT PROBED
---------------------------------
Selection reads `bank_hud.npz`'s `hud` column, which `data/hud.py` produced from
native 480 px pixels and which was validated against a human at MAE 0.012 for
health and 0.025 for spirit (`scripts/score_hud_annotation.py`). This is offline
labelling of *real* frames, so none of the reward probe's 0.116 residual is
involved. Where the HUD cannot see something -- knockdown, projectiles, position
-- there is no gym here, and inventing one from the probe would be worse than
not having it.

WHAT THE GAME STATE ADDED
-------------------------
The four gyms above are all the HUD can see, and this file used to say so:
"`knockdown` and `projectile_pressure`, the two situations most worth drilling
after blocking, are not in the HUD -- they need a pixel detector or game state
logged at capture time." The state is logged now
(`dll/include/sfe/player_state.hpp`, verified against the game over 66 380
frames), so the mechanic gyms below select on what actually happened rather
than on a proxy for it: blockstun beginning, a guard breaking, a body on the
floor.

Those gyms are keyed to the three mechanics the agent has to master, in the
order they matter. Blocking first, because an agent that does not block loses
in seconds rather than minutes.

DODGING IS STILL NOT HERE, AND `hit_clean` IS NOT IT
-----------------------------------------------------
Nothing in the sidecar sees projectiles, so a gym claiming to drill dodging
would be selecting on a guess. What the labels do support is the *outcome*
dodging avoids: taking damage while not in blockstun -- a clean hit. Drilling
that teaches avoidance in general (block, or move) rather than evasion
specifically, and it is named for what it selects instead of what one might
hope it teaches. Real projectile pressure needs a detector, and that is honest
work still to do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HP1, HP2, SPIRIT1, SPIRIT2, COMBO1, COMBO2 = range(6)

# Columns of the state array, from sokubot/data/state.py::STATE_CHANNELS.
DX, FACING, GUARDING, WRONGBLOCK, CRUSHED, KNOCKDOWN, AIRBORNE = range(7)
# "Close" in game units on a stage about 1200 wide. Blocking is only a decision
# at a range where the opponent can reach; at full screen the same frame is a
# neutral one and drilling it as a blocking rep teaches nothing.
NEAR = 250.0


def build(Hd: np.ndarray, E: np.ndarray, horizon: int, history: int,
          combo_min: float, low_hp: float,
          S: np.ndarray | None = None, SV: np.ndarray | None = None,
          span: float = 1200.0) -> dict:
    """-> {name: (starts [N], sides [N])}, sides being 0 for P1."""
    n = len(Hd)
    # A window must stay inside one replay, in both directions: `history-1`
    # frames of context behind and `horizon` of future ahead.
    ok = np.ones(n, dtype=bool)
    ok[: history - 1] = False
    ok[n - horizon - 1 :] = False
    for k in range(1, horizon + 1):
        ok[: n - k] &= (E[k:] == E[: n - k])
    for k in range(1, history):
        ok[k:] &= (E[: n - k] == E[k:])
    idx = np.flatnonzero(ok)

    # Red on a player's own bar is damage being done TO them, so combo1 high
    # means P1 is the one being hit. See rl/reward.py::_sides.
    c1, c2 = Hd[idx, COMBO1], Hd[idx, COMBO2]
    hp1, hp2 = Hd[idx, HP1], Hd[idx, HP2]

    gyms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, mask_p1, mask_p2):
        st = np.concatenate([idx[mask_p1], idx[mask_p2]])
        sd = np.concatenate([np.zeros(mask_p1.sum(), np.int64),
                             np.ones(mask_p2.sum(), np.int64)])
        order = np.argsort(st, kind="mergesort")
        gyms[name] = (st[order], sd[order])

    # Being combo'd: red on my own bar. The mechanic is blocking.
    add("under_pressure", c1 >= combo_min, c2 >= combo_min)
    # Doing the comboing: red on theirs. The mechanic is extending it.
    add("pressuring", c2 >= combo_min, c1 >= combo_min)
    # Behind on health with no combo in flight -- where the agent was observed
    # giving up (HANDOFF §10).
    calm = (c1 < combo_min) & (c2 < combo_min)
    add("losing", calm & (hp1 <= low_hp) & (hp1 < hp2),
        calm & (hp2 <= low_hp) & (hp2 < hp1))
    add("neutral", calm & (hp1 > low_hp) & (hp2 > low_hp),
        calm & (hp2 > low_hp) & (hp1 > low_hp))

    if S is None:
        return gyms

    # ---- mechanic gyms, from the game's own state ------------------------
    # Windows containing a padded label are dropped outright. `label_valid` is
    # 0 on frames the re-capture did not cover, filled by repeating a neighbour;
    # selecting a gym on an invented frame would drill a situation that never
    # happened.
    if SV is not None:
        keep = np.ones(len(idx), dtype=bool)
        for k in range(0, horizon + 1):
            keep &= SV[idx + k]
    else:
        keep = np.ones(len(idx), dtype=bool)

    def soon(col: int, side: int) -> np.ndarray:
        """Does `col` turn on for `side` within the horizon, having been off?"""
        now = S[idx, side, col] > 0.5
        later = np.zeros(len(idx), dtype=bool)
        for k in range(1, horizon + 1):
            later |= S[idx + k, side, col] > 0.5
        return (~now) & later & keep

    near = np.abs(S[idx, 0, DX]) * span < NEAR

    # BLOCKING. The rep is the frame *before* blockstun starts, because that is
    # where holding away is still a decision -- once in blockstun the input is
    # already committed. Restricted to close range for the same reason.
    add("blocking",
        (soon(GUARDING, 0) | soon(WRONGBLOCK, 0)) & near,
        (soon(GUARDING, 1) | soon(WRONGBLOCK, 1)) & near)

    # GUARD BREAK. Crushed is 0.04% of frames and the most punishing thing that
    # happens to a defender, so it would never be sampled by accident.
    add("guard_break", soon(CRUSHED, 0), soon(CRUSHED, 1))

    # KNOCKDOWN, both chairs. On the floor is the situation the brief named
    # explicitly; the same frame from the other seat is okizeme, which is where
    # combos are extended.
    down1 = (S[idx, 0, KNOCKDOWN] > 0.5) & keep
    down2 = (S[idx, 1, KNOCKDOWN] > 0.5) & keep
    add("knocked_down", down1, down2)
    add("okizeme", down2 & near, down1 & near)

    # A clean hit: damage arriving without blockstun. Named for what it
    # selects; see the docstring on why this is not a dodging gym.
    hurt1 = np.zeros(len(idx), dtype=bool)
    hurt2 = np.zeros(len(idx), dtype=bool)
    for k in range(1, horizon + 1):
        hurt1 |= Hd[idx + k, COMBO1] >= combo_min
        hurt2 |= Hd[idx + k, COMBO2] >= combo_min
    blocked1 = np.zeros(len(idx), dtype=bool)
    blocked2 = np.zeros(len(idx), dtype=bool)
    for k in range(0, horizon + 1):
        blocked1 |= S[idx + k, 0, GUARDING] > 0.5
        blocked2 |= S[idx + k, 1, GUARDING] > 0.5
    add("hit_clean", hurt1 & ~blocked1 & keep, hurt2 & ~blocked2 & keep)
    return gyms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--combo-min", type=float, default=0.02,
                    help="red on the health bar counting as an active combo. "
                         "0.02 of a bar is above hud.py's RED_FLOOR of 0.018, so "
                         "it is a real hit rather than the reader's noise.")
    ap.add_argument("--low-hp", type=float, default=0.35)
    a = ap.parse_args()

    b = np.load(a.bank.expanduser())
    if "hud" not in b.files:
        raise SystemExit(f"{a.bank} has no `hud`; build it with build_hud_bank")
    Hd, E = b["hud"].astype(np.float32), b["ep"]
    # The state array is optional: a bank built before the corpus was
    # re-captured has none, and the four HUD gyms are still worth having on
    # their own. Saying which happened matters, though -- silently producing
    # four gyms where nine were expected is the kind of thing that reads as
    # "the mechanic gyms did not select anything".
    S = b["state"].astype(np.float32) if "state" in b.files else None
    SV = b["state_valid"] if "state_valid" in b.files else None
    if S is None:
        print("note: bank has no `state`; building the four HUD gyms only. "
              "Rebuild with build_hud_bank once the corpus has state.csv.gz "
              "for the mechanic gyms.\n")
    gyms = build(Hd, E, a.horizon, a.history, a.combo_min, a.low_hp, S, SV)

    total = sum(len(v[0]) for v in gyms.values())
    print(f"bank {len(Hd)} steps, {int(E.max())+1} replays | horizon "
          f"{a.horizon}, history {a.history}\n")
    print("  gym               pairs    share   replays   mean hp(me)  mean red(me)")
    meta = {}
    for name, (st, sd) in gyms.items():
        mine_hp = np.where(sd == 0, Hd[st, HP1], Hd[st, HP2])
        mine_red = np.where(sd == 0, Hd[st, COMBO1], Hd[st, COMBO2])
        reps = len(np.unique(E[st]))
        meta[name] = {"pairs": int(len(st)), "replays": int(reps),
                      "mean_hp_mine": float(mine_hp.mean()),
                      "mean_red_mine": float(mine_red.mean())}
        print(f"  {name:<16} {len(st):7d}  {len(st)/max(total,1):6.1%}  "
              f"{reps:7d}   {mine_hp.mean():10.3f}  {mine_red.mean():12.4f}")

    np.savez(a.out, **{f"{k}_starts": v[0] for k, v in gyms.items()},
             **{f"{k}_sides": v[1] for k, v in gyms.items()},
             names=np.array(list(gyms)), horizon=a.horizon, history=a.history,
             fingerprint=str(b["fingerprint"]) if "fingerprint" in b.files else "")
    (a.out.with_suffix(".json")).write_text(json.dumps(
        {"bank": str(a.bank), "horizon": a.horizon, "gyms": meta}, indent=1))

    print(f"\n-> {a.out}")
    up = meta.get("blocking") or meta["under_pressure"]
    print(f"\nThe blocking gym has {up['pairs']} pairs across {up['replays']} "
          f"replays, at mean\nhealth {up['mean_hp_mine']:.2f} and mean red "
          f"{up['mean_red_mine']:.3f} on the agent's own bar.")
    print("Every pair carries its side: 'under_pressure' means the agent sits in "
          "the chair\nbeing hit. Sampling a side at random would put it in the "
          "attacker's seat half\nthe time and drill the opposite mechanic.")
    if any(k in gyms for k in ("blocking", "knocked_down")):
        print("\nStill NOT built: projectile pressure, and so no dodging gym. "
              "Nothing in\nthe sidecar sees projectiles, and `hit_clean` "
              "selects the outcome dodging\navoids rather than the mechanic. "
              "That one needs a pixel detector.")
    else:
        print("\nNOT built, because the HUD cannot see them: knockdown, "
              "projectile pressure,\nand anything positional. Those need a "
              "pixel detector or state logged at capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
