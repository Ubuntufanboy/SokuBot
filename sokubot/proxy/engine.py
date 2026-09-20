"""A deterministic miniature fighting game, built to reproduce Soku's problems.

WHY A PROXY AT ALL
------------------
The question that matters -- does a world model need interventional data to
learn what actions do -- cannot be answered on Soku, because getting
counterfactual pairs out of the real game is a reverse-engineering task of
unknown length (see `future_paper/findings/08`). Here the simulator is ours, so
a counterfactual is one function call: save the state, apply action A, restore,
apply action B.

WHAT MAKES IT A FAIR PROXY, AND WHAT WOULD MAKE IT A BAD ONE
-------------------------------------------------------------
A toy that is easy in the ways Soku is hard proves nothing. The design targets
the five measured pathologies, and `validate.py` checks each one against the
real numbers before any conclusion is drawn from it:

  1. Most frames are NON-ACTIONABLE. In Soku 62.6% of frames are in untech,
     hitstop or knockdown, where no input changes anything. Here stun does the
     same job.
  2. The interesting event is RARE and BINARY. Soku's `guarding` fires on 4.5%
     of frames, so its whole entropy is 0.185 nats and a summed loss prices it
     at about half a percent.
  3. The action must PRECEDE the event it causes. A Soku block needs the
     defender already holding away when the attack connects -- median 44 frames
     early. An intervention applied at the moment of contact is applied after
     the decision that mattered.
  4. HELD ACTIONS LEAK INTO STATE. A player who has walked backwards for a
     second has revealed their input through their position, which is why
     conditioning on 12 frames of history destroys 2.03x of the action
     information at k=8.
  5. The behaviour policy has HIDDEN STATE. This is the one that decides the
     whole experiment: if the logged state contains everything the policy
     conditions on, the back-door criterion is satisfied and observational data
     identifies the causal dynamics -- counterfactuals would be unnecessary by
     construction, and a positive result would be an artifact of the toy. So
     the scripted players carry an `intent` that is never written to the state.

THE ENGINE
----------
Two players on a line. An attack has startup, active and recovery frames; input
is ignored throughout. Contact during the active window is a BLOCK if the
defender is holding away at that instant, otherwise a HIT. Blocks cost chip and
a little blockstun; hits cost damage and a lot of hitstun.

Deterministic by construction: integer positions, no floating point in the
transition, no RNG inside `step`. All randomness lives in the policy, which is
seeded and reproducible, so a rollout is a pure function of (state, actions).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

STAGE = 200          # half-width; x in [-STAGE, STAGE]
REACH = 60           # attack range
SPEED = 4
STARTUP, ACTIVE, RECOVERY = 5, 3, 8
HITSTUN, BLOCKSTUN = 20, 16
DAMAGE, CHIP = 100, 12
MAX_HP = 1000

# Actions, as a discrete set. `guard` is not an action: holding away IS the
# guard, exactly as in Soku, which is what makes requirement 3 hold.
IDLE, LEFT, RIGHT, ATTACK = 0, 1, 2, 3
N_ACTIONS = 4


@dataclass(frozen=True)
class Fighter:
    x: int
    hp: int = MAX_HP
    stun: int = 0          # forced inactivity: no input has any effect
    atk: int = -1          # frames into the current attack, -1 when not attacking
    # `stance` is holding away; `guarding` is a block that CONNECTED. Keeping
    # them separate is the whole point: the first version conflated them, so
    # guarding became a deterministic function of the action, association and
    # causal effect were both 1.0, and no confound was possible. Soku's
    # `guarding` is ACT_RIGHTBLOCK -- blockstun from a real attack -- and
    # holding away is merely what makes it possible.
    stance: int = 0        # holding away this frame; NOT recorded in observe()
    guarding: int = 0      # a block actually landed; blockstun
    facing: int = 1

    @property
    def actionable(self) -> bool:
        return self.stun == 0 and self.atk < 0


@dataclass(frozen=True)
class State:
    p: tuple  # (Fighter, Fighter)
    t: int = 0

    @property
    def over(self) -> bool:
        return any(f.hp <= 0 for f in self.p) or self.t > 6000


def initial(x0: int = -120, x1: int = 120) -> State:
    return State(p=(Fighter(x=x0, facing=1), Fighter(x=x1, facing=-1)))


def _away(me: Fighter, other: Fighter) -> int:
    """The action that moves `me` away from `other`."""
    return LEFT if other.x > me.x else RIGHT


def step(s: State, a0: int, a1: int) -> State:
    """(state, both actions) -> next state. Pure, integer, no RNG."""
    f = list(s.p)
    acts = [a0, a1]

    # --- start attacks / apply movement, only where the player can act -----
    nxt = []
    for i, me in enumerate(f):
        a = acts[i]
        if me.stun > 0:
            # Blockstun keeps `guarding` lit, matching ACT_RIGHTBLOCK's span.
            nxt.append(replace(me, stun=me.stun - 1, stance=0))
            continue
        if me.atk >= 0:
            nxt.append(replace(me, atk=me.atk + 1, stance=0, guarding=0))
            continue
        other = f[1 - i]
        facing = 1 if other.x > me.x else -1
        if a == ATTACK:
            nxt.append(replace(me, atk=0, facing=facing, stance=0, guarding=0))
        elif a in (LEFT, RIGHT):
            dx = -SPEED if a == LEFT else SPEED
            x = max(-STAGE, min(STAGE, me.x + dx))
            # Holding away is the guard STANCE. It only becomes a block if an
            # attack arrives while it is held -- resolved below.
            st = 1 if a == _away(me, other) else 0
            nxt.append(replace(me, x=x, facing=facing, stance=st, guarding=0))
        else:
            nxt.append(replace(me, facing=facing, stance=0, guarding=0))

    # --- resolve contact ---------------------------------------------------
    out = list(nxt)
    for i, me in enumerate(nxt):
        if not (STARTUP <= me.atk < STARTUP + ACTIVE):
            continue
        d = nxt[1 - i]
        if abs(me.x - d.x) > REACH or d.stun > 0:
            continue
        if d.stance:
            out[1 - i] = replace(out[1 - i], hp=max(0, d.hp - CHIP),
                                 stun=BLOCKSTUN, guarding=1)
        else:
            out[1 - i] = replace(out[1 - i], hp=max(0, d.hp - DAMAGE),
                                 stun=HITSTUN, guarding=0)

    # --- retire finished attacks -------------------------------------------
    out = [replace(m, atk=-1) if m.atk >= STARTUP + ACTIVE + RECOVERY else m
           for m in out]
    return State(p=tuple(out), t=s.t + 1)


# BOTH players, because an attack's STARTUP is visible on screen -- Soku
# telegraphs every move for several frames before the hitbox goes live, and its
# state is [2, 33], carrying the opponent's `hitboxes` and `action_frame` too.
# An earlier version recorded only the ego player's attack and left the
# opponent's hidden, which manufactured a confounder the real game does not
# have: the scripted players react to an incoming attack, so hiding it made
# the reaction look like unexplained intent. What stays hidden is INTENT, which
# is genuinely unobservable in both.
CHANNELS = ("x", "hp", "stun", "atk", "guarding", "facing", "dx", "actionable",
            "o_hp", "o_stun", "o_atk", "o_guarding", "o_actionable")


def observe(s: State, me: int) -> list:
    """The recorded state, ego-ordered. Deliberately a LOSSY SUMMARY -- it
    carries no policy intent, which is what leaves the back door open."""
    a, b = s.p[me], s.p[1 - me]
    # `stance` is deliberately ABSENT: the recorded state does not say which
    # way the player is currently holding, exactly as Soku's 33 channels do
    # not. It leaks only through position drift.
    return [a.x / STAGE, a.hp / MAX_HP, a.stun / HITSTUN, a.atk / 20.0,
            float(a.guarding), float(a.facing), (b.x - a.x) / STAGE,
            float(a.actionable),
            b.hp / MAX_HP, b.stun / HITSTUN, b.atk / 20.0,
            float(b.guarding), float(b.actionable)]
