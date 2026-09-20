"""Scripted players with HIDDEN state, and the data generators.

THE HIDDEN INTENT IS THE POINT
------------------------------
If the behaviour policy conditioned only on the recorded state, the back-door
criterion would be satisfied and observational data would identify the causal
dynamics exactly -- counterfactuals would then be unnecessary BY CONSTRUCTION,
and a positive result for them would be an artifact of the toy rather than a
finding. So each player carries an `intent` that is never observed:

    intent in {RUSH, TURTLE, BAIT}, switching on a hidden timer.

Two players in identical recorded states act differently depending on intent,
and intent correlates with what happens next -- a TURTLE holds away and is
therefore both more likely to block AND in a situation where the opponent
attacks less. That is exactly the confound that makes the observational
association between "holding away" and "blocking" larger than the causal
effect, which is the flaw found in the real Soku measurement.
"""

from __future__ import annotations

import numpy as np

from .engine import (ATTACK, IDLE, LEFT, REACH, RIGHT, STAGE, State, initial,
                     observe, step)

RUSH, TURTLE, BAIT = 0, 1, 2


class Scripted:
    """One player. `intent` is hidden from everything that gets recorded."""

    def __init__(self, rng, aggression=0.5):
        self.rng = rng
        self.aggression = aggression
        self.intent = RUSH
        self.timer = 0

    def act(self, s: State, me: int) -> int:
        a, b = s.p[me], s.p[1 - me]
        if not a.actionable:
            return IDLE                      # nothing has any effect here
        if self.timer <= 0:
            # Normalised, not hand-balanced: with aggression above 0.85 the
            # turtle share went negative and numpy rejected the draw.
            w = np.array([self.aggression,
                          max(0.05, 1.0 - self.aggression - 0.15), 0.15])
            self.intent = int(self.rng.choice([RUSH, TURTLE, BAIT], p=w / w.sum()))
            self.timer = int(self.rng.integers(20, 70))
        self.timer -= 1

        gap = abs(b.x - a.x)
        toward = LEFT if b.x < a.x else RIGHT
        away = RIGHT if b.x < a.x else LEFT
        # A committed opponent is the cue every intent reacts to, differently.
        threat = b.atk >= 0
        if self.intent == TURTLE:
            if threat and gap < REACH * 2:
                return away
            return away if gap < REACH * 1.3 else IDLE
        if self.intent == BAIT:
            if threat and gap < REACH * 1.5:
                return away
            return toward if gap > REACH * 2 else IDLE
        # RUSH
        if gap <= REACH * 1.2 and self.rng.random() < 0.65:
            return ATTACK
        if threat and gap < REACH and self.rng.random() < 0.4:
            return away
        return toward if gap > REACH * 0.7 else IDLE


def rollout(seed: int, max_t: int = 1200):
    """One match. Returns (obs[T,2,C], actions[T,2], states[T])."""
    rng = np.random.default_rng(seed)
    pols = [Scripted(rng, aggression=float(rng.uniform(0.35, 0.7)))
            for _ in range(2)]
    s = initial(x0=int(rng.integers(-150, -40)), x1=int(rng.integers(40, 150)))
    O, A, S = [], [], []
    for _ in range(max_t):
        if s.over:
            break
        acts = (pols[0].act(s, 0), pols[1].act(s, 1))
        O.append([observe(s, 0), observe(s, 1)])
        A.append(list(acts))
        S.append(s)
        s = step(s, acts[0], acts[1])
    return np.array(O, np.float32), np.array(A, np.int64), S


def corpus(n_matches: int, seed0: int = 0):
    """Observational data: what a corpus of recorded matches looks like."""
    O, A, E = [], [], []
    for m in range(n_matches):
        o, a, _ = rollout(seed0 + m)
        if len(o) < 40:
            continue
        O.append(o); A.append(a); E.append(np.full(len(o), m, np.int32))
    return (np.concatenate(O), np.concatenate(A), np.concatenate(E))


def counterfactual_pairs(n: int, seed0: int = 10_000, horizon: int = 16):
    """THE THING THE REAL GAME COULD NOT GIVE US.

    From one visited state, roll the SAME state forward under two different
    actions for player 0, with player 1's action sequence held fixed. Both
    branches share a start, so the difference is causal by construction --
    no conditioning, no back door, no confound.
    """
    rng = np.random.default_rng(seed0)
    pre, act_a, act_b, fut_a, fut_b = [], [], [], [], []
    m = 0
    while len(pre) < n:
        o, a, S = rollout(seed0 + m)
        m += 1
        if len(S) < horizon + 4:
            continue
        for _ in range(max(1, n // 200)):
            t = int(rng.integers(1, len(S) - horizon - 1))
            s0 = S[t]
            # The opponent's actions are FIXED across both branches: varying
            # them too would measure a different match, not an intervention.
            opp = [int(a[min(t + k, len(a) - 1)][1]) for k in range(horizon)]
            aa = int(rng.integers(0, 4))
            bb = int(rng.integers(0, 4))
            if aa == bb:
                bb = (bb + 1) % 4
            branch = []
            for first in (aa, bb):
                s, traj = s0, []
                for k in range(horizon):
                    mine = first if k < 4 else IDLE   # hold the intervention 4 ticks
                    s = step(s, mine, opp[k])
                    traj.append(observe(s, 0))
                branch.append(np.array(traj, np.float32))
            pre.append(observe(s0, 0))
            act_a.append(aa); act_b.append(bb)
            fut_a.append(branch[0]); fut_b.append(branch[1])
            if len(pre) >= n:
                break
    return (np.array(pre, np.float32), np.array(act_a, np.int64),
            np.array(act_b, np.int64), np.array(fut_a, np.float32),
            np.array(fut_b, np.float32))


# =========================================================================
# On-policy data: unconfounded WITHOUT branching
# =========================================================================
# The human corpus is confounded because the scripted players condition on
# `intent`, which is never recorded. An agent of our own has no such privacy:
# it conditions on exactly the observation we log, plus independent noise. That
# closes the back door by construction, so
#
#     P(s' | o, a) = P(s' | o, do(a))
#
# and ordinary next-state regression on its data identifies the causal
# dynamics. If this holds, frame-precise counterfactual branching is
# unnecessary and on-policy collection is enough -- which is the difference
# between an open-ended reverse-engineering task and pointing the existing live
# loop at the problem.


class Observational:
    """Acts on the RECORDED observation only, plus independent noise.

    Deliberately crude. The point is not that it plays well but that nothing it
    conditions on is hidden from the model that will be trained on its data.
    """

    def __init__(self, rng, eps=0.18, hold=6):
        self.rng = rng
        self.eps = eps
        self.hold = hold
        self._noise = -1
        self._left = 0

    def act(self, s, me):
        # TEMPORALLY CORRELATED EXPLORATION.
        #
        # i.i.d. per-frame noise cannot coexist with a multi-frame commitment:
        # at eps 0.18 the chance of holding a direction through a 5-frame
        # startup is 0.82^5 = 0.37, so two thirds of attempted blocks were
        # broken by a random action mid-stance. The on-policy corpus blocked on
        # 0.2% of frames and the arm trained on it saw a third of the events
        # the others did. Holding each exploratory action for a few frames
        # keeps the noise INDEPENDENT of the hidden state -- which is all
        # unconfoundedness requires -- while letting a stance survive.
        if self._left > 0:
            self._left -= 1
            return self._noise
        if self.rng.random() < self.eps:
            self._noise = int(self.rng.integers(0, 4))
            self._left = int(self.rng.integers(2, self.hold))
            return self._noise
        o = observe(s, me)
        if o[7] < 0.5:                              # not actionable
            return IDLE
        dx, o_atk, o_act = o[6], o[10], o[12]
        toward = LEFT if dx < 0 else RIGHT
        away = RIGHT if dx < 0 else LEFT
        gap = abs(dx) * STAGE
        # `o_atk >= 0` is any attack in progress, INCLUDING startup: the whole
        # point of the telegraph is that it is visible before the hitbox is
        # live, and a policy that waits for `> 0` reacts a frame too late to
        # ever block. The first version did, and produced a corpus with zero
        # blocks in 47k frames -- unusable, since a model cannot learn about an
        # event that never happens.
        if o_atk >= 0 and gap < REACH * 1.3:
            return away
        if gap <= REACH * 1.2 and self.rng.random() < 0.55:
            return ATTACK
        return toward if gap > REACH * 0.6 else IDLE


def onpolicy_corpus(n_matches, seed0=200_000, eps=0.18):
    """Same engine, same everything -- only the behaviour policy changes."""
    O, A, E = [], [], []
    for m in range(n_matches):
        rng = np.random.default_rng(seed0 + m)
        pols = [Observational(rng, eps) for _ in range(2)]
        s = initial(x0=int(rng.integers(-150, -40)), x1=int(rng.integers(40, 150)))
        o_, a_ = [], []
        for _ in range(1200):
            if s.over:
                break
            acts = (pols[0].act(s, 0), pols[1].act(s, 1))
            o_.append([observe(s, 0), observe(s, 1)])
            a_.append(list(acts))
            s = step(s, acts[0], acts[1])
        if len(o_) < 40:
            continue
        O.append(np.array(o_, np.float32)); A.append(np.array(a_, np.int64))
        E.append(np.full(len(o_), m, np.int32))
    return np.concatenate(O), np.concatenate(A), np.concatenate(E)
