"""How much does the SIMULATOR's prediction depend on the buttons?

    python -m scripts.state_action_effect ~/rl/mv2/base/sim.pt ~/rl/h2/s0/sim.pt \
        --names h12 h2 --cache ~/rl/mv2_corpus.npz

WHY THIS IS THE MEASUREMENT THAT MATTERS
-----------------------------------------
Prediction skill and action sensitivity are different quantities, and this
project spent a long time optimising the first while needing the second. The
simulator reads `guarding` off the situation at AUC 0.993 one step ahead and
moves its guard prediction by 0.015 when the defender's stick is flipped from
toward to away -- 6% of the spread its predictions show across situations.
Pressing an attack button moves predicted `hp` by 0.004 sigma.

A policy cannot learn a mechanic whose button does not move the outcome inside
the model, however well the model predicts that outcome from context. So:

  effect   the change in the predicted next state from an intervention on the
           buttons, holding the state history fixed. This is what RL consumes.
  spread   the standard deviation of the same prediction across DIFFERENT real
           states. This is what next-state regression is scored on.

The ratio is the honest summary, because an effect is only small relative to
something.

Step 1 only, and every input is real corpus data -- no rollout, so nothing here
can be blamed on autoregressive feedback.

BINARY CHANNELS ARE READ AS PROBABILITIES
------------------------------------------
The head emits logits for flags. Comparing a logit difference against the
corpus standard deviation of a 0/1 flag mixes two different units and inflates
the number by roughly 20x -- a mistake made once already while writing this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.soku import BUTTONS
from sokubot.data.state import CH
from sokubot.model.state_dynamics import load_sim
from sokubot.model.state_head import BINARY

WATCH = ("guarding", "airborne", "hp", "spirit", "vx", "x")
PRESS = ("left", "right", "up", "down", "a", "b", "c")


def read(ns, i, sig):
    """Prediction for channel i: probability for a flag, sigma for the rest."""
    v = ns[:, -1, 0, i]
    return torch.sigmoid(v) if i in set(BINARY) else v / sig[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sims", type=Path, nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--near", type=float, default=250.0,
                    help="only states with the players this close, where the "
                         "stick is a real decision rather than irrelevant")
    a = ap.parse_args()
    names = a.names or [p.parent.name for p in a.sims]

    d = np.load(a.cache)
    S, P, A, E = d["S"], d["P"], d["A"], d["E"]
    sig = torch.as_tensor(S.reshape(-1, S.shape[-1]).std(0).clip(1e-6))

    for name, path in zip(names, a.sims):
        model, meta = load_sim(path, "cpu")
        H = int(meta["history"])
        ok = np.ones(len(S), bool); ok[-(H + 2):] = False
        for k in range(1, H + 2):
            ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
        ok &= np.abs(S[:, 0, CH["dx"]]) < (a.near / 1200.0)
        idx = np.random.default_rng(7).choice(
            np.flatnonzero(ok), min(a.n, int(ok.sum())), replace=False)
        w = idx[:, None] + np.arange(H)[None, :]
        s = torch.as_tensor(S[w]).float(); p = torch.as_tensor(P[w]).float()
        off = torch.as_tensor(A[w]).float().clone(); off[:] = 0.0
        dx = s[:, -1, 0, CH["dx"]]
        L, R = BUTTONS.index("left"), BUTTONS.index("right")

        with torch.no_grad():
            base, _ = model(s, p, off)
            print(f"\n=== {name}  (history {H}) ===")
            print("%7s " % "button" + " ".join("%10s" % c for c in WATCH))
            for b in PRESS:
                act = off.clone(); act[:, :, :, BUTTONS.index(b)] = 1.0
                ns, _ = model(s, p, act)
                print("%7s " % b + " ".join(
                    "%10.4f" % float((read(ns, CH[c], sig)
                                      - read(base, CH[c], sig)).abs().mean())
                    for c in WATCH))

            # The blocking intervention specifically: away vs toward, which is
            # what guarding IS in this game.
            out = {}
            for tag, away in (("away", True), ("toward", False)):
                act = off.clone()
                left = (dx > 0) if away else (dx < 0)
                act[left, :, :, L] = 1.0
                act[~left, :, :, R] = 1.0
                ns, _ = model(s, p, act)
                out[tag] = torch.sigmoid(ns[:, -1, 0, CH["guarding"]])
            eff = float((out["away"] - out["toward"]).mean())
            spread = float(out["away"].std())
            print("  guard: away-toward %+.5f | spread over states %.5f | "
                  "ratio %.4f" % (eff, spread, abs(eff) / max(spread, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
