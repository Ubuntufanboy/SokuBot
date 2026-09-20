"""Micro world model on the proxy, with and without counterfactual data.

THE ONE QUESTION
----------------
Does a world model trained on observational data alone learn what actions DO,
or only what they CORRELATE with? On Soku this was unanswerable: the causal
effect could not be measured, so the model's response had no honest denominator
(`future_paper/findings/08`). Here the engine is ours, so the truth is known --
`validate.py` measures it at +0.0262 for away-vs-toward on guarding -- and the
observational association overstates it by 1.72x.

So the prediction is sharp, and falsifiable both ways:

  obs-only  should land near the ASSOCIATION (+0.045), because that is what
            the data it saw says, and it has no way to tell the two apart.
  obs+cf    should land near the CAUSAL effect (+0.026), because it has seen
            the same state played two ways.

If both land in the same place, counterfactual data buys nothing here and the
argument for chasing it in Soku collapses.

The counterfactual arm gets NO EXTRA GRADIENT STEPS -- it swaps a fraction of
each batch for interventional pairs. Otherwise a win would just be more
training.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import CHANNELS, LEFT, N_ACTIONS, REACH, RIGHT, observe, step
from .policy import (corpus, counterfactual_pairs, onpolicy_corpus,
                     rollout)

GI = CHANNELS.index("guarding")
C = len(CHANNELS)
H = 4


class WM(nn.Module):
    """(state history, action) -> next state. Flags as logits, rest as deltas."""

    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C * H + N_ACTIONS, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, C))

    def forward(self, hist, act):
        x = torch.cat([hist.flatten(1), F.one_hot(act, N_ACTIONS).float()], -1)
        d = self.net(x)
        nxt = hist[:, -1] + d
        return nxt, d


def windows(O, A, E, n):
    ok = np.ones(len(O), bool); ok[-(H + 1):] = False
    for k in range(1, H + 2):
        ok[:len(O) - k] &= (E[k:] == E[:len(O) - k])
    return np.flatnonzero(ok)


def train(seed, cf_frac, steps=6000, bs=256, lr=1e-3, n_match=400,
          verbose=True, guard_weight=1.0, onpolicy=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Equal match counts, because with temporally correlated exploration the
    # two corpora now block at comparable RATES (3.6% on-policy vs 2.3% human).
    # The previous 6x scaling was compensating for a starvation that came from
    # i.i.d. noise breaking every stance, and it compensated by the wrong
    # factor in the wrong direction -- realised counts are the thing to check,
    # not the intended ones.
    O, A, E = (onpolicy_corpus(n_match, seed0=seed * 7919)
               if onpolicy else corpus(n_match, seed0=seed * 7919))
    if verbose:
        from .engine import CHANNELS as _C
        g = O[:, 0, _C.index("guarding")]
        print(f"  corpus {len(O)} frames, guard {g.mean()*100:.2f}% "
              f"({int(g.sum())} frames)", flush=True)
    idx = windows(O, A, E, H)
    CFn = 20000 if cf_frac > 0 else 0
    if CFn:
        pre, aa, ab, fa, fb = counterfactual_pairs(CFn, seed0=seed * 7919 + 500_000)
        pre_t = torch.as_tensor(pre); fa_t = torch.as_tensor(fa)
        fb_t = torch.as_tensor(fb)
        aa_t = torch.as_tensor(aa); ab_t = torch.as_tensor(ab)

    m = WM()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                                pct_start=0.1)
    for i in range(steps):
        n_cf = int(bs * cf_frac)
        n_ob = bs - n_cf
        b = rng.choice(idx, n_ob)
        w = b[:, None] + np.arange(H + 1)[None, :]
        hist = torch.as_tensor(O[w][:, :H, 0])
        act = torch.as_tensor(A[b + H - 1, 0])
        tgt = torch.as_tensor(O[w][:, H, 0])
        pred, _ = m(hist, act)
        # PRICE THE RARE CHANNEL. `guarding` fires on 2.5% of frames, so an
        # MSE summed over 13 channels makes "never guards" nearly optimal
        # whatever the causal structure -- which would suppress the action
        # effect in BOTH arms, which is the pattern observed.
        w = torch.ones(C); w[GI] = guard_weight
        loss = (((pred - tgt) ** 2) * w).mean()

        if n_cf:
            # Interventional pairs: SAME start, two actions, two futures. The
            # history is the start state repeated, since the branch point is
            # all the counterfactual pins down.
            j = torch.as_tensor(rng.choice(len(pre_t), n_cf))
            h0 = pre_t[j].unsqueeze(1).expand(-1, H, -1)
            for act_t, fut_t in ((aa_t[j], fa_t[j]), (ab_t[j], fb_t[j])):
                p, _ = m(h0, act_t)
                loss = loss + 0.5 * (((p - fut_t[:, 0]) ** 2) * w).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sched.step()
        if verbose and i % 2000 == 0:
            print(f"  step {i:5d} loss {float(loss.detach()):.5f}", flush=True)
    return m


@torch.no_grad()
def action_effect(m, n=4000, seed=99):
    """The model's OWN causal effect: same state, away vs toward, 3 steps."""
    rng = np.random.default_rng(seed)
    got = {"away": 0.0, "toward": 0.0}
    vals = []
    trials, mm = 0, 0
    while trials < n:
        o, a, S = rollout(900_000 + mm); mm += 1
        for _ in range(30):
            if len(S) < 8 or trials >= n:
                break
            t = int(rng.integers(1, len(S) - 4))
            s0 = S[t]
            if not s0.p[0].actionable:
                continue
            d = s0.p[1].x - s0.p[0].x
            if abs(d) > REACH * 2:
                continue
            h0 = torch.as_tensor(np.array([observe(s0, 0)] * H, np.float32)
                                 ).unsqueeze(0)
            for tag, mine in (("away", LEFT if d > 0 else RIGHT),
                              ("toward", RIGHT if d > 0 else LEFT)):
                hist, best = h0.clone(), 0.0
                for _k in range(3):
                    p, _ = m(hist, torch.tensor([mine]))
                    best = max(best, float(torch.sigmoid(p[0, GI] * 4 - 2)))
                    hist = torch.cat([hist[:, 1:], p.unsqueeze(1)], 1)
                got[tag] += best
                vals.append(best)
            trials += 1
    eff = (got["away"] - got["toward"]) / max(trials, 1)
    # THE EFFECT IS ONLY SMALL RELATIVE TO SOMETHING. Reported against the
    # model's own output spread, because an absolute difference between a
    # compressed regression output and a ground-truth probability is a
    # comparison of two different units -- the error made three times in this
    # project already (see future_paper/findings/07).
    sd = float(np.std(vals)) if vals else float("nan")
    return eff, sd, float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--cf-frac", type=float, default=0.35)
    ap.add_argument("--guard-weight", type=float, default=1.0)
    ap.add_argument("--out", default="/tmp/proxy_result.json")
    a = ap.parse_args()
    res = {}
    for name, cf, op in (("human_obs", 0.0, False),
                         ("onpolicy_obs", 0.0, True),
                         ("human_plus_cf", a.cf_frac, False)):
        res[name] = []
        for s in a.seeds:
            print(f"=== {name} seed {s} ===", flush=True)
            m = train(s, cf, steps=a.steps, guard_weight=a.guard_weight,
                      onpolicy=op)
            eff, sd, mu = action_effect(m)
            res[name].append({"eff": eff, "sd": sd, "mean": mu,
                              "ratio": eff / sd if sd else float("nan")})
            print(f"  effect {eff:+.5f}  spread {sd:.5f}  "
                  f"ratio {eff/sd if sd else float('nan'):+.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    # Truth in the same units: a 2.5% binary has spread sqrt(p(1-p)) = 0.156,
    # so the causal effect is 0.0262 / 0.156 = 0.168 of its own spread.
    TRUE_RATIO = 0.0262 / (0.025 * 0.975) ** 0.5
    print(f"\nTRUE causal +0.0262, spread 0.1561, ratio {TRUE_RATIO:+.4f}")
    for k, v in res.items():
        r = [x["ratio"] for x in v]
        e = [x["eff"] for x in v]
        print(f"{k:12s} eff mean {np.mean(e):+.5f} | ratio "
              + " ".join(f"{x:+.4f}" for x in r)
              + f" | mean ratio {np.mean(r):+.4f}"
              + f" ({np.mean(r)/TRUE_RATIO*100:.0f}% of truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
