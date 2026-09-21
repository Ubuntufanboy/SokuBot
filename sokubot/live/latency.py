"""Per-decision latency, summarised the way the control loop is judged.

The loop is not judged on its mean. A decision that lands late is played late, and
an agent whose p50 is comfortable and whose p99 is twice the period is an agent
that stutters exactly when something is happening on screen. `Pilot` used to keep
one exponential average and a late COUNT, which cannot tell those two apart.

Only decisions that produced an action are recorded by the caller: the first
`history` decisions of a session return no action because the window is still
filling, and they skip the policy, so counting them would flatter the loop.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def summarise(samples_ms: Iterable[float], budget_ms: float) -> dict:
    """n, p50/p90/p99/max in ms, and how many decisions exceeded `budget_ms`."""
    a = np.asarray(list(samples_ms), dtype=np.float64)
    out = {"n": int(a.size), "budget_ms": float(budget_ms)}
    if a.size == 0:
        out.update(p50=None, p90=None, p99=None, max=None,
                   over_budget=0, over_budget_frac=None)
        return out
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    over = int((a > budget_ms).sum())
    out.update(p50=float(p50), p90=float(p90), p99=float(p99),
               max=float(a.max()), over_budget=over,
               over_budget_frac=over / a.size)
    return out


def format_summary(s: dict) -> str:
    if not s["n"]:
        return "no decisions yet (nothing has produced an action)"
    verdict = "WITHIN budget" if s["p99"] <= s["budget_ms"] else "p99 OVER budget"
    return (f"{s['n']} decisions | p50 {s['p50']:.1f}  p90 {s['p90']:.1f}  "
            f"p99 {s['p99']:.1f}  max {s['max']:.1f} ms | budget "
            f"{s['budget_ms']:.1f} ms | over budget {s['over_budget']} "
            f"({100 * s['over_budget_frac']:.1f}%) | {verdict}")
