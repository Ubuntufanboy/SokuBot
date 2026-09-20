"""Read a state-GRPO run's log and say what actually moved, per gym.

    python -m scripts.report_state_grpo ~/rl/grpo_state

The headline `net` is an average over eleven drills plus the unfiltered corpus,
and an average is exactly the wrong instrument for the question being asked. A
policy that learns to block and forgets to attack can hold `net` flat while two
gyms move hard in opposite directions -- and "blocking improved" is the claim
this whole redesign exists to be able to make or refuse. So every number here is
per gym, and the pooled figure is reported last rather than first.

Reported against the FIRST evaluation, not against zero: at step 1 the policy
and the frozen reference are the same weights, so that row is the run's own
measurement of its noise floor and every later row should be read against it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokubot.data.state import FULL_HP


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--last", type=int, default=5,
                    help="average the final N evaluations, so a single noisy "
                         "row cannot be quoted as the result")
    a = ap.parse_args()

    log = json.loads((a.run.expanduser() / "log.json").read_text())
    cfg = json.loads((a.run.expanduser() / "config.json").read_text())
    evals = [r for r in log if "eval_net" in r]
    if not evals:
        raise SystemExit(f"{a.run}/log.json has no evaluations yet")
    gyms = sorted(k[4:-4] for k in evals[-1] if k.startswith("gym_")
                  and k.endswith("_net"))

    print(f"{a.run}  |  simulator {Path(cfg['sim']).parent.name} step "
          f"{cfg['sim_step']}  |  horizon {cfg['horizon']} steps "
          f"({cfg['horizon']*cfg['ticks']*1000/60:.0f} ms)")
    print(f"damage_mode {cfg['damage_mode']} | proj_feedback "
          f"{cfg['proj_feedback']} | {cfg['starts']}x{cfg['group_size']} "
          f"rollouts | {cfg['bank_replays']} replays\n")

    first, last = evals[0], evals[-a.last:]

    def mean(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"evaluated {len(evals)} times, steps {evals[0]['step']} .. "
          f"{evals[-1]['step']} | averaging the last {len(last)}")
    print(f"\n  {'gym':<20} {'net @1':>10} {'net @end':>10} {'delta':>10} "
          f"{'HP/step':>9}")
    rows = []
    for g in gyms:
        k = f"gym_{g}_net"
        a0, a1 = first.get(k, float("nan")), mean(last, k)
        rows.append((a1 - a0, g, a0, a1))
    for d, g, a0, a1 in sorted(rows, reverse=True):
        print(f"  {g:<20} {a0:+10.5f} {a1:+10.5f} {d:+10.5f} {d*FULL_HP:+9.1f}")

    n0, n1 = first["eval_net"], mean(last, "eval_net")
    print(f"\n  {'(corpus)':<20} {n0:+10.5f} {n1:+10.5f} {n1-n0:+10.5f} "
          f"{(n1-n0)*FULL_HP:+9.1f}")
    print(f"\nstep 1 is the noise floor: policy and reference are the same "
          f"weights there, so |{n0:+.5f}| is what 'no difference' measures as.")

    # `eval_guard` is the side-swapped rate against the frozen reference and is
    # the one to quote; `guard_rate` is the training rollouts' own rate, which
    # is confounded by which gym the batch came from. Fall back rather than
    # print nan, because a run launched before the eval-side rate existed still
    # has the training-side one and the mechanic is the point.
    key = "eval_guard" if "eval_guard" in evals[-1] else "guard_rate"
    src = "eval, side-swapped" if key == "eval_guard" else "training rollouts"
    tail = log[-a.last * 8:]
    g0 = first.get(key, log[0].get(key, float("nan")))
    g1 = mean(last if key == "eval_guard" else tail, key)
    print(f"guarding rate {g0:.4f} -> {g1:.4f} ({g1-g0:+.4f})  [{src}]")

    # Drift, because the failure mode of every previous GRPO run was the policy
    # leaving the distribution the world model was fit to rather than the
    # optimiser stalling: entropy to 0.05 and a 33% press rate in one run,
    # entropy to the 25.4 maximum in another. Both showed up here first.
    quarters = [log[i * len(log) // 4:(i + 1) * len(log) // 4] for i in range(4)]
    print(f"\ndrift over the run (corpus press rate 0.0976):")
    print(f"  {'quarter':<9} {'entropy':>8} {'press':>8} {'idle':>7} "
          f"{'KL_ref':>9} {'KL':>8}")
    for i, q in enumerate(quarters):
        if not q:
            continue
        print(f"  {i+1:<9} {mean(q,'entropy'):8.2f} {mean(q,'press_rate'):8.4f} "
              f"{mean(q,'idle_rate'):7.4f} {mean(q,'kl_ref'):9.4f} "
              f"{mean(q,'kl'):8.5f}")
    print(f"  entropy floor {log[-1].get('ent_floor', float('nan')):.2f}, "
          f"multiplier {log[-1].get('ent_alpha', float('nan')):.3f} "
          f"(inert while entropy is above the floor)")
    best = max(evals, key=lambda r: r["eval_net"])
    print(f"best corpus net {best['eval_net']:+.5f} "
          f"({best['eval_net']*FULL_HP:+.1f} HP/step) at step {best['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
