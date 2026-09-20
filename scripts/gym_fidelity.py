"""Where does the simulator over- or under-predict damage, per gym?

    python -m scripts.gym_fidelity --sim ~/rl/fix5/sim.pt --corpus ~/corpus

An RL gain is only worth as much as the simulator's honesty in the situation it
was won in, and honesty is not uniform across situations. This rolls the
simulator forward under the buttons the two humans ACTUALLY pressed, from each
gym's own start states, and compares the health it predicts against the health
the game really produced. No policy is involved, so nothing here is about how
well the agent plays -- only about whether the environment it played in was
telling the truth in that corner.

WHAT MOTIVATED IT
-----------------
After 7600 GRPO steps the agent had found +6.4 HP/step of extra damage DEALT in
`spell_incoming` -- a situation where real players deal 20.8 HP and take 155.6.
Attacking your way out of a spell declaration is exactly the kind of thing a
learned simulator can be wrong about generously, and "the agent got better at
X" and "the simulator is optimistic about X" produce identical training curves.
This tells them apart.

WHAT THE BIAS DOES AND DOES NOT PROVE
-------------------------------------
Positive bias means the simulator predicts MORE damage than really happened.
It is a map of where the simulator is confused, NOT a quantity to subtract from
an RL gain, and the difference matters:

  * An RL gain is `net(agent) - net(reference)`, both measured INSIDE the
    simulator. A bias that is constant across actions cancels exactly in that
    difference, so a gym can carry a large bias and still report an honest
    improvement.
  * What cannot be measured here is whether the bias is ACTION-DEPENDENT --
    whether the agent has found particular buttons the simulator is especially
    wrong about. That is the real hazard, and it is unmeasurable offline for
    the reason model-based RL is hard at all: the outcome of the agent's
    actions was never played, so there is no ground truth to compare against.

So a large bias is a reason to distrust a gym's result and to check it against
the real game before building on it -- not a proof that the result is fake. The
gyms where the bias is small ARE the ones whose numbers can be quoted without
that caveat, which is the useful half of this measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.data import state_bank
from sokubot.data.state import CH, FULL_HP
from sokubot.model.state_dynamics import load_sim
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.state_arena import (StateArena, StateGRPOConfig, StateObs,
                                    StatePolicyOpponent, corpus_stats)
from scripts.build_gyms import build as build_gyms
from scripts.state_preflight import _replay_rollout
from scripts.train_state_grpo import valid_starts


@torch.no_grad()
def fidelity(arena, S, P, A, idx, sides, H, horizon, device, chunk=512,
             driver=None):
    """Predicted vs real damage over `horizon`, for these (start, side) pairs.

    `driver` is None to replay the RECORDED buttons, or a policy to sample both
    chairs from. The two answer different questions and the gap between them is
    the point:

      human replay   how wrong is the simulator on the action manifold it was
                     trained on?
      prior policy   how wrong is it on actions that are human in their MARGINAL
                     rates but not in their sequencing -- which is what any
                     sampled policy produces, including the frozen reference an
                     RL gain is measured against?

    If the second is much worse than the first, then every RL number measured in
    this simulator sits on a distortion that the human-replay check does not
    see, and the size of that distortion is not something the agent-minus-
    reference difference can be assumed to cancel.
    """
    got = {"deal_pred": [], "deal_real": [], "take_pred": [], "take_real": []}
    for lo in range(0, len(idx), chunk):
        ix = idx[lo:lo + chunk]
        sd = sides[lo:lo + chunk]
        # The window starts `H-1` before the decision frame, matching the gym
        # convention and the trainer's `off = arange(H) - (H - 1)`.
        w = (ix - (H - 1))[:, None] + np.arange(H + horizon)[None, :]
        s = torch.as_tensor(S[w]).to(device)
        p = torch.as_tensor(P[w]).to(device)
        a = torch.as_tensor(A[w]).float().to(device)
        if driver is None:
            pred = _replay_rollout(arena, s[:, :H], p[:, :H], a, horizon)
        else:
            side0 = torch.zeros(len(ix), dtype=torch.long, device=device)
            tr = arena.rollout(s[:, :H], p[:, :H], a[:, :H - 1], side0,
                               driver, StatePolicyOpponent(driver))
            pred = tr["states"][:, 1:]
        me = torch.as_tensor(sd).to(device).long()
        them = 1 - me

        def drop(x, who, t0, t1):
            g = who[:, None, None].expand(-1, x.shape[1], 1)
            h = x[..., CH["hp"]].gather(-1, g).squeeze(-1)
            return (h[:, t1] - h[:, t0]).clamp(max=0.0).abs()

        # Real: from the decision frame to `horizon` steps later. Predicted:
        # from the same decision frame, whose value the rollout does not
        # change, to the rollout's last state.
        got["deal_real"].append(drop(s, them, H - 1, H + horizon - 1))
        got["take_real"].append(drop(s, me, H - 1, H + horizon - 1))
        base = s[:, H - 1:H]
        seq = torch.cat([base, pred], dim=1)
        got["deal_pred"].append(drop(seq, them, 0, horizon))
        got["take_pred"].append(drop(seq, me, 0, horizon))
    return {k: float(torch.cat(v).mean()) * FULL_HP for k, v in got.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("~/corpus").expanduser()])
    ap.add_argument("--cache", type=Path,
                    default=Path("~/rl/bank_state.npz").expanduser())
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--starts", type=int, default=2048)
    ap.add_argument("--driver", default="human", choices=("human", "prior"),
                    help="what supplies the buttons. 'human' replays the "
                         "recording; 'prior' samples both chairs from the "
                         "corpus-prior policy -- human marginal rates, but "
                         "sampled per tick rather than sequenced, which is what "
                         "every RL rollout actually contains.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sim, meta = load_sim(a.sim, a.device)
    H, slots, ticks = int(meta["history"]), int(meta["slots"]), int(meta["ticks"])
    S, P, A, E, V, names = state_bank.load(a.corpus, ticks, slots, a.cache)
    s_mu, s_sd, p_mu, p_sd = corpus_stats(S, P)
    obs = StateObs(s_mu, s_sd, p_mu, p_sd, slots).to(a.device)
    arena = StateArena(sim, obs, StateGRPOConfig(horizon=a.horizon), H, ticks)

    gyms = build_gyms(S, P, V, E, a.horizon, H)
    starts = valid_starts(E, V, H, a.horizon)
    rng = np.random.default_rng(7)
    sets = {"(corpus)": (rng.choice(starts, min(a.starts, len(starts))),
                         rng.integers(0, 2, min(a.starts, len(starts))))}
    for k in sorted(gyms):
        st, sd = gyms[k]
        if len(st):
            pick = rng.choice(len(st), min(a.starts, len(st)), replace=False)
            sets[k] = (st[pick], sd[pick])

    driver = None
    if a.driver == "prior":
        torch.manual_seed(0)
        driver = SokuPolicy(obs.dim, H, ticks).to(a.device)
        p1 = A.reshape(-1, 20)[:, :10].astype(np.float32)
        lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                         float(p1[:, 2].mean()), float(p1[:, 3].mean())])
        ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                         float(p1[:, 0].mean()), float(p1[:, 1].mean())])
        driver.set_action_prior(lr_p / lr_p.sum(), ud_p / ud_p.sum(),
                                p1[:, 4:10].mean(0))
        driver.eval()
    print(f"\nsimulator {a.sim} | horizon {a.horizon} steps "
          f"({a.horizon*ticks*1000/60:.0f} ms) | driver: {a.driver}\n")
    print(f"  {'gym':<20} {'deal pred':>10} {'real':>7} {'bias':>7}   "
          f"{'take pred':>10} {'real':>7} {'bias':>7}")
    out = {}
    for name, (ix, sd) in sets.items():
        r = fidelity(arena, S, P, A, ix, sd, H, a.horizon, a.device,
                     driver=driver)
        db = r["deal_pred"] - r["deal_real"]
        tb = r["take_pred"] - r["take_real"]
        out[name] = {**r, "deal_bias": db, "take_bias": tb, "n": int(len(ix))}
        print(f"  {name:<20} {r['deal_pred']:10.1f} {r['deal_real']:7.1f} "
              f"{db:+7.1f}   {r['take_pred']:10.1f} {r['take_real']:7.1f} "
              f"{tb:+7.1f}")
    print("\ngame HP over the WINDOW (an RL gain quoted per step is 1/horizon "
          "of this).\n`bias` positive = the simulator predicts MORE damage "
          "than really happened.\nA constant bias cancels in agent-minus-"
          "reference; what it marks is a region\nwhere the simulator is "
          "confused, and so a region where an action-dependent\nerror the agent "
          "can exploit is more likely. Check those against the real game\n"
          "before building on them; quote the low-bias gyms freely.")
    print("\n  net bias (deal - take), the part that does NOT cancel if it "
          "varies with actions:")
    for name, r in sorted(out.items(), key=lambda kv: -abs(kv[1]["deal_bias"]
                                                           - kv[1]["take_bias"])):
        nb = r["deal_bias"] - r["take_bias"]
        print(f"    {name:<20} {nb:+7.1f} HP/window  ({nb/a.horizon:+.1f} "
              f"HP/step)")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
