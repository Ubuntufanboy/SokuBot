"""Which buttons can the world model actually feel?

    python -m scripts.button_sensitivity --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --bank ~/bank_hud.npz --probe ~/gate_base/reward_probe.npz

WHY PER BUTTON, WHEN action_sensitivity ALREADY EXISTS
-------------------------------------------------------
`scripts/action_sensitivity.py` asks whether the imagined world responds to the
agent *at all*, aggregated over whole action sequences. It answers yes, and that
was enough while the question was "can RL work here".

The question now is narrower and the aggregate hides it. `scripts/block_effect.py`
found the model registers *whether* a horizontal direction is held (37% of a
latent standard deviation) but almost not *which* one (4.6%) -- and blocking,
which is the single mechanic most responsible for the agent losing, is entirely
about which. An aggregate sensitivity that pools LEFT with the attack buttons
reports a healthy number while the specific channel the agent needs is dead.

So this measures one button at a time, and the three mechanics worth mastering
map onto it directly:

    blocking, dodging   LEFT / RIGHT / DOWN     (directional, positional)
    combos              A / B / C / D           (attack buttons)
    movement            UP                      (flight, the Soku-specific one)

WHERE THE BLINDNESS LIVES
-------------------------
Two different faults produce the same symptom, and they need different fixes, so
both are measured:

  **conditioning**  the action encoder maps the button to a vector that barely
                    differs from its absence. Then no predictor could use it, and
                    the fix is upstream of the world model entirely.
  **predictor**     the conditioning vector is distinct but the predictor's
                    output barely moves. Then the information is present and
                    unused, which is a training-objective problem.

Reported as `cond` and `latent` below. A button with a large `cond` and a small
`latent` is being ignored; one with a small `cond` cannot be expressed.

READ THE SCALE, NOT THE NUMBER
-------------------------------
Latent displacement is reported as a fraction of the across-batch spread of the
latent itself, because an L2 of 0.04 means nothing until you know the states
being separated are 0.93 apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.model.loading import load_world_model
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import GRPOConfig, ImaginedArena, ProbeHead
from scripts.eval_policy import RCFG
from scripts.train_grpo import valid_starts

BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C", "D", "CHANGE", "SPELL")
HP1 = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--starts", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=Path("button_sensitivity.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    wm, cfg, _ = load_world_model(a.wm, a.device)
    b = np.load(a.bank.expanduser())
    Z = torch.from_numpy(b["z"]).to(a.device)
    At = torch.from_numpy(b["a"]).to(a.device)
    E = b["ep"]

    d = np.load(a.probe.expanduser(), allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    arena = ImaginedArena(wm, ProbeHead(probe).to(a.device),
                          GRPOConfig(horizon=a.horizon, reward=RCFG),
                          cfg.history, cfg.action_ticks)

    T = a.horizon
    starts = valid_starts(E, cfg.history, T)
    rng = np.random.default_rng(a.seed)
    idx = torch.from_numpy(rng.choice(starts, size=a.starts, replace=False)).to(a.device)
    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Z[idx[:, None] + off[None, :]].float()
    a_hist = At[idx[:, None] + off[None, :-1]].float()
    base = At[idx[:, None] + torch.arange(T, device=a.device)[None, :]].float()

    print(f"{len(idx)} starts x {T} steps | the opponent's true inputs are held "
          f"fixed in every arm,\nso only the agent's own button varies.\n")

    with torch.no_grad():
        # Reference: the agent presses nothing at all. Every button is measured
        # as a departure from the same place, so the numbers are comparable to
        # each other rather than to whatever the human happened to be doing.
        off_plan = base.clone()
        off_plan[..., :10] = 0.0
        z_off = wm.rollout(z_ctx, off_plan, a_hist)
        spread = float(z_off[:, -1].std(0).mean())
        hp_off = arena.probe(z_off)[:, -1, HP1]

        # Conditioning vectors, for the same two action chunks.
        cond_off = wm.action_encoder(off_plan.reshape(len(idx) * T, -1)
                                     if off_plan.dim() == 3 else off_plan)

    rows = []
    with torch.no_grad():
        for i, name in enumerate(BUTTONS):
            on_plan = off_plan.clone()
            on_plan[..., i] = 1.0
            z_on = wm.rollout(z_ctx, on_plan, a_hist)
            l2 = float((z_on[:, -1] - z_off[:, -1]).norm(dim=-1).mean())
            cos = float(F.cosine_similarity(z_on[:, -1], z_off[:, -1], dim=-1).mean())
            hp_on = arena.probe(z_on)[:, -1, HP1]
            dhp = float((hp_on - hp_off).mean())
            cond_on = wm.action_encoder(on_plan.reshape(len(idx) * T, -1)
                                        if on_plan.dim() == 3 else on_plan)
            dcond = float((cond_on - cond_off).norm(dim=-1).mean()
                          / (cond_off.norm(dim=-1).mean() + 1e-9))
            rows.append({"button": name, "latent_l2": l2,
                         "latent_frac_of_spread": l2 / max(spread, 1e-9),
                         "cosine": cos, "d_health": dhp, "cond_rel": dcond})

    rows.sort(key=lambda r: -r["latent_frac_of_spread"])
    print("  button    latent move   cond move   d(P1 health)")
    print("            (% of spread)  (relative)")
    for r in rows:
        print(f"  {r['button']:<8} {r['latent_frac_of_spread']:11.1%} "
              f"{r['cond_rel']:11.3f}  {r['d_health']:+13.5f}")
    print(f"\n  latent spread across starts, for scale: {spread:.4f}")

    res = {"wm": str(a.wm), "horizon": T, "n": len(idx), "spread": spread,
           "buttons": rows}
    by = {r["button"]: r for r in rows}
    lr = 0.5 * (by["LEFT"]["latent_frac_of_spread"] + by["RIGHT"]["latent_frac_of_spread"])
    atk = np.mean([by[k]["latent_frac_of_spread"] for k in ("A", "B", "C", "D")])
    res["directional_vs_attack"] = float(lr / max(atk, 1e-9))
    print("\n" + "=" * 68)
    print(f"directional (LEFT/RIGHT) {lr:.1%} vs attack (A-D) {atk:.1%} "
          f"-- ratio {lr / max(atk, 1e-9):.2f}")
    if lr < 0.5 * atk:
        print("The model feels attacks far more than directions. Blocking and "
              "dodging are\nboth directional, so those two mechanics are the "
              "ones it cannot currently\nlearn -- and a gym for either would "
              "train against noise.")
    else:
        print("Directional and attack sensitivity are comparable; the "
              "directional channel\nis not obviously the bottleneck.")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
