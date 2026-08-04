"""Does the imagined world respond to what the agent does?

    python -m scripts.action_sensitivity --ckpt /root/ckpt/best.pt

This is the precondition for every form of RL inside a world model, and it is
not the same question the skill metric answers. A predictor can score well at
"what happens next" by learning the momentum of the scene while ignoring its
action conditioning entirely -- most of the next frame is determined by the
current one. Such a model is useful for prediction and useless for control: no
policy can change the reward, so the advantage of every rollout in a group is
measurement noise and the gradient points nowhere.

The test rolls the *same* start state forward under deliberately different
action sequences and asks how far the probed game state diverges. Two spreads
are compared at each horizon:

  across-action   std of probed health over different action sequences from one
                  identical start. This is the part a policy can control.
  across-start    std over different start states. This is the part it cannot.

The ratio is what matters. If across-action is a small fraction of across-start,
the model is telling us that what you do barely matters compared to where you
are, and GRPO in imagination cannot work however well the reward is designed.

A neutral-action control is included: rolling the same start forward twice under
identical actions must diverge by exactly zero, which confirms the measurement
is reading action sensitivity rather than sampling noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import ProbeHead
from sokubot.rl.policy import IDX_LEFT, IDX_RIGHT, IDX_UP

HP1, HP2 = 0, 1


def action_set(ticks: int, dim: int, device) -> dict[str, torch.Tensor]:
    """A handful of maximally different things one player can do."""
    def blank():
        return torch.zeros(ticks, dim, device=device)

    out = {"neutral": blank()}
    for name, col in (("left", IDX_LEFT), ("right", IDX_RIGHT), ("up", IDX_UP),
                      ("attack_a", 4), ("attack_b", 5), ("attack_c", 6)):
        a = blank()
        a[:, col] = 1.0
        out[name] = a
    a = blank()                       # everything at once: the loudest input
    a[:, [IDX_RIGHT, 4, 5, 6]] = 1.0
    out["all_in"] = a
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon2/reward_probe.npz"))
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo/bank.npz"))
    ap.add_argument("--starts", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/action_sensitivity.json"))
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    d = np.load(a.probe, allow_pickle=True)
    head = ProbeHead(LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"],
                                 ysd=d["ysd"], W=d["W"],
                                 names=[str(x) for x in d["names"]])).to(a.device)

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(a.device)
    ep = bank["ep"]
    lo = cfg.history - 1
    ok = np.flatnonzero(np.diff(ep) == 0)
    ok = ok[(ok >= lo) & (ok < len(ep) - a.horizon - 1)]
    rng = np.random.default_rng(0)
    idx = torch.from_numpy(rng.choice(ok, size=a.starts, replace=False)).to(a.device)
    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Z[idx[:, None] + off[None, :]].float()

    acts = action_set(cfg.action_ticks, cfg.action_dim, a.device)
    names = list(acts)
    S, P = a.starts, a.horizon

    # [n_actions, starts, horizon, 2] -- the agent is P1 throughout, so the
    # opponent's ten columns stay zero and only the agent's input varies.
    traces = []
    for name in names:
        a_plan = acts[name][None, None].expand(S, P, cfg.action_ticks, cfg.action_dim)
        a_hist = torch.zeros(S, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                             device=a.device)
        zi = wm.rollout(z_ctx, a_plan.contiguous(), a_hist)
        traces.append(head(zi)[..., [HP1, HP2]])
    T = torch.stack(traces)

    # Control: the same actions twice must produce identical latents.
    rep = wm.rollout(z_ctx, acts["neutral"][None, None].expand(
        S, P, cfg.action_ticks, cfg.action_dim).contiguous(),
        torch.zeros(S, cfg.history - 1, cfg.action_ticks, cfg.action_dim,
                    device=a.device))
    determinism = float((head(rep)[..., [HP1, HP2]] - T[0]).abs().max())

    print(f"world model step {blob.get('step','?')} | {S} starts | "
          f"{len(names)} action sequences | horizon {P}")
    print(f"determinism check (want exactly 0): {determinism:.3e}\n")
    print(f"{'h':>3} {'sec':>5} {'across-action':>14} {'across-start':>13} "
          f"{'ratio':>8} {'max pair gap':>13}")
    print("-" * 62)

    curve = []
    for h in range(P):
        across_action = float(T[:, :, h].std(dim=0).mean())
        across_start = float(T[:, :, h].mean(dim=0).std(dim=0).mean())
        # Biggest divergence any two action sequences produce from one start.
        gap = float((T[:, :, h].max(dim=0).values - T[:, :, h].min(dim=0).values).mean())
        curve.append({"h": h + 1, "seconds": (h + 1) * cfg.frame_skip / 60,
                      "across_action": across_action, "across_start": across_start,
                      "ratio": across_action / (across_start + 1e-12),
                      "max_pair_gap": gap})
        if h + 1 <= 4 or (h + 1) % 4 == 0:
            c = curve[-1]
            print(f"{c['h']:3d} {c['seconds']:5.2f} {across_action:14.5f} "
                  f"{across_start:13.5f} {c['ratio']:8.4f} {gap:13.5f}")

    final = curve[-1]
    print("-" * 62)
    print(f"per-action mean health at h={P} (P1 / P2):")
    for i, n in enumerate(names):
        print(f"   {n:10s} {float(T[i, :, -1, 0].mean()):+.4f} "
              f"{float(T[i, :, -1, 1].mean()):+.4f}")
    print()
    r = final["ratio"]
    if r < 0.02:
        verdict = ("the model is effectively action-blind; imagined-rollout RL "
                   "cannot work on this checkpoint")
    elif r < 0.10:
        verdict = ("actions move the imagined state far less than the start does; "
                   "expect a very weak learning signal")
    else:
        verdict = "actions meaningfully steer the imagined state"
    print(f"RATIO AT FULL HORIZON: {r:.4f}  ->  {verdict}")
    a.out.write_text(json.dumps({"determinism": determinism, "actions": names,
                                 "curve": curve}, indent=2))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
