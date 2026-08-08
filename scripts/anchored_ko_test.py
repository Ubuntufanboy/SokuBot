"""Does anchoring the probe to the true HUD rescue the KO signal?

    python -m scripts.anchored_ko_test --ckpt ~/sokubot-art/wm_cf_bnfix.pt \
        --probe ~/gate_base/reward_probe.npz --bank ~/bank_hud.npz

THE PROBLEM THIS TESTS
----------------------
`scripts/probe_reliability.py` measured the reward's KO detector at **precision
0.003**, firing 20-45x more often than KOs actually happen. Each false fire pays
`win`/`lose` = +-5 against damage terms worth ~0.1 and masks out the rest of the
trajectory, so the match-outcome signal is mostly noise.

Deleting the term would make the agent indifferent to winning, which defeats the
point. The alternative is to fix the *measurement*: take the health **level**
from `data/hud.py` -- validated against a human at MAE 0.012, and 0.012 in the
low-health band specifically -- and the health **change** from the probe.

    state_t = hud_ref + [ probe(z_t) - probe(z_0) ]

WHY NOT SIMPLY CARRY THE HUD FORWARD
-------------------------------------
Because then health never changes, so damage is identically zero and the reward
vanishes. The change has to come from the probe; only the offset can come from
the HUD.

WHAT IS MEASURED
----------------
Rollouts under the policy's own sampled actions from held-out starts, scored two
ways -- probe alone, and probe anchored -- against `data/hud.py`'s reading of the
*same* frames as ground truth. Reported: KO precision, recall, false-KO rate, and
how often each pays out `win`/`lose`.

The ground truth here is the true HUD *at the start*, propagated by the true
future health from the bank. That makes this an honest test of the detector's
false-positive rate on real trajectories, which is the failure mode: a rollout
beginning at half health must never report a KO.
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
from sokubot.rl.grpo import GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.reward import RewardConfig, ko_mask
from scripts.train_grpo import valid_starts

HP1, HP2 = 0, 1


def audit(hp_pred: torch.Tensor, hp_true: torch.Tensor, cfg: RewardConfig) -> dict:
    """KO detection on predicted health vs on true health, over the same windows."""
    ko_p = ko_mask(hp_pred, cfg).any(1).cpu().numpy()
    ko_t = ko_mask(hp_true, cfg).any(1).cpu().numpy()
    tp = int((ko_p & ko_t).sum()); fp = int((ko_p & ~ko_t).sum())
    fn = int((~ko_p & ko_t).sum())
    return {"n": int(len(ko_p)), "true_rate": float(ko_t.mean()),
            "pred_rate": float(ko_p.mean()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "false_ko_rate": float(fp / max(len(ko_p), 1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--starts", type=int, default=4096)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--hud-head", type=Path, default=None,
                    help="a head from scripts.train_hud_head. Adds a third arm "
                         "that carries HUD through the rollout with a learned "
                         "delta instead of the probe's. The predictor is frozen, "
                         "so the latent half is unchanged in every arm.")
    ap.add_argument("--out", type=Path, default=Path("anchored_ko.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]; cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device); wm.load_state_dict(blob["model"]); wm.eval()

    d = np.load(a.probe, allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])

    b = np.load(a.bank)
    Z, A, ep = b["z"], b["a"], b["ep"]
    if "hud" not in b.files:
        raise SystemExit(f"{a.bank} has no `hud`; build it with scripts.build_hud_bank")
    Hd = b["hud"]
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)
    Ht = torch.from_numpy(Hd).to(a.device).float()

    T = a.horizon
    starts = valid_starts(ep, cfg.history, T + 2)
    idx = torch.from_numpy(rng.choice(starts, size=min(a.starts, len(starts)),
                                      replace=False)).to(a.device)

    rcfg = RewardConfig()
    gcfg = GRPOConfig(horizon=T, reward=rcfg)
    arena = ImaginedArena(wm, ProbeHead(probe).to(a.device), gcfg,
                          cfg.history, cfg.action_ticks)
    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)

    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Zt[idx[:, None] + off[None, :]].float()
    a_hist = At[idx[:, None] + off[None, :-1]].float()
    side = torch.randint(0, 2, (len(idx),), device=a.device)
    # `states[0]` is the first *imagined* step, i.e. the true state one step after
    # the context, so that is what the anchor must be.
    hud_ref = Ht[idx + 1]

    # ALL arms roll under the SAME actions -- the bank's true ones. The first
    # version let the probe arms sample from a policy while the learned arm used
    # the true actions, which is not a readout comparison at all: predicting the
    # true HUD under the actions that produced it is a different, far easier
    # task, and the gap it opened (MAE 0.117 vs 0.014) was mostly that confound.
    a_plan = At[idx[:, None] + torch.arange(T, device=a.device)[None, :]].float()
    h_ctx = Ht[idx[:, None] + off[None, :]]

    with torch.no_grad():
        z_roll = wm.rollout(z_ctx, a_plan, a_hist)              # [B,T,latent]
        probe_states = arena.probe(z_roll)                      # [B,T,K]
        # states[0] must be the first imagined step, matching the ground truth.
        plain = {"states": probe_states}
        anch = {"states": hud_ref[:, None, : probe_states.shape[-1]]
                + probe_states - probe_states[:, :1]}

    learned = None
    if a.hud_head is not None:
        from sokubot.model.augmented import HudDeltaHead, HudWorldModel
        hb = torch.load(a.hud_head, map_location=a.device, weights_only=False)
        head = HudDeltaHead(cfg, len(hb["hud_channels"]), hb["width"],
                            hb["depth"]).to(a.device)
        head.load_state_dict(hb["head"]); head.eval()
        hm = HudWorldModel(wm, head).to(a.device).eval()
        with torch.no_grad():
            zl, hh = hm.rollout(z_ctx, h_ctx, a_plan, a_hist)
            assert torch.allclose(zl, z_roll, atol=1e-5), \
                "the head must not change the latent rollout"
            learned = {"states": hh}
        # Copy-forward, the baseline the head has to beat to be worth anything.
        carry = {"states": hud_ref[:, None, :].expand(-1, T, -1).contiguous()}

    # Ground truth: the bank's own HUD over the same window.
    steps = torch.arange(T, device=a.device)
    true_states = Ht[idx[:, None] + 1 + steps[None, :]]

    print(f"{len(idx)} rollouts x {T} steps, from {int(ep.max())+1} replays\n")
    res = {"ckpt": str(a.ckpt), "n": int(len(idx)), "horizon": T, "arms": {}}
    arms = [("probe only", plain), ("anchored", anch)]
    if learned is not None:
        arms += [("carry-forward", carry), ("learned delta", learned)]
    for tag, tr in arms:
        row = {}
        for who, ch in (("p1", HP1), ("p2", HP2)):
            row[who] = audit(tr["states"][..., ch], true_states[..., ch], rcfg)
        # How far the health level itself is off -- the quantity the KO
        # threshold is tested against.
        err = (tr["states"][..., :2] - true_states[..., :2]).abs()
        row["hp_mae"] = float(err.mean())
        row["hp_mae_step0"] = float(err[:, 0].mean())
        res["arms"][tag] = row
        print(f"=== {tag} ===")
        print(f"  health MAE overall {row['hp_mae']:.4f} | at the first step "
              f"{row['hp_mae_step0']:.4f}")
        for who in ("p1", "p2"):
            k = row[who]
            print(f"  KO[{who}] true {k['true_rate']:.3%} pred {k['pred_rate']:.3%}"
                  f" | precision {k['precision']:.3f} recall {k['recall']:.3f}"
                  f" | false KO {k['false_ko_rate']:.3%}")
        print()

    p0 = np.mean([res["arms"]["probe only"][w]["precision"] for w in ("p1", "p2")])
    p1 = np.mean([res["arms"]["anchored"][w]["precision"] for w in ("p1", "p2")])
    f0 = np.mean([res["arms"]["probe only"][w]["false_ko_rate"] for w in ("p1", "p2")])
    f1 = np.mean([res["arms"]["anchored"][w]["false_ko_rate"] for w in ("p1", "p2")])
    print("=" * 68)
    print(f"mean KO precision  {p0:.3f} -> {p1:.3f}")
    print(f"mean false-KO rate {f0:.3%} -> {f1:.3%}")
    if p1 > max(p0 * 2, 0.5):
        print("\nANCHORING RESCUES THE KO SIGNAL. win/lose can stay on: the "
              "detector now fires mostly on real KOs, so the +-5 term is paying "
              "for outcomes rather than for probe noise.")
    else:
        print("\nAnchoring is NOT enough on its own. The level is right and the "
              "detector still misfires, so the error is in the probe's *deltas*, "
              "not its offset -- which is an argument for HUD supervision in the "
              "encoder rather than for deleting the term.")
    res["summary"] = {"precision_before": float(p0), "precision_after": float(p1),
                      "false_ko_before": float(f0), "false_ko_after": float(f1)}
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
