"""What does a real player score on the reward GRPO is optimising?

    python -m scripts.human_baseline --ckpt /root/ckpt_cf/best_bnfix.pt

GRPO reports numbers like "net vs frozen init +0.00162" and "as P1 +0.0085
dealt / -0.0057 taken". Those are damage in health-bar units over a four-step
window of 0.27 s, and on their own they mean nothing: there is no way to tell
whether +0.0085 is strong play, weak play, or noise without knowing what a
person scores on the same scale.

This measures that. Real captures, real buttons, the same reward function and
the same window length, so the agent's numbers can be read against a human's.

TWO ARMS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
-------------------------------------------------
  real       Probe the actual encoder latents of what happened. This is what a
             human earned in the real game, as our reward function reads it.
             The honest yardstick, and it is also a check on the reward: if
             real play scores near zero here, the reward is not measuring the
             game.

  imagined   Feed the human's recorded buttons into the world model from the
             same start and score the rollout. This is the human's play *as
             GRPO's environment sees it*, which is the arm the agent's numbers
             are directly comparable to -- the agent is only ever scored inside
             the world model, so comparing it to real-game damage would flatter
             or punish it for the model's own error.

The gap between the two arms is itself worth reading: it is how much the
imagined environment distorts a known-good policy.

WINNERS AND LOSERS ARE SEPARATED
--------------------------------
Averaging over everyone measures the average player, and half of those lost.
The number the agent actually has to clear is the winner's, so the split is
reported. Winners are decided by which health bar is higher at the end of the
capture, read with the same probe.

Damage is also reported per minute, because 0.0085 per 0.27 s is not a quantity
anyone has intuition for, and a health bar per minute is.
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
from sokubot.rl.reward import RewardConfig, compute_rewards


def load_head(path: Path, device) -> ProbeHead:
    d = np.load(path, allow_pickle=True)
    return ProbeHead(LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"],
                                 ysd=d["ysd"], W=d["W"],
                                 names=[str(x) for x in d["names"]])).to(device)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt_cf/best_bnfix.pt"))
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo_h4c/bank.npz"))
    ap.add_argument("--probe", type=Path,
                    default=Path("/root/horizon_bnfix/reward_probe.npz"))
    ap.add_argument("--probe-encoder", type=Path,
                    default=Path("/root/horizon_bnfix/reward_probe_encoder.npz"),
                    help="probe fit on encoder latents, for the `real` arm. The "
                         "calibrated one is fit on predictor outputs and is the "
                         "right instrument only for the imagined arm.")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--samples", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("/root/human_baseline.json"))
    a = ap.parse_args()
    dev, H = a.device, a.horizon

    blob = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = dev
    wm = LeWorldModel(cfg).to(dev)
    wm.load_state_dict(blob["model"])
    wm.eval()

    head = load_head(a.probe, dev)
    head_enc = load_head(a.probe_encoder, dev) if a.probe_encoder.exists() else head
    if head_enc is head:
        print("note: no encoder-fit probe found; using the calibrated one for "
              "both arms, which slightly misreads the real latents\n")

    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).to(dev)
    A = torch.from_numpy(bank["a"]).to(dev)
    ep = bank["ep"]

    # Winner per replay. Not from the final frame: captures do not all end on
    # the deciding hit, and the probe carries a small constant bias between the
    # hp1 and hp2 readouts, which on final-frame health alone declared P1 the
    # winner of all 200 replays. The lowest point each bar reaches over the
    # whole capture is far more robust -- the loser is whoever got closest to
    # zero -- and the bias mostly cancels in the comparison of two minima.
    hp_all = []
    for s0 in range(0, len(Z), 65536):
        hp_all.append(head_enc(Z[s0 : s0 + 65536].float())[:, :2].cpu())
    hp_all = torch.cat(hp_all).numpy()
    edges = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
    winner, margin = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m1, m2 = hp_all[lo_:hi_, 0].min(), hp_all[lo_:hi_, 1].min()
        winner.append(0 if m1 > m2 else 1)
        margin.append(abs(m1 - m2))
    winner, margin = np.array(winner), np.array(margin)
    print(f"{len(winner)} replays; P1 won {(winner == 0).sum()}, "
          f"P2 won {(winner == 1).sum()}  "
          f"(median margin {np.median(margin):.3f} bar)")

    lo = cfg.history - 1
    ok = np.zeros(len(ep), dtype=bool)
    ok[lo : len(ep) - H - 1] = ep[lo : len(ep) - H - 1] == ep[lo + H + 1 :]
    idx_all = np.flatnonzero(ok)
    rng = np.random.default_rng(0)
    idx = rng.choice(idx_all, size=min(a.samples, len(idx_all)), replace=False)
    rep_of = ep[idx]
    print(f"{len(idx)} windows of {H} steps ({H*cfg.frame_skip/60:.2f} s) "
          f"from {len(idx_all)} eligible\n")

    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020)
    off = torch.arange(cfg.history, device=dev) - (cfg.history - 1)
    steps = torch.arange(H + 1, device=dev)

    acc = {}
    for arm in ("real", "imagined"):
        for side in (0, 1):
            acc[(arm, side)] = {"dealt": [], "taken": [], "ret": []}

    idx_t = torch.from_numpy(idx).to(dev)
    for s in range(0, len(idx_t), a.chunk):
        base = idx_t[s : s + a.chunk]
        B = len(base)
        # Actions really pressed over the window, [B, H+1, ticks, 20].
        acts = A[base[:, None] + steps[None, :]].float()

        # --- real: probe what actually happened -------------------------
        z_real = Z[base[:, None] + steps[None, :]].float()      # [B,H+1,latent]
        st_real = head_enc(z_real)

        # --- imagined: the same buttons through the world model ---------
        z_win = Z[base[:, None] + off[None, :]].float()
        a_win = A[base[:, None] + off[None, :]][:, :-1].float()
        zs = []
        for t in range(H + 1):
            act = acts[:, t]
            zhat = wm.predictor(z_win, wm.action_encoder(
                torch.cat([a_win, act[:, None]], dim=1)))[:, -1]
            zs.append(zhat)
            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            if cfg.history > 1:
                a_win = torch.cat([a_win[:, 1:], act[:, None]], dim=1)
        st_imag = head(torch.stack(zs, dim=1))

        for arm, states in (("real", st_real), ("imagined", st_imag)):
            for side in (0, 1):
                sd = torch.full((B,), side, dtype=torch.long, device=dev)
                r, alive, terms = compute_rewards(states, acts[:, 1:], sd, rcfg)
                acc[(arm, side)]["dealt"].append(terms["dealt"].sum(1).cpu())
                acc[(arm, side)]["taken"].append(terms["taken"].sum(1).cpu())
                acc[(arm, side)]["ret"].append(r.sum(1).cpu())

    won = torch.from_numpy((winner[rep_of] == 0).astype(bool))   # P1 won this rep
    per_min = 60.0 / (H * cfg.frame_skip / 60.0)

    res = {"horizon": H, "seconds": H * cfg.frame_skip / 60.0,
           "windows": len(idx), "per_minute_factor": per_min, "arms": {}}
    print(f"{'arm':9} {'who':16} {'dealt':>9} {'taken':>9} {'net':>9} "
          f"{'net/min':>9}")
    print("-" * 68)
    for arm in ("real", "imagined"):
        for side in (0, 1):
            d = torch.cat(acc[(arm, side)]["dealt"])
            t = torch.cat(acc[(arm, side)]["taken"])
            # This side won its match: for side 0 that is `won`, for side 1 the
            # complement. Scoring the winner's play is the bar the agent has to
            # clear, not the average of everybody including the person who lost.
            w = won if side == 0 else ~won
            for label, m in (("all", torch.ones_like(w)), ("winners", w),
                             ("losers", ~w)):
                if m.sum() == 0:
                    continue
                dd, tt = float(d[m].mean()), float(t[m].mean())
                res["arms"][f"{arm}/P{side+1}/{label}"] = {
                    "dealt": dd, "taken": tt, "net": dd + tt,
                    "net_per_min": (dd + tt) * per_min, "n": int(m.sum())}
                print(f"{arm:9} {f'P{side+1} {label}':16} {dd:+9.4f} {tt:+9.4f} "
                      f"{dd+tt:+9.4f} {(dd+tt)*per_min:+9.3f}")
    print("-" * 68)

    ra = res["arms"]
    g = lambda k, f="dealt": ra[k][f] if k in ra else float("nan")
    mean2 = lambda a_, b_: np.nanmean([a_, b_])

    rl = mean2(g("real/P1/all"), g("real/P2/all"))
    ia = mean2(g("imagined/P1/all"), g("imagined/P2/all"))
    wn = mean2(g("real/P1/winners"), g("real/P2/winners"))
    res["summary"] = {"real_dealt": rl, "imagined_dealt": ia,
                      "winner_real_dealt": wn, "distortion": ia / max(rl, 1e-9)}
    print(f"\nHUMAN THROUGHPUT, averaged over both chairs so the probe's "
          f"P1/P2 bias cancels:")
    print(f"  in the real game        {rl:.4f} per {res['seconds']:.2f} s "
          f"= {rl*per_min:.2f} bars/min")
    print(f"  through the world model {ia:.4f} per {res['seconds']:.2f} s "
          f"= {ia*per_min:.2f} bars/min   ({ia/max(rl,1e-9):.2f}x)")
    print(f"  winners only, real      {wn:.4f}")
    print(f"\nCompare GRPO's `as P1 +x` against the *imagined* figure: the "
          f"agent is only ever scored inside the world model.")
    print(f"Net is near zero for both humans by construction -- two people of "
          f"similar skill trade evenly -- so damage *dealt* is the throughput "
          f"to compare, and net only means something against a fixed opponent.")
    a.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
