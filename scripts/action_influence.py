"""Action Influence Curve: how far apart do two futures drift, given one state?

    python -m scripts.action_influence ~/rl/mv2/base/sim.pt --names h12 \
        --cache ~/rl/mv2_corpus.npz --out aic.json

    delta_k(s, A, B) = d(F^A_{t+k}, F^B_{t+k})
    AIC_k            = E_{s,A,B}[delta_k]
    H_eps(s, A, B)   = min { k : delta_k > eps }

Rolled from ONE start state with two different action sequences, so the only
thing that differs between the branches is the button. This is the quantity RL
actually depends on -- not "can the model predict the future", but "does the
future it predicts depend on what the agent does".

FOUR DISTANCES, AND WHAT IS AND IS NOT AVAILABLE
------------------------------------------------
The requested `d_pixel` and `d_perceptual` need a decoder. This simulator
predicts state, not frames, and there is no state->pixel renderer in the
project, so reporting them would mean inventing them. These four are the ones
that are real here:

  d_state    L2 over all channels in per-channel corpus sigma. The blunt
             instrument: everything counts, like a pixel metric does.
  d_kin      position and velocity only -- the part a human would SEE differ.
             Closest in spirit to d_pixel: two futures that differ here look
             different on screen.
  d_latent   L2 in the simulator's own trunk activations, i.e. the model's
             internal notion of "a different situation". The analogue of a
             perceptual metric, and it is the model's perception, which is the
             relevant one when asking what the model can act on.
  d_reward   |R(F^A) - R(F^B)| under the training reward. THE IMPORTANT ONE:
             two futures can be nearly identical in every other metric and
             differ by a knockdown. A model whose d_reward is flat cannot
             support policy learning however good its d_state looks.

H_eps IS EXPECTED TO BE BIMODAL, NOT NOISY
-------------------------------------------
Some inputs do nothing whatever the horizon -- pressing attack during hitstun,
holding a direction while airborne and committed. Others resolve in a frame.
So the mean of H_eps is close to meaningless and the DISTRIBUTION is the
result, including the censored share that never crosses eps at all. Reporting
a mean here would hide exactly the structure worth seeing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.soku import BUTTONS
from sokubot.data.state import CH, STATE_CHANNELS
from sokubot.model.state_dynamics import feed_proj, load_sim, hold_dead
from sokubot.model.state_head import BINARY

KIN = ("x", "y", "vx", "vy", "dx", "dy")


def branch_actions(a0, dx, kind, rng):
    """Two action blocks over the whole rollout, differing by the branch."""
    L, R = BUTTONS.index("left"), BUTTONS.index("right")
    a = a0.clone()
    a[:, :, :, :] = 0.0
    if kind == "away_toward":
        left = dx > 0
        b = a.clone()
        a[left, :, :, L] = 1.0; a[~left, :, :, R] = 1.0      # away
        b[left, :, :, R] = 1.0; b[~left, :, :, L] = 1.0      # toward
        return a, b
    if kind == "attack_idle":
        b = a.clone()
        a[:, :, :, BUTTONS.index("b")] = 1.0
        return a, b
    if kind == "jump_idle":
        b = a.clone()
        a[:, :, :, BUTTONS.index("up")] = 1.0
        return a, b
    if kind == "random":
        b = a.clone()
        for t in (a, b):
            t[:, :, :, :10] = torch.as_tensor(
                rng.random(t[:, :, :, :10].shape) < 0.15,
                dtype=t.dtype, device=t.device)
        return a, b
    raise ValueError(kind)


@torch.no_grad()
def roll(model, s, p, acts, steps, want_latent=False):
    """Rollout returning states and, optionally, trunk activations per step."""
    H = s.shape[1]
    binr = torch.tensor(BINARY, device=s.device)
    cur_s, cur_p = s, p
    states, lats = [], []
    for k in range(steps):
        a = acts[:, k:k + H]
        if want_latent:
            B, T = cur_s.shape[:2]
            x = torch.cat([cur_s.reshape(B, T, -1), cur_p.reshape(B, T, -1),
                           a.reshape(B, T, -1)], -1)
            h = model.embed(x) + model.pos[:, :T]
            for blk in model.blocks:
                h = blk(h, causal=True)
            lats.append(model.norm(h)[:, -1])
        ns, np_ = model(cur_s, cur_p, a)
        nxt = ns[:, -1:].index_copy(
            -1, binr, torch.sigmoid(ns[:, -1:].index_select(-1, binr)))
        # Its own loop, so it must hold dead channels itself (state_dynamics.hold_dead).
        nxt = hold_dead(nxt, cur_s[:, -1:], model)
        cur_s = torch.cat([cur_s[:, 1:], nxt], 1)
        cur_p = torch.cat([cur_p[:, 1:],
                           feed_proj(np_[:, -1:], model.proj_feedback)], 1)
        states.append(cur_s[:, -1])
    return (torch.stack(states, 1),
            torch.stack(lats, 1) if want_latent else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sims", type=Path, nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--n", type=int, default=1536)
    ap.add_argument("--near", type=float, default=250.0)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    names = a.names or [p.parent.name for p in a.sims]

    d = np.load(a.cache)
    S, P, A, E = d["S"], d["P"], d["A"], d["E"]
    sig = S.reshape(-1, S.shape[-1]).std(0)
    # A CHANNEL THAT NEVER VARIES CANNOT BE NORMALISED BY ITS OWN SPREAD.
    #
    # `ax`, the horizontal gravity term, has a corpus standard deviation of
    # EXACTLY zero. Clipping it to 1e-6 and dividing turns any float difference
    # between two rollouts into ~1e6, and the first version of this script duly
    # reported d_state of 22,326 for one model and 0.29 for another -- five
    # orders of magnitude of pure artifact, from one dead channel out of 33.
    # Same trap as the pooled MSE that was 57.8% `action_frame`.
    live = sig > 1e-3
    dead = [STATE_CHANNELS[i] for i in range(len(sig)) if not live[i]]
    if dead:
        print(f"  excluding dead channels from d_state: {dead}", flush=True)
    sig = np.where(live, sig, 1.0)
    sig_t = torch.as_tensor(sig, dtype=torch.float32, device=a.device)
    live_t = torch.as_tensor(np.flatnonzero(live), device=a.device)
    kin = torch.tensor([CH[c] for c in KIN], device=a.device)
    hp, gd = CH["hp"], CH["guarding"]
    out = {"steps": a.steps, "eps": a.eps, "runs": {}}

    for name, path in zip(names, a.sims):
        model, meta = load_sim(path, a.device)
        H = int(meta["history"])
        span = H + a.steps
        ok = np.ones(len(S), bool); ok[-span:] = False
        for j in range(1, span + 1):
            ok[:len(S) - j] &= (E[j:] == E[:len(S) - j])
        ok &= np.abs(S[:, 0, CH["dx"]]) < (a.near / 1200.0)
        idx = np.random.default_rng(11).choice(
            np.flatnonzero(ok), min(a.n, int(ok.sum())), replace=False)
        w = idx[:, None] + np.arange(span)[None, :]
        s = torch.as_tensor(S[w][:, :H]).float().to(a.device)
        p = torch.as_tensor(P[w][:, :H]).float().to(a.device)
        a0 = torch.as_tensor(A[w]).float().to(a.device)
        dx = s[:, -1, 0, CH["dx"]]
        rng = np.random.default_rng(3)

        for kind in ("away_toward", "attack_idle", "jump_idle", "random"):
            aA, aB = branch_actions(a0, dx, kind, rng)
            sA, lA = roll(model, s, p, aA, a.steps, True)
            sB, lB = roll(model, s, p, aB, a.steps, True)
            dstate = (((sA - sB) / sig_t).index_select(-1, live_t)
                      ).norm(dim=-1).mean(-1)                       # [B,K]
            dkin = ((sA - sB).index_select(-1, kin)
                    / sig_t.index_select(0, kin)).norm(dim=-1).mean(-1)
            dlat = (lA - lB).norm(dim=-1) / (lA.norm(dim=-1) + 1e-6)
            # Reward divergence: the damage exchange the training reward reads,
            # per step, as a fraction of a health bar.
            rA = sA[:, :, 1, hp] - sA[:, :, 0, hp]
            rB = sB[:, :, 1, hp] - sB[:, :, 0, hp]
            drew = (rA - rB).abs()
            ent = {}
            for tag, dd in (("d_state", dstate), ("d_kin", dkin),
                            ("d_latent", dlat), ("d_reward", drew)):
                cur = dd.detach().cpu().numpy()
                ent[tag] = {"aic": cur.mean(0).tolist(),
                            "p50": np.median(cur, 0).tolist(),
                            "p90": np.percentile(cur, 90, 0).tolist()}
                # H_eps: first crossing per start state, with the never-crossed
                # share reported rather than dropped.
                he = {}
                scale = cur.max() if tag == "d_latent" else 1.0
                for e in a.eps:
                    thr = e * (cur.max() if tag != "d_reward" else 0.02)
                    cross = cur > thr
                    first = np.where(cross.any(1), cross.argmax(1) + 1, -1)
                    he[str(e)] = {
                        "median_k": float(np.median(first[first > 0]))
                        if (first > 0).any() else None,
                        "censored_frac": float((first < 0).mean()),
                        "hist": np.bincount(first[first > 0],
                                            minlength=a.steps + 1).tolist()}
                ent[tag]["H_eps"] = he
            out["runs"][f"{name}/{kind}"] = ent
            print(f"{name:8s} {kind:13s} "
                  f"d_state k1 {dstate[:,0].mean():.4f} k8 {dstate[:,7].mean():.4f} "
                  f"k32 {dstate[:,-1].mean():.4f} | "
                  f"d_reward k8 {drew[:,7].mean():.5f} k32 {drew[:,-1].mean():.5f}",
                  flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
