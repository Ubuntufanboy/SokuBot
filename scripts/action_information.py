"""I_k = I(A_t ; F_{t+k} | F_t), estimated from the corpus. No world model.

    python -m scripts.action_information --cache ~/rl/mv2_corpus.npz --out ik.json

WHAT THIS IS FOR
----------------
Every claim about the simulator being action-insensitive needs a denominator:
how much information about the future IS in the action, given the present? That
is a property of the game and the corpus, not of any model, and until it is
measured "the effect is small" is a statement without a scale.

THE ESTIMATOR
-------------
    I(A; Y | X) = H(Y | X) - H(Y | X, A)

so two predictors are fitted per horizon -- one given the state, one given the
state and the action -- and the drop in conditional entropy IS the information,
in nats. Both are the same architecture, same budget, same data, same
early-stopped fit; only the input differs.

  binary channels     H is cross-entropy, so the difference is an exact
                      conditional-MI estimate for that channel.
  continuous channels Gaussian entropy, so the difference is
                      0.5 * log(var_base / var_act).

This is a LOWER BOUND. A predictor that fits worse leaves information on the
table and understates I. That direction is the safe one for the argument being
made with it -- a lower bound on what is available, compared against what the
simulator actually uses, can only understate the simulator's shortfall.

WHY TWO CONTEXT LENGTHS
-----------------------
`--context 1` is the quantity as written, conditioning on the current frame.
`--context 12` conditions on the same history the simulator gets. The gap
between them is the information-theoretic statement of the measured result that
a 12-step model responds to buttons half as much as a 2-step one: conditioning
on more past makes the action more redundant, and this says by exactly how many
nats rather than by anecdote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, STATE_CHANNELS
from sokubot.model.state_head import BINARY, CONTINUOUS

_BINSET = {STATE_CHANNELS[i] for i in BINARY}
REPORT = ("guarding", "hp", "x", "vx", "airborne", "spirit", "dx", "untech")


class Predictor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(),
            nn.Linear(width, out_dim))

    def forward(self, x):
        return self.net(x)


def fit(Xtr, Ytr, Xva, Yva, binr, steps, lr, dev, bs=1024, seed=0):
    """-> per-channel held-out conditional entropy, in nats."""
    torch.manual_seed(seed)
    m = Predictor(Xtr.shape[1], Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(seed)
    is_bin = torch.zeros(Ytr.shape[1], dtype=torch.bool, device=dev)
    is_bin[binr] = True
    for _ in range(steps):
        ix = torch.as_tensor(rng.integers(0, len(Xtr), bs), device=dev)
        pred = m(Xtr[ix])
        tgt = Ytr[ix]
        lb = F.binary_cross_entropy_with_logits(
            pred[:, is_bin], tgt[:, is_bin]) if is_bin.any() else 0.0
        lc = F.mse_loss(pred[:, ~is_bin], tgt[:, ~is_bin])
        (lb + lc).backward()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        out = torch.cat([m(Xva[i:i + 4096]) for i in range(0, len(Xva), 4096)])
        H = torch.zeros(Ytr.shape[1], device=dev)
        # Binary: cross-entropy in nats is the conditional entropy directly.
        if is_bin.any():
            H[is_bin] = F.binary_cross_entropy_with_logits(
                out[:, is_bin], Yva[:, is_bin], reduction="none").mean(0)
        # Continuous: Gaussian entropy of the residual, constants cancel in the
        # difference so only 0.5*log(var) is kept.
        res = (out[:, ~is_bin] - Yva[:, ~is_bin]).var(0).clamp(min=1e-12)
        H[~is_bin] = 0.5 * torch.log(res)
    return H.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 24, 32])
    ap.add_argument("--contexts", type=int, nargs="+", default=[1, 12])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--actionable", action="store_true",
                    help="restrict to frames where player 0 CAN act. Without "
                         "this the average runs over the 65%% of frames in "
                         "untech, hitstop or knockdown, where the input "
                         "provably cannot matter -- so the unconditioned "
                         "number is a statement about how often the game lets "
                         "you act, mixed into one about what acting does.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    d = np.load(a.cache)
    S, A, E = d["S"], d["A"], d["E"]
    N, _, C = S.shape
    flat = S.reshape(N, -1)
    mu, sd = flat.mean(0), flat.std(0).clip(1e-6)
    binr = np.array(BINARY)
    # Targets are player 0's channels. Standardised for the continuous ones so
    # the Gaussian term is comparable across channels; binary left as 0/1.
    tgt_cols = np.arange(C)
    out = {"horizons": a.horizons, "contexts": a.contexts,
           "actionable_only": bool(a.actionable),
           "channels": list(STATE_CHANNELS), "report": list(REPORT), "I": {}}

    for ctx in a.contexts:
        for k in a.horizons:
            span = ctx + k
            ok = np.ones(N, bool); ok[-span:] = False
            for j in range(1, span + 1):
                ok[:N - j] &= (E[j:] == E[:N - j])
            if a.actionable:
                z = S[:, 0]
                ok &= ((z[:, CH["untech"]] <= 0) & (z[:, CH["hitstop"]] <= 0)
                       & (z[:, CH["knockdown"]] < 0.5)
                       & (z[:, CH["crushed"]] < 0.5))
            pool = np.flatnonzero(ok)
            idx = np.random.default_rng(0).choice(
                pool, min(a.n, len(pool)), replace=False)
            w = idx[:, None] + np.arange(ctx)[None, :]
            X = flat[w].reshape(len(idx), -1)
            X = (X - np.tile(mu, ctx)) / np.tile(sd, ctx)
            # The action block at t: both players' buttons over the frame skip.
            Aq = A[idx + ctx - 1].reshape(len(idx), -1)
            Y = S[idx + ctx - 1 + k, 0][:, tgt_cols].copy()
            for c in CONTINUOUS:
                Y[:, c] = (Y[:, c] - mu[c]) / sd[c]

            cut = int(len(idx) * 0.85)
            dev = a.device
            T = lambda z: torch.as_tensor(z, dtype=torch.float32, device=dev)
            Xtr, Xva = T(X[:cut]), T(X[cut:])
            XAtr = T(np.concatenate([X[:cut], Aq[:cut]], 1))
            XAva = T(np.concatenate([X[cut:], Aq[cut:]], 1))
            Ytr, Yva = T(Y[:cut]), T(Y[cut:])

            Hb = np.mean([fit(Xtr, Ytr, Xva, Yva, binr, a.steps, a.lr, dev,
                              seed=s) for s in range(a.seeds)], 0)
            Ha = np.mean([fit(XAtr, Ytr, XAva, Yva, binr, a.steps, a.lr, dev,
                              seed=s) for s in range(a.seeds)], 0)
            I = np.maximum(Hb - Ha, 0.0)     # negative means the fit, not info
            # H_base is kept, not just the difference. Without it there is no
            # denominator: `guarding` has a 4.5% base rate, so its whole
            # entropy is 0.185 nats and an absolute 0.014 looks negligible
            # beside position's 0.10 -- while being a comparable SHARE of what
            # there was to explain. Reporting only I ranks channels by how
            # common they are.
            out["I"][f"ctx{ctx}_k{k}"] = {
                "total_nats": float(I.sum()),
                "per_channel": {STATE_CHANNELS[i]: float(I[i])
                                for i in range(C)},
                "H_base": {STATE_CHANNELS[i]: float(Hb[i]) for i in range(C)},
                "frac_explained": {
                    STATE_CHANNELS[i]: float(I[i] / Hb[i])
                    if STATE_CHANNELS[i] in _BINSET and Hb[i] > 1e-9 else None
                    for i in range(C)}}
            print(f"ctx {ctx:2d} k {k:2d} | total {I.sum():7.4f} nats | "
                  + " ".join(f"{c} {I[CH[c]]:+.4f}" for c in REPORT),
                  flush=True)
            a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
