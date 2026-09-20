"""Per-channel, per-step rollout error for one or more simulators.

    python -m scripts.sim_channel_table ~/rl/mv_base/sim.pt ~/rl/mv_move/sim.pt \
        --names base move --corpus ~/corpus --replays 400 --seed 0

WHY THIS EXISTS RATHER THAN THE AGGREGATE SKILL NUMBER
-------------------------------------------------------
`train_state_dynamics` reports skill pooled over kinematic channels, and that
number cannot show the thing this project's central open question turns on.
Every channel but one degrades smoothly with horizon, which is what compounding
autoregressive error looks like. `guarding` is already at half a sigma by step
TWO and then flat -- flat-from-the-start is the signature of a variable the
model cannot see at all, and a mean over 28 channels hides it completely.

Errors are reported in each channel's own corpus standard deviation, because
the channels are incommensurable: `action_frame` reaches magnitudes of 441
while `hp` is a fraction of one. A pooled raw MSE over them is 57.8%
`action_frame` and measures nothing else -- three runs were abandoned on that
number before it was caught.

The validation split is RECONSTRUCTED with the trainer's own rule (by replay,
`rng(seed).choice(n_ep, n_ep // 8)`), so the windows scored here are windows no
model in the comparison was trained on. Passing a different --replays or --seed
than the run used silently scores on training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_state_dynamics import cached_sequences
from sokubot.data.state import STATE_CHANNELS
from sokubot.model.state_dynamics import index_moves, load_sim, rollout

STEPS = (1, 2, 4, 8, 16, 24, 32)
SHOW = ("guarding", "hp", "x", "dx", "spirit", "airborne", "untech",
        "action_frame", "vx", "combo_hits")
# Flags, whose head emits logits and whose sigma-error is dominated by
# calibration rather than by knowledge -- see `flag_skill`.
FLAGS = ("guarding", "airborne")


def flag_skill(logit: torch.Tensor, target: torch.Tensor,
               base: float) -> tuple[float, float]:
    """-> (BCE skill against the base rate, AUC). Two numbers, because one of
    them cannot answer the question on its own.

    The sigma table cannot separate knowledge from calibration for a rare flag.
    Measured on fix5, `guarding` sits at 2.48 sigma of error at EVERY horizon
    -- but the base rate is 3.2% and the model emits ~0.45, so nearly all of
    that number is one constant offset.

    BCE skill does not fix that: cross-entropy punishes a confident wrong
    probability, so a model that ranks guarding frames perfectly and calibrates
    badly still scores far below zero. fix5 reports -2.26, which says its
    guarding output is worse than a constant -- true, and useless for deciding
    WHY.

    AUC is the one that answers it. It reads only the ordering, so any constant
    offset or squashing leaves it unchanged: 0.5 is "this output carries no
    information about when the flag is set", and that -- not the magnitude of
    the error -- is the claim about partial observability.
    """
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="mean")
    b = torch.full_like(logit, float(np.log(base / (1 - base))))
    ref = F.binary_cross_entropy_with_logits(b, target, reduction="mean")
    skill = float(1.0 - bce / ref.clamp(min=1e-9))

    y = (target.reshape(-1) > 0.5)
    x = logit.reshape(-1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return skill, float("nan")
    # Rank-sum AUC, ties averaged -- the flags are predicted from a linear head
    # and exact ties do occur.
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return skill, float(auc)


def load_corpus(a):
    """Parsed corpus arrays, via the trainer's own verified cache so an
    evaluation and the run it scores can never disagree about their input."""
    return cached_sequences(a.corpus, a.replays, a.slots, a.frame_skip, a.cache)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sims", type=Path, nargs="+")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=400)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--flag-feedback", default="sigmoid",
                    choices=("sigmoid", "hard", "sample"),
                    help="what the binary channels look like when the rollout "
                         "eats its own output; see state_dynamics._unroll_impl")
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz to hold the parsed corpus. Parsing 400 replays "
                         "of CSV is the fragile step -- on the 1080 Ti box it "
                         "segfaulted inside libpython on 4 of 7 attempts with "
                         "identical arguments -- so it is done ONCE, verified "
                         "by read-back, and every later run reads the artifact")
    a = ap.parse_args()
    names = a.names or [p.parent.name for p in a.sims]
    if len(names) != len(a.sims):
        raise SystemExit("--names must match the number of simulators")

    S, P, A, E, M = load_corpus(a)
    n_ep = int(E.max()) + 1
    val_ep = set(np.random.default_rng(a.seed).choice(
        n_ep, max(1, n_ep // 8), replace=False).tolist())
    vm = np.isin(E, list(val_ep))
    # Sigma comes from the WHOLE corpus, not the validation slice, so the unit
    # is the same one the training run reported and the two are comparable.
    sig = S.reshape(-1, S.shape[-1]).std(0).clip(1e-6)
    Sv, Pv, Av, Ev, Mv = S[vm], P[vm], A[vm], E[vm], M[vm]
    print(f"  val {len(Sv)} steps / {len(val_ep)} replays", flush=True)

    rows, flags, aucs = {}, {}, {}
    for name, path in zip(names, a.sims):
        model, meta = load_sim(path, a.device)
        H = int(meta["history"])
        span = H + max(STEPS)
        ok = np.ones(len(Sv), bool); ok[-span:] = False
        for k in range(1, span + 1):
            ok[:len(Sv) - k] &= (Ev[k:] == Ev[:len(Sv) - k])
        # Same windows for every simulator in the comparison: the seed is fixed
        # here rather than taken from the loop, so two models are never scored
        # on two different draws and called different.
        idx = np.random.default_rng(1234).choice(
            np.flatnonzero(ok), min(a.windows, int(ok.sum())), replace=False)
        w = idx[:, None] + np.arange(span)[None, :]
        s = torch.as_tensor(Sv[w]).to(a.device)
        p = torch.as_tensor(Pv[w]).to(a.device)
        act = torch.as_tensor(Av[w]).to(a.device)
        mv = None
        if model.move_dim:
            vocab = meta.get("move_vocab")
            if vocab is None:
                raise SystemExit(f"{path}: move model with no saved vocabulary")
            mv = torch.as_tensor(index_moves(Mv[w][:, :H], np.asarray(vocab))
                                 ).to(a.device)
        pred = rollout(model, s[:, :H], p[:, :H], act, max(STEPS), mv,
                       a.flag_feedback)

        tab, fsk, fauc = {}, {}, {}
        for ch in FLAGS:
            i = STATE_CHANNELS.index(ch)
            base = float(np.clip(S[:, :, i].mean(), 1e-4, 1 - 1e-4))
            for h in STEPS:
                fsk[(ch, h)], fauc[(ch, h)] = flag_skill(
                    pred[:, h - 1, :, i], s[:, H + h - 1, :, i], base)
        for h in STEPS:
            pr, tg = pred[:, h - 1], s[:, H + h - 1]
            for ch in SHOW:
                i = STATE_CHANNELS.index(ch)
                v = pr[:, :, i]
                # Flags come out of the head as logits; comparing a logit to a
                # 0/1 label would report a large error for a correct model.
                if ch in ("guarding", "airborne"):
                    v = torch.sigmoid(v)
                tab[(ch, h)] = float((v - tg[:, :, i]).abs().mean()) / sig[i]
        rows[name] = tab
        flags[name] = fsk
        aucs[name] = fauc
        print(f"  {name}: history {H} ticks {meta['ticks']} "
              f"move_dim {model.move_dim} n_moves {model.n_moves}", flush=True)

    hdr = "".join(f"{h:>8d}" for h in STEPS)
    for name in names:
        print(f"\n{name} -- mean |error| in units of each channel's sigma")
        print(f"{'channel':<14}{hdr}")
        for ch in SHOW:
            print(f"{ch:<14}" + "".join(f"{rows[name][(ch, h)]:8.3f}"
                                        for h in STEPS))
    print("\nflag AUC -- 0.5 means the output says nothing about WHEN the "
          "flag is set")
    print(f"{'model/flag':<22}{hdr}")
    for name in names:
        for ch in FLAGS:
            print(f"{name + '/' + ch:<22}" + "".join(
                f"{aucs[name][(ch, h)]:8.3f}" for h in STEPS))

    print("\nflag BCE skill vs the base rate (calibration AND ranking)")
    print(f"{'model/flag':<22}{hdr}")
    for name in names:
        for ch in FLAGS:
            print(f"{name + '/' + ch:<22}" + "".join(
                f"{flags[name][(ch, h)]:+8.4f}" for h in STEPS))

    if len(names) == 2:
        b, m = names
        print(f"\ndelta ({m} - {b}); negative is better for {m}")
        print(f"{'channel':<14}{hdr}")
        for ch in SHOW:
            print(f"{ch:<14}" + "".join(
                f"{rows[m][(ch, h)] - rows[b][(ch, h)]:+8.3f}" for h in STEPS))
        print(f"\ndelta flag AUC ({m} - {b}); POSITIVE is better for {m}")
        print(f"{'flag':<14}{hdr}")
        for ch in FLAGS:
            print(f"{ch:<14}" + "".join(
                f"{aucs[m][(ch, h)] - aucs[b][(ch, h)]:+8.4f}"
                for h in STEPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
