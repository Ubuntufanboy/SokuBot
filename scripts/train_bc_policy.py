"""Train the GRPO policy to imitate real play, as an initialisation for it.

    python -m scripts.train_bc_policy --bank /root/grpo_prior/bank.npz

WHY THIS AND NOT THE EARLIER BC HEAD
------------------------------------
The head trained alongside the decoder reached 0.910 accuracy against a 0.896
base rate and a mean per-button AUC of 0.564, which is a coin flip with a lean.
Three things were wrong with it and are fixed here.

It predicted twenty independent Bernoullis. The game stores `lr` and `ud` as
signed integers, so left+right and up+down cannot co-occur and no such frame
exists in the corpus; a model spending capacity on impossible outputs is a model
not spending it on the distinction that matters. This trains the same factored
distribution `SokuPolicy` samples from -- two 3-way categoricals and six
independent buttons.

It used unweighted cross-entropy on a signal that is positive a tenth of the
time, where "never press anything" is a strong local optimum worth 0.90
accuracy. Positive-class weighting removes that optimum.

And it shared an optimiser with a decoder at half loss weight, so it was never
the objective being optimised.

WHAT IT IS FOR
--------------
Two things, and the second matters more than the first.

A policy that plays like the corpus keeps imagined rollouts inside the
distribution the world model was trained on. Uniform initialisation holds 44% of
buttons per tick against a human 9.85%, so every rollout from it is
extrapolation, and GRPO optimising against extrapolation is the leading
explanation for why it collapses to something arbitrary and ties random.

And it is a competent starting point rather than a random one. GRPO is then
improvement on top of imitation, which is the usual shape of these pipelines,
instead of discovery from nothing inside a model whose causal beliefs are known
to be wrong in at least one respect.

The architecture is `SokuPolicy` exactly, so the result loads straight into
train_grpo with --init.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.probe import _auc
from sokubot.rl.policy import (FREE_BUTTONS, IDX_DOWN, IDX_LEFT, IDX_RIGHT,
                               IDX_UP, SokuPolicy)

BUTTON_NAMES = ("a", "b", "c", "d", "change", "spell")


def targets_from(actions: torch.Tensor):
    """[N,ticks,10] one-hot buttons -> (lr index, ud index, six free buttons)."""
    lr = (actions[..., IDX_LEFT] > 0.5).long() + 2 * (actions[..., IDX_RIGHT] > 0.5).long()
    ud = (actions[..., IDX_UP] > 0.5).long() + 2 * (actions[..., IDX_DOWN] > 0.5).long()
    return lr, ud, actions[..., list(FREE_BUTTONS)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, default=Path("/root/grpo_prior/bank.npz"))
    ap.add_argument("--wm", type=Path, default=Path("/root/ckpt_cf/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("/root/bc"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    cfg: Config = torch.load(a.wm, map_location="cpu", weights_only=False)["cfg"]
    bank = np.load(a.bank)
    Z = torch.from_numpy(bank["z"]).float()
    A = torch.from_numpy(bank["a"]).float()
    ep = bank["ep"]
    H = cfg.history

    # Windows must not straddle a replay boundary, and the split is by replay so
    # neighbouring near-duplicate frames cannot appear on both sides of it.
    edges = np.flatnonzero(np.diff(ep)) + 1
    starts = []
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ep)]):
        if hi - lo > H:
            starts.append(np.arange(lo + H - 1, hi))
    starts = np.concatenate(starts)
    eps_of = ep[starts]
    uniq = np.unique(ep)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * a.val_frac))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in eps_of])
    tr_idx = torch.from_numpy(starts[~is_val])
    va_idx = torch.from_numpy(starts[is_val])
    print(f"{len(tr_idx)} train windows, {len(va_idx)} val windows from "
          f"{len(uniq)} replays ({n_val} held out)", flush=True)

    dev = torch.device(a.device)
    Z, A = Z.to(dev), A.to(dev)
    tr_idx, va_idx = tr_idx.to(dev), va_idx.to(dev)
    off = torch.arange(H, device=dev) - (H - 1)

    policy = SokuPolicy(cfg.latent_dim, H, cfg.action_ticks).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    # Each button's own rate, so a spell card pressed 0.6% of the time is not
    # weighted like a direction held 28% of the time.
    rates = A.reshape(-1, cfg.action_dim)[:, list(FREE_BUTTONS)].mean(0)
    pos_weight = ((1 - rates) / rates.clamp_min(1e-4)).clamp(max=50.0)
    print("positive weights: " + ", ".join(
        f"{n} {w:.1f}" for n, w in zip(BUTTON_NAMES, pos_weight.tolist())), flush=True)

    def batch(idx_pool, n):
        sel = idx_pool[torch.randint(len(idx_pool), (n,), device=dev)]
        z = Z[sel[:, None] + off[None, :]]
        return z, A[sel][..., :10]           # the agent is P1 in the bank layout

    def loss_on(z, act):
        d_lr, d_ud, d_btn = policy.distributions(
            z, torch.zeros(len(z), dtype=torch.long, device=dev))
        t_lr, t_ud, t_btn = targets_from(act)
        l = (F.cross_entropy(d_lr.logits.reshape(-1, 3), t_lr.reshape(-1))
             + F.cross_entropy(d_ud.logits.reshape(-1, 3), t_ud.reshape(-1))
             + F.binary_cross_entropy_with_logits(d_btn.logits, t_btn,
                                                  pos_weight=pos_weight))
        return l, (d_lr, d_ud, d_btn), (t_lr, t_ud, t_btn)

    @torch.no_grad()
    def evaluate(n=32768):
        policy.eval()
        z, act = batch(va_idx, n)
        l, (d_lr, d_ud, d_btn), (t_lr, t_ud, t_btn) = loss_on(z, act)
        lr_acc = float((d_lr.logits.argmax(-1) == t_lr).float().mean())
        ud_acc = float((d_ud.logits.argmax(-1) == t_ud).float().mean())
        # Majority-class baselines, which is what the earlier head was scoring.
        lr_base = float(torch.bincount(t_lr.reshape(-1), minlength=3).max() / t_lr.numel())
        ud_base = float(torch.bincount(t_ud.reshape(-1), minlength=3).max() / t_ud.numel())
        probs = torch.sigmoid(d_btn.logits).reshape(-1, len(FREE_BUTTONS)).cpu().numpy()
        truth = t_btn.reshape(-1, len(FREE_BUTTONS)).cpu().numpy()
        aucs = {n_: _auc(probs[:, i], truth[:, i])
                for i, n_ in enumerate(BUTTON_NAMES)}
        policy.train()
        return {"val_loss": float(l), "lr_acc": lr_acc, "lr_base": lr_base,
                "ud_acc": ud_acc, "ud_base": ud_base, "auc": aucs,
                "auc_mean": float(np.nanmean(list(aucs.values())))}

    hist, t0 = [], time.time()
    best = -1.0
    for step in range(1, a.steps + 1):
        z, act = batch(tr_idx, a.batch_size)
        l, _, _ = loss_on(z, act)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.log_every == 0 or step == 1:
            ev = evaluate()
            ev.update({"step": step, "train_loss": float(l),
                       "elapsed_h": (time.time() - t0) / 3600})
            hist.append(ev)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            if ev["auc_mean"] > best:
                best = ev["auc_mean"]
                torch.save({"policy": policy.state_dict(), "cfg": cfg,
                            "step": step, "auc_mean": best}, a.out / "policy.pt")
            print(f"step {step:6d} | train {float(l):.4f} val {ev['val_loss']:.4f} "
                  f"| lr {ev['lr_acc']:.3f}/{ev['lr_base']:.3f} "
                  f"ud {ev['ud_acc']:.3f}/{ev['ud_base']:.3f} "
                  f"| AUC {ev['auc_mean']:.4f}", flush=True)

    ev = hist[-1]
    print(f"\nbest mean AUC {best:.4f} (the earlier head managed 0.564)")
    print("per-button AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in ev["auc"].items()))
    print(f"-> {a.out/'policy.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
