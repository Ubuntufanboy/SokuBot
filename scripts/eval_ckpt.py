"""Evaluate a checkpoint with BatchNorm statistics recalibrated first.

    python -m scripts.eval_ckpt --ckpt /root/ckpt/sokubot.pt --val /root/corpus/val.pt

WHY RECALIBRATION
-----------------
Both projectors end in a non-affine BatchNorm. In `eval()` mode it normalises by
running statistics accumulated with momentum 0.1 over past batches. Early in a
long cosine schedule the learning rate is near maximum and the encoder's output
distribution moves faster than those statistics track it, so evaluation
normalises by numbers that describe a model from several hundred steps ago.

The symptom is unmistakable and was observed on the 200 h run: training loss
moving smoothly (0.034 -> 0.050 -> 0.058) while held-out loss bounced
0.127 -> 1.143 -> 0.079. A held-out loss above 1.0 means predictions worse than
outputting the mean, which no model does while its training loss is 0.058.
Nothing is wrong with the model; the yardstick is stale.

Recalibration fixes it: run forward passes in `train()` mode under no_grad so
the running statistics re-estimate against the *current* weights, then evaluate.
This changes only how activations are normalised at eval time -- no gradients,
no parameter updates.

Using the held-out inputs to re-estimate them is safe here. The statistics are
per-channel means and variances of the inputs; the task is self-supervised, so
there are no labels to leak, and the model and the copy-last-frame baseline are
both normalised the same way, keeping the comparison fair.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from scripts.train_full import evaluate


@torch.no_grad()
def recalibrate_bn(model: LeWorldModel, cache: dict, batch: int = 64,
                   passes: int = 1) -> int:
    """Re-estimate BatchNorm running stats against the current weights."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    if not bns:
        return 0
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # None => cumulative average, not exponential
    model.train()                  # train mode updates running stats
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]
    for _ in range(passes):
        for i in range(0, len(O), batch):
            model(O[i : i + batch].to(device), A[i : i + batch].to(device).float())
    model.eval()
    return len(bns)


@torch.no_grad()
def predictor_skill(model, Z, A, ep, cfg, n: int = 8192, seed: int = 0) -> float:
    """One-step `1 - MSE_model / MSE_identity` against a bank of real latents.

    The same quantity `train_full.evaluate` reports, computed from a bank so it
    costs nothing and needs no video decode. At or below zero the predictor is
    no better than copying the previous latent forward.
    """
    import numpy as np
    dev = next(model.parameters()).device
    H = cfg.history
    same = np.zeros(len(ep), dtype=bool)
    same[: len(ep) - (H + 1)] = ep[: len(ep) - (H + 1)] == ep[H + 1 :]
    ok = np.flatnonzero(same)
    ok = ok[ok >= H - 1]
    idx = torch.from_numpy(np.random.default_rng(seed).choice(
        ok, size=min(n, len(ok)), replace=False)).to(dev)
    off = torch.arange(H, device=dev) - (H - 1)
    se_m = se_i = 0.0
    for s in range(0, len(idx), 4096):
        base = idx[s : s + 4096]
        zw = Z[base[:, None] + off[None, :]].float()
        acts = A[base[:, None] + off[None, :]].float()
        zhat = model.predictor(zw, model.action_encoder(acts))[:, -1]
        zt = Z[base + 1].float()
        se_m += float(((zhat - zt) ** 2).mean(1).sum())
        se_i += float(((zw[:, -1] - zt) ** 2).mean(1).sum())
    return 1.0 - se_m / se_i


def assert_predictor_sane(model, Z, A, ep, cfg, floor: float = 0.0,
                          what: str = "checkpoint") -> float:
    """Refuse to roll a predictor that cannot beat copying the last latent.

    `ckpt_cf/best.pt` was saved with BatchNorm running statistics that had never
    been recalibrated -- the fine-tune scored a recalibrated *copy* and wrote the
    original -- and measured skill -6.15 while its own blob recorded +0.82. It
    was then used as the default by the horizon ablation, the reward probe it
    fits, and every rollout diagnostic downstream. Nothing failed loudly; the
    numbers just quietly stopped meaning anything. One line here would have
    caught it, so here is the line.
    """
    sk = predictor_skill(model, Z, A, ep, cfg)
    if sk < floor:
        raise SystemExit(
            f"{what} has one-step skill {sk:+.4f}, at or below the floor "
            f"{floor:+.2f}: its predictor is no better than copying the "
            f"previous latent, so any rollout through it is noise. If this is "
            f"a fine-tuned checkpoint, its BatchNorm running statistics are "
            f"probably stale -- see scripts.eval_ckpt.recalibrate_bn.")
    return sk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=Path("/root/corpus/val.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cache = torch.load(args.val, map_location="cpu", weights_only=False)
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = args.device
    step = blob.get("step", "?")

    model = LeWorldModel(cfg).to(args.device)
    model.load_state_dict(blob["model"])

    stale = evaluate(model, cache, cfg)
    n = recalibrate_bn(copy.deepcopy(model).to(args.device), cache)  # warm check
    model2 = LeWorldModel(cfg).to(args.device)
    model2.load_state_dict(blob["model"])
    recalibrate_bn(model2, cache)
    fresh = evaluate(model2, cache, cfg)

    if args.json:
        print(json.dumps({"step": step, "stale": stale, "recalibrated": fresh}, indent=2))
        return
    print(f"checkpoint step {step}  ({n} BatchNorm layers recalibrated)")
    print(f"{'':16s} {'val_pred':>9} {'identity':>9} {'skill':>9}")
    print(f"{'running stats':16s} {stale['val_pred']:9.4f} {stale['identity']:9.4f} "
          f"{stale['skill']:+9.4f}")
    print(f"{'recalibrated':16s} {fresh['val_pred']:9.4f} {fresh['identity']:9.4f} "
          f"{fresh['skill']:+9.4f}")


if __name__ == "__main__":
    main()
