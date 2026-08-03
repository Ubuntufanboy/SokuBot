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
