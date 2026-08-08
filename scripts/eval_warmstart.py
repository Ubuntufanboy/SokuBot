"""Score a warm-started model at the new resolution **before** any training.

    python -m scripts.eval_warmstart --ckpt /root/wm_cf_bnfix.pt \
        --val /root/corpus/val.pt --image-size 448

WHY THIS IS THE EXPERIMENT THAT MATTERS
---------------------------------------
Four 448 training runs, spanning a 4x learning-rate sweep and a 10x change in the
auxiliary loss weights, all share one signature: **the best held-out score is the
first eval, at step 4000, and everything after it is worse.**

    lr 2e-4, aux full   step 4000  -3.08   step 8000  -0.36
    lr 1e-4, aux full   step 4000  +0.657  step 8000  +0.304
    lr 5e-5, aux full   step 4000  +0.437  step 8000  -0.638
    lr 1e-4, aux /10    step 4000  +0.701  step 8000  -1.23

Step 4000 is the eval closest to initialisation. So the hypothesis is not about
learning rates or loss weights at all: the 224 model, with its positional
embedding re-gridded to a 32x32 grid, may already be a good 448 model, and
training at 448 may simply be degrading it.

That is testable in a minute and settles what four hours of training could not.
If step-0 skill is close to the step-4000 numbers, the 448 "retrain" has been
buying nothing and the deliverable is the re-gridded checkpoint itself.

WHAT IT DOES
------------
Exactly what `train_full`'s warm start does -- re-grid `encoder.pos_embed`,
load everything else untouched -- then recalibrate BatchNorm against the val
cache and evaluate. No optimiser, no steps. The BatchNorm pass is not optional:
the running statistics were estimated at 224 over 256 patches, and reading them
at 1024 patches without re-estimation measures the stale statistics rather than
the model (`docs/BUGS.md` §1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sokubot.config import Config
from sokubot.model.encoder import resize_pos_embed
from sokubot.model.world_model import LeWorldModel
from scripts.eval_ckpt import recalibrate_bn
from scripts.train_full import evaluate, eval_batch_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("warmstart_eval.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = dict(blob["model"])
    old_cfg = blob.get("cfg")
    old_size = getattr(old_cfg, "image_size", a.image_size)

    cfg = (Config.soku448() if a.image_size == 448
           else Config.soku(image_size=a.image_size))
    cfg.device = a.device
    # The checkpoint may predate the HUD head; build the model to match what the
    # weights actually contain, so the load is strict about everything else.
    cfg.hud_coef = 0.0 if "hud_head.weight" not in sd else cfg.hud_coef

    if old_size != a.image_size:
        og, ng = old_size // cfg.patch_size, a.image_size // cfg.patch_size
        sd["encoder.pos_embed"] = resize_pos_embed(sd["encoder.pos_embed"], og, ng)
        print(f"re-gridded pos_embed {og}x{og} -> {ng}x{ng} "
              f"({old_size} -> {a.image_size} px)")
    else:
        print(f"no re-grid needed; checkpoint is already {old_size} px")

    model = LeWorldModel(cfg).to(a.device)
    missing = model.load_state_dict(sd, strict=False)
    if missing.missing_keys:
        print(f"  freshly initialised: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"  ignored: {missing.unexpected_keys}")

    cache = torch.load(a.val, map_location="cpu", weights_only=False)
    n = len(cache["obs"])
    eb = a.eval_batch or eval_batch_for(cfg)
    print(f"val cache: {n} windows, batch {eb}")

    before = evaluate(model, cache, cfg, batch=eb)
    print(f"\nas loaded          skill {before['skill']:+.4f} | "
          f"val {before['val_pred']:.4f} | identity {before['identity']:.4f}")
    recalibrate_bn(model, cache, batch=eb)
    after = evaluate(model, cache, cfg, batch=eb)
    print(f"BatchNorm re-est.  skill {after['skill']:+.4f} | "
          f"val {after['val_pred']:.4f} | identity {after['identity']:.4f}")

    print("\n" + "=" * 68)
    print("Compare against the step-4000 evals of the training runs. If this is "
          "close to\nthem, training at the new resolution is not adding anything "
          "and the\ndeliverable is this checkpoint, re-gridded and BN-"
          "recalibrated.")
    a.out.write_text(json.dumps(
        {"ckpt": str(a.ckpt), "image_size": a.image_size, "steps_trained": 0,
         "as_loaded": before, "bn_recalibrated": after}, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
