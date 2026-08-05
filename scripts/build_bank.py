"""Encode a bank of start states for one checkpoint, and stamp it with that
checkpoint's fingerprint.

    python -m scripts.build_bank --ckpt /root/ckpt_cf/best_bnfix.pt \
        --out /root/bank_bnfix.npz

`train_grpo.build_bank` already does this inline, but the diagnostics
(`action_effect_test`, `aggression_test`, `latent_drift_test`) all read a bank
they cannot build, so the ones on disk were whatever the last training run
happened to leave behind. `bank_best.npz` and `bank_cf.npz` both predate the
fingerprint guard and carry no record of which weights encoded them.

Latents mean nothing except relative to the encoder that produced them, so this
refuses to write a bank without stamping it, and prints the one-step skill of
the pairing it just created as a receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from scripts.eval_ckpt import predictor_skill
from scripts.train_grpo import build_bank, model_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()
    print(f"{a.ckpt}  fingerprint {model_fingerprint(wm)}  "
          f"recorded skill {blob.get('skill')}", flush=True)

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    np.random.default_rng(a.seed).shuffle(rows)
    Z, A, E = build_bank(rows, manifest, wm, cfg, a.device, a.replays, a.out)

    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)
    sk = predictor_skill(wm, Zt, At, E, cfg)
    print(f"\n{len(Z)} latents from {E.max() + 1} replays")
    print(f"one-step skill of this checkpoint on this bank: {sk:+.4f}")
    if sk <= 0:
        print("WARNING: the predictor is no better than copying the previous "
              "latent. Rolling this forward produces noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
