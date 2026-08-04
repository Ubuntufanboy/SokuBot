"""Is the behaviour-cloned policy actually weak, or just badly scored?

    python -m scripts.eval_bc --heads /root/heads/heads.pt --wm /root/ckpt/best.pt

Accuracy on this task is close to meaningless. Buttons are held roughly a tenth
of the time, so "never press anything" scores about 0.90, and the training log's
0.910 against a 0.896 base rate is consistent both with a policy that has
learned a little and with one that has learned nothing and simply drifted to
the majority class.

AUC separates those. It asks whether the head ranks frames where a button *was*
pressed above frames where it was not, which is unaffected by class imbalance
and by where the decision threshold sits. Per button, because the answer will
not be uniform: a policy might predict movement well and attacks not at all.

  AUC ~0.50   no signal; the head is a constant and cannot be an opponent
  AUC ~0.60   weak but real; usable with positive-class weighting
  AUC >0.70   genuinely predictive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.soku import ACTION_COLUMNS
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import _auc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heads", type=Path, default=Path("/root/heads/heads.pt"))
    ap.add_argument("--wm", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--val", type=Path, default=Path("/root/corpus/val.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()

    wmb = torch.load(a.wm, map_location=a.device, weights_only=False)
    cfg: Config = wmb["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(wmb["model"])
    wm.eval()

    hb = torch.load(a.heads, map_location=a.device, weights_only=False)
    print(f"heads checkpoint step {hb.get('step','?')}", flush=True)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_heads import BCPolicy

    bc = BCPolicy(cfg.latent_dim, cfg.seq_len).to(a.device)
    bc.load_state_dict(hb["bc"])
    bc.eval()

    cache = torch.load(a.val, map_location="cpu", weights_only=False)
    O, A = cache["obs"], cache["actions"]

    scores, truth = [], []
    with torch.no_grad():
        for i in range(0, len(O), a.batch):
            obs = O[i : i + a.batch].to(a.device)
            act = A[i : i + a.batch].to(a.device).float()
            z = wm.encode(obs)                       # [B, T, latent]
            # Same framing train_heads used: the whole window in, the buttons of
            # its last step out.
            logits = bc(z.float())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            truth.append(act[:, -1].amax(dim=1).cpu().numpy())
    S = np.concatenate(scores)
    Y = (np.concatenate(truth) > 0.5).astype(np.float32)

    print(f"{len(S)} val windows\n")
    print(f"{'button':12s} {'base rate':>10} {'AUC':>8}")
    print("-" * 32)
    aucs = {}
    for k, name in enumerate(ACTION_COLUMNS):
        auc = _auc(S[:, k], Y[:, k])
        aucs[name] = auc
        print(f"{name:12s} {Y[:, k].mean():10.4f} {auc:8.4f}")
    vals = [v for v in aucs.values() if np.isfinite(v)]
    mean = float(np.mean(vals))
    print("-" * 32)
    print(f"{'mean':12s} {'':10s} {mean:8.4f}")
    print()
    if mean < 0.55:
        verdict = "no usable signal -- not viable as an opponent"
    elif mean < 0.65:
        verdict = "weak but real -- retrain with positive-class weighting"
    else:
        verdict = "predictive -- usable as a human-like opponent"
    print(f"VERDICT: {verdict}")
    Path(a.heads).parent.joinpath("bc_auc.json").write_text(
        json.dumps({"auc": aucs, "mean": mean, "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
