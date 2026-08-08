"""Append a KNOCK OUT channel to the reward probe.

    python -m scripts.fit_banner_channel --probe ~/gate_base/reward_probe.npz \
        --bank ~/bank_hud.npz --labels ~/blp_labels.npz \
        --out ~/gate_base/reward_probe_banner.npz

WHAT THIS PRODUCES
------------------
A copy of the reward probe with one extra output row, `ko_banner`, landing at
index `reward.KO_BANNER`. `ProbeHead` needs no change -- it applies whatever
matrix it is handed -- so `RewardConfig(ko_source="banner")` starts working the
moment the reward is pointed at this file.

The channel is a **linear** map from the latent, like every other probe channel,
and for the same reason `probe.py` gives: a deeper head could recover the banner
from a latent that does not linearly expose it, which is exactly the property the
reward needs and would therefore hide the failure it exists to detect.

WHY A SEPARATE FILE RATHER THAN AN EDIT IN PLACE
------------------------------------------------
Because `train_grpo` and `train_ac` refuse to run when the probe's fingerprint
disagrees with the world model's, and every recorded number -- including the
+0.00147 baseline -- was measured with the original probe. Overwriting it would
silently change what those runs meant. The new file carries the same fingerprint
and is opt-in at the command line.

THE THRESHOLD IS NOT A FREE PARAMETER
--------------------------------------
`RewardConfig.ko_banner_threshold` defaults to 0.5, and this script reports what
that threshold actually buys on held-out replays so the number is chosen from
data rather than from the fact that 0.5 looks natural. Precision matters far more
than recall here: a missed KO forgoes a bonus, while a false KO pays +-5 and
masks the rest of the trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokubot.rl.reward import KO_BANNER
from scripts.train_banner import CLASSES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True,
                    help="the npz written by banner_latent_probe --label-cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--val-frac", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = dict(np.load(a.probe.expanduser(), allow_pickle=True))
    names = [str(x) for x in d["names"]]
    if len(names) != KO_BANNER:
        raise SystemExit(
            f"probe has {len(names)} channels {names}; ko_banner must land at "
            f"index {KO_BANNER}, so exactly {KO_BANNER} are expected first. "
            f"Reward reads that index positionally.")
    if "ko_banner" in names:
        raise SystemExit(f"{a.probe} already has a ko_banner channel")

    b = np.load(a.bank.expanduser())
    Z, E = b["z"], b["ep"]
    lab = np.load(a.labels.expanduser())
    y = lab["y"]
    n = len(y)
    if n > len(Z):
        raise SystemExit(f"{n} labels but only {len(Z)} latents in the bank")
    ep = E[:n]
    t = (y == CLASSES.index("knockout")).astype(np.float32)
    print(f"{n} labelled frames over {len(np.unique(ep))} replays | "
          f"{int(t.sum())} knockout ({t.mean():.3%})")

    # The probe's own standardisation, reused rather than refitted, so the new
    # row lives in the same input space as the six it is joining. Refitting would
    # make this channel silently incompatible with the others.
    zmu, zsd = d["zmu"].astype(np.float32), d["zsd"].astype(np.float32)
    X = (Z[:n].astype(np.float32) - zmu) / zsd

    reps = np.unique(ep)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(reps)
    te_rep = set(reps[: max(1, int(len(reps) * a.val_frac))].tolist())
    te = np.array([i for i in range(n) if ep[i] in te_rep])
    tr = np.array([i for i in range(n) if ep[i] not in te_rep])

    def fit(idx):
        Xi = np.concatenate([X[idx], np.ones((len(idx), 1), np.float32)], axis=1)
        A_ = Xi.T @ Xi + a.alpha * np.eye(Xi.shape[1], dtype=np.float32)
        return np.linalg.solve(A_, Xi.T @ t[idx])

    w_tr = fit(tr)
    s = np.concatenate([X[te], np.ones((len(te), 1), np.float32)], axis=1) @ w_tr
    tt = t[te]
    print(f"\nheld out {len(te_rep)} of {len(reps)} replays ({len(te)} frames)")
    print("  threshold  precision  recall  fires/1000 frames")
    rows = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        p = s >= thr
        tp = float((p & (tt > 0)).sum())
        prec = tp / max(p.sum(), 1)
        rec = tp / max(tt.sum(), 1)
        rows.append({"threshold": thr, "precision": prec, "recall": rec,
                     "fire_rate": float(p.mean())})
        print(f"  {thr:9.2f}  {prec:9.3f}  {rec:6.3f}  {p.mean()*1000:8.1f}")

    # Refit on everything for the shipped weights -- the held-out split above was
    # to choose and report the operating point, not to hold back data from a
    # channel whose whole job is to be as good as the labels allow.
    w = fit(np.arange(n))

    W = np.asarray(d["W"], dtype=np.float32)
    ymu = np.asarray(d["ymu"], dtype=np.float32)
    ysd = np.asarray(d["ysd"], dtype=np.float32)
    # The probe applies `(z - zmu)/zsd @ W * ysd + ymu`. This row is fitted
    # directly in output units, so it takes ysd 1 and ymu 0 and the ridge's bias
    # is folded into ymu -- which keeps `LinearProbe` untouched.
    d["W"] = np.concatenate([W, w[:-1, None]], axis=1)
    d["ymu"] = np.concatenate([ymu, np.array([w[-1]], np.float32)])
    d["ysd"] = np.concatenate([ysd, np.array([1.0], np.float32)])
    d["names"] = np.array(names + ["ko_banner"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    print(f"\nchannels now {list(d['names'])}")
    print(f"-> {a.out}")
    print("\nUse with: RewardConfig(ko_source='banner') and --probe "
          f"{a.out}\nThe fingerprint is carried over unchanged, so the "
          "world-model check still applies.")
    (a.out.with_suffix(".report.json")).write_text(json.dumps(
        {"n": int(n), "knockout": int(t.sum()), "operating_points": rows},
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
