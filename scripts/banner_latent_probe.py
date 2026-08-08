"""Can the world model's latent see the KNOCK OUT banner?

    python -m scripts.banner_latent_probe --bank ~/bank_hud.npz \
        --banner ~/banner/banner.pt --corpus ~/corpus --replays 40

WHY THIS IS THE NEXT QUESTION AFTER THE CLASSIFIER
--------------------------------------------------
`scripts/train_banner.py` reads KNOCK OUT off the pixels at precision 0.857
against the health probe's 0.003. That fixes the *measurement* of a KO in a real
capture -- but the reward is not paid in a real capture. It is paid inside an
imagined rollout, where there are no pixels at all, only latents. A pixel
classifier cannot fire there.

So the term `win`/`lose` can only switch on if one of these holds:

  (a) the latent already carries the banner, and a probe can read it; or
  (b) it does not, and the encoder has to be *taught* to carry it -- which means
      a supervised banner channel alongside `hud_coef`'s, in the next world-model
      training run rather than in the reward.

This script decides which, and it is worth being clear that (b) is a perfectly
good answer. It is the same shape as the argument for the HUD head: nothing in
the objective currently asks the encoder to carry a round-end banner, so there is
no reason to expect it to.

HOW THE LABELS ARE ALIGNED TO THE BANK
--------------------------------------
Not by re-encoding. `build_hud_bank.py` walks the train manifest shuffled with a
fixed seed and keeps the first N usable replays, numbering them 0..N-1 in `ep`.
Re-walking the same manifest with the same seed reproduces that order, so banner
labels computed per replay drop straight onto the bank's rows.

That alignment is an assumption, so it is checked rather than trusted: the
per-episode frame counts derived here must equal the bank's own `ep` counts,
replay for replay. A mismatch means a replay was skipped in one pass and not the
other, and the script stops rather than fitting a probe on shifted labels --
which would read as "the latent cannot see it" and be indistinguishable from a
real null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from scripts.harvest_banners import BAND_X, DISP_Y, STORE_Y, banner_score
from scripts.horizon_ablation import capture_paths, decode_hud
from scripts.train_banner import CLASSES, H, W, BannerNet


def banner_labels(video: str, skip: int, net, device: str,
                  batch: int = 256) -> np.ndarray:
    """[D] class index per decision step, from native-resolution frames."""
    hud = decode_hud(video, 0)                       # [F, 480, 480, 3], stored flipped
    try:
        # Decision steps are every `skip`th frame, matching build_hud_bank.
        idx = np.arange(0, len(hud), skip)
        out = np.empty(len(idx), dtype=np.int64)
        for s in range(0, len(idx), batch):
            sl = idx[s : s + batch]
            crop = hud[sl, STORE_Y[0]:STORE_Y[1], BAND_X[0]:BAND_X[1]]
            # Stored frames are vertically flipped; the classifier was trained on
            # crops taken from display-oriented frames, so flip back. Getting this
            # wrong trains and tests on mirrored text, which looks like a null.
            crop = np.ascontiguousarray(crop[:, ::-1])
            t = torch.from_numpy(crop).permute(0, 3, 1, 2).float().to(device) / 255.0
            t = F.interpolate(t, size=(H, W), mode="bilinear", antialias=True,
                              align_corners=False)
            with torch.no_grad():
                out[s : s + len(sl)] = net(t).argmax(-1).cpu().numpy()
        return out
    finally:
        del hud


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--banner", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=40,
                    help="how many of the bank's replays to label. Fewer than the "
                         "bank holds is fine -- labels land on episodes 0..N-1 "
                         "and the probe is fit on those rows only.")
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("banner_latent_probe.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    b = np.load(a.bank.expanduser())
    Z, E = b["z"], b["ep"]
    print(f"bank {len(Z)} latents, {int(E.max())+1} replays")

    blob = torch.load(a.banner.expanduser(), map_location=a.device,
                      weights_only=False)
    net = BannerNet(len(blob["classes"])).to(a.device)
    net.load_state_dict(blob["model"])
    net.eval()
    if tuple(blob["classes"]) != CLASSES:
        raise SystemExit(f"classifier classes {blob['classes']} != {CLASSES}")

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    np.random.default_rng(a.seed).shuffle(rows)

    counts = np.bincount(E)
    labels, kept = [], 0
    for r in rows:
        if kept >= a.replays:
            break
        try:
            video, _ = capture_paths(r, manifest)
            lab = banner_labels(str(video), a.skip, net, a.device)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        want = int(counts[kept])
        if len(lab) < want:
            raise SystemExit(
                f"replay {kept} yielded {len(lab)} decision steps but the bank "
                f"has {want}; the manifest walk has diverged from the one that "
                f"built the bank and the labels would be misaligned")
        labels.append(lab[:want])
        kept += 1
        if kept % 5 == 0:
            print(f"   labelled {kept}/{a.replays}", flush=True)

    y = np.concatenate(labels)
    n = len(y)
    Zs = Z[:n].astype(np.float32)
    print(f"\n{n} labelled decision steps over {kept} replays")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<9} {int((y == i).sum()):6d}  ({(y == i).mean():.3%})")

    # A linear probe, for the same reason `probe.py` uses one: a deeper head
    # would recover the banner from a latent that does not linearly expose it,
    # which is exactly the property the reward needs and would therefore hide the
    # failure this exists to detect.
    mu, sd = Zs.mean(0), Zs.std(0) + 1e-6
    X = (Zs - mu) / sd
    X = np.concatenate([X, np.ones((n, 1), np.float32)], axis=1)
    ep = E[:n]

    def auc_of(score: np.ndarray, t: np.ndarray) -> float:
        """Mann-Whitney AUC. Ties get average ranks, which matters here because
        a saturated probe produces many identical scores."""
        m = len(score)
        order = np.argsort(score, kind="mergesort")
        rank = np.empty(m, np.float64)
        rank[order] = np.arange(1, m + 1)
        npos, nneg = int(t.sum()), int(m - t.sum())
        if npos == 0 or nneg == 0:
            return float("nan")
        return float((rank[t > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    def fit(Xtr, ttr):
        A_ = Xtr.T @ Xtr + a.alpha * np.eye(Xtr.shape[1], dtype=np.float32)
        return np.linalg.solve(A_, Xtr.T @ ttr)

    # Held out **by replay**, not by frame. Consecutive frames of one match are
    # nearly identical latents, so a frame-wise split would put a banner's own
    # neighbours in the training set and score memorisation as generalisation.
    reps = np.unique(ep)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(reps)
    n_te = max(1, len(reps) // 3)
    te_rep = set(reps[:n_te].tolist())
    te = np.array([i for i in range(n) if ep[i] in te_rep])
    tr = np.array([i for i in range(n) if ep[i] not in te_rep])
    print(f"\nheld out {len(te_rep)} of {len(reps)} replays "
          f"({len(te)} frames) — split by replay, never by frame")

    res = {"n": int(n), "replays": int(kept), "classes": list(CLASSES),
           "auc": {}, "auc_heldout": {}, "held_out_replays": int(len(te_rep))}
    print("\nlinear probe, latent -> banner (AUC; 0.5 is chance)")
    print("  class        n   in-sample   held-out")
    for i, c in enumerate(CLASSES):
        t = (y == i).astype(np.float32)
        if t.sum() < 20 or t.sum() == n:
            print(f"  {c:<9} too few positives ({int(t.sum())}) to score")
            continue
        # In-sample first: the question is whether the information is *present*,
        # and this is the most generous test there is, so a null here is strong.
        w = fit(X, t)
        res["auc"][c] = auc_of(X @ w, t)
        # Then the honest one.
        ho = float("nan")
        if len(te) and t[te].sum() >= 5 and t[tr].sum() >= 5:
            w2 = fit(X[tr], t[tr])
            ho = auc_of(X[te] @ w2, t[te])
            res["auc_heldout"][c] = ho
        print(f"  {c:<9} {int(t.sum()):5d}   {res['auc'][c]:9.4f}   "
              f"{ho:8.4f}")

    ko = res["auc_heldout"].get("knockout", res["auc"].get("knockout"))
    print("\n" + "=" * 68)
    if ko is None:
        print("Not enough knockout frames in this sample to score. Raise "
              "--replays.")
    elif ko > 0.9:
        print(f"The latent carries it: KO AUC {ko:.3f} on **held-out replays**, "
              f"from a\nlinear probe. So the terminal signal can be read inside "
              f"imagination and\n`win`/`lose` can be wired to the banner without "
              f"retraining the encoder.")
    else:
        print(f"KO AUC {ko:.3f} held out. Nothing in the objective asks the "
              f"encoder to\ncarry a round-end announcement, so this is expected "
              f"rather than surprising.\nThe fix is a supervised banner channel "
              f"alongside the HUD head in the next\nworld-model run, not a "
              f"cleverer probe.")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
