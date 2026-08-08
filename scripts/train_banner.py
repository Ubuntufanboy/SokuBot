"""Classify the banner the game draws across the screen, from 120 hand labels.

    python -m scripts.train_banner --labels banners_label --out ~/banner

WHY A CLASSIFIER AND NOT A HEALTH THRESHOLD
-------------------------------------------
The reward's `win`/`lose` term is worth +-5 against damage terms worth ~0.1, and
it currently fires when probed health crosses 0.06. `scripts/probe_reliability.py`
measured that detector at **precision 0.003** -- twenty to forty-five false fires
per real KO -- and `scripts/anchored_ko_test.py` showed the error is in the
probe's per-step deltas, so anchoring the level to `data/hud.py` moved precision
from 0.005 to 0.015, i.e. not at all.

But the game *announces* the event in letters half a screen wide. Reading the
announcement is a different and much easier problem than inferring the state that
caused it, and it stays inside the project's rule: pixels only, no game memory.

WHAT MAKES THIS HARD, AND HOW IT IS EVALUATED
---------------------------------------------
120 labels, and the classes that matter are the rare ones: 8 `knockout`, 8
`down`, against 82 `none`. Two consequences drive every choice below.

**The split is by capture, never by frame.** The 120 candidates come from 29
replays. Two crops from one replay share a stage, a pair of characters and a
colour palette, so a model that memorised the background would score beautifully
on a random frame split and collapse on a new match. `GroupKFold` on the capture
id makes the reported number mean "a replay it has never seen".

**Everything reported is out-of-fold.** With 8 positives per class a single
train/test split resolves nothing, so every sample is predicted by a model that
did not train on its replay, and the confusion matrix is accumulated over the
folds. Multiple seeds are averaged because at this size one seed is a coin toss.

The number that decides whether `win`/`lose` can be switched on is not overall
accuracy -- at 68% `none` a constant predictor gets 68%. It is **precision on
`knockout`**: how often a fired KO is a real one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordered so the rare, decision-relevant classes keep stable indices.
CLASSES = ("none", "round", "start", "down", "knockout", "other")
H, W = 48, 112


def load(labels_dir: Path):
    cand = {c["id"]: c for c in json.loads((labels_dir / "candidates.json").read_text())}
    xs, ys, groups, ids = [], [], [], []
    for line in (labels_dir / "labels.csv").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("id,"):
            continue
        sid, _, lab = line.partition(",")
        lab = lab.strip()
        if not lab:
            continue                      # the annotator could not tell; not data
        if lab not in CLASSES:
            raise SystemExit(f"row {sid}: unknown label {lab!r}, want {CLASSES}")
        i = int(sid)
        xs.append(np.load(labels_dir / "crops" / f"{i:03d}.npy"))
        ys.append(CLASSES.index(lab))
        groups.append(cand[i]["capture"])
        ids.append(i)
    raw = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    # antialias=True is the point, not a nicety: this is a 3.5x downsample and
    # the glyphs that separate DOWN from KNOCK OUT are thin outlines. Plain
    # bilinear sampling drops them between grid points and the two classes end
    # up looking like the same blue smear.
    x = F.interpolate(raw, size=(H, W), mode="bilinear", antialias=True,
                      align_corners=False)
    return x, np.array(ys), np.array(groups), np.array(ids)


class BannerNet(nn.Module):
    """~60k parameters. Small on purpose: 120 samples cannot train more.

    Global average pooling rather than a flatten, so the head cannot key on
    *where* in the band something appeared -- only on what it looks like. The
    banner's vertical position shifts between START and KNOCK OUT, and with a
    flatten the model would happily learn the position instead of the glyphs and
    then fail on any capture whose timing differs.
    """

    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 64))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.f(x).mean(dim=(2, 3)))


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Photometric jitter plus a small shift. No horizontal flip -- it is text."""
    B = x.shape[0]
    dev = x.device
    gain = 1.0 + 0.4 * (torch.rand(B, 1, 1, 1, device=dev, generator=g) - 0.5)
    bias = 0.2 * (torch.rand(B, 1, 1, 1, device=dev, generator=g) - 0.5)
    # Per-channel gain as well: stages differ a lot in colour cast, and the
    # detector that produced these candidates keys on blue, so a model allowed to
    # rely on absolute channel levels would inherit that bias.
    cgain = 1.0 + 0.3 * (torch.rand(B, 3, 1, 1, device=dev, generator=g) - 0.5)
    x = (x * gain * cgain + bias).clamp(0, 1)
    sx = int(torch.randint(-6, 7, (1,), device=dev, generator=g).item())
    sy = int(torch.randint(-3, 4, (1,), device=dev, generator=g).item())
    x = torch.roll(x, shifts=(sy, sx), dims=(2, 3))
    x = x + 0.02 * torch.randn(x.shape, device=dev, generator=g)
    return x.clamp(0, 1)


def train_fold(xtr, ytr, xte, steps, lr, device, seed, batch=32):
    g = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    net = BannerNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # Inverse-frequency weights. Without them the loss is minimised by predicting
    # `none` for everything, which is 68% accurate and useless -- and with 8
    # examples of `knockout` the gradient from that class would otherwise be
    # noise against the majority.
    cnt = torch.bincount(ytr, minlength=len(CLASSES)).float()
    w = torch.where(cnt > 0, cnt.sum() / cnt.clamp(min=1), torch.zeros_like(cnt))
    w = (w / w[w > 0].mean()).to(device)
    net.train()
    for _ in range(steps):
        idx = torch.randint(0, len(xtr), (min(batch, len(xtr)),), device=device,
                            generator=g)
        xb = augment(xtr[idx], g)
        loss = F.cross_entropy(net(xb), ytr[idx], weight=w, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net, net(xte).softmax(-1).cpu().numpy()


def group_kfold(groups: np.ndarray, k: int, seed: int):
    """Folds that never split a capture across train and test."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    for f in range(k):
        te_g = set(uniq[f::k])
        te = np.array([i for i, g in enumerate(groups) if g in te_g])
        tr = np.array([i for i, g in enumerate(groups) if g not in te_g])
        if len(te) and len(tr):
            yield tr, te


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=Path("banners_label"))
    ap.add_argument("--out", type=Path, default=Path("banner"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5,
                    help="the dataset is small enough that one seed is a coin "
                         "toss; out-of-fold probabilities are averaged over this "
                         "many independent runs")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    x, y, groups, ids = load(a.labels)
    n = len(y)
    print(f"{n} labels over {len(np.unique(groups))} captures")
    for c, k in enumerate(CLASSES):
        print(f"  {k:<9} {int((y == c).sum()):3d}")
    X = x.contiguous().to(a.device)
    Y = torch.from_numpy(y).long().to(a.device)

    prob = np.zeros((n, len(CLASSES)), dtype=np.float64)
    for seed in range(a.seeds):
        for tr, te in group_kfold(groups, a.folds, seed):
            _, p = train_fold(X[tr], Y[tr], X[te], a.steps, a.lr, a.device, seed)
            prob[te] += p
        print(f"  seed {seed} done", flush=True)
    prob /= a.seeds
    pred = prob.argmax(1)

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print("\nout-of-fold confusion (rows = truth, cols = predicted)")
    print("           " + " ".join(f"{c[:5]:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:<9}" + " ".join(f"{v:6d}" for v in cm[i]))

    print("\nper class")
    rows = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        rec = tp / max(cm[i].sum(), 1)
        prec = tp / max(cm[:, i].sum(), 1)
        rows[c] = {"n": int(cm[i].sum()), "recall": float(rec),
                   "precision": float(prec)}
        print(f"  {c:<9} n {cm[i].sum():3d} | recall {rec:6.1%} | "
              f"precision {prec:6.1%}")

    # The two numbers the reward actually depends on.
    none_i = CLASSES.index("none")
    fp = 1.0 - cm[none_i, none_i] / max(cm[none_i].sum(), 1)
    ko = CLASSES.index("knockout")
    dn = CLASSES.index("down")
    # Round-end vs match-end confusion is the one mistake with a real cost: it
    # pays a match-outcome reward for a round the agent might still lose.
    ko_as_dn = cm[ko, dn] / max(cm[ko].sum(), 1)
    dn_as_ko = cm[dn, ko] / max(cm[dn].sum(), 1)
    print(f"\n`none` called a banner (false-positive rate) {fp:6.1%}")
    print(f"knockout mistaken for down                   {ko_as_dn:6.1%}")
    print(f"down mistaken for knockout                   {dn_as_ko:6.1%}")

    print("\n" + "=" * 68)
    kp = rows["knockout"]["precision"]
    kr = rows["knockout"]["recall"]
    if kp >= 0.5:
        print(f"KO precision {kp:.1%} at recall {kr:.1%}. The probe's detector "
              f"runs at 0.3%,\nso this is the trigger `win`/`lose` should use. "
              f"Wire it into RewardConfig\nvia the banner rather than a health "
              f"threshold.")
    else:
        print(f"KO precision {kp:.1%} at recall {kr:.1%} -- not yet good enough "
              f"to pay +-5.\nWith {rows['knockout']['n']} knockout labels the "
              f"limit is most likely data, not\narchitecture: the next lever is "
              f"more labels of the rare classes, not a\nbigger model.")

    # A model trained on everything, for labelling the unlabelled candidates.
    net, _ = train_fold(X, Y, X[:1], a.steps, a.lr, a.device, 0)
    torch.save({"model": net.state_dict(), "classes": list(CLASSES),
                "hw": [H, W], "oof": rows}, a.out / "banner.pt")
    (a.out / "report.json").write_text(json.dumps(
        {"n": n, "captures": int(len(np.unique(groups))), "classes": list(CLASSES),
         "confusion": cm.tolist(), "per_class": rows,
         "none_false_positive": float(fp),
         "knockout_as_down": float(ko_as_dn),
         "down_as_knockout": float(dn_as_ko)}, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
