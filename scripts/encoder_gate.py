"""Can a DEDICATED supervised net read position off the screen? The gate.

    python -m scripts.encoder_gate --corpus ~/corpus --replays 120 --steps 3000

WHY THIS RUNS BEFORE THE ENCODER
---------------------------------
Position has failed to come out of pixels five times, and the numbers are on
record: JEPA 0.540, inverse dynamics 0.651, class-balanced IDM 0.688, direct dx
supervision over 2003 replays 0.620, play-area-mirror augmentation 0.6047 --
all against a 0.956 ceiling for the same probe reading the HUD. That history is
the entire reason the project moved to a state-space world model.

But every one of those was an attempt to make a SHARED 192-dimensional latent
hold "what is happening" and "where everything is" at once, trained with a JEPA
objective, and the conclusion drawn was that the compression keeps the first.
None of them was a network whose only job is to look at a frame and say where
the characters are.

So this asks the narrow question, with nothing else in the loss and no
bottleneck to fight over. Two characters on a 2D plane at 480x480 is not a hard
vision problem on its face -- but "on its face" is what the last five attempts
also looked like, so it gets measured before a day is spent on the full encoder.

WHAT WOULD COUNT AS PASSING
---------------------------
`dx` R^2 above ~0.9 and side-accuracy (does it know who is on the left) above
~0.95. Anything near the old 0.62 means the failure was never about the shared
latent and the whole pixels-to-state plan needs rethinking rather than scaling.

SPLIT BY REPLAY, NOT BY FRAME
-----------------------------
Consecutive frames of a fighting game are nearly identical, so a frame-level
split lets the model memorise its own validation set and report a number that
means nothing. Every metric here is on replays never seen in training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sokubot.data.state import CH, STAGE_SPAN, read_state

# The targets that matter, and why each is here:
#   dx      the mechanic the whole redesign turned on -- "which way is away"
#   x1,x2   absolute stage position, what corner pressure is about
#   y1,y2   height, which decides whether an attack can even connect
#   hp1,hp2 the reward. The HUD reads these at MAE 0.012 from pixels already,
#           so they are the CONTROL: a model that fails on health is broken in
#           some way that has nothing to do with the hard question.
#   absdx   THE DIAGNOSTIC. Separation WITHOUT its sign, so it needs no idea
#           which sprite is which. If this is learnable while `dx` is not, the
#           blocker is player IDENTITY -- associating an on-screen character
#           with a player slot -- and not geometry at all. That distinction
#           decides whether the fix is resolution or the HUD.
TARGETS = ("dx", "absdx", "x1", "y1", "x2", "y2", "hp1", "hp2")


def targets_from(state: np.ndarray) -> np.ndarray:
    return np.stack([
        state[:, 0, CH["dx"]], np.abs(state[:, 0, CH["dx"]]),
        state[:, 0, CH["x"]], state[:, 0, CH["y"]],
        state[:, 1, CH["x"]], state[:, 1, CH["y"]],
        state[:, 0, CH["hp"]], state[:, 1, CH["hp"]],
    ], axis=1).astype(np.float32)


def load_pairs(corpus: Path, n_replays: int, per_replay: int, size: int,
               seed: int = 0):
    """(frames [N,3,S,S] uint8, targets [N,T], replay id [N]). Held in RAM.

    Nothing is written to disk: the box with the video has 2 GB free and the
    whole sample is a few hundred megabytes in memory.
    """
    import cv2
    rng = np.random.default_rng(seed)
    dirs = []
    for w in sorted(corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    X, Y, R = [], [], []
    kept = 0
    for d in dirs:
        if kept >= n_replays:
            break
        vid, sc = d / "video.mp4", d / "state.csv.gz"
        if not (vid.exists() and sc.exists()):
            continue
        try:
            state, _proj, _act, valid = read_state(sc)
        except (ValueError, OSError):
            continue
        hp = state[:, :, CH["hp"]]
        # In a live battle, not a menu or a load screen. Frames outside one
        # carry stale or zeroed state, and training on them teaches the encoder
        # to read a number off a screen that does not contain it.
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        idx = np.flatnonzero(ok)
        if len(idx) < per_replay * 2:
            continue
        pick = np.sort(rng.choice(idx, per_replay, replace=False))
        cap = cv2.VideoCapture(str(vid))
        got, tgt = [], []
        for f in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok_read, frame = cap.read()
            if not ok_read:
                continue
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            got.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tgt.append(f)
        cap.release()
        if len(got) < per_replay // 2:
            continue
        X.append(np.stack(got))
        Y.append(targets_from(state[np.array(tgt)]))
        R.append(np.full(len(got), kept, np.int32))
        kept += 1
        if kept % 10 == 0:
            print(f"  {kept}/{n_replays} replays, {sum(len(x) for x in X)} frames",
                  flush=True)
    if not X:
        raise SystemExit("no usable replays found")
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(R))


class Net(nn.Module):
    """Conv stack with a selectable head, because the head IS the question.

    `gap` ends in global average pooling. That is the obvious choice and it is
    self-defeating for this task: averaging over the spatial axes is precisely
    the operation that discards WHERE something is. Measured here -- health
    R^2 0.975 while dx sat at chance -- and it is architecturally the same
    mistake as the CLS-token pooling the old encoder used, which
    `spatial_probe.py` scored at 0.540 for the same question.

    `spatial` is a soft-argmax (spatial softmax) head, the standard way to read
    coordinates out of a feature map: each of K channels is softmaxed over the
    spatial grid and reduced to its expected (x, y). Position survives by
    construction instead of having to be smuggled through a mean. The pooled
    vector is kept alongside it, because health and the flags are genuinely
    non-spatial and a coordinate head should not have to carry them.
    """

    def __init__(self, n_out: int, width: int = 32, head: str = "spatial",
                 keypoints: int = 32, downs: int = 5, backbone: str = "scratch",
                 in_ch: int = 3):
        super().__init__()
        if backbone == "resnet18":
            # PRETRAINED FEATURES, because the scratch nets are almost certainly
            # data-starved rather than blind: 1.3-2.4M parameters trained on
            # 5000 frames plateaued at the SAME R^2 0.775 whether they were fed
            # 224px or 480px, and a ceiling that does not move with four times
            # the pixels is a ceiling on what the model can infer, not on what
            # it can see. ImageNet features are the standard answer to that and
            # cost nothing to try.
            from torchvision.models import ResNet18_Weights, resnet18
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Drop avgpool+fc: the whole point is to keep the spatial map.
            self.body = nn.Sequential(*(list(m.children())[:-2]))
            ch = 512
            self.kind = head
            self._build_head(n_out, head, keypoints, ch)
            return
        c = width
        layers, ch = [], in_ch
        # `downs` stride-2 blocks. At 5 a 224px frame becomes a 7x7 map, so one
        # cell covers 32 input pixels -- and a character is only ~30-45 px tall
        # after the 480->224 downsample. A soft-argmax cannot be more precise
        # than its grid, so this is the knob that decides whether the head has
        # anything to localise WITH.
        for i in range(downs):
            layers += [nn.Conv2d(ch, c, 3, 2, 1), nn.BatchNorm2d(c), nn.GELU(),
                       nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.GELU()]
            ch, c = c, min(c * 2, 256)
        self.body = nn.Sequential(*layers)
        self.kind = head
        self._build_head(n_out, head, keypoints, ch)

    def _build_head(self, n_out, head, keypoints, ch):
        if head == "gap":
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(ch, 256), nn.GELU(),
                                      nn.Linear(256, n_out))
        elif head == "spatial":
            self.kp = nn.Conv2d(ch, keypoints, 1)
            self.head = nn.Sequential(
                nn.Linear(2 * keypoints + ch, 256), nn.GELU(),
                nn.Linear(256, n_out))
        else:
            raise ValueError(f"head {head!r}; want gap or spatial")

    def forward(self, x, return_feat: bool = False):
        """`return_feat` hands back the conv map alongside the prediction, so a
        second head (the projectile occupancy map) can read the same features
        without a second forward pass."""
        f = self.body(x)
        if self.kind == "gap":
            out = self.head(f)
            return (out, f) if return_feat else out
        B, _, H, W = f.shape
        heat = self.kp(f).flatten(2).softmax(-1).view(B, -1, H, W)
        ys = torch.linspace(-1, 1, H, device=f.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=f.device).view(1, 1, 1, W)
        # Expected coordinate of each keypoint: the whole point is that this is
        # a POSITION, not a magnitude, so no averaging over space happens.
        ex = (heat * xs).sum((2, 3))
        ey = (heat * ys).sum((2, 3))
        pooled = f.mean((2, 3))
        out = self.head(torch.cat([ex, ey, pooled], dim=1))
        return (out, f) if return_feat else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=120)
    ap.add_argument("--per-replay", type=int, default=40)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="scratch",
                    choices=("scratch", "resnet18"),
                    help="`resnet18` uses ImageNet-pretrained features with the "
                         "classifier removed, keeping the spatial map.")
    ap.add_argument("--downs", type=int, default=5,
                    help="stride-2 blocks. Fewer means a finer feature map and "
                         "a coarser-to-finer soft-argmax.")
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz to reuse decoded frames from. Decoding is 6.7 min "
                         "per run and every arm uses the same frames, so the "
                         "comparison is also exact rather than merely similar.")
    ap.add_argument("--tag", default="", help="label for the log")
    ap.add_argument("--head", default="spatial", choices=("spatial", "gap"),
                    help="`gap` is global average pooling, which discards "
                         "position by construction and is kept only as the "
                         "control that demonstrates it. `spatial` is a "
                         "soft-argmax head.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    t0 = time.time()
    if a.cache and a.cache.exists():
        d = np.load(a.cache)
        X, Y, R = d["X"], d["Y"], d["R"]
        print(f"frames from cache {a.cache}: {len(X)} at {X.shape[1]}px",
              flush=True)
        if X.shape[1] != a.size:
            raise SystemExit(f"cache is {X.shape[1]}px, run asked for {a.size}")
    else:
        print(f"sampling {a.per_replay} frames from up to {a.replays} replays "
              f"at {a.size}px ...", flush=True)
        X, Y, R = load_pairs(a.corpus, a.replays, a.per_replay, a.size, a.seed)
        if a.cache:
            a.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.cache, X=X, Y=Y, R=R)
            print(f"  cached -> {a.cache}", flush=True)
    print(f"  {len(X)} frames from {R.max()+1} replays in "
          f"{(time.time()-t0)/60:.1f} min, {X.nbytes/1e9:.2f} GB in RAM",
          flush=True)

    reps = np.unique(R)
    rng = np.random.default_rng(a.seed)
    val_reps = set(rng.choice(reps, max(1, int(len(reps) * a.val_frac)),
                              replace=False).tolist())
    vm = np.isin(R, list(val_reps))
    print(f"  train {int((~vm).sum())} frames / {len(reps)-len(val_reps)} replays"
          f" | val {int(vm.sum())} / {len(val_reps)} replays  (SPLIT BY REPLAY)",
          flush=True)

    dev = a.device
    Xtr = torch.from_numpy(X[~vm]).permute(0, 3, 1, 2).contiguous()
    Ytr = torch.from_numpy(Y[~vm])
    Xva = torch.from_numpy(X[vm]).permute(0, 3, 1, 2).contiguous().to(dev)
    Yva = torch.from_numpy(Y[vm]).to(dev)
    mu, sd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)

    net = Net(len(TARGETS), head=a.head, downs=a.downs,
              backbone=a.backbone).to(dev)
    with torch.no_grad():
        grid = net.body(torch.zeros(1, 3, a.size, a.size, device=dev)).shape[-1]
    print(f"  head: {a.head} | {a.downs} downsamples | feature grid "
          f"{grid}x{grid} ({a.size/grid:.0f} px per cell)", flush=True)
    print(f"  net {sum(p.numel() for p in net.parameters())/1e6:.2f}M params",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    mu_d, sd_d = mu.to(dev), sd.to(dev)
    log = []
    for step in range(a.steps):
        i = torch.from_numpy(rng.choice(len(Xtr), a.batch))
        xb = Xtr[i].to(dev).float().div_(255.0)
        yb = ((Ytr[i].to(dev) - mu_d) / sd_d)
        loss = F.mse_loss(net(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                pr = []
                for j in range(0, len(Xva), 128):
                    pr.append(net(Xva[j:j+128].float().div(255.0)) * sd_d + mu_d)
                pr = torch.cat(pr)
                ss_res = ((pr - Yva) ** 2).sum(0)
                ss_tot = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp(min=1e-9)
                r2 = (1 - ss_res / ss_tot).cpu().numpy()
                # "Which side is the opponent on" -- the question every one of
                # the five failed attempts was actually being scored on.
                side_acc = float(((pr[:, 0] > 0) == (Yva[:, 0] > 0)).float().mean())
                r2_abs = float(r2[TARGETS.index("absdx")])
                dx_err = float((pr[:, 0] - Yva[:, 0]).abs().mean()) * STAGE_SPAN
            net.train()
            rec = {"step": step, "loss": float(loss), "side_acc": side_acc,
                   "dx_mae_units": dx_err,
                   **{f"r2_{t}": float(v) for t, v in zip(TARGETS, r2)}}
            log.append(rec)
            print(f"step {step:5d} | loss {float(loss):.4f} | dx {r2[0]:+.3f} "
                  f"|dx| {r2_abs:+.3f} side {side_acc:.3f} | "
                  f"x1 {float(r2[TARGETS.index('x1')]):+.3f} "
                  f"y1 {float(r2[TARGETS.index('y1')]):+.3f} | "
                  f"hp {float(r2[TARGETS.index('hp1')]):+.3f}", flush=True)

    best = log[-1]
    print("\n--- held-out R2 by target ---")
    for t in TARGETS:
        print(f"  {t:<5} {best['r2_' + t]:+.4f}")
    print(f"\n  side accuracy {best['side_acc']:.4f}  (five earlier attempts: "
          f"0.540 0.651 0.688 0.620 0.605; HUD ceiling 0.956)")
    print(f"  |dx| (sign-free separation) {best['r2_absdx']:+.4f}  <- if this "
          f"works and dx does not,\n       the blocker is player IDENTITY, "
          f"not geometry")
    verdict = ("PASS -- position is readable; build the encoder"
               if best["r2_dx"] > 0.9 and best["side_acc"] > 0.95 else
               "FAIL -- position is not coming out of pixels even with a "
               "dedicated net; rethink before scaling")
    print(f"\n{verdict}")
    if a.out:
        a.out.write_text(json.dumps({"log": log, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
