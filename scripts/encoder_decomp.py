"""Split the problem: geometry is easy, identity is one bit. Learn them apart.

    python -m scripts.encoder_decomp --cache /dev/shm/gate224.npz --steps 6000

WHY THE PREVIOUS FRAMING WAS THE PROBLEM
-----------------------------------------
Asking a network for "Player 1's x" entangles two jobs in every output:

  geometry  where are the two characters on screen -- visible, continuous, and
            exactly what a conv net is good at;
  identity  WHICH of them is Player 1 -- a single bit, not visible in the play
            area at all, and readable only from the HUD (the top-left portrait,
            or the palette in a mirror match).

Measured, that entanglement is fatal. Two heads (global-average and
soft-argmax) both scored health R^2 0.98, absolute x ~0.50, and `dx` at
CHANCE -- side accuracy 0.524 against a 0.506 base rate. A model that had found
the characters would get `dx` free, since on-screen separation needs no identity
at all. What it had actually learned was the CAMERA MIDPOINT: a predictor that
knows the midpoint perfectly and guesses mean separation scores x1 +0.733 and dx
+0.000, which is the shape of the observed result.

Nor is there a temporal shortcut: Player 1 starts on the left only 69% of the
time, and the two swap sides about 11 times a minute, so tracking from a
round-start anchor drifts almost immediately.

THE DECOMPOSITION
-----------------
Predict everything in LEFT-TO-RIGHT screen order, which needs no identity:

    xL, yL, xR, yR, absdx        pure geometry
    hp1, hp2                     already solved at 0.98 -- the HUD states these
                                 by POSITION, so they never needed identity
    p1_is_left                   the one bit, isolated, as a classifier

and reconstruct what the policy wants:

    dx = (xR - xL) * (+1 if p1_is_left else -1)

Now the geometric head is never punished for a binding it cannot see, and the
binding is a single supervised bit that can be learned from the HUD region
alone. If `absdx` and the left/right coordinates come out strong while
`p1_is_left` stays at chance, the remaining problem is precisely and only
character identification -- which is a resolution and data question with an
obvious answer, not a mystery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sokubot.data.state import CH, STAGE_SPAN
from scripts.encoder_gate import Net, load_pairs

# Left-to-right ordered, so none of these needs to know who is who.
GEO = ("xL", "yL", "xR", "yR", "absdx", "hp1", "hp2")


def decompose(Y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The gate's targets -> (geometry [N,7], p1_is_left [N]).

    `Y_raw` columns are (dx, absdx, x1, y1, x2, y2, hp1, hp2) from
    `encoder_gate.targets_from`. dx is x2 - x1, so dx > 0 means player 1 is to
    the LEFT of player 2.
    """
    dx, absdx = Y_raw[:, 0], Y_raw[:, 1]
    x1, y1, x2, y2 = Y_raw[:, 2], Y_raw[:, 3], Y_raw[:, 4], Y_raw[:, 5]
    p1_left = (dx > 0).astype(np.float32)
    xL = np.where(p1_left > 0.5, x1, x2)
    yL = np.where(p1_left > 0.5, y1, y2)
    xR = np.where(p1_left > 0.5, x2, x1)
    yR = np.where(p1_left > 0.5, y2, y1)
    geo = np.stack([xL, yL, xR, yR, absdx, Y_raw[:, 6], Y_raw[:, 7]], 1)
    return geo.astype(np.float32), p1_left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--replays", type=int, default=150)
    ap.add_argument("--per-replay", type=int, default=50)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--downs", type=int, default=4)
    ap.add_argument("--backbone", default="scratch",
                    choices=("scratch", "resnet18"))
    ap.add_argument("--head", default="spatial", choices=("spatial", "gap"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    if a.cache and a.cache.exists():
        d = np.load(a.cache)
        X, Yr, R = d["X"], d["Y"], d["R"]
        print(f"frames from cache {a.cache}: {len(X)} at {X.shape[1]}px", flush=True)
    else:
        X, Yr, R = load_pairs(a.corpus, a.replays, a.per_replay, a.size, a.seed)
        if a.cache:
            np.savez(a.cache, X=X, Y=Yr, R=R)
    Y, P = decompose(Yr)
    print(f"  p1 is left on {100*P.mean():.1f}% of frames "
          f"(a constant predictor scores that)", flush=True)

    reps = np.unique(R)
    val_reps = set(rng.choice(reps, max(1, int(len(reps) * a.val_frac)),
                              replace=False).tolist())
    vm = np.isin(R, list(val_reps))
    print(f"  train {int((~vm).sum())} / val {int(vm.sum())}  (SPLIT BY REPLAY)",
          flush=True)

    dev = a.device
    Xtr = torch.from_numpy(X[~vm]).permute(0, 3, 1, 2).contiguous()
    Ytr, Ptr = torch.from_numpy(Y[~vm]), torch.from_numpy(P[~vm])
    Xva = torch.from_numpy(X[vm]).permute(0, 3, 1, 2).contiguous().to(dev)
    Yva, Pva = torch.from_numpy(Y[vm]).to(dev), torch.from_numpy(P[vm]).to(dev)
    Yraw_va = torch.from_numpy(Yr[vm]).to(dev)
    mu, sd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-6)
    mu_d, sd_d = mu.to(dev), sd.to(dev)

    # One extra output: the identity bit, as a logit beside the geometry.
    net = Net(len(GEO) + 1, head=a.head, downs=a.downs,
              backbone=a.backbone).to(dev)
    with torch.no_grad():
        grid = net.body(torch.zeros(1, 3, a.size, a.size, device=dev)).shape[-1]
    print(f"  head {a.head} | grid {grid}x{grid} | "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    log = []
    for step in range(a.steps):
        i = torch.from_numpy(rng.choice(len(Xtr), a.batch))
        xb = Xtr[i].to(dev).float().div_(255.0)
        o = net(xb)
        loss_geo = F.mse_loss(o[:, :len(GEO)], (Ytr[i].to(dev) - mu_d) / sd_d)
        loss_id = F.binary_cross_entropy_with_logits(o[:, -1], Ptr[i].to(dev))
        loss = loss_geo + loss_id
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                pr = []
                for j in range(0, len(Xva), 128):
                    pr.append(net(Xva[j:j+128].float().div(255.0)))
                pr = torch.cat(pr)
                geo = pr[:, :len(GEO)] * sd_d + mu_d
                idl = pr[:, -1]
                ss_res = ((geo - Yva) ** 2).sum(0)
                ss_tot = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp(min=1e-9)
                r2 = (1 - ss_res / ss_tot).cpu().numpy()
                id_acc = float(((idl > 0).float() == Pva).float().mean())
                # Reconstruct the thing the policy actually needs, from the two
                # halves -- this is the number that has to clear 0.9.
                sep = (geo[:, 2] - geo[:, 0])
                sign = torch.where(idl > 0, 1.0, -1.0)
                dx_hat = sep * sign
                dx_true = Yraw_va[:, 0]
                r2_dx = float(1 - ((dx_hat - dx_true) ** 2).sum()
                              / ((dx_true - dx_true.mean()) ** 2).sum())
                side = float(((dx_hat > 0) == (dx_true > 0)).float().mean())
            net.train()
            rec = {"step": step, "loss": float(loss), "id_acc": id_acc,
                   "r2_dx_reconstructed": r2_dx, "side_acc": side,
                   **{f"r2_{t}": float(v) for t, v in zip(GEO, r2)}}
            log.append(rec)
            print(f"step {step:5d} | id_acc {id_acc:.3f} | absdx "
                  f"{float(r2[GEO.index('absdx')]):+.3f} xL "
                  f"{float(r2[GEO.index('xL')]):+.3f} xR "
                  f"{float(r2[GEO.index('xR')]):+.3f} | hp "
                  f"{float(r2[GEO.index('hp1')]):+.3f} | RECON dx {r2_dx:+.3f} "
                  f"side {side:.3f}", flush=True)

    b = log[-1]
    print("\n--- geometry (identity-free) ---")
    for t in GEO:
        print(f"  {t:<6} {b['r2_' + t]:+.4f}")
    print(f"\n--- identity (the one bit) ---\n  p1_is_left accuracy "
          f"{b['id_acc']:.4f}  (constant-predictor baseline "
          f"{max(P.mean(), 1-P.mean()):.3f})")
    print(f"\n--- reconstructed, what the policy needs ---\n  dx R2 "
          f"{b['r2_dx_reconstructed']:+.4f}   side {b['side_acc']:.4f}")
    geo_ok = b["r2_absdx"] > 0.8
    id_ok = b["id_acc"] > 0.9
    print(f"\nGEOMETRY {'OK' if geo_ok else 'FAILS'} | IDENTITY "
          f"{'OK' if id_ok else 'FAILS'}")
    if geo_ok and not id_ok:
        print("=> the blocker is character IDENTIFICATION, not geometry. The "
              "HUD names\n   who is who; give the net the resolution to read "
              "it, or more data.")
    elif not geo_ok:
        print("=> geometry itself is not being learned, which is a deeper "
              "problem than\n   identity and points at resolution or the "
              "camera.")
    if a.out:
        a.out.write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
