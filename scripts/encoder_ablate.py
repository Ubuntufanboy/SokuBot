"""Encoder ablations: build the frames once, then vary one thing at a time.

    python -m scripts.encoder_ablate build --corpus ~/vcorpus --out ~/enc224 \
        --replays 149 --per-replay 150 --size 224 --offsets 0 2 4 8
    python -m scripts.encoder_ablate train --cache ~/enc224 --delta 2 \
        --out ~/enc/base.pt

WHY A CACHE WITH SEVERAL OFFSETS
---------------------------------
Decoding video is the expensive part of an encoder run and it is identical
across arms, so it happens once. The cache stores each sampled moment as a
STACK of frames at several lags rather than a single pair, which makes `delta`
a slice instead of a rebuild -- and delta is the knob most likely to matter.

WHAT encoder2 SAID, AND WHAT THAT IMPLIES
------------------------------------------
encoder2 reached mean state R^2 +0.359 with its best checkpoint at step 3000 of
12000. Peaking at a quarter of the schedule and decaying after is overfitting,
not a capacity ceiling, and it points at DATA before architecture: 149 replays
x 60 frames is ~9k samples for a 224px regression.

Its velocity channels came out at vx +0.02 and vy -0.08 -- at or below the
corpus mean. Two mechanisms could explain that, and they suggest different
fixes:

  grid    at downs=4 a 224px input becomes a 14x14 map, so one cell covers 16
          input pixels. Two frames 33 ms apart move a character about 3.7 px,
          which is a QUARTER of one cell. Finer grid (fewer downs) or more
          pixels would give the head something to localise with.
  gap     33 ms is simply a short baseline. delta=8 is 133 ms and moves the
          same character ~15 px, which is visible -- at the cost of measuring
          an average velocity over a longer window rather than the
          instantaneous one.

Differencing predicted POSITIONS cannot work and is not offered as an arm: x
scores R^2 0.73, so its residual is ~0.52 sigma, and against a stage-scale
sigma that is far larger than the ~20 units a character moves in two frames.
The difference of two such estimates is dominated by their noise. Velocity has
to be seen directly or not at all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.data.state import (CH, PROJ_FEATURES, STATE_CHANNELS, read_state)

N_STATE = len(STATE_CHANNELS)
# proj_n and proj_hb were always in the state vector and were simply never
# asked for. The policy was handed a CONSTANT for both -- measured in a real
# match, truth varied 0..8 with projectiles present on 75% of steps while the
# encoder emitted 4.74 with a standard deviation of exactly zero. Zoning the
# bot from across the screen attacks a channel it cannot see at all.
SUPERVISED = ("x", "y", "dx", "dy", "vx", "vy", "hp", "spirit", "airborne",
              "timestop", "proj_n", "proj_hb")
# The most urgent object in the air, per player. The extractor already sorts
# each player's slots danger-first (live hitbox, then nearest to its target),
# so slot 0 is the one a human would be reacting to. Predicting all eight slots
# is a set-prediction problem that scored R^2 0.042; predicting the single most
# dangerous one is a regression.
N_PROJ_OUT = 2 * len(PROJ_FEATURES)


# ----------------------------------------------------------------- build
def build(a) -> int:
    import cv2
    rng = np.random.default_rng(a.seed)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    offs = sorted(set(int(o) for o in a.offsets))
    if offs[0] != 0:
        raise SystemExit("offsets must include 0 (the current frame)")
    K, S = len(offs), a.size

    dirs = []
    for w in sorted(Path(a.corpus).expanduser().glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    dirs = dirs[:a.replays] if a.replays else dirs

    # ONE STREAMING PASS, upper-bound allocation.
    #
    # The previous version planned first and extracted second, holding every
    # replay's state AND projectile arrays in a list. `read_state` returns
    # 8-33 MB of projectiles per replay, so at 2003 replays that plan was ~20 GB
    # on a 16 GB box -- it died silently, no log, no traceback, nothing for a
    # monitor to see. Now the memmap is allocated at the upper bound, one
    # replay is resident at a time, and `meta["n"]` records how much of it is
    # real.
    #
    # Frames come from SEEKS rather than a sequential walk. Sequential is right
    # when sampling 150 frames of a 6000-frame video; at 7 frames it means
    # decoding the whole file to reach the last pick, which is what would have
    # made this an eight-hour job.
    N_MAX = len(dirs) * a.per_replay
    gb = N_MAX * K * 3 * S * S / 1e9
    print(f"{len(dirs)} replays x {a.per_replay} = up to {N_MAX} samples, "
          f"{K} offsets {offs} at {S}px -> up to {gb:.1f} GB", flush=True)

    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.uint8,
                                  shape=(N_MAX, K, 3, S, S))
    Y = np.zeros((N_MAX, 2, len(SUPERVISED)), np.float32)
    Q = np.zeros((N_MAX, 2, len(PROJ_FEATURES)), np.float32)
    L = np.zeros(N_MAX, np.float32)
    R = np.zeros(N_MAX, np.int32)
    sup = [CH[n] for n in SUPERVISED]
    fill_sum, fill_n = np.zeros(N_STATE, np.float64), 0

    w = 0
    kept = 0
    t0 = time.time()
    for ri, d in enumerate(dirs):
        vid, sc = d / "video.mp4", d / "state.csv.gz"
        if not (vid.exists() and sc.exists()):
            continue
        try:
            st, pr, _a, valid = read_state(sc)
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        idx = np.flatnonzero(ok)
        idx = idx[idx >= max(a.min_index, max(offs))]
        if len(idx) < a.per_replay:
            del st, pr
            continue
        pick = np.sort(rng.choice(idx, a.per_replay, replace=False))
        cap = cv2.VideoCapture(str(vid))
        for f in pick:
            f = int(f)
            frames = []
            for o in offs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - o)
                got, fr = cap.read()
                if not got:
                    break
                fr = cv2.resize(fr, (S, S), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
            if len(frames) != K:
                continue
            for k in range(K):
                X[w, k] = frames[k]
            s = st[f]
            p1_left = float(s[0, CH["dx"]] > 0)
            lr = (0, 1) if p1_left > 0.5 else (1, 0)
            Y[w] = s[list(lr)][:, sup]
            Q[w] = pr[f][list(lr)][:, 0, :]
            L[w] = p1_left
            R[w] = ri
            fill_sum += s.mean(0).astype(np.float64)
            fill_n += 1
            w += 1
        cap.release()
        del st, pr
        kept += 1
        if kept % 25 == 0:
            el = time.time() - t0
            rate = el / max(kept, 1)
            print(f"  {kept} replays kept ({ri+1} seen), {w} samples, "
                  f"{el/60:.1f} min, {rate:.1f}s each, eta "
                  f"{rate*(len(dirs)-ri-1)/60:.0f} min", flush=True)
    X.flush()
    # X is allocated at the upper bound, so `n` is what is actually real in it.
    meta = {"n": int(w), "offsets": offs, "size": S,
            "supervised": list(SUPERVISED),
            "proj_features": list(PROJ_FEATURES),
            "fill": (fill_sum / max(fill_n, 1)).tolist(),
            "per_replay": a.per_replay, "replays": int(kept)}
    np.save(out / "Y.npy", Y[:w]); np.save(out / "Q.npy", Q[:w])
    np.save(out / "L.npy", L[:w]); np.save(out / "R.npy", R[:w])
    # READ IT BACK BEFORE DECLARING IT GOOD.
    #
    # On 2026-08-16 a 13.4 GB cache on .141 was written, reported success, and
    # then failed with EIO 84.6% of the way in -- btrfs had logged 2714
    # checksum errors and the NVMe was handing back data that did not match
    # what was written. The first eight arms read from page cache and passed;
    # every arm after the cache was evicted died with a bare segfault and no
    # traceback, and the run script's `rc=0` was `date`'s exit status rather
    # than python's, so nothing said a word. A full sequential read costs
    # seconds against a build that costs a quarter of an hour.
    del X
    print("verifying the cache reads back ...", flush=True)
    chk = np.load(out / "X.npy", mmap_mode="r")
    try:
        step = max(1, len(chk) // 200)
        for i in range(0, len(chk), step):
            int(chk[i].sum())              # forces every page of the sample
        with open(out / "X.npy", "rb", buffering=0) as fh:
            got = 0
            while True:
                blk = fh.read(64 << 20)
                if not blk:
                    break
                got += len(blk)
    except OSError as e:
        print(f"CACHE IS UNREADABLE: {e}\n  the file on disk does not match "
              f"what was written -- do not train on it", flush=True)
        return 3
    want = (out / "X.npy").stat().st_size
    if got != want:
        print(f"CACHE SHORT: read {got} of {want} bytes", flush=True)
        return 3
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"cache -> {out}  ({w} samples, verified {got/1e9:.1f} GB readable, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return 0


# ----------------------------------------------------------------- train
def train(a) -> int:
    from scripts.encoder_gate import Net
    cache = Path(a.cache).expanduser()
    meta = json.loads((cache / "meta.json").read_text())
    offs = meta["offsets"]
    if a.delta not in offs:
        raise SystemExit(f"delta {a.delta} not cached; have {offs}")
    kd = offs.index(a.delta)
    X = np.load(cache / "X.npy", mmap_mode="r")
    Y = np.load(cache / "Y.npy"); L = np.load(cache / "L.npy")
    R = np.load(cache / "R.npy")
    qf = list(meta.get("proj_features", []))
    Q = np.load(cache / "Q.npy") if (cache / "Q.npy").exists() else None
    if a.proj_mode != "slots":
        # Must happen BEFORE the target tensor is assembled: the head is sized
        # from its width, and dropping the columns afterwards leaves a net with
        # 38 outputs scored against 24 weights.
        Q = None
    n = len(Y)
    sup = list(meta["supervised"])
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    reps = np.unique(R)
    val_reps = set(rng.choice(reps, max(1, int(len(reps) * a.val_frac)),
                              replace=False).tolist())
    vm = np.isin(R, list(val_reps))
    tr_i, va_i = np.flatnonzero(~vm), np.flatnonzero(vm)
    print(f"{n} samples | train {len(tr_i)} / val {len(va_i)} (SPLIT BY REPLAY)"
          f" | delta {a.delta} | input {a.input}", flush=True)

    # State and projectile targets share one head; they are standardised
    # together and split again only for reporting, because the projectile
    # block is the whole point of this build and must not hide inside a mean.
    flat = [Y.reshape(n, -1)]
    if Q is not None:
        flat.append(Q.reshape(n, -1))
    n_state_out = Y.shape[1] * Y.shape[2]
    Yt = torch.from_numpy(np.concatenate(flat, 1))
    Lt = torch.from_numpy(L)
    mu, sd = Yt[tr_i].mean(0), Yt[tr_i].std(0).clamp(min=1e-4)
    dev = a.device
    mu_d, sd_d = mu.to(dev), sd.to(dev)

    in_ch = 3 if a.input == "single" else 6
    net = Net(Yt.shape[1] + 1, width=a.width, head=a.head, downs=a.downs,
              in_ch=in_ch, keypoints=a.keypoints).to(dev)
    with torch.no_grad():
        grid = net.body(torch.zeros(1, in_ch, meta["size"], meta["size"],
                                    device=dev)).shape[-1]
    npar = sum(p.numel() for p in net.parameters())
    print(f"net {npar/1e6:.2f}M params | grid {grid}x{grid} "
          f"({meta['size']/grid:.0f} px per cell)", flush=True)

    # Per-channel loss weights: 1.0 everywhere, `--vel-weight` on vx/vy. The
    # velocity channels are the ones that failed, and an unweighted MSE over
    # standardised targets spends equal capacity on channels already at 0.75.
    wv = torch.ones(len(sup))
    for i, nme in enumerate(sup):
        if nme in ("vx", "vy"):
            wv[i] = a.vel_weight
    wv = wv.repeat(2)
    if Q is not None:
        wv = torch.cat([wv, torch.full((Q.shape[1] * Q.shape[2],),
                                       a.proj_weight)])
    wv = wv.to(dev)

    phead = None
    if a.proj_mode == "heatmap":
        from scripts.encoder_proj_head import ProjHead, render_targets, GRID
        from sokubot.data.state import PROJ_FEATURES, STAGE_SPAN
        with torch.no_grad():
            ch = net.body(torch.zeros(1, in_ch, meta["size"], meta["size"],
                                      device=dev)).shape[1]
        phead = ProjHead(ch, coords=a.proj_coords).to(dev)
        # Where (x, y) sit inside the per-player output block, so the head can
        # be handed the encoder's own estimate of where the players are.
        # `sup` is the list of channel NAMES the cache was built with, not
        # their indices into STATE_CHANNELS -- the two are easy to confuse and
        # only one of them is what `meta.json` stores.
        xo, yo = list(sup).index("x"), list(sup).index("y")
        Qraw = np.load(cache / "Q.npy")
        pi = PROJ_FEATURES.index("present")
        dxi, dyi = PROJ_FEATURES.index("dx"), PROJ_FEATURES.index("dy")
        print(f"projectile heatmap: {GRID}x{GRID} cells over +-600 units, "
              f"{sum(p.numel() for p in phead.parameters())/1e3:.0f}k params",
              flush=True)
    params = list(net.parameters()) + (list(phead.parameters()) if phead else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)

    def batch(ix):
        """-> [B, in_ch, S, S] float on device."""
        xb = X[ix]                                   # [B,K,3,S,S] uint8
        cur = torch.from_numpy(np.ascontiguousarray(xb[:, 0])).to(dev).float()
        if a.input == "single":
            return cur.div_(255.0)
        old = torch.from_numpy(np.ascontiguousarray(xb[:, kd])).to(dev).float()
        if a.input == "diff":
            # Motion made explicit: the net is handed the difference rather
            # than having to learn subtraction. Centred at 0.5 so it stays in
            # the same numeric range as the image.
            return torch.cat([cur.div(255.0),
                              (cur - old).div(510.0).add_(0.5)], 1)
        return torch.cat([old.div(255.0), cur.div(255.0)], 1)

    va_chunks = [va_i[i:i + 128] for i in range(0, len(va_i), 128)]
    Yva = Yt[va_i].to(dev); Lva = Lt[va_i].to(dev)
    best, best_step, log = -1e9, -1, []
    t0 = time.time()
    for step in range(a.steps):
        ix = np.sort(rng.choice(tr_i, a.batch, replace=False))
        xb = batch(ix)
        if phead is not None:
            o, feat = net(xb, return_feat=True)
        else:
            o = net(xb)
        tgt = (Yt[ix].to(dev) - mu_d) / sd_d
        loss = ((o[:, :-1] - tgt) ** 2 * wv).mean() + \
            F.binary_cross_entropy_with_logits(o[:, -1], Lt[ix].to(dev))
        if phead is not None:
            q = torch.from_numpy(Qraw[ix]).to(dev)[:, :, None, :]
            hmap = render_targets(q, pi, dxi, dyi, STAGE_SPAN)
            # The encoder's OWN predicted positions, detached: the projectile
            # head is allowed to use them, but its gradient must not reach back
            # and reshape the position channels to whatever makes projectiles
            # easier. Those channels have their own supervision and are the
            # best-scoring thing this encoder does.
            pl = None
            if a.proj_coords:
                pl = o[:, :-1].reshape(-1, 2, len(sup))[:, :, [xo, yo]].detach()
            plog, pcoord = phead(feat, pl)
            # BOTH terms. The map alone leaves the soft-argmax read biased
            # toward the centre (measured: +210 for a true +300 on a fitted
            # single example); the coordinate alone is the regression that
            # already failed.
            true_xy = torch.stack([q[..., 0, dxi], q[..., 0, dyi]], -1) * STAGE_SPAN
            live = (q[..., 0, pi] > 0.5).float().unsqueeze(-1)
            coord_l = (((pcoord - true_xy) / 600.0) ** 2 * live).sum() \
                / live.sum().clamp(min=1)
            loss = loss + a.proj_weight * (phead.loss(plog, hmap) + coord_l)
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                pr = torch.cat([net(batch(c)) for c in va_chunks])
                pred = pr[:, :-1] * sd_d + mu_d
                ssr = ((pred - Yva) ** 2).sum(0)
                sst = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp(min=1e-9)
                r2 = (1 - ssr / sst).cpu().numpy()
                idacc = float(((pr[:, -1] > 0).float() == Lva).float().mean())
            # The heatmap head is the POINT of these arms, so it needs its own
            # numbers: a coordinate error in game units and how often the
            # predicted peak cell contains a real projectile. Without this the
            # log shows only the state channels and the experiment is invisible.
            pstat = {}
            if phead is not None:
                with torch.no_grad():
                    qv = torch.from_numpy(Qraw[va_i]).to(dev)[:, :, None, :]
                    live = (qv[..., 0, pi] > 0.5)
                    if live.any():
                        cs = []
                        for c in va_chunks:
                            _, f2 = net(batch(c), return_feat=True)
                            cs.append(phead(f2)[1])
                        pc = torch.cat(cs)
                        txy = torch.stack([qv[..., 0, dxi], qv[..., 0, dyi]], -1) * STAGE_SPAN
                        err = (pc - txy).norm(dim=-1)[live]
                        base = txy.norm(dim=-1)[live]
                        pstat = {"proj_err_u": float(err.mean()),
                                 "proj_base_u": float(base.mean()),
                                 "proj_live_frac": float(live.float().mean())}
            net.train()
            per = r2[:n_state_out].reshape(2, len(sup)).mean(0)
            per_q = (r2[n_state_out:].reshape(2, len(qf)).mean(0)
                     if Q is not None else np.zeros(0))
            mean_r2 = float(per.mean())
            if mean_r2 > best:
                best, best_step = mean_r2, step
                torch.save({"net": net.state_dict(),
                            # Without this the head trains and is then thrown
                            # away on save, which is worse than not training it.
                            "proj_head": (phead.state_dict() if phead else None),
                            "proj_mode": a.proj_mode,
                            "proj_coords": bool(a.proj_coords),
                            "mu": mu, "sd": sd,
                            "size": meta["size"], "slots": 8, "downs": a.downs,
                            "n_state": n_state_out,
                            "n_proj": int(Yt.shape[1] - n_state_out),
                            "supervised": sup, "delta": a.delta,
                            "proj_features": qf, "r2_proj": per_q.tolist(),
                            "fill": meta["fill"], "r2": r2.tolist(),
                            "id_acc": idacc, "step": step,
                            "input": a.input, "width": a.width,
                            "keypoints": a.keypoints},
                           Path(a.out).expanduser())
            rec = {"step": step, "mean_r2": mean_r2, "id_acc": idacc, **pstat,
                   **{f"r2_{c}": float(v) for c, v in zip(sup, per)}}
            log.append(rec)
            pj = ("  || proj " + " ".join(f"{c} {float(v):+.3f}"
                                          for c, v in zip(qf, per_q))
                  if Q is not None else "")
            if pstat:
                # against `base`, the error of predicting "at the player":
                # below that number the head is doing something.
                pj = (f"  || projmap err {pstat['proj_err_u']:.0f}u vs "
                      f"baseline {pstat['proj_base_u']:.0f}u")
            print(f"step {step:6d} | mean {mean_r2:+.4f} | "
                  + " ".join(f"{c} {float(v):+.3f}" for c, v in zip(sup, per))
                  + pj + f" | id {idacc:.3f} | {(time.time()-t0)/60:.1f}m",
                  flush=True)

    res = {"arm": a.name, "best_mean_r2": best, "best_step": best_step,
           "steps": a.steps, "delta": a.delta, "input": a.input,
           "downs": a.downs, "width": a.width, "keypoints": a.keypoints,
           "lr": a.lr, "wd": a.wd, "vel_weight": a.vel_weight,
           "batch": a.batch, "n_train": int(len(tr_i)), "params": npar,
           "grid": int(grid), "cache": str(cache), "log": log}
    Path(str(a.out) + ".json").expanduser().write_text(json.dumps(res, indent=1))
    print(f"\nBEST mean R2 {best:+.4f} at step {best_step} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--replays", type=int, default=0)
    b.add_argument("--per-replay", type=int, default=150)
    b.add_argument("--size", type=int, default=224)
    b.add_argument("--offsets", type=int, nargs="+", default=[0, 2, 4, 8])
    b.add_argument("--min-index", type=int, default=0,
                   help="eligibility floor, independent of --offsets. Set it to "
                        "the widest offset any COMPARABLE cache uses, so a "
                        "narrower cache samples the same frames.")
    b.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--name", default="arm")
    t.add_argument("--delta", type=int, default=2)
    t.add_argument("--input", default="pair", choices=("pair", "diff", "single"))
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--downs", type=int, default=4)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--keypoints", type=int, default=32)
    t.add_argument("--head", default="spatial", choices=("spatial", "gap"))
    t.add_argument("--vel-weight", type=float, default=1.0)
    t.add_argument("--proj-mode", default="slots",
                   choices=("slots", "heatmap", "none"),
                   help="how projectiles are supervised. 'slots' regresses the "
                        "danger-slot coordinates from the pooled vector and is "
                        "what FAILED: it peaked at step 4500 and then went "
                        "negative, dragging dy and both velocities down with "
                        "it. 'heatmap' predicts an occupancy map over positions "
                        "relative to the target player and reads the coordinate "
                        "off it by soft-argmax. 'none' drops the block entirely "
                        "-- the control that tests whether the failing outputs "
                        "were costing the state channels.")
    t.add_argument("--proj-coords", action="store_true",
                   help="give the heatmap head a coordinate grid and the "
                        "encoder's own predicted player positions. Without "
                        "these it is a conv stack over a SCREEN-space feature "
                        "map asked for a PLAYER-RELATIVE occupancy map, a "
                        "frame conversion 3x3 kernels cannot express.")
    t.add_argument("--proj-weight", type=float, default=1.0,
                   help="loss weight on the projectile block. The policy was "
                        "blind to projectiles in every match so far, so this "
                        "is the channel group worth paying for.")
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    a = ap.parse_args()
    return build(a) if a.cmd == "build" else train(a)


if __name__ == "__main__":
    raise SystemExit(main())
