"""Pixels -> the FULL observation the policy eats. The real encoder.

    python -m scripts.train_encoder --corpus ~/vcorpus --replays 149 \
        --per-replay 120 --steps 12000 --out ~/rl/encoder.pt

The gate answered the narrow question -- geometry is readable and data-limited,
identity is not readable at all -- on 7 targets. The policy needs 178: two
players x 33 state channels, plus two players x 8 projectile slots x 7
features. This trains that.

LEFT/RIGHT ORDER, NOT PLAYER ORDER
-----------------------------------
Measured on the gate: asking for "player 1's x" scores R^2 0.48, asking for
"the left character's x" scores 0.83, from the same pixels and the same net.
The difference is that player order requires knowing which sprite is which, and
that bit is simply not in the play area -- seven arms across two resolutions,
three feature grids and four data scales never moved it off 0.52.

So the encoder never tries to name PLAYER order. It reports the scene in screen
order, which is what a camera can actually see.

BUT "WHICH PLAYER" AND "WHICH CHARACTER" ARE DIFFERENT QUESTIONS
----------------------------------------------------------------
Player order is invisible: nothing in the play area distinguishes P1 from P2,
which is why seven arms left that bit at 0.52. A CHARACTER is not invisible --
Reimu and Suwako do not look alike, and Soku forces different palettes in a
mirror match. So this also asks, per screen row, WHICH OF THE TWENTY characters
is standing there.

That is what retires the identity probe. `visionstate.calibrate` held a
direction for a full second and watched who moved, costing ~3 s of not playing
at the start of a match and ~6% of playing time whenever it had to be redone;
and because the answer was then PINNED, every crossup after it was wrong until
the next one. The agent knows which character it picked, so a per-row character
classifier turns that into a per-frame lookup that cannot get stuck.

Labels come from the replay header (`pipeline/repparse.py`, offsets 0x0E and
0x3F, measured over 3010 replays and confirmed against the video), staged as
`--chars`. The head is 2 x 20 logits appended AFTER the regression block, so a
checkpoint without it still loads.

TWO CHANNELS ARE EXPECTED TO FAIL AND ARE KEPT ANYWAY
-----------------------------------------------------
`hitboxes` and `hurtboxes` are counts of internal collision volumes; nothing
draws them. `untech` is a counter whose semantics are unconfirmed even in the
extractor. They are trained and REPORTED per channel rather than dropped, so
the encoder's own scorecard says which parts of the observation are real and
which are the model's best guess -- the policy will consume both, and it should
be visible which is which.
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

from sokubot.data.state import (CH, PROJ_FEATURES, STAGE_SPAN, STATE_CHANNELS,
                                read_camera, read_state, to_screen,
                                valid_camera)
from scripts.encoder_gate import Net
from sokubot.model.char_head import CharHead

N_STATE = len(STATE_CHANNELS)
N_PROJF = len(PROJ_FEATURES)

# ONLY THE CHANNELS THAT ARE ACTUALLY IN THE PIXELS.
#
# The first full-observation run supervised all 178 outputs and made the
# encoder WORSE at the things it could already do: health fell from R^2 0.98 in
# the 7-target gate to 0.65, and x from 0.83 to 0.63, on identical data and
# architecture. Twenty-seven of the thirty-three channels scored under 0.15 and
# eleven were negative -- fitting that noise costs capacity, and it was taken
# from the channels that mattered.
#
# So the encoder is asked only for what a camera can answer. Everything else is
# filled at inference with the CORPUS MEAN, which normalises to exactly zero in
# the policy's input space -- the honest encoding of "I do not know", rather
# than a guess dressed as a measurement.
#
# `vx`/`vy` are here despite scoring 0.036/0.011 single-frame, because that is
# not a visibility problem: velocity is a derivative and one frame cannot carry
# it. They get two frames now.
SUPERVISED = ("x", "y", "dx", "dy", "vx", "vy", "hp", "spirit", "airborne",
              "timestop")


N_CHAR = 20          # Soku's roster; pipeline.repparse.CHARACTER_NAMES


def build(corpus: Path, n_replays: int, per_replay: int, size: int, slots: int,
          seed: int = 0, delta: int = 2, chars: dict | None = None):
    """Frame PAIRS plus the left/right-ordered supervised channels.

    `delta` frames back for the second image. Velocity is a displacement, so
    the network needs two moments to see one; at 60 fps and a stage 1200 units
    wide a two-frame gap turns a typical speed into a handful of pixels, which
    is small but real, where a single frame carries none of it.
    """
    import cv2
    rng = np.random.default_rng(seed)
    dirs = []
    for w in sorted(corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    X, S, P, L, R, C, SX = [], [], [], [], [], [], []
    kept = 0
    for d in dirs:
        if kept >= n_replays:
            break
        vid, sc = d / "video.mp4", d / "state.csv.gz"
        if not (vid.exists() and sc.exists()):
            continue
        # A capture with no character label is skipped rather than given a
        # placeholder: a wrong identity label is worse than one fewer replay.
        lab = None if chars is None else chars.get(d.name)
        if chars is not None and lab is None:
            continue
        try:
            st, pr, _a, valid = read_state(sc)
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        idx = np.flatnonzero(ok)
        if len(idx) < per_replay * 2:
            continue
        idx = idx[idx >= delta]
        if len(idx) < per_replay * 2:
            continue
        pick = np.sort(rng.choice(idx, per_replay, replace=False))
        cap = cv2.VideoCapture(str(vid))
        frames, rows = [], []
        for f in pick:
            pair = []
            for off in (delta, 0):          # older first, then current
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - off)
                got, fr = cap.read()
                if not got:
                    break
                fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
                pair.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(pair) != 2:
                continue
            # Stored CHANNELS-FIRST. `permute(...).contiguous()` later would
            # make a second full copy of the whole array, and with two stacked
            # frames that copy is what took the machine down: 5.4 GB of samples
            # became 14 GB resident and the kernel killed it.
            frames.append(np.concatenate(pair, axis=2).transpose(2, 0, 1))
            rows.append(f)
        cap.release()
        if len(frames) < per_replay // 2:
            continue
        rows = np.array(rows)
        s_sel, p_sel = st[rows], pr[rows][:, :, :slots]
        # p1_is_left: dx is x_opponent - x_me for player 0, so dx > 0 means
        # player 1 sits to the LEFT of player 2.
        p1_left = (s_sel[:, 0, CH["dx"]] > 0)
        order = np.where(p1_left[:, None], np.array([0, 1]), np.array([1, 0]))
        take = np.take_along_axis(s_sel, order[:, :, None], axis=1)
        takep = np.take_along_axis(p_sel, order[:, :, None, None], axis=1)
        X.append(np.stack(frames))
        S.append(take.astype(np.float32))
        P.append(takep.astype(np.float32))
        L.append(p1_left.astype(np.float32))
        R.append(np.full(len(frames), kept, np.int32))
        if chars is not None:
            # Reordered by the SAME p1_left permutation as the state, so the
            # label describes the row the network is looking at.
            pair = np.array([lab["p1_char"], lab["p2_char"]], np.int64)
            C.append(np.where(p1_left[:, None], pair[None, :], pair[::-1][None, :]))
            # WHERE each row actually is, in the frame the network sees. This
            # is the whole point of the camera columns: `take` is world-space
            # and left/right ordered, so projecting it gives the screen x of
            # the character the head is being asked to name. Zeros (and a
            # False mask) for a capture predating the camera columns.
            cam = read_camera(sc)
            if len(cam) >= int(rows.max()) + 1 and valid_camera(cam)[rows].all():
                wx = take[:, :, CH["x"]] * STAGE_SPAN
                wy = take[:, :, CH["y"]] * STAGE_SPAN
                sx, _sy = to_screen(wx, wy, cam[rows])
                SX.append(sx.astype(np.float32))
            else:
                SX.append(np.full((len(frames), 2), np.nan, np.float32))
        kept += 1
        if kept % 20 == 0:
            print(f"  {kept}/{n_replays} replays, {sum(len(x) for x in X)} frames",
                  flush=True)
    if not X:
        raise SystemExit("no usable replays")
    return (np.concatenate(X), np.concatenate(S), np.concatenate(P),
            np.concatenate(L), np.concatenate(R),
            np.concatenate(C) if C else None,
            np.concatenate(SX) if SX else None)


def _decide_acc(logits: "torch.Tensor", truth: "torch.Tensor") -> float:
    """The metric the head exists for, not the one it is trained on.

    Per-row top-1 accuracy answers "can it name the character". What the live
    loop asks is narrower and easier: GIVEN the character the agent picked,
    which of the two rows is it? So score each frame by comparing the two rows'
    logit for that one class and taking the larger.

    Both rows are used as the query in turn, so a head that always prefers the
    left row cannot score above chance. Mirror matches are excluded from the
    average and reported separately by the caller if wanted -- with both rows
    the same class the question has no answer from character alone, which is
    what palette is for.
    """
    ok, n = 0.0, 0
    for row in (0, 1):
        q = truth[:, row]                       # "my" character
        keep = truth[:, 0] != truth[:, 1]       # not a mirror match
        if keep.sum() == 0:
            continue
        sl = logits[keep, 0].gather(1, q[keep, None]).squeeze(1)
        sr = logits[keep, 1].gather(1, q[keep, None]).squeeze(1)
        pick_left = sl > sr
        correct = pick_left if row == 0 else ~pick_left
        ok += float(correct.float().sum())
        n += int(keep.sum())
    return ok / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--replays", type=int, default=149)
    ap.add_argument("--per-replay", type=int, default=120)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--downs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--delta", type=int, default=2,
                    help="frames between the two stacked images")
    ap.add_argument("--chars", type=Path, default=None,
                    help="JSON of capture-dir -> {p1_char,p2_char,...}. With "
                         "it the encoder also classifies each screen row's "
                         "character, which is what replaces the identity probe.")
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz to hold the decoded frames. Built on first use "
                         "and reused after. Decoding 100 replays is minutes of "
                         "the run and identical across arms, so an ablation "
                         "that rebuilds it per arm pays for it every time and "
                         "-- worse -- lets the frame SAMPLE differ between "
                         "arms that are supposed to differ in one thing.")
    ap.add_argument("--char-head", default="attn", choices=("attn", "linear"),
                    help="`linear` appends 2*n_char logits to the shared output "
                         "layer -- measured at identity-decision 0.4987, i.e. "
                         "chance, because a soft-argmax head pooled globally "
                         "cannot say WHICH of two things it is looking at. "
                         "`attn` pools the feature map under two learned "
                         "attention maps and orders them by x centroid, so "
                         "left/right is computed rather than predicted.")
    ap.add_argument("--attn-supervise", type=float, default=1.0,
                    help="weight pulling each attention map's x centroid onto "
                         "the character it is meant to be naming, using the "
                         "camera columns. Without this the maps settle on "
                         "fixed screen regions -- measured correlation to the "
                         "characters' real separation was +0.0012 -- and the "
                         "head names characters at 8x chance while deciding "
                         "identity at chance. 0 disables it.")
    ap.add_argument("--char-weight", type=float, default=1.0,
                    help="weight on the character loss. An auxiliary task "
                         "competes for capacity whatever its form -- tripling "
                         "the projectile weight cost ~0.013 of state R2 either "
                         "way -- so this is reported, not assumed harmless.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    chars = json.loads(a.chars.read_text()) if a.chars else None
    if chars:
        print(f"  character labels for {len(chars)} captures", flush=True)
    if a.cache and a.cache.exists():
        z = np.load(a.cache)
        X, S, P, L, Rep = z["X"], z["S"], z["P"], z["L"], z["Rep"]
        C = z["C"] if "C" in z.files and z["C"].size else None
        SX = z["SX"] if "SX" in z.files and z["SX"].size else None
        # A cache built WITH labels must still produce a control arm when
        # --chars is omitted, or the control silently grows the head it exists
        # to be compared against.
        if chars is None:
            C = SX = None
        print(f"  cache {a.cache} -> {len(X)} frames "
              f"(chars {'yes' if C is not None else 'no'})", flush=True)
        if chars is not None and C is None:
            raise SystemExit(
                f"{a.cache} was built without character labels but --chars was "
                "given. Delete it and rebuild, rather than training a head on "
                "a cache that cannot supply its target.")
    else:
        X, S, P, L, Rep, C, SX = build(a.corpus, a.replays, a.per_replay,
                                       a.size, a.slots, a.seed, a.delta, chars)
        if a.cache:
            np.savez(a.cache, X=X, S=S, P=P, L=L, Rep=Rep,
                     C=C if C is not None else np.zeros(0, np.int64),
                     SX=SX if SX is not None else np.zeros(0, np.float32))
            print(f"  cached -> {a.cache} "
                  f"({a.cache.stat().st_size / 1e9:.1f} GB)", flush=True)
    # What the unsupervised channels become at inference: the corpus mean, which
    # the policy's normaliser maps to exactly zero.
    fill = S.reshape(-1, N_STATE).mean(0)
    print(f"  {len(X)} frames in {(time.time()-t0)/60:.1f} min, "
          f"{X.nbytes/1e9:.2f} GB (peak needs ~2x this during the split)",
          flush=True)

    sup = [CH[n] for n in SUPERVISED]
    n_state = 2 * len(sup)
    n_proj = 0                      # dropped: the block scored R2 0.042
    Y = S[:, :, sup].reshape(len(S), -1)
    print(f"  supervising {len(SUPERVISED)} of {N_STATE} channels x2 = "
          f"{n_state} outputs: {', '.join(SUPERVISED)}", flush=True)
    print(f"  projectiles NOT predicted (measured R2 0.042); filled as absent",
          flush=True)

    reps = np.unique(Rep)
    val = set(rng.choice(reps, max(1, int(len(reps) * a.val_frac)),
                         replace=False).tolist())
    vm = np.isin(Rep, list(val))
    dev = a.device
    Xtr = torch.from_numpy(np.ascontiguousarray(X[~vm]))
    Ytr, Ltr = torch.from_numpy(Y[~vm]), torch.from_numpy(L[~vm])
    Xva = torch.from_numpy(np.ascontiguousarray(X[vm]))
    Yva, Lva = torch.from_numpy(Y[vm]).to(dev), torch.from_numpy(L[vm]).to(dev)
    Ctr = torch.from_numpy(C[~vm]) if C is not None else None
    Cva = torch.from_numpy(C[vm]).to(dev) if C is not None else None
    SXtr = torch.from_numpy(SX[~vm]) if SX is not None else None
    SXva = torch.from_numpy(SX[vm]).to(dev) if SX is not None else None
    # PER-SAMPLE, NOT ALL-OR-NOTHING.
    #
    # This used to require every screen position in the cache to be finite. A
    # capture predating the camera columns -- or one where a single sampled
    # frame fell on an invalid rect -- would then switch supervision off for
    # the WHOLE run while printing one line, and the arm would quietly become
    # the unsupervised arm it was meant to be compared against. The camera is
    # valid on 99.3% of frames, not 100%, so that was a live risk rather than a
    # hypothetical one.
    SXok_tr = (torch.from_numpy(np.isfinite(SX[~vm]).all(1))
               if SX is not None else None)
    have_sx = SX is not None and bool(np.isfinite(SX).any())
    if SX is not None:
        frac = float(np.isfinite(SX).all(1).mean())
        print(f"  screen positions on {frac:.4f} of frames; attention "
              f"supervision {'ON' if have_sx else 'OFF'}", flush=True)
    # The source array is now redundant and is the largest thing in the process.
    del X, S, P
    import gc; gc.collect()
    import resource
    print(f"  peak RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB",
          flush=True)
    print(f"  train {len(Xtr)} / val {len(Xva)}  (SPLIT BY REPLAY)", flush=True)

    mu, sd = Ytr.mean(0), Ytr.std(0).clamp(min=1e-4)
    mu_d, sd_d = mu.to(dev), sd.to(dev)
    # Output layout, and it is load-bearing that the regression block stays
    # FIRST: [state | p1_left | char_left | char_right]. `mu`/`sd` index the
    # first n_state outputs, so appending never disturbs them and a checkpoint
    # trained without the character head still loads.
    n_char = N_CHAR if C is not None else 0
    attn_char = bool(n_char) and a.char_head == "attn"
    # With the attention head the character logits leave the shared output
    # layer entirely, so the main head keeps exactly the shape it had before
    # the identity work started.
    n_tail = 0 if attn_char else 2 * n_char
    net = Net(n_state + 1 + n_tail, head="spatial", downs=a.downs,
              in_ch=6).to(dev)
    i_id = n_state                      # the p1_left logit
    i_ch = n_state + 1                  # 2 * n_char logits, `linear` head only
    chead = None
    if attn_char:
        with torch.no_grad():
            _, _f = net(torch.zeros(1, 6, a.size, a.size, device=dev),
                        return_feat=True)
        chead = CharHead(_f.shape[1], n_char).to(dev)
        print(f"  character head: attention-pooled, "
              f"{sum(q.numel() for q in chead.parameters()) / 1e3:.0f}k params",
              flush=True)
    print(f"  net {sum(p.numel() for p in net.parameters())/1e6:.2f}M params",
          flush=True)
    params = list(net.parameters()) + (list(chead.parameters()) if chead else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.1)
    best = -1e9
    for step in range(a.steps):
        i = torch.from_numpy(rng.choice(len(Xtr), a.batch))
        xb = Xtr[i].to(dev).float().div_(255.0)
        o, feat = net(xb, return_feat=True)
        loss = (F.mse_loss(o[:, :n_state], (Ytr[i].to(dev) - mu_d) / sd_d)
                + F.binary_cross_entropy_with_logits(o[:, i_id], Ltr[i].to(dev)))
        if n_char:
            cb = Ctr[i].to(dev)                       # [B, 2] left, right
            if attn_char:
                cl, cx = chead(feat)
            else:
                cl = o[:, i_ch:i_ch + 2 * n_char].view(-1, 2, n_char)
                cx = None
            loss = loss + a.char_weight * 0.5 * (
                F.cross_entropy(cl[:, 0], cb[:, 0])
                + F.cross_entropy(cl[:, 1], cb[:, 1]))
            if attn_char and have_sx and a.attn_supervise > 0:
                # Put the attention where the character IS. The head already
                # sorts its two maps by x centroid, so target row 0 with the
                # left character and row 1 with the right -- the same ordering
                # the class labels use.
                m_ = SXok_tr[i].to(dev)
                if m_.any():
                    loss = loss + a.attn_supervise * F.mse_loss(
                        cx[m_], SXtr[i].to(dev)[m_])
        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                pr, pc, px = [], [], []
                for j in range(0, len(Xva), 128):
                    ob, fb = net(Xva[j:j+128].to(dev).float().div(255.0),
                                 return_feat=True)
                    pr.append(ob)
                    if attn_char:
                        _cl, _cx = chead(fb)
                        pc.append(_cl); px.append(_cx)
                pr = torch.cat(pr)
                pc = torch.cat(pc) if pc else None
                px = torch.cat(px) if px else None
                pred = pr[:, :n_state] * sd_d + mu_d
                ssr = ((pred - Yva) ** 2).sum(0)
                sst = ((Yva - Yva.mean(0)) ** 2).sum(0).clamp(min=1e-9)
                r2 = (1 - ssr / sst).cpu().numpy()
                idacc = float(((pr[:, i_id] > 0).float() == Lva).float().mean())
                chacc, decacc = float("nan"), float("nan")
                attn_r, attn_mae = float("nan"), float("nan")
                if n_char:
                    cl = (pc if attn_char
                          else pr[:, i_ch:i_ch + 2 * n_char].view(-1, 2, n_char))
                    chacc = float((cl.argmax(-1) == Cva).float().mean())
                    decacc = _decide_acc(cl, Cva)
                    if px is not None and SXva is not None:
                        # Is the attention actually ON the characters? The
                        # unsupervised head scored +0.0012 here, which is what
                        # made it useless despite naming characters at 8x
                        # chance.
                        a_ = px.flatten().float()
                        b_ = SXva.flatten().float()
                        fin = torch.isfinite(b_) & torch.isfinite(a_)
                        if fin.sum() > 8:
                            a_, b_ = a_[fin], b_[fin]
                            attn_mae = float((a_ - b_).abs().mean())
                            attn_r = float(torch.corrcoef(
                                torch.stack([a_, b_]))[0, 1])
            net.train()
            key = {n: float(r2[SUPERVISED.index(n)]) for n in
                   ("x", "y", "dx", "vx", "hp", "spirit")}
            mean_state = float(np.mean(r2[:n_state]))
            if mean_state > best:
                best = mean_state
                torch.save({"net": net.state_dict(), "mu": mu, "sd": sd,
                            "size": a.size, "slots": a.slots, "downs": a.downs,
                            "n_state": n_state, "n_proj": n_proj,
                            "n_char": n_char,
                            "char_head": a.char_head if n_char else None,
                            "char_state": (chead.state_dict() if chead else None),
                            "feat_ch": (int(_f.shape[1]) if attn_char else 0),
                            "supervised": list(SUPERVISED), "delta": a.delta,
                            "fill": fill.tolist(),
                            "r2": r2.tolist(), "id_acc": idacc,
                            "char_acc": chacc, "decide_acc": decacc,
                            "attn_r": attn_r, "attn_mae": attn_mae,
                            "step": step},
                           a.out)
            extra = "" if not n_char else f" | char {chacc:.3f} decide {decacc:.3f}"
            if n_char and np.isfinite(attn_r):
                extra += f" attn r {attn_r:+.3f} mae {attn_mae:.3f}"
            print(f"step {step:6d} | mean R2 {mean_state:+.3f} | x "
                  f"{key['x']:+.3f} dx {key['dx']:+.3f} vx {key['vx']:+.3f} "
                  f"hp {key['hp']:+.3f} spirit {key['spirit']:+.3f} "
                  f"y {key['y']:+.3f} | id {idacc:.3f}{extra}", flush=True)

    d = torch.load(a.out, map_location="cpu", weights_only=False)
    r2 = np.array(d["r2"])
    print(f"\nbest checkpoint: step {d['step']}, mean state R2 "
          f"{np.mean(r2[:n_state]):+.4f}\n")
    print("  per-channel R2, worst first:")
    per = r2[:len(SUPERVISED)]
    for i in np.argsort(per):
        print(f"    {SUPERVISED[i]:<12} {per[i]:+.4f}")
    print(f"  p1_left accuracy {d['id_acc']:.4f}  <- expected to be chance. "
          f"Nothing in the play area names Player 1.")
    if d.get("n_char"):
        print(f"  character accuracy {d['char_acc']:.4f} per row "
              f"(chance {1 / N_CHAR:.3f})")
        print(f"  IDENTITY DECISION  {d['decide_acc']:.4f}  <- the number that "
              f"matters: given the character the agent picked, does it pick the "
              f"right row? Compare against the live nearest-neighbour tracker's "
              f"0.51 and the 3 s probe it replaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
