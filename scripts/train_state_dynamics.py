"""Train the state-space simulator, and gate it on the mechanic it exists for.

    python -m scripts.train_state_dynamics --corpus ~/corpus --steps 20000

NO VIDEO IS READ. The whole corpus of state is 1.8 GB against 63.5 GB of
frames, so this trains on a 4 GB card from CSVs, and the thing RL rolls forward
in never touches a pixel. See model/state_dynamics.py for the seven
measurements that led here.

THE GATE IS NOT THE LOSS
------------------------
A simulator that scores well on next-step MSE can still be useless for RL: at
frame-skip 5 most channels barely move, so "copy the input" is already a good
prediction. Two things are therefore reported, and the second is the one that
decides:

  skill      1 - mse/identity_mse over an AUTOREGRESSIVE rollout, per horizon.
             0 means the model is no better than standing still.
  block_gain does holding AWAY from the opponent produce more guarding in the
             simulator than holding TOWARD? This is `scripts/block_effect.py`
             asked in state space. On the pixel-latent model it was flat --
             -0.00025 +- 0.00761 -- which is precisely why that model could
             not teach blocking however long GRPO ran against it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.state import CH, PROJ_FEATURES, STATE_CHANNELS, read_state
from sokubot.model.state_dynamics import (StateDynamics, build_move_vocab,
                                           index_moves, rollout, unroll)
from sokubot.model.state_head import (BINARY, CONTINUOUS, default_pos_weight,
                                      projectile_loss)

BUTTONS = ("up", "down", "left", "right", "a", "b", "c", "d", "change", "spell")


def load_sequences(corpus: Path, replays: int, slots: int, skip: int,
                   name: str = "") -> tuple[np.ndarray, ...]:
    """-> (state [N,2,C], proj [N,2,K,F], actions [N,ticks,20], ep [N],
    move [N,2]).

    `move` is the nominal action id, which `read_state` has always returned and
    every consumer has always thrown away.
    """
    dirs = []
    for w in sorted(corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    if not dirs:
        dirs = [d for d in sorted(corpus.iterdir()) if d.is_dir()]

    S, P, A, E, M = [], [], [], [], []
    kept = 0
    for d in dirs:
        # 0 means ALL, as it does in state_bank.find_replays. Read literally it meant "stop
        # before the first one", and `--replays 0` trained on nothing.
        if replays and kept >= replays:
            break
        sc = next((c for c in ([d / name] if name else
                               [d / "state_s5.csv", d / "state.csv.gz",
                                d / "inputs.csv.gz"]) if c.exists()), None)
        inp = next((c for c in (d / "inputs.csv", d / "inputs.csv.gz")
                    if c.exists()), None)
        if sc is None or inp is None:
            continue
        try:
            st, pr, mv, valid = read_state(sc)
            btn = read_actions(inp)
        except (ValueError, OSError, KeyError):
            continue
        # A pre-strided sidecar is already at decision rate; a full-rate one is
        # not. Getting this backwards pairs state at frame t with the buttons
        # from frame 5t, which trains the simulator on a lie.
        strided = sc.name.startswith("state_s")
        if strided and skip != 5:
            # state_s5.csv IS the stride-5 subsample; re-striding it would
            # silently produce a 25-frame step labelled as 5.
            continue
        n = len(st) if strided else len(st) // skip
        rows = np.arange(n) if strided else np.arange(n) * skip
        if len(rows) < 32:
            continue
        # Action CHUNK for decision step d covers source frames [d*skip, +skip).
        chunks = np.stack([btn[i * skip:i * skip + skip] for i in range(n)
                           if i * skip + skip <= len(btn)])
        n = min(n, len(chunks))
        if n < 32:
            continue
        S.append(st[rows][:n]); P.append(pr[rows][:n, :, :slots])
        A.append(chunks[:n]); E.append(np.full(n, kept, np.int32))
        M.append(mv[rows][:n])
        kept += 1
    if not S:
        raise SystemExit(f"no usable sidecars under {corpus}")
    print(f"  {kept} replays, {sum(len(s) for s in S)} decision steps")
    return (np.concatenate(S), np.concatenate(P), np.concatenate(A),
            np.concatenate(E), np.concatenate(M))


def cached_sequences(corpus: Path, replays: int, slots: int, skip: int,
                     cache: Path | None = None, name: str = ""):
    """`load_sequences` behind a verified cache.

    Parsing hundreds of full-rate state CSVs is the slowest and least reliable
    step in every consumer of this corpus -- it takes minutes, and on one box
    it segfaulted inside libpython on 4 of 7 attempts with identical arguments.
    Doing it once and writing an artifact makes every later run fast AND makes
    the fragile step something that either succeeded or did not, rather than
    something each run re-rolls the dice on.

    The cache is READ BACK before it is trusted. A file written through a
    memory fault looks exactly like a good one until something consumes it,
    and by then it is the silent input to everything downstream.
    """
    key = f"{replays}_{slots}_{skip}_{name}"
    if cache and Path(cache).exists():
        d = np.load(cache)
        if str(d["key"]) == key:
            print(f"  corpus from cache {cache}", flush=True)
            return d["S"], d["P"], d["A"], d["E"], d["M"]
        print(f"  cache key {str(d['key'])!r} != {key!r}, rebuilding", flush=True)
    out = load_sequences(corpus, replays, slots, skip, name)
    if not np.isfinite(out[0]).all():
        raise SystemExit("loaded state contains non-finite values")
    if cache:
        cache = Path(cache)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, S=out[0], P=out[1], A=out[2], E=out[3], M=out[4], key=key)
        d = np.load(tmp)
        if not (np.array_equal(d["S"], out[0]) and np.array_equal(d["M"], out[4])
                and np.array_equal(d["A"], out[2])):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{tmp}: read-back does not match what was "
                             f"written; refusing to keep it")
        tmp.replace(cache)
        print(f"  corpus cached to {cache} (verified)", flush=True)
    return out


def read_actions(path: Path) -> np.ndarray:
    """inputs.csv -> [N, 20] float, p1 buttons then p2 buttons."""
    import csv, gzip
    op = gzip.open if path.suffix == ".gz" else open
    cols = [f"p{p}_{b}" for p in (1, 2) for b in BUTTONS]
    with op(path, "rt", newline="") as fh:
        r = csv.DictReader(fh)
        return np.array([[float(row[c]) for c in cols] for row in r],
                        dtype=np.float32)


def guard_onset_windows(S, idx, H):
    """Of the legal windows, the ones where a block actually STARTS.

    Not "guarding is on somewhere in the window": a block already in progress
    is predicted by copying the flag forward, which is the part the model
    already does. The thing it fails at is the TRANSITION -- the frame where an
    attack arrives and the defender's guard comes up -- so that is what the gym
    is made of.

    Either player, because the simulator predicts both and the corpus has each
    of them defending about half the time.
    """
    g = S[:, :, CH["guarding"]] > 0.5              # [N, 2]
    onset = np.zeros(len(S), bool)
    onset[:-1] = (g[1:] & ~g[:-1]).any(1)
    # A window is a training example for its own prediction targets, which are
    # steps 1..H of the span, so an onset anywhere in that range counts.
    w = idx[:, None] + np.arange(1, H + 1)[None, :]
    return idx[onset[w].any(1)]


def attack_arrival_frames(S):
    """Frames where an attack goes LIVE -- the step a blocking decision resolves.

    Selecting WINDOWS containing an arrival was tried first and does nothing: a
    12-step window spans 60 frames, so 31.6% of them already contain an arrival
    somewhere, and drawing 30% of each batch from those changes the batch
    barely at all. The arrival is one step of the twelve a window is scored on,
    so the weight has to be per step.

    Either player's hitbox, because the simulator predicts both and each is
    attacking about half the time.
    """
    hb = S[:, :, CH["hitboxes"]] > 0.01                 # [N, 2]
    live = np.zeros(len(S), bool)
    live[1:] = (hb[1:] & ~hb[:-1]).any(1)
    return live


def batches(S, P, A, E, H, bs, steps, rng, device, M=None, guard_frac=0.0,
            ct=None):
    """Windows of H+1 that never cross a replay boundary.

    `guard_frac` replaces that share of every batch with windows containing a
    guard onset. THE GYM: if the simulator cannot predict blocking because
    blocking is 4% of frames, this fixes it; if it cannot predict blocking
    because the 33 channels do not determine it, this changes nothing. The two
    have been indistinguishable in every measurement so far and they imply
    completely different next steps.
    """
    ok = np.ones(len(S), bool)
    ok[-(H + 1):] = False
    for k in range(1, H + 2):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    idx = np.flatnonzero(ok)
    gidx = guard_onset_windows(S, idx, H) if guard_frac > 0 else None
    if gidx is not None:
        print(f"  guard gym: {len(gidx)} of {len(idx)} windows contain a guard "
              f"onset ({100*len(gidx)/len(idx):.2f}%), drawing "
              f"{guard_frac:.0%} of each batch from them", flush=True)
        if len(gidx) < 100:
            raise SystemExit("too few guard-onset windows to train a gym on")
    n_g = int(round(bs * guard_frac)) if gidx is not None else 0
    for _ in range(steps):
        b = rng.choice(idx, bs - n_g) if n_g else rng.choice(idx, bs)
        if n_g:
            b = np.concatenate([b, rng.choice(gidx, n_g)])
        w = b[:, None] + np.arange(H + 1)[None, :]
        yield (torch.as_tensor(S[w]).to(device),
               torch.as_tensor(P[w]).to(device),
               torch.as_tensor(A[w]).to(device),
               None if M is None else torch.as_tensor(M[w]).to(device),
               None if ct is None else
               torch.as_tensor(ct[w]).float().to(device))


# A channel whose one-step change has a standard deviation below this carries
# no signal to predict. `ax` is exactly 0.0 -- the horizontal gravity term never
# changes -- and several others fall here once the step shrinks.
DEAD_STD = 1e-4


def delta_scale(S, E, cont_idx):
    """-> (scale, live_mask) for standardising the continuous loss.

    THE CLAMP WAS THE BUG
    ---------------------
    This used to standardise by `max(std, 1e-2)`, which does something
    catastrophic to a channel that does not vary: `ax` has a true one-step
    delta std of EXACTLY 0.0, so the clamp handed it a divisor of 0.01 and any
    float noise the model emitted on it was scored as a large error. The
    objective then spent capacity fitting noise on dead channels, which is why
    every run showed the loss falling while every rollout horizon degraded and
    the best checkpoint was always early.

    It also explains why frame_skip=1 was WORSE rather than better: at 16.7 ms
    steps 11 of 28 continuous channels fall under the old clamp, against 2 of
    28 at 83 ms. Shrinking the step made more channels effectively constant, so
    more of the loss became noise-fitting. The step size was never the variable.

    So dead channels are now EXCLUDED rather than amplified, and the survivors
    are standardised by their real spread with no clamp to distort them.
    """
    same = np.zeros(len(S), bool)
    same[:-1] = E[1:] == E[:-1]
    d = (S[1:][same[:-1]][:, :, cont_idx]
         - S[:-1][same[:-1]][:, :, cont_idx]).reshape(-1, len(cont_idx))
    sd = d.std(0)
    # A NON-FINITE SCALE IS NOT A DEAD CHANNEL.
    #
    # `sd > DEAD_STD` is False for NaN as well as for zero, so a corrupted read
    # arrives here disguised as "this channel carries no signal" and the run
    # continues with a quiet line in the log. That happened: one arm dropped
    # `spirit` and `spirit_delay`, went NaN at step 500 and kept training for
    # 1500 more steps while its twin on identical data was fine. Dead channels
    # are a real and expected condition; non-finite ones are a broken run.
    if not np.isfinite(sd).all():
        bad = [STATE_CHANNELS[cont_idx[j]] for j in range(len(sd))
               if not np.isfinite(sd[j])]
        raise SystemExit(
            f"delta scale is non-finite for {bad}; the loaded state contains "
            f"NaN or inf. This is a corrupt read, not a dead channel -- "
            f"re-run, and if it repeats check the filesystem.")
    live = sd > DEAD_STD
    # The divisor for a dead channel is irrelevant -- the mask zeroes it -- but
    # it must not be 0 or the division produces NaN before the mask applies.
    return (np.where(live, sd, 1.0).astype(np.float32),
            live.astype(np.float32))


def flag_weights(S, binr, ref=0.045):
    """Per-flag loss weight, inversely proportional to the flag's entropy.

    A summed per-channel loss prices a channel by how COMMON it is.
    `guarding` fires on 4.5% of frames, so its whole entropy is 0.185 nats and
    it is worth about 0.5% of the binary term -- the model allocates capacity
    accordingly and is right to. Measured against the corpus, the action
    explains 15% of the uncertainty that remains about guarding after the
    state is known, against 21% for `airborne`: the two are comparable in the
    only units that matter, and only the raw nats make blocking look
    negligible. Weighting by 1/H equalises what each channel is asked to
    contribute rather than letting base rate decide it.

        CAPPED AT A REFERENCE RATE, because 1/H rewards rarity and rarity is not
    learnability. Uncapped, `crushed` -- 0.15% of frames, and measured to carry
    0.1% of its remaining uncertainty in the action -- takes 16x the weight of
    `guarding`, i.e. the objective would spend most of its binary budget on the
    least predictable channel in the state. Capping at the rate of the channel
    this is meant to rescue treats everything rarer than guarding the same as
    guarding, and still discounts the common ones.
    """
    q = np.clip(S[:, :, binr].reshape(-1, len(binr)).mean(0), 1e-5, 1 - 1e-5)
    H = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    r = float(ref)
    cap = 1.0 / -(r * np.log(r) + (1 - r) * np.log(1 - r))
    w = np.minimum(1.0 / H, cap)
    return w / w.mean()


def step_loss(model, s, p, a, pw, dsc, live, unroll_steps=0, unroll_w=1.0,
              mv=None, move_w=1.0, fw=None, cw=None, idm_w=0.0,
              horizon_w=None):
    """Teacher-forced next-step loss, plus an autoregressive term.

    The rollout term is what makes this a SIMULATOR rather than a one-step
    predictor: RL consumes the model's own output, and a model never asked to
    do that learns average dynamics that look fine at h16 and are worse than a
    no-op at h1.
    """
    ns, np_, mlog = model(s[:, :-1], p[:, :-1], a[:, :-1],
                          None if mv is None else mv[:, :-1], want_moves=True)
    tgt_s, tgt_p = s[:, 1:], p[:, 1:]
    cont = torch.tensor(CONTINUOUS, device=s.device)
    binr = torch.tensor(BINARY, device=s.device)
    # `live` zeroes the channels with nothing to predict; the mean is over the
    # live ones only, so dropping a channel does not quietly shrink the loss.
    err = ((ns.index_select(-1, cont) - tgt_s.index_select(-1, cont)) / dsc) ** 2
    l_c = (err * live).sum() / (live.sum() * err.shape[0] * err.shape[1]
                                * err.shape[2])
    l_b_raw = F.binary_cross_entropy_with_logits(
        ns.index_select(-1, binr), tgt_s.index_select(-1, binr),
        pos_weight=pw.to(s.device), reduction="none")
    if fw is not None:
        l_b_raw = l_b_raw * fw.to(s.device)
    # `cw` concentrates the gradient on frames where the action can matter at
    # all: 62.6% of corpus frames are in untech, hitstop or knockdown, where
    # no input changes anything, and averaging over them dilutes the
    # action-conditional signal by about three times.
    l_b = (l_b_raw * cw).sum() / cw.sum() / l_b_raw.shape[-1] \
        if cw is not None else l_b_raw.mean()
    l_p, pm = projectile_loss(np_, tgt_p)
    # 0.3 on the binary term: it is a classification over flags that are
    # already rare, and at weight 1.0 with pos_weights it swamped the
    # kinematics the rollout depends on.
    total = l_c + 0.3 * l_b + 0.3 * l_p
    m = {"cont": float(l_c.detach()), "bin": float(l_b.detach()), **pm}

    if idm_w > 0 and model.head_idm is not None:
        # Read the action back out of the model's own prediction. Reported as
        # accuracy per button so a degenerate solution is visible: if idm_acc
        # climbs while the action-sensitivity probe does not, the model is
        # smuggling the action into a channel nobody checks rather than making
        # its dynamics depend on it.
        il = model.idm_logits(s[:, :-1], ns)
        at = (a[:, :-1].reshape(*il.shape) > 0.5).float()
        l_i = F.binary_cross_entropy_with_logits(il, at)
        total = total + idm_w * l_i
        m["idm"] = float(l_i.detach())
        m["idm_acc"] = float(((il > 0) == (at > 0.5)).float().mean())

    if mlog is not None:
        tgt_m = mv[:, 1:].long()
        l_m = F.cross_entropy(mlog.reshape(-1, mlog.shape[-1]),
                              tgt_m.reshape(-1))
        total = total + move_w * l_m
        m["move"] = float(l_m.detach())
        m["move_acc"] = float((mlog.argmax(-1) == tgt_m).float().mean())

    if unroll_steps > 0:
        H = s.shape[1] - unroll_steps
        if H >= 2:
            pred = unroll(model, s[:, :H], p[:, :H], a, unroll_steps,
                          None if mv is None else mv[:, :H])
            tgt = s[:, H:H + unroll_steps]
            e = ((pred.index_select(-1, cont) - tgt.index_select(-1, cont))
                 / dsc) ** 2
            if horizon_w is None:
                l_r = (e * live).sum() / (live.sum() * e.shape[0] * e.shape[1]
                                          * e.shape[2])
                total = total + unroll_w * l_r
                m["unroll"] = float(l_r.detach())
            else:
                # ONE LOSS PER HORIZON, WEIGHTED BY LEARNED UNCERTAINTY.
                #
                # A free weight per horizon collapses: gradient descent puts it
                # where the loss is already lowest, which is k=1, and teacher
                # forcing comes back through the side door. Kendall & Gal's
                # form has the log term, so the optimum is sigma_k^2 = L_k --
                # each horizon is normalised by its OWN difficulty instead of
                # abandoned for being hard.
                per = []
                for k in range(e.shape[1]):
                    ek = e[:, k]
                    per.append((ek * live).sum()
                               / (live.sum() * ek.shape[0] * ek.shape[1]))
                ls = horizon_w[:len(per)]
                l_r = sum(pk / (2 * torch.exp(ls[i])) + ls[i] / 2
                          for i, pk in enumerate(per))
                total = total + unroll_w * l_r
                m["unroll"] = float(sum(per).detach() / len(per))
                m["hw"] = " ".join(f"{float(torch.exp(x/2)):.2f}" for x in ls)
    return total, m


@torch.no_grad()
def eval_rollout(model, S, P, A, E, H, horizons, device, n=256, seed=0,
                 M=None):
    """Autoregressive skill per horizon, against standing still."""
    rng = np.random.default_rng(seed)
    span = H + max(horizons)
    ok = np.ones(len(S), bool); ok[-span:] = False
    for k in range(1, span + 1):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    idx = rng.choice(np.flatnonzero(ok), min(n, int(ok.sum())), replace=False)
    w = idx[:, None] + np.arange(span)[None, :]
    s = torch.as_tensor(S[w]).to(device); p = torch.as_tensor(P[w]).to(device)
    a = torch.as_tensor(A[w]).to(device)
    mv = None if M is None else torch.as_tensor(M[w][:, :H]).to(device)
    pred = rollout(model, s[:, :H], p[:, :H], a, max(horizons), mv)
    cont = torch.tensor(CONTINUOUS, device=device)

    # STANDARDISED, AND SPLIT BY GROUP.
    #
    # A pooled raw MSE over 28 incommensurable channels is not a measurement.
    # Measured on this corpus it is 57.8% `action_frame` and 41.3% `untech` --
    # two frame counters normalised by 60 that reach magnitudes of 441 -- which
    # leaves under 1% for position, velocity, health and every flag combined.
    # That number reported -5.42 for a model whose kinematic skill at the same
    # horizon was +0.61, and three runs were abandoned or redesigned on it.
    sc = torch.as_tensor(S[:, :, np.array(CONTINUOUS)].reshape(
        -1, len(CONTINUOUS)).std(0).clip(1e-6)).to(device)
    names = [STATE_CHANNELS[i] for i in CONTINUOUS]
    KIN = [names.index(x) for x in
           ("dx", "dy", "x", "y", "vx", "vy", "ay") if x in names]
    kin = torch.tensor(KIN, device=device)
    out = {}
    for h in horizons:
        tgt = s[:, H + h - 1].index_select(-1, cont) / sc
        pr = pred[:, h - 1].index_select(-1, cont) / sc
        # Identity: the last OBSERVED state, held. Any skill above 0 is skill
        # over a simulator that predicts nothing ever changes.
        idn = s[:, H - 1].index_select(-1, cont) / sc
        out[f"skill_h{h}"] = float(
            1.0 - F.mse_loss(pr.index_select(-1, kin),
                             tgt.index_select(-1, kin))
            / F.mse_loss(idn.index_select(-1, kin),
                         tgt.index_select(-1, kin)).clamp(min=1e-9))
        out[f"all_h{h}"] = float(
            1.0 - F.mse_loss(pr, tgt) / F.mse_loss(idn, tgt).clamp(min=1e-9))

    # PER-CHANNEL ERROR IN EACH CHANNEL'S OWN SIGMA.
    #
    # The aggregate skill numbers cannot show the thing that matters here.
    # `guarding` sits at 0.51 sigma by step two and stays flat while every
    # other channel degrades smoothly -- flat-from-the-start is the signature
    # of a variable the model cannot see, and it is invisible inside a mean
    # over 28 channels.
    gi = CH["guarding"]
    gsig = float(S[:, :, gi].std().clip(1e-6))
    for h in horizons:
        g_pr = torch.sigmoid(pred[:, h - 1, :, gi])
        g_tg = s[:, H + h - 1, :, gi]
        out[f"guard_sigma_h{h}"] = float((g_pr - g_tg).abs().mean() / gsig)
    return out


@torch.no_grad()
def block_gain(model, S, P, A, E, H, device, n=512, horizon=8, seed=0,
               M=None):
    """Does holding AWAY produce more guarding than holding TOWARD?

    THE GATE. Two rollouts from identical real start states, differing only in
    the defender's stick: away from the opponent versus toward. Guarding in
    Hisoutensoku IS holding away, so a simulator that cannot reproduce this
    cannot teach blocking, and no amount of RL against it will help.

    Reported as the difference in mean predicted `guarding` over the rollout.
    The pixel-latent model scored -0.00025 +- 0.00761 here: flat, and slightly
    the wrong sign.
    """
    rng = np.random.default_rng(seed)
    span = H + horizon
    ok = np.ones(len(S), bool); ok[-span:] = False
    for k in range(1, span + 1):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    # Near, and the defender able to act -- the frames where blocking is a
    # decision at all.
    near = np.abs(S[:, 0, CH["dx"]]) < (250.0 / 1200.0)
    ok &= near
    pool = np.flatnonzero(ok)
    if len(pool) < 32:
        return {"block_gain": float("nan"), "block_n": 0}
    idx = rng.choice(pool, min(n, len(pool)), replace=False)
    w = idx[:, None] + np.arange(span)[None, :]
    s = torch.as_tensor(S[w]).to(device); p = torch.as_tensor(P[w]).to(device)
    a = torch.as_tensor(A[w]).to(device)
    mv = None if M is None else torch.as_tensor(M[w][:, :H]).to(device)

    # dx > 0 means the opponent is to P1's right, so AWAY for P1 is left.
    dx = s[:, H - 1, 0, CH["dx"]]
    L, R = BUTTONS.index("left"), BUTTONS.index("right")
    res = {}
    for tag, away in (("away", True), ("toward", False)):
        af = a.clone()
        af[:, :, :, L] = 0.0; af[:, :, :, R] = 0.0
        left = (dx > 0) if away else (dx < 0)
        af[left, :, :, L] = 1.0
        af[~left, :, :, R] = 1.0
        pred = rollout(model, s[:, :H], p[:, :H], af, horizon, mv)
        res[tag] = float(torch.sigmoid(pred[:, :, 0, CH["guarding"]]).mean())
    return {"block_away": res["away"], "block_toward": res["toward"],
            "block_gain": res["away"] - res["toward"], "block_n": len(idx)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=400, help="replay directories to read; 0 = all")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frame-skip", type=int, default=0,
                    help="frames per decision step; 0 uses cfg.frame_skip (5). "
                         "1 means one step is one frame, which is the arm that "
                         "asks whether the dynamics are learnable at all before "
                         "any loss term is blamed for h1 being negative.")
    ap.add_argument("--unroll", type=int, default=4,
                    help="autoregressive steps in the training loss. 0 is the "
                         "teacher-forced objective that scored -1.02 at h1.")
    ap.add_argument("--unroll-weight", type=float, default=1.0)
    ap.add_argument("--proj-feedback", default="sigmoid",
                    choices=("sigmoid", "raw"),
                    help="how the projectile head's output re-enters the "
                         "rollout. 'sigmoid' squashes the WHOLE tensor, which "
                         "is what every checkpoint to date was trained through "
                         "and is a defect: dx/dy/vx/vy are regression outputs "
                         "that can be negative, so their sign -- the entire "
                         "content of 'is that bullet coming at me' -- is "
                         "destroyed from step 1 of every rollout. 'raw' "
                         "squashes only `present` and `hb`. The fix cannot be "
                         "switched on under an existing model (measured: "
                         "-0.048 kinematic skill at h8 on fix5/sim.pt); it has "
                         "to be trained in, which is what this flag is for.")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move-dim", type=int, default=0,
                    help="width of the nominal-action-id embedding; 0 keeps "
                         "the model blind to move identity, which is what "
                         "every checkpoint before this was")
    ap.add_argument("--move-weight", type=float, default=1.0,
                    help="weight on the next-move cross-entropy")
    ap.add_argument("--move-min-count", type=int, default=32,
                    help="ids rarer than this share the OOV row")
    ap.add_argument("--idm", type=float, default=0.0,
                    help="weight on reading the action back out of the "
                         "model's own prediction; a lower bound on the action "
                         "information the dynamics actually carries")
    ap.add_argument("--state-history", type=int, default=0,
                    help="state frames visible, inside the same action "
                         "history. 0 keeps them equal.")
    ap.add_argument("--flag-entropy-weight", action="store_true",
                    help="weight each flag's loss by 1/H(flag) so rare "
                         "channels are not priced by their base rate")
    ap.add_argument("--ct-weight", type=float, default=0.0,
                    help="fraction of the loss weight moved onto frames where "
                         "the character can act (c_t); 0 disables")
    ap.add_argument("--horizon-weights", action="store_true",
                    help="one unroll loss per horizon with learned "
                         "uncertainty weighting instead of one pooled term")
    ap.add_argument("--pos-weight-max", type=float, default=5.0,
                    help="clamp on the per-flag BCE pos_weight. 1.0 makes the "
                         "flag heads calibrated; anything above it trades "
                         "calibration for gradient share on rare classes.")
    ap.add_argument("--action-skip", action="store_true",
                    help="a direct zero-initialised path from the buttons to "
                         "the output head. The action otherwise reaches the "
                         "prediction only through six transformer blocks, and "
                         "reweighting the loss where it matters was a null, "
                         "which points at the path rather than the objective.")
    ap.add_argument("--arrival-weight", type=float, default=0.0,
                    help="extra weight on the binary loss at the STEP an "
                         "attack goes live. Unlike --guard-frac this keeps "
                         "both outcomes: the measured defect is predicting "
                         "P(guard|toward)=0.12 where the game says 0.014, and "
                         "a diet of guard onsets would widen that.")
    ap.add_argument("--guard-frac", type=float, default=0.0,
                    help="share of each batch drawn from windows containing a "
                         "guard onset. 0 is the ordinary uniform mix.")
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz holding the parsed corpus, so sibling arms share "
                         "one parse instead of re-rolling the slowest and "
                         "least reliable step in the run")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    cfg = Config()

    print(f"loading state from {a.corpus} (no video) ...", flush=True)
    skip = a.frame_skip or cfg.frame_skip
    print(f"  frame_skip {skip} -> one decision step is {skip*1000/60:.0f} ms",
          flush=True)
    S, P, A, E, M = cached_sequences(a.corpus, a.replays, a.slots, skip,
                                     a.cache, a.name)
    # Held-out replays, not held-out frames: consecutive frames are nearly
    # identical, so a frame split would let the model memorise its own val set.
    n_ep = int(E.max()) + 1
    val_ep = set(np.random.default_rng(a.seed).choice(
        n_ep, max(1, n_ep // 8), replace=False).tolist())
    vm = np.isin(E, list(val_ep))
    Sv, Pv, Av, Ev, Mv = S[vm], P[vm], A[vm], E[vm], M[vm]
    S, P, A, E, M = S[~vm], P[~vm], A[~vm], E[~vm], M[~vm]
    print(f"  train {len(S)} steps / {n_ep - len(val_ep)} replays | "
          f"val {len(Sv)} / {len(val_ep)}", flush=True)
    for tag, arr in (("state", S), ("proj", P), ("val state", Sv)):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{tag} contains {int((~np.isfinite(arr)).sum())} non-finite "
                f"values after loading; refusing to train on it")

    # The vocabulary is built on the TRAINING split only. Built over both, a
    # move that appears solely in validation would get its own embedding row
    # trained on nothing, and the run would report a vocabulary it cannot
    # actually predict.
    vocab, n_moves = None, 0
    if a.move_dim > 0:
        vocab = build_move_vocab(M, a.move_min_count)
        n_moves = len(vocab)
        M, Mv = index_moves(M, vocab), index_moves(Mv, vocab)
        oov = float((Mv == 0).mean())
        print(f"  move vocabulary {n_moves} ids (>= {a.move_min_count} "
              f"occurrences), val OOV {oov*100:.3f}%", flush=True)
    else:
        M = Mv = None

    model = StateDynamics(cfg, a.slots, a.width, a.depth,
                          history=a.history, ticks=skip,
                          proj_feedback=a.proj_feedback,
                          n_moves=n_moves, move_dim=a.move_dim,
                          idm=a.idm > 0,
                          state_history=a.state_history,
                          act_skip=a.action_skip).to(a.device)
    # One log-variance per unrolled horizon, trained alongside the weights.
    horizon_w = None
    if a.horizon_weights and a.unroll > 0:
        horizon_w = torch.zeros(a.unroll, device=a.device, requires_grad=True)
    # `c_t`: can this character act at all? Exact from the state channels, so
    # it needs neither the world model nor a counterfactual to define.
    ct = None
    if a.ct_weight > 0 or a.arrival_weight > 0:
        z = S[:, 0]
        act = ((z[:, CH["untech"]] <= 0) & (z[:, CH["hitstop"]] <= 0)
               & (z[:, CH["knockdown"]] < 0.5) & (z[:, CH["crushed"]] < 0.5))
        w = np.full(len(S), 1.0 - a.ct_weight, dtype=np.float32) \
            + a.ct_weight * act.astype(np.float32)
        if a.ct_weight > 0:
            print(f"  c_t: {act.mean()*100:.1f}% of frames are actionable; "
                  f"moving {a.ct_weight:.0%} of the loss weight onto them",
                  flush=True)
        if a.arrival_weight > 0:
            arr = attack_arrival_frames(S)
            w = w * (1.0 + a.arrival_weight * arr.astype(np.float32))
            print(f"  arrival: {arr.mean()*100:.2f}% of frames are the step an "
                  f"attack goes live; weighting the binary loss there "
                  f"{1.0 + a.arrival_weight:.1f}x", flush=True)
        ct = w
    fw = None
    if a.flag_entropy_weight:
        fw = torch.as_tensor(flag_weights(S, np.array(BINARY)),
                             dtype=torch.float32, device=a.device)
        names = [STATE_CHANNELS[i] for i in BINARY]
        top = sorted(zip(names, fw.tolist()), key=lambda t: -t[1])[:4]
        print("  flag weights: " + " ".join(f"{n} {v:.2f}" for n, v in top),
              flush=True)
    print(f"  simulator {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params, input {model.in_dim}", flush=True)
    params = list(model.parameters()) + ([horizon_w] if horizon_w is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    # Clamped far lower than the head's default 50. That clamp exists to keep
    # `crushed` visible at a 0.11% base rate, but it also pushes every flag's
    # predicted probability far above its true rate -- measured, `guarding`
    # came out at 0.52 against 4.66% -- which makes the block gate unreadable.
    # BCE with pos_weight w has minimiser w*p/(w*p + 1 - p), a monotone
    # distortion, so any w != 1 is a deliberate miscalibration. Measured at
    # w=5: the game blocks 0.578 of arriving attacks when holding away and the
    # model predicts 0.834, against a formula value of 0.873 -- the model is
    # calibrated FOR ITS LOSS, and the loss is what is skewed. At w=1 the
    # minimiser is p, so the predicted causal effect should land near the
    # measured +0.564 instead of +0.741.
    #
    # The reason w>1 existed is gradient share for rare flags, and
    # --flag-entropy-weight is the tool for that: it scales a CHANNEL's loss
    # without moving where that channel's optimum sits.
    pw = default_pos_weight().clamp(max=a.pos_weight_max)
    cont_idx = np.array(CONTINUOUS)
    sc_np, live_np = delta_scale(S, E, cont_idx)
    dsc = torch.as_tensor(sc_np).to(a.device)
    live = torch.as_tensor(live_np).to(a.device)
    dead = [STATE_CHANNELS[cont_idx[j]] for j in range(len(cont_idx))
            if live_np[j] == 0]
    print(f"  {int(live_np.sum())}/{len(cont_idx)} continuous channels carry "
          f"signal; dropped {dead}", flush=True)
    print(f"  delta scale over live channels: min {sc_np[live_np>0].min():.5f} "
          f"max {sc_np[live_np>0].max():.5f}", flush=True)
    rng = np.random.default_rng(a.seed)
    log, t0 = [], time.time()

    for i, (s, p, act, mv, cwb) in enumerate(batches(S, P, A, E, a.history,
                                                    a.batch_size, a.steps, rng,
                                                    a.device, M, a.guard_frac,
                                                    ct)):
        cw = None
        if cwb is not None:
            # A floor rather than a hard gate: a non-actionable frame still has
            # dynamics worth predicting, it just cannot teach anything about
            # the action.
            # `ct` already carries the full per-frame weight, actionability
            # and arrival combined, so it is used as-is rather than re-mixed.
            cw = cwb[:, :-1, None, None]
        loss, m = step_loss(model, s, p, act, pw, dsc, live, a.unroll,
                            a.unroll_weight, mv, a.move_weight, fw, cw,
                            a.idm, horizon_w)
        if not torch.isfinite(loss):
            raise SystemExit(
                f"loss is {float(loss)} at step {i}; stopping. A run that "
                f"keeps going here writes checkpoints of a destroyed model "
                f"and reports NaN skill for hours.")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if i % a.eval_every == 0 or i == a.steps - 1:
            model.eval()
            ev = eval_rollout(model, Sv, Pv, Av, Ev, a.history, (1, 2, 4, 8, 16),
                              a.device, M=Mv)
            ev.update(block_gain(model, Sv, Pv, Av, Ev, a.history, a.device,
                                 M=Mv))
            # Calibration: a flag predicted far from its base rate makes the
            # block gate unreadable however large the gain looks.
            ev["guard_base"] = float(Sv[:, :, CH["guarding"]].mean())
            model.train()
            ev.update(step=i, loss=float(loss.detach()), **m,
                      elapsed_h=(time.time() - t0) / 3600)
            log.append(ev)
            print(f"step {i:6d} | loss {float(loss.detach()):.4f} | KIN skill "
                  f"h1 {ev['skill_h1']:+.3f} h4 {ev['skill_h4']:+.3f} "
                  f"h16 {ev['skill_h16']:+.3f} | guard sigma h2 "
                  f"{ev['guard_sigma_h2']:.3f} h8 {ev['guard_sigma_h8']:.3f}"
                  + (f" | move acc {m['move_acc']:.4f}" if "move_acc" in m else "")
                  + f" | BLOCK gain {ev['block_gain']:+.4f} "
                  f"(away {ev['block_away']:.3f} toward {ev['block_toward']:.3f} "
                  f"base {ev['guard_base']:.3f})",
                  flush=True)
            (a.out / "log.json").write_text(json.dumps(log, indent=1))
            # Best by skill_h1, not by loss. The previous run peaked at step
            # 3000 (h1 -0.043) and degraded monotonically to -5.4 by 10000
            # while the loss kept falling; saving on loss would have kept the
            # worst model of the run.
            if ev["skill_h1"] >= max(c["skill_h1"] for c in log):
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": i,
                            "slots": a.slots, "history": a.history,
                            "width": a.width, "depth": a.depth, "ticks": skip,
                            "proj_feedback": a.proj_feedback,
                            "n_moves": n_moves, "move_dim": a.move_dim,
                            "move_vocab": vocab, "idm": a.idm > 0,
                            "state_history": a.state_history,
                            # Architecture, not bookkeeping: load_sim rebuilds
                            # from these keys, and an absent act_skip silently
                            # reconstructs a DIFFERENT model than was trained.
                            "act_skip": a.action_skip},
                           a.out / "best_h1.pt")
            # `ticks` goes into sim.pt too. Leaving it out was survivable
            # only because every run so far used the config default; a run
            # with --frame-skip 1 wrote a checkpoint whose stride could not
            # be recovered from the file at all.
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": i,
                        "slots": a.slots, "history": a.history,
                        "width": a.width, "depth": a.depth, "ticks": skip,
                        "proj_feedback": a.proj_feedback,
                        "n_moves": n_moves, "move_dim": a.move_dim,
                        "move_vocab": vocab, "guard_frac": a.guard_frac,
                        "idm": a.idm > 0, "state_history": a.state_history,
                        "flag_entropy_weight": a.flag_entropy_weight,
                        "ct_weight": a.ct_weight,
                        "arrival_weight": a.arrival_weight,
                        "act_skip": a.action_skip,
                        "pos_weight_max": a.pos_weight_max,
                        "horizon_weights": a.horizon_weights},
                       a.out / "sim.pt")
    print(f"\n-> {a.out}/sim.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
