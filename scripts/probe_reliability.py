"""Is the reward probe trustworthy at low health, and does its KO detector fire on noise?

    python -m scripts.probe_reliability --corpus ~/corpus \
        --ckpt ~/sokubot-art/best_bnfix.pt \
        --probe ~/sokubot-art/reward_probe.npz --out ~/probe_reliability

WHY THIS RUNS BEFORE ANY REWARD CHANGE
--------------------------------------
The most interesting behaviour from the first live match is that the agent gives
up once its health is low. There is a mechanistic candidate -- `compute_rewards`
returns an `alive` mask and GRPO divides by `alive.sum()`, so imagined
trajectories that reach a KO stop contributing gradient, leaving low-health
states with the *least* training signal exactly where fighting back matters.

But that story assumes the reward *means* something down there, and the reward is
read through a probe whose residual is around 0.13 of a bar against a label
standard deviation of 0.32. Health also lives in [0, 1], so at true health 0.05
the residual cannot be symmetric -- it is squeezed against the floor -- and
`data/hud.py` documents a red-floor of 0.018 and an end-of-match heal on top of
that. If the probe is unreliable below the KO threshold then the reward was never
meaningful there, and no amount of reweighting the start distribution helps.

So this script answers three questions, in order of how much they would change:

  1. **Bias and noise versus true health.** If the probe systematically reads low
     when health is low, the KO threshold fires early and the agent is being
     told it is dead while it is alive.
  2. **The KO detector's error rates.** `rl/reward.ko_mask` is run on probed
     health and on true HUD health over the same windows, and the disagreement is
     counted. A false positive pays `lose` (-5) against damage terms that live
     around 0.1, and masks out every later step.
  3. **How rare low-health states actually are** among corpus start states --
     the other half of the hypothesis, and the half that `scripts/build_bank.py`
     could fix by oversampling.

WHAT IS EXCLUDED, AND WHY THAT IS NOT CHEATING
----------------------------------------------
`read_trace` flags frames where its own reading is untrustworthy: `healing` (the
end-of-match heal), `flash` (a screen effect washed out the HUD and the previous
reading was held) and `clamped` (a physically impossible drop was rejected).
Those are bad *labels*, not bad probe outputs, and scoring the probe against them
measures the HUD reader. They are dropped, and the dropped fraction is reported
so the exclusion cannot hide a problem.

TWO ARMS, BECAUSE GRPO DOES NOT READ ENCODER LATENTS
----------------------------------------------------
`encoder` reads latents of real frames -- the ceiling, and what
`human_baseline.py`'s `real` arm uses. `imagined` rolls the predictor one step
under the true actions and reads that, which is what the reward actually sees
inside `ImaginedArena`. A probe can be fine on the first and useless on the
second; only the second bears on the give-up behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe, fit_ridge
from sokubot.rl.reward import RewardConfig, ko_mask
from scripts.horizon_ablation import (TARGETS, capture_paths, encode_all,
                                      load_replay, rollout_starts)

# Bin edges in bars of health. Deliberately fine below 0.2: the whole question is
# about the region between the KO threshold (0.06) and the alive margin (0.16),
# and a uniform ten-bin split would put that entire regime in one bucket.
BINS = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.16, 0.25, 0.40,
                 0.60, 0.80, 1.01])


def load_probe(path: Path) -> tuple[LinearProbe, dict]:
    d = np.load(path, allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    meta = {"alpha": float(d["alpha"]) if "alpha" in d.files else None,
            "fingerprint": str(d["fingerprint"]) if "fingerprint" in d.files else None}
    return probe, meta


def residual_table(pred_hp: np.ndarray, true_hp: np.ndarray) -> list[dict]:
    """Bias and noise of the health reading, bucketed by *true* health.

    Bucketing by the truth rather than by the prediction matters: bucketing by
    the prediction would sort on the very quantity whose error is being
    measured, and regression to the mean alone would then manufacture a bias
    that looks exactly like the one being looked for.
    """
    rows = []
    idx = np.digitize(true_hp, BINS) - 1
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        if n < 32:
            rows.append({"lo": float(BINS[b]), "hi": float(BINS[b + 1]),
                         "n": n, "bias": None, "noise": None, "mae": None})
            continue
        res = pred_hp[m] - true_hp[m]
        rows.append({"lo": float(BINS[b]), "hi": float(BINS[b + 1]), "n": n,
                     "bias": float(res.mean()), "noise": float(res.std()),
                     "mae": float(np.abs(res).mean()),
                     "frac": float(n) / len(true_hp)})
    return rows


def window_starts(ep: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Starts of length-(window+1) runs that stay inside one replay and are all valid.

    Two things this must not do. It must not let a window straddle a capture
    boundary, where health jumps back to full and every detector fires. And it
    must not be built by first deleting the invalid rows and then windowing what
    is left -- that silently splices across the gap, so a window can span a
    screen flash and read as a health cliff that never happened. Validity is
    therefore applied as a *requirement on every frame of the window*, with the
    time axis left intact.
    """
    keep = []
    run = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])   # prefix sums
    edges = np.flatnonzero(np.diff(ep)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ep)]):
        if hi - lo <= window + 1:
            continue
        s = np.arange(lo, hi - window - 1)
        # All of s..s+window valid <=> the prefix sum advances by window+1.
        full = (run[s + window + 1] - run[s]) == (window + 1)
        keep.append(s[full])
    return np.concatenate(keep) if keep else np.array([], dtype=np.int64)


def ko_audit(seq_pred: np.ndarray, seq_true: np.ndarray,
             cfg: RewardConfig) -> dict:
    """Run the reward's own KO detector on probed and on true health, and diff.

    Both arguments are [n, window+1] health sequences over the same windows.
    """
    if len(seq_pred) == 0:
        return {"n": 0}
    ko_pred = ko_mask(torch.from_numpy(seq_pred).float(), cfg).any(1).numpy()
    ko_true = ko_mask(torch.from_numpy(seq_true).float(), cfg).any(1).numpy()
    n = len(seq_pred)

    tp = int((ko_pred & ko_true).sum())
    fp = int((ko_pred & ~ko_true).sum())
    fn = int((~ko_pred & ko_true).sum())
    tn = int((~ko_pred & ~ko_true).sum())
    return {"n": int(n), "window": int(seq_pred.shape[1] - 1),
            "true_ko_rate": float(ko_true.mean()),
            "pred_ko_rate": float(ko_pred.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            # Of the KOs the reward believes in, how many really happened.
            "precision": float(tp / max(tp + fp, 1)),
            # Of the real KOs, how many the reward notices.
            "recall": float(tp / max(tp + fn, 1)),
            # The number that matters most: a false positive pays -5 and masks
            # out every later step of the rollout.
            "false_ko_per_window": float(fp / max(n, 1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=None,
                    help="the deployed reward probe. Without it a fresh probe is "
                         "fit on a disjoint half of the replays, which measures "
                         "the representation rather than the artefact in use.")
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=9000)
    ap.add_argument("--ko-window", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("probe_reliability"))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or (a.out / "cache")

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    model = LeWorldModel(cfg).to(a.device)
    model.load_state_dict(blob["model"])
    model.eval()

    manifest = a.corpus / "val" / "manifest.jsonl"
    if not manifest.exists():
        manifest = a.corpus / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} captures in {manifest}", flush=True)

    Zs, Ys, Ms, As, eps = [], [], [], [], []
    kept = 0
    for r in rows:
        if kept >= a.replays:
            break
        try:
            video, inputs = capture_paths(r, manifest)
            obs, act, lab, msk = load_replay(video, inputs, cfg, a.max_frames,
                                             cache, return_masks=True)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        z = encode_all(model, obs, a.device)
        Zs.append(z); Ys.append(lab); Ms.append(msk); As.append(act)
        eps.append(np.full(len(z), kept, dtype=np.int32))
        kept += 1
        if kept % 5 == 0:
            print(f"  {kept}/{a.replays} replays", flush=True)
    if kept < 4:
        raise SystemExit(f"only {kept} replays loaded; need at least 4")

    Z = np.concatenate(Zs); Y = np.concatenate(Ys)
    M = np.concatenate(Ms); A = np.concatenate(As); ep = np.concatenate(eps)
    del Zs, Ys, Ms, As, eps

    # Bad HUD readings are bad labels, so they cannot score the probe.
    good = ~M.any(axis=1)
    dropped = {m: float(M[:, i].mean()) for i, m in
               enumerate(("healing", "flash", "clamped"))}
    print(f"masked out {1 - good.mean():.3%} of frames "
          f"({', '.join(f'{k} {v:.3%}' for k, v in dropped.items())})", flush=True)

    # ---- the probe under test ----
    if a.probe is not None:
        probe, meta = load_probe(a.probe)
        fit_ep = np.array([], dtype=int)
        print(f"probe: {a.probe} (alpha {meta['alpha']}, "
              f"fingerprint {meta['fingerprint']}), targets {probe.names}",
              flush=True)
    else:
        n_fit = max(1, int(kept * a.fit_frac))
        fit_ep = np.arange(n_fit)
        m = np.isin(ep, fit_ep) & good
        probe = fit_ridge(Z[m], Y[m], names=list(TARGETS), alpha=a.alpha)
        meta = {"alpha": a.alpha, "fingerprint": None}
        print(f"probe: fitted fresh on {n_fit} replays, {int(m.sum())} rows",
              flush=True)

    # Columns of Y are TARGETS; the probe may have been fit with fewer (the
    # deployed one predates cards1/cards2). Align by name rather than position.
    try:
        hp1_i, hp2_i = probe.names.index("hp1"), probe.names.index("hp2")
    except ValueError:
        raise SystemExit(f"probe has no hp channels; names are {probe.names}")
    ty1, ty2 = TARGETS.index("hp1"), TARGETS.index("hp2")

    eval_m = good & ~np.isin(ep, fit_ep)
    print(f"scoring on {int(eval_m.sum())} held-out rows", flush=True)

    result = {"ckpt": str(a.ckpt), "probe": str(a.probe) if a.probe else None,
              "probe_meta": meta, "replays": kept,
              "masked_fraction": float(1 - good.mean()), "masked": dropped,
              "eval_rows": int(eval_m.sum()), "arms": {}}

    rcfg = RewardConfig()
    H, W = cfg.history, a.ko_window
    chairs = (("p1", hp1_i, ty1), ("p2", hp2_i, ty2))

    # Window starts are shared by both arms so the two are scored on exactly the
    # same situations. They need `history - 1` latents of context behind them for
    # the rollout, and `window` labels ahead of them.
    ws = window_starts(ep, eval_m, W)
    ws = ws[ws >= H - 1]
    print(f"{len(ws)} fully-valid {W}-step windows", flush=True)

    for arm in ("encoder", "imagined"):
        if arm == "encoder":
            zz, yy = Z[eval_m], Y[eval_m]
        else:
            # One predictor step under the *true* actions, so any error is the
            # model's rather than a policy's.
            idx = np.flatnonzero(eval_m)
            idx = idx[(idx >= H - 1) & (idx + 1 < len(ep))]
            idx = idx[eval_m[idx + 1] & (ep[idx] == ep[idx + 1])]
            if len(idx) < 256:
                print(f"  {arm}: only {len(idx)} usable starts, skipping")
                continue
            zz = rollout_starts(model, Z, A, idx, 1, H, a.device)[:, 0]
            yy = Y[idx + 1]

        pred = probe.predict(zz)
        arm_out = {"n": int(len(zz))}
        for who, pi, ti in chairs:
            arm_out[who] = {
                "bins": residual_table(pred[:, pi], yy[:, ti]),
                "overall_bias": float((pred[:, pi] - yy[:, ti]).mean()),
                "overall_noise": float((pred[:, pi] - yy[:, ti]).std()),
            }

        # The KO detector runs on a *sequence*, so auditing it needs sequences
        # rather than the independent readings above. For `encoder` those are
        # consecutive probed frames; for `imagined` they are a genuine `window`-
        # step rollout under the true actions -- which is what GRPO's detector
        # actually consumes, and the only version of this number that bears on
        # the give-up behaviour.
        if len(ws) < 64:
            arm_out["ko"] = {who: {"n": 0} for who, _, _ in chairs}
        else:
            off = np.arange(W + 1)
            if arm == "encoder":
                seq_z = probe.predict(Z[ws[:, None] + off[None, :]].reshape(-1, Z.shape[1]))
                seq_p = seq_z.reshape(len(ws), W + 1, -1)
            else:
                roll = rollout_starts(model, Z, A, ws, W, H, a.device)  # [n,W,latent]
                # Prepend the encoder latent the rollout started from, so the
                # sequence has W+1 entries aligned with labels ws..ws+W.
                seq = np.concatenate([Z[ws][:, None], roll], axis=1)
                seq_p = probe.predict(seq.reshape(-1, seq.shape[-1])
                                      ).reshape(len(ws), W + 1, -1)
            seq_t = Y[ws[:, None] + off[None, :]]
            arm_out["ko"] = {
                who: ko_audit(np.ascontiguousarray(seq_p[:, :, pi]),
                              np.ascontiguousarray(seq_t[:, :, ti]), rcfg)
                for who, pi, ti in chairs
            }
        result["arms"][arm] = arm_out

    # ---- how rare is low health, among the states GRPO starts from ----
    true_hp = np.concatenate([Y[good][:, ty1], Y[good][:, ty2]])
    result["health_distribution"] = {
        "below_ko_threshold": float((true_hp <= rcfg.ko_threshold).mean()),
        "below_alive_margin": float(
            (true_hp <= rcfg.ko_threshold + rcfg.ko_alive_margin).mean()),
        "below_0.25": float((true_hp <= 0.25).mean()),
        "median": float(np.median(true_hp)),
    }

    (a.out / "probe_reliability.json").write_text(json.dumps(result, indent=1))

    # ---- report ----
    print()
    for arm, d in result["arms"].items():
        print(f"=== {arm} ({d['n']} rows) ===")
        print(f"{'true hp':>13} {'n':>8} {'bias':>9} {'noise':>9} {'mae':>9}")
        for row in d["p1"]["bins"]:
            if row["bias"] is None:
                print(f"{row['lo']:5.2f}-{row['hi']:<5.2f} {row['n']:>8} "
                      f"{'--':>9} {'--':>9} {'--':>9}")
            else:
                print(f"{row['lo']:5.2f}-{row['hi']:<5.2f} {row['n']:>8} "
                      f"{row['bias']:>+9.4f} {row['noise']:>9.4f} "
                      f"{row['mae']:>9.4f}")
        for who in ("p1", "p2"):
            k = d["ko"][who]
            if k.get("n"):
                print(f"  KO[{who}] over {a.ko_window}-step windows: "
                      f"true {k['true_ko_rate']:.3%} pred {k['pred_ko_rate']:.3%} "
                      f"| precision {k['precision']:.3f} recall {k['recall']:.3f} "
                      f"| false KO {k['false_ko_per_window']:.3%} of windows")
        print()

    h = result["health_distribution"]
    print(f"health: median {h['median']:.3f} | "
          f"<= ko_threshold {h['below_ko_threshold']:.3%} | "
          f"<= alive_margin {h['below_alive_margin']:.3%} | "
          f"<= 0.25 {h['below_0.25']:.3%}")
    print(f"\n-> {a.out}/probe_reliability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
