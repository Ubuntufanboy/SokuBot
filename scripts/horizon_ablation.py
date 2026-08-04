"""How far can GRPO plan before the reward signal stops being trustworthy?

    python -m scripts.horizon_ablation --corpus /root/corpus --ckpt /root/ckpt/best.pt

GRPO optimises a policy against rewards computed *inside imagination*: roll the
predictor forward under candidate actions and read health, spirit and combo off
the resulting latents with the linear probe. That is only sound while the
imagined latents still lie close enough to the real ones for the probe to read
them. Each autoregressive step feeds the predictor its own output, so error
compounds; past some horizon the probe is reading noise and the policy is being
rewarded for hallucinations.

This measures where that happens, and the answer sets the rollout length.

WHAT IS MEASURED
----------------
For each start state the predictor is rolled forward P steps under the actions
that were *actually* taken, so the only error being measured is the world
model's own. At each horizon k four numbers are compared, all from the same
probe:

  ceiling   probe on the real latent at t+k -- the best the probe can do, and
            roughly flat in k. Everything else should be read against this, not
            against 1.0.
  imagined  probe on the rolled-out latent at t+k. This is the deployment
            number: the reward GRPO would actually see.
  frozen    probe on the latent at t, held constant. The "assume nothing
            changes" baseline. Imagined rollouts have to beat this or the
            rollout is contributing nothing and a myopic policy is as good.
  refit     a *fresh* probe fit on the imagined latents at horizon k. This
            separates two failure modes that look identical from the deployment
            number alone: if refit stays high while imagined collapses, the
            information survived and only the readout drifted, which is fixed by
            calibrating the probe on imagined latents rather than by shortening
            the horizon.

THE SPLIT
---------
Replays are split into a probe-fitting set and an evaluation set, and no replay
appears in both. `ridge_probe`'s own split is over rows, which leaks badly here:
consecutive frames of one match are near-duplicates, so a random row split puts
near-copies of the training data in the test set. The milestone-4 gate reported
R^2 0.959 under that row split; the by-replay ceiling printed here is the honest
version of the same number and should be expected to be lower.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.hud import read_trace
from sokubot.data.soku import decode_frames, read_actions
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import fit_ridge

TARGETS = ("hp1", "hp2", "spirit1", "spirit2", "combo1", "combo2")
HUD_W = HUD_H = 480


def decode_hud(path: str, max_frames: int) -> np.ndarray:
    """Native-resolution frames for HUD reading, capped at `max_frames`."""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    n = len(p.stdout) // (HUD_W * HUD_H * 3)
    return np.frombuffer(p.stdout[: n * HUD_W * HUD_H * 3],
                         dtype=np.uint8).reshape(n, HUD_H, HUD_W, 3)


def capture_paths(row: dict, manifest: Path) -> tuple[Path, Path]:
    """(video, inputs.csv) for a manifest row, derived exactly as discover_captures.

    The manifest records only ``video``; the CSV is its sibling. Resolving it
    here rather than inside the load path keeps a missing file a hard error
    instead of a per-replay 'skip' line.
    """
    video = Path(row.get("video", ""))
    if not video.is_absolute():
        video = manifest.parent / video
    inputs = video.parent / "inputs.csv"
    if not video.exists():
        raise FileNotFoundError(f"missing video {video}")
    if not inputs.exists():
        raise FileNotFoundError(f"missing inputs {inputs}")
    return video, inputs


def load_replay(video: Path, inputs: Path, cfg: Config, max_frames: int,
                cache_dir: Path | None = None):
    """-> (obs [D,S,S,3] uint8, actions [D,ticks,20] float32, labels [D,K] float32).

    Decision step d corresponds to source frame ``d * frame_skip``: that is the
    contract `sokubot.data.soku` trains under, and getting it wrong here would
    shift every label by a fraction of a second while still looking plausible.

    Labels and action chunks are cached: they depend only on the capture, not on
    the checkpoint, and producing them means decoding the whole video at native
    resolution, which dominates the runtime of any re-run.
    """
    skip = cfg.frame_skip
    key = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_dir / f"{video.parent.name}-{max_frames}-{skip}.npz"

    if key is not None and key.exists():
        with np.load(key) as d:
            labels, chunks = d["labels"], d["chunks"]
    else:
        hud = decode_hud(str(video), max_frames)
        if len(hud) < skip * 8:
            raise ValueError(f"only {len(hud)} frames decoded")
        tr = read_trace(hud, smooth=3)
        del hud
        acts_full = read_actions(inputs)
        n = min(len(tr.hp1) // skip, len(acts_full) // skip)
        idx = np.arange(n) * skip
        labels = np.stack([getattr(tr, t)[idx] for t in TARGETS], axis=1).astype(np.float32)
        chunks = np.stack([acts_full[d * skip : d * skip + skip]
                           for d in range(n)]).astype(np.float32)
        if key is not None:
            np.savez_compressed(key, labels=labels, chunks=chunks)

    obs = list(decode_frames(video, cfg.image_size, skip))
    acts = None

    del acts
    D = min(len(obs), len(labels), len(chunks))
    if max_frames > 0:
        D = min(D, max_frames // skip)
    if D < 16:
        raise ValueError(f"only {D} decision steps")
    return np.stack(obs[:D]), chunks[:D], labels[:D]


@torch.no_grad()
def encode_all(model: LeWorldModel, obs: np.ndarray, device: str,
               batch: int = 256) -> np.ndarray:
    """[D,S,S,3] uint8 -> [D, latent]. Encoder takes uint8 and scales on device."""
    out = []
    for i in range(0, len(obs), batch):
        x = torch.from_numpy(np.ascontiguousarray(obs[i : i + batch]))
        x = x.permute(0, 3, 1, 2).to(device, non_blocking=True)
        out.append(model.encode(x.unsqueeze(1))[:, 0].float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def rollout_starts(model: LeWorldModel, Z: np.ndarray, A: np.ndarray,
                   starts: np.ndarray, P: int, H: int, device: str,
                   batch: int = 256) -> np.ndarray:
    """-> [n_starts, P, latent] imagined latents under the true action sequence."""
    Zt = torch.from_numpy(Z).to(device)
    At = torch.from_numpy(A).to(device)
    out = []
    for i in range(0, len(starts), batch):
        s = torch.from_numpy(starts[i : i + batch]).to(device)
        off = torch.arange(H, device=device) - (H - 1)          # -(H-1) .. 0
        z_ctx = Zt[s[:, None] + off[None, :]]                   # [B,H,latent]
        a_hist = At[s[:, None] + off[None, :-1]]                # [B,H-1,ticks,A]
        a_plan = At[s[:, None] + torch.arange(P, device=device)[None, :]]
        out.append(model.rollout(z_ctx, a_plan, a_hist).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="fraction of replays used to fit the probe (disjoint)")
    ap.add_argument("--horizon", type=int, default=48,
                    help="max rollout length in decision steps (15 Hz)")
    ap.add_argument("--starts", type=int, default=150, help="start states per replay")
    ap.add_argument("--max-frames", type=int, default=9000,
                    help="cap source frames per replay; 0 for the whole capture")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-2, 1.0, 1e2, 1e3, 1e4, 1e5, 1e6],
                    help="ridge strengths to sweep; the gate used 1e-2, which on "
                         "42k standardised rows is effectively unregularised")
    ap.add_argument("--out", type=Path, default=Path("/root/horizon"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    model = LeWorldModel(cfg).to(a.device)
    model.load_state_dict(blob["model"])
    model.eval()
    H, P, skip = cfg.history, a.horizon, cfg.frame_skip
    print(f"world model step {blob.get('step','?')} | history {H} | frame_skip {skip} "
          f"| horizon {P} steps = {P*skip/60:.2f}s", flush=True)

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rows)
    rows = rows[: a.replays]
    n_fit = max(1, int(len(rows) * a.fit_frac))
    fit_rows, eval_rows = rows[:n_fit], rows[n_fit:]
    print(f"{len(fit_rows)} replays to fit the probe, {len(eval_rows)} to evaluate "
          f"(disjoint)", flush=True)

    # ---- gather real and imagined latents for every replay ----
    def gather(rows, tag):
        """-> dict of stacked arrays, or None if nothing usable came back."""
        Zr, ZIr, Lr, Fr, allz, ally = [], [], [], [], [], []
        for k, r in enumerate(rows):
            video, inputs = capture_paths(r, manifest)
            try:
                obs, act, lab = load_replay(video, inputs, cfg, a.max_frames,
                                            a.out / "cache")
            except ValueError as exc:            # too short to window; not a bug
                print(f"   skip {r['replay_id'][:12]}: {exc}", flush=True)
                continue
            Z = encode_all(model, obs, a.device)
            del obs
            allz.append(Z)
            ally.append(lab)
            D = len(Z)
            lo, hi = H - 1, D - P - 1
            if hi > lo:
                starts = rng.choice(np.arange(lo, hi), size=min(a.starts, hi - lo),
                                    replace=False).astype(np.int64)
                ZIr.append(rollout_starts(model, Z, act, starts, P, H, a.device))
                off = np.arange(P) + 1
                Zr.append(Z[starts[:, None] + off[None, :]])
                Lr.append(lab[starts[:, None] + off[None, :]])
                Fr.append(Z[starts])
            if (k + 1) % 5 == 0:
                print(f"   {tag} {k+1}/{len(rows)}", flush=True)
        if not allz:
            return None
        return {"imag": np.concatenate(ZIr), "real": np.concatenate(Zr),
                "lab": np.concatenate(Lr), "froz": np.concatenate(Fr),
                "all_z": np.concatenate(allz), "all_y": np.concatenate(ally)}

    fit = gather(fit_rows, "fit")
    if fit is None:
        raise SystemExit(f"no replay produced usable frames out of {len(fit_rows)}")
    print(f"fit pool: {len(fit['all_z'])} real latents, "
          f"{fit['imag'].shape[0]} rollouts x {P} steps", flush=True)
    ev = gather(eval_rows, "eval")
    if ev is None:
        raise SystemExit("no evaluation replay produced usable frames")

    # ---- two candidate readouts, each swept over regularisation strength ----
    # `real` is fit on encoder outputs only, the way the milestone-4 gate was.
    # `calibrated` is fit on predictor outputs pooled over every horizon, which
    # is the distribution the reward actually has to be read from.
    ny = len(TARGETS)
    fi = fit["imag"].reshape(-1, cfg.latent_dim)
    fl = fit["lab"].reshape(-1, ny)
    probes: dict[str, dict[float, object]] = {"real": {}, "calibrated": {}}
    for al in a.alphas:
        probes["real"][al] = fit_ridge(fit["all_z"], fit["all_y"],
                                       names=list(TARGETS), alpha=al)
        probes["calibrated"][al] = fit_ridge(fi, fl, names=list(TARGETS), alpha=al)
    del fi, fl

    # Pick each variant's alpha on the fit replays' own rollouts, never on the
    # evaluation replays -- selecting on the test set would inflate every number
    # that follows.
    MIN_STD_SEL = 1e-3
    sel_i = fit["imag"][:, ::8].reshape(-1, cfg.latent_dim)
    sel_l = fit["lab"][:, ::8].reshape(-1, ny)
    best = {}
    for kind, byalpha in probes.items():
        scored = []
        for al, pr in byalpha.items():
            r2 = pr.r2(sel_i, sel_l, min_std=MIN_STD_SEL)
            v = [x for x in r2.values() if np.isfinite(x)]
            scored.append((float(np.mean(v)) if v else -np.inf, al))
        scored.sort(reverse=True)
        best[kind] = scored[0][1]
        print(f"alpha for {kind:11s}: {scored[0][1]:g}  "
              + "  ".join(f"{al:g}:{s:+.3f}" for s, al in sorted(scored, key=lambda x: x[1])),
              flush=True)
    del sel_i, sel_l

    probe = probes["real"][best["real"]]
    probe_cal = probes["calibrated"][best["calibrated"]]

    imag = [ev["imag"][:, j] for j in range(P)]
    real = [ev["real"][:, j] for j in range(P)]
    labs = [ev["lab"][:, j] for j in range(P)]
    froz = [ev["froz"] for _ in range(P)]

    # ---- metrics per horizon ----
    # A target whose spread across the evaluation sample is this small cannot be
    # scored: R^2's denominator is that spread. Health and spirit move in units
    # of ~0.01 of a bar, so 1e-3 is well below anything real.
    MIN_STD = 1e-3
    curve = []
    for j in range(P):
        ZI, ZR, Y, ZF = imag[j], real[j], labs[j], froz[j]
        n = len(ZI)

        # A fresh probe needs enough held-out rows to be worth fitting: with
        # fewer test rows than latent dimensions it reports its own overfitting,
        # not the model's.
        refit = {t: float("nan") for t in TARGETS}
        if n >= 10 * cfg.latent_dim:
            cut = int(n * 0.8)
            perm = np.random.default_rng(a.seed + j).permutation(n)
            tr_i, te_i = perm[:cut], perm[cut:]
            refit = fit_ridge(ZI[tr_i], Y[tr_i], names=list(TARGETS)).r2(
                ZI[te_i], Y[te_i], min_std=MIN_STD)

        num = np.linalg.norm(ZI - ZR, axis=1)
        den = np.linalg.norm(ZR, axis=1) + 1e-8
        cos = ((ZI * ZR).sum(1) /
               (np.linalg.norm(ZI, axis=1) * np.linalg.norm(ZR, axis=1) + 1e-8))
        curve.append({
            "h": j + 1,
            "seconds": (j + 1) * skip / 60.0,
            "n": int(n),
            "ceiling": probe.r2(ZR, Y, min_std=MIN_STD),
            "imagined": probe.r2(ZI, Y, min_std=MIN_STD),
            "calibrated": probe_cal.r2(ZI, Y, min_std=MIN_STD),
            "frozen": probe_cal.r2(ZF, Y, min_std=MIN_STD),
            "refit": refit,
            "label_std": {t: float(s) for t, s in zip(TARGETS, Y.std(0))},
            "drift_rel_l2": float(np.mean(num / den)),
            "cosine": float(np.mean(cos)),
        })

    def mean(d):
        v = [x for x in d.values() if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    # Usable horizon: imagined must retain 90% of the ceiling and still beat the
    # do-nothing baseline. Both conditions matter -- a high R^2 that merely ties
    # `frozen` means the rollout added no information.
    # The deployed readout is the calibrated one, so it is what sets the horizon.
    #
    # Two things this deliberately does NOT do. It does not average over targets
    # the reward cannot use: spirit scores ~0 even from real latents, so
    # including it drags the mean below any threshold and reports a horizon of
    # zero for a model whose health readings are fine. And it does not require
    # beating the frozen baseline at short horizons -- over one step "nothing
    # changed" is very nearly true, so frozen is *supposed* to be strong there.
    # The margin over frozen grows with horizon; the retained fraction of the
    # ceiling is what shrinks. Those are the two things worth reporting.
    KEY = "calibrated"

    def mean_of(d, ts):
        v = [d[t] for t in ts if np.isfinite(d[t])]
        return float(np.mean(v)) if v else float("nan")

    scored = [t for t in TARGETS if curve[0]["ceiling"][t] > 0.15]
    dropped = [t for t in TARGETS if t not in scored]
    usable = 0
    for c in curve:
        ceil = mean_of(c["ceiling"], scored)
        if ceil > 0 and mean_of(c[KEY], scored) >= 0.90 * ceil:
            usable = c["h"]                       # largest, not first-break
    beats = max((c["h"] for c in curve
                 if mean_of(c[KEY], scored) > mean_of(c["frozen"], scored) + 0.02),
                default=0)

    print()
    print("=" * 88)
    print(f"{'h':>3} {'sec':>5} {'ceiling':>8} {'calibr':>8} {'real-fit':>9} "
          f"{'frozen':>8} {'refit':>8} {'drift':>7} {'cos':>6}")
    print("=" * 88)
    for c in curve:
        if c["h"] <= 8 or c["h"] % 4 == 0:
            print(f"{c['h']:3d} {c['seconds']:5.2f} {mean(c['ceiling']):8.3f} "
                  f"{mean(c['calibrated']):8.3f} {mean(c['imagined']):9.3f} "
                  f"{mean(c['frozen']):8.3f} {mean(c['refit']):8.3f} "
                  f"{c['drift_rel_l2']:7.3f} {c['cosine']:6.3f}")
    print("=" * 88)
    print("per-target, calibrated probe on imagined latents:")
    print(f"{'h':>3} " + "".join(f"{t:>9}" for t in TARGETS))
    for c in curve:
        if c["h"] in (1, 2, 4, 8, 16, 24, 32, 48):
            print(f"{c['h']:3d} " + "".join(f"{c['calibrated'][t]:9.3f}" for t in TARGETS))
    print("=" * 88)
    print(f"per-target ceiling (real latents, by-replay split), "
          f"with the spread R^2 is measured against:")
    for t in TARGETS:
        print(f"   {t:9s} R2 {curve[0]['ceiling'][t]:+7.3f}   "
              f"label std {curve[0]['label_std'][t]:.4f}")
    print(f"   (n = {curve[0]['n']} start states)")
    print()
    print(f"scored on {scored}"
          + (f"; dropped {dropped} (not decodable even from real latents)"
             if dropped else ""))
    print(f"USABLE HORIZON : {usable} steps = {usable*skip/60:.2f}s "
          f"(calibrated probe still at >=90% of its ceiling)")
    print(f"BEATS FROZEN   : {beats} steps = {beats*skip/60:.2f}s "
          f"(rollout beats assuming nothing changed)")
    print("=" * 78)

    res = {"ckpt": str(a.ckpt), "step": blob.get("step"), "history": H,
           "frame_skip": skip, "horizon": P, "targets": list(TARGETS),
           "replays_fit": len(fit_rows), "replays_eval": len(eval_rows),
           "alpha_real": best["real"], "alpha_calibrated": best["calibrated"],
           "usable_horizon": usable, "beats_frozen": beats, "curve": curve}
    (a.out / "horizon.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out/'horizon.json'}")

    # The calibrated readout is what GRPO reads rewards with, so persist it
    # rather than making the training run refit it.
    np.savez(a.out / "reward_probe.npz", zmu=probe_cal.zmu, zsd=probe_cal.zsd,
             ymu=probe_cal.ymu, ysd=probe_cal.ysd, W=probe_cal.W,
             names=np.array(probe_cal.names), alpha=best["calibrated"])
    print(f"wrote {a.out/'reward_probe.npz'}")

    try:
        plot(res, a.out / "horizon.png")
        print(f"wrote {a.out/'horizon.png'}")
    except Exception as exc:
        print(f"(no chart: {exc})")
    return 0


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = res["curve"]
    h = [c["h"] for c in curve]
    sec = [c["seconds"] for c in curve]

    def m(key):
        return [float(np.mean(list(c[key].values()))) for c in curve]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("#12101a")
    for row in ax:
        for x in row:
            x.set_facecolor("#1a1724")
            x.grid(alpha=0.15, color="#8b83a8")
            x.tick_params(colors="#c9c2dc")
            for s in x.spines.values():
                s.set_color("#3a3450")

    a0 = ax[0][0]
    a0.plot(h, m("ceiling"), color="#4ade80", lw=2, label="ceiling (real latent)")
    a0.plot(h, m("calibrated"), color="#d9a441", lw=2.5,
            label="calibrated probe (deployed)")
    a0.plot(h, m("refit"), color="#60a5fa", lw=1.6, ls="--", label="refit per horizon")
    a0.plot(h, m("frozen"), color="#f87171", lw=1.6, ls=":", label="frozen (do nothing)")
    u = res["usable_horizon"]
    if u:
        a0.axvspan(1, u, color="#4ade80", alpha=0.08)
        a0.axvline(u, color="#4ade80", lw=1, ls="--")
        a0.text(u + 0.6, 0.05, f" usable: {u} steps\n {u*res['frame_skip']/60:.2f}s",
                color="#4ade80", fontsize=9, va="bottom")
    a0.set_title("Reward fidelity vs planning horizon", color="#efe9fb", fontsize=12)
    a0.set_xlabel("horizon (decision steps @ 15 Hz)", color="#c9c2dc")
    a0.set_ylabel("probe R² (mean over targets)", color="#c9c2dc")
    a0.legend(facecolor="#221d31", edgecolor="#3a3450", labelcolor="#efe9fb", fontsize=9)

    a1 = ax[0][1]
    for t, col in zip(res["targets"],
                      ["#d9a441", "#f0b860", "#60a5fa", "#93c5fd", "#f87171", "#fca5a5"]):
        a1.plot(h, [c["calibrated"][t] for c in curve], color=col, lw=1.8, label=t)
    a1.set_title("Per-target decay (imagined latents)", color="#efe9fb", fontsize=12)
    a1.set_xlabel("horizon (decision steps)", color="#c9c2dc")
    a1.set_ylabel("probe R²", color="#c9c2dc")
    a1.legend(facecolor="#221d31", edgecolor="#3a3450", labelcolor="#efe9fb",
              fontsize=8, ncol=2)

    a2 = ax[1][0]
    a2.plot(sec, [c["drift_rel_l2"] for c in curve], color="#c084fc", lw=2)
    a2.set_title("Latent drift  ‖ẑ−z‖ / ‖z‖", color="#efe9fb", fontsize=12)
    a2.set_xlabel("seconds of imagination", color="#c9c2dc")
    a2.set_ylabel("relative L2 error", color="#c9c2dc")

    a3 = ax[1][1]
    a3.plot(sec, [c["cosine"] for c in curve], color="#34d399", lw=2)
    a3.set_title("Direction retained  cos(ẑ, z)", color="#efe9fb", fontsize=12)
    a3.set_xlabel("seconds of imagination", color="#c9c2dc")
    a3.set_ylabel("cosine similarity", color="#c9c2dc")

    fig.suptitle(f"Horizon ablation — world model step {res.get('step')}, "
                 f"{res['replays_eval']} held-out replays",
                 color="#efe9fb", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    raise SystemExit(main())
