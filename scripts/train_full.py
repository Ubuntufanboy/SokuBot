"""The full-corpus training run, with periodic held-out evaluation.

    python -m scripts.train_full --corpus /root/corpus --steps 320000

`sokubot.train`'s CLI trains and checkpoints but never looks at held-out data.
Over six hours that is too long to be flying blind: a collapse, or a loader that
silently starts repeating one capture, looks identical to healthy training from
the loss curve alone.

Evaluation runs through `train()`'s callback, inside one continuous run --
calling train() in chunks would rebuild the optimiser and restart the learning
rate schedule at every eval point.

Metrics per eval, all on the same cached val windows the model never trains on:

  val_pred   MSE against the true next latent
  identity   MSE of "next latent looks like this latent" -- the number that
             makes val_pred meaningful, because at 15 Hz a pass-through
             predictor already scores well
  skill      1 - val_pred/identity; 0 is pass-through, 1 is perfect
  inv_dyn    can a linear probe recover the buttons from a latent pair
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import replace

from sokubot.config import Config
from sokubot.model.encoder import resize_pos_embed
from sokubot.data.soku import ACTION_COLUMNS, build_soku_dataset
from sokubot.losses import prediction_loss
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import inverse_dynamics_probe
from sokubot.train import save_checkpoint, set_seed, train


def eval_batch_for(cfg: Config, at_224: int = 64) -> int:
    """Eval batch that holds activation memory roughly constant across resolutions.

    Hardcoding 64 was fine at 224 on a 32 GB card. At 448 each image carries 4x
    the patches, and 64 of them will not fit in 4 GB -- the run dies at its first
    eval, after the training steps have already proved they fit, which is a
    confusing way to fail.
    """
    return max(4, int(at_224 * (224 / cfg.image_size) ** 2))


# Display-orientation rows of the play area at native 480 px; the HUD lives above
# and below (data/hud.py: health 34-74, spirit and cards 428-470). Frames in the
# val cache are stored vertically flipped, which is why this converts.
DISP_PLAY_Y = (80, 420)
NATIVE_PX = 480


def _play_band(image_size: int) -> tuple[int, int]:
    lo = int(round((NATIVE_PX - DISP_PLAY_Y[1]) * image_size / NATIVE_PX))
    hi = int(round((NATIVE_PX - DISP_PLAY_Y[0]) * image_size / NATIVE_PX))
    return lo, hi


@torch.no_grad()
def mirror_sensitivity(model: LeWorldModel, cache: dict, cfg: Config,
                       batch: int) -> dict:
    """How far the latent moves when the play area is mirrored horizontally.

    The metric that actually matters for control, measured in the training loop
    so a run can be checkpointed on it. `scripts/spatial_probe.py` asks the same
    question with a linear probe and is the authority; this is the cheap
    in-loop version, and it needs no fitting and no held-out split because it
    compares a latent against itself.

    Reported as a fraction of the across-batch latent spread, because an L2 of
    0.04 means nothing until you know the states being separated are 0.93 apart.
    `full` mirrors everything including the HUD and is the ceiling: whatever the
    encoder can notice at all, it notices there.

    Exists because the first inverse-dynamics run selected `best.pt` on skill
    while spatial sensitivity peaked at step 4000 and settled lower -- so the
    checkpoint that was best at the thing we cared about was overwritten by one
    that was better at something else.
    """
    device = next(model.parameters()).device
    O = cache["obs"]
    lo, hi = _play_band(cfg.image_size)
    out = {}
    base, play, full = [], [], []
    for i in range(0, len(O), batch):
        o = O[i : i + batch].to(device, non_blocking=True)
        b = o[:, :1]                                   # one frame per window
        m_play = b.clone()
        m_play[..., lo:hi, :] = torch.flip(m_play[..., lo:hi, :], dims=[-1])
        m_full = torch.flip(b, dims=[-1])
        base.append(model.encode(b)[:, 0])
        play.append(model.encode(m_play)[:, 0])
        full.append(model.encode(m_full)[:, 0])
    z0 = torch.cat(base).float()
    spread = float(z0.std(0).mean()) + 1e-9
    out["mirror_play"] = float((torch.cat(play).float() - z0).norm(dim=-1).mean()) / spread
    out["mirror_full"] = float((torch.cat(full).float() - z0).norm(dim=-1).mean()) / spread
    out["latent_spread"] = spread
    return out


@torch.no_grad()
def evaluate(model: LeWorldModel, cache: dict, cfg: Config,
             batch: int | None = None) -> dict:
    batch = batch or eval_batch_for(cfg)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    O, A = cache["obs"], cache["actions"]

    tot = ident = 0.0
    seen = 0
    zt, zn, acts = [], [], []
    for i in range(0, len(O), batch):
        obs = O[i : i + batch].to(device, non_blocking=True)
        a = A[i : i + batch].to(device).float()
        out = model(obs, a)
        z, tgt = out.z, out.z[:, 1:]
        n = obs.shape[0]
        tot += float(prediction_loss(out.zhat, z).item()) * n
        ident += float(F.mse_loss(z[:, :-1], tgt).item()) * n
        seen += n
        zt.append(z[:, :-1].reshape(-1, cfg.latent_dim).cpu().numpy())
        zn.append(tgt.reshape(-1, cfg.latent_dim).cpu().numpy())
        acts.append(a[:, :-1].amax(dim=2).reshape(-1, cfg.action_dim).cpu().numpy())

    if was_training:
        model.train()
    vp, idm = tot / seen, ident / seen
    probe = inverse_dynamics_probe(
        np.concatenate(zt), np.concatenate(zn),
        (np.concatenate(acts) > 0.5).astype(np.float32), names=list(ACTION_COLUMNS))
    return {"val_pred": vp, "identity": idm,
            "skill": 1.0 - vp / idm if idm > 0 else float("nan"),
            "inv_dyn_auc": probe.auc_mean, "n": seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--steps", type=int, default=320_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--ckpt-every", type=int, default=5_000)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("/root/ckpt"))
    ap.add_argument("--log", type=Path, default=Path("/root/train_log.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="peak LR. The 5e-4 tuned on 16k-step runs diverged at "
                         "75k of a 320k-step schedule -- see module docstring.")
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--eval-batch", type=int, default=0,
                    help="batch for the eval and BatchNorm recalibration passes; "
                         "0 scales it from the image size. Both are extra peak "
                         "memory on top of a training step that already fits, so "
                         "on a small card this is the knob that decides whether "
                         "the run survives its first eval.")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile. It is worth 1.38x when it works, "
                         "but Inductor needs a C compiler on PATH and fails at "
                         "the first step without one -- which on a box with no "
                         "sudo is not something a training run can fix.")
    ap.add_argument("--shuffle-buffer", type=int, default=0,
                    help="windows held per worker; 0 derives it from --shuffle-gb")
    ap.add_argument("--shuffle-gb", type=float, default=20.0,
                    help="TOTAL host RAM for shuffle buffers across all workers. "
                         "The original run used 4096 windows per worker, which is "
                         "2.5 GB each at 224 px and 9.8 GB at 448 -- affordable "
                         "only on the 754 GB box it was written for.")
    ap.add_argument("--image-size", type=int, default=224,
                    help="448 is the considered value for this game; 224 was "
                         "LeWorldModel's PushT default and throws away 4.6x the "
                         "pixels of a 480x480 capture, which is why spirit "
                         "probes at R^2 0.05")
    ap.add_argument("--cf-coef", type=float, default=Config.cf_coef,
                    help="action-discrimination weight, applied throughout")
    ap.add_argument("--hud-coef", type=float, default=Config.hud_coef,
                    help="supervised HUD readout weight")
    ap.add_argument("--idm-coef", type=float, default=Config.idm_coef,
                    help="inverse-dynamics weight. This is the term that decides "
                         "what the representation keeps: prediction prefers the "
                         "predictable, inverse dynamics prefers the "
                         "controllable. 0 reproduces the JEPA-only objective "
                         "whose encoder cannot tell that the characters swapped "
                         "sides (spatial_probe AUC 0.540).")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="continue from these weights instead of random init. "
                         "Optimiser state and LR schedule restart.")
    args = ap.parse_args()

    # Before anything long-running. `on_eval` writes best.pt into --ckpt-dir at
    # the first eval, and `save_checkpoint` -- the only thing that used to
    # create that directory -- does not run until training finishes. On a fresh
    # --ckpt-dir this killed the run at step 5000, eighteen minutes in, with the
    # eval already computed and then thrown away along with everything else.
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    make = Config.soku448 if args.image_size == 448 else Config.soku
    cfg = make(device=args.device, batch_size=args.batch_size,
               num_workers=args.num_workers, total_steps=args.steps,
               warmup_steps=args.warmup, lr=args.lr, seed=args.seed,
               cf_coef=args.cf_coef, hud_coef=args.hud_coef,
               idm_coef=args.idm_coef, compile=not args.no_compile)
    if args.image_size not in (224, 448):
        cfg = replace(cfg, image_size=args.image_size)
    print(f"image {cfg.image_size} px, patch {cfg.patch_size} -> "
          f"{cfg.num_patches} patches | cf_coef {cfg.cf_coef} "
          f"hud_coef {cfg.hud_coef} idm_coef {cfg.idm_coef}", flush=True)
    cache = torch.load(args.corpus / "val.pt", map_location="cpu", weights_only=False)
    # The cache stores raw frames at whatever resolution it was built for, and a
    # mismatch does not surface until the first eval -- thousands of steps and
    # tens of minutes in, where it reads as a crash in the eval rather than as a
    # setup error. Checked here, against the one thing that cannot be wrong.
    got = tuple(cache["obs"].shape[-2:])
    if got != (cfg.image_size, cfg.image_size):
        raise SystemExit(
            f"{args.corpus / 'val.pt'} holds {got[0]}x{got[1]} frames but this "
            f"run trains at {cfg.image_size}. Rebuild it:\n"
            f"  python -m scripts.build_val_cache --image-size {cfg.image_size} "
            f"--manifest-root {args.corpus}")
    print(f"val cache: {cache['obs'].shape[0]} windows", flush=True)

    set_seed(args.seed)
    model = LeWorldModel(cfg)
    if args.init_from is not None:
        # Weights only. The optimiser state and the cosine schedule restart,
        # which is the honest thing to say about it: this continues training
        # from a set of weights, it does not resume an interrupted run.
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = dict(blob["model"])
        old_cfg = blob.get("cfg")
        old_size = getattr(old_cfg, "image_size", cfg.image_size)
        if old_size != cfg.image_size:
            # Re-grid the one tensor tied to the patch count. Everything else --
            # the stride-14 patch convolution, every transformer block, the CLS
            # token, the projector, the predictor -- is resolution-independent,
            # so 222 of 223 tensors transfer untouched. See
            # model.encoder.resize_pos_embed for why.
            og, ng = old_size // cfg.patch_size, cfg.image_size // cfg.patch_size
            sd["encoder.pos_embed"] = resize_pos_embed(
                sd["encoder.pos_embed"], og, ng)
            print(f"warm start {old_size} -> {cfg.image_size} px: pos_embed "
                  f"re-gridded {og}x{og} -> {ng}x{ng}", flush=True)
        # The HUD head is new when warm-starting a model trained without it.
        missing = model.load_state_dict(sd, strict=False)
        if missing.missing_keys:
            print(f"  freshly initialised: {missing.missing_keys}", flush=True)
        if missing.unexpected_keys:
            print(f"  ignored from checkpoint: {missing.unexpected_keys}", flush=True)
        print(f"initialised from {args.init_from} (step {blob.get('step')}); "
              f"optimiser and LR schedule start fresh", flush=True)
    rep = model.param_report()
    print("params: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in rep.items()), flush=True)

    # The shuffle buffer holds decoded windows, so its cost scales with the
    # square of the image size: 4096 windows is 2.5 GB at 224 px and 9.8 GB at
    # 448, *per worker*. The original 320k run had 754 GB of host RAM and could
    # ignore that; on anything smaller it is an out-of-memory kill in a DataLoader
    # worker, which surfaces only as "worker exited unexpectedly".
    #
    # So the default is a memory budget rather than a window count, and it holds
    # the same ~2.5 GB the original run used whatever the resolution.
    win_bytes = cfg.seq_len * 3 * cfg.image_size ** 2
    # A TOTAL budget split across workers, not a per-worker one. Each worker
    # holds its own buffer, so a per-worker figure multiplies by the worker count
    # and silently blows past host RAM: 16 workers at 2.5 GB each is 40 GB, which
    # on this 62 GB box leaves nothing for the val cache or the parent.
    workers = max(1, cfg.num_workers)
    shuffle_buffer = args.shuffle_buffer or max(
        128, int(args.shuffle_gb * 1e9 / (win_bytes * workers)))
    print(f"shuffle buffer {shuffle_buffer} windows x {workers} workers = "
          f"{shuffle_buffer * win_bytes * workers / 1e9:.1f} GB total", flush=True)
    ds = build_soku_dataset(cfg, [str(args.corpus / "train")],
                            shuffle_buffer=shuffle_buffer, seed=args.seed)

    hours = sum(json.loads(l)["frames"] for l in
                (args.corpus / "train" / "manifest.jsonl").read_text().splitlines()) / 60 / 3600
    windows = hours * 3600 * 15
    print(f"train: {hours:.2f} h ~= {windows/1e6:.2f}M windows | "
          f"{args.steps} steps x {cfg.batch_size} = "
          f"{args.steps*cfg.batch_size/windows:.2f} epochs", flush=True)

    curve, t0 = [], time.time()

    def on_eval(m, step, hist):
        # Evaluate a copy with BatchNorm statistics re-estimated against the
        # current weights. With a long cosine schedule the learning rate stays
        # near maximum for tens of thousands of steps, and the running stats lag
        # far enough behind that held-out loss becomes meaningless -- this run
        # reported val 1.1431 at step 30k while training loss was 0.058. The
        # copy keeps training's own running stats untouched.
        from scripts.eval_ckpt import recalibrate_bn
        # Both of these forward the val cache in batches, and both defaulted to
        # 64 -- fine at 224 on a 32 GB card, an out-of-memory kill at 448 on 4 GB
        # *after* the training steps have already proved they fit. Releasing the
        # training step's cached blocks first matters too: the allocator holds
        # them, and a deepcopy plus a recalibration pass needs the room.
        eb = args.eval_batch or eval_batch_for(cfg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        probe_model = copy.deepcopy(m)
        recalibrate_bn(probe_model, cache, batch=eb)
        ev = evaluate(probe_model, cache, cfg, batch=eb)
        ev.update(mirror_sensitivity(probe_model, cache, cfg, batch=eb))
        # Keep the weights that were actually scored. Saving `m` instead writes
        # a checkpoint whose BatchNorm statistics are not the ones the recorded
        # skill describes -- mild here, since ordinary training keeps them close
        # (+0.8689 as saved against +0.8642 recalibrated), and catastrophic in
        # finetune_action, where counterfactual negatives drag the running stats
        # off-distribution and this same pattern shipped a checkpoint measuring
        # -6.15 while its own blob recorded +0.82.
        eval_sd = {k: v.detach().cpu().clone()
                   for k, v in probe_model.state_dict().items()}
        del probe_model
        ev["step"] = step
        tail = hist[-20:] if hist else []
        ev["train_pred"] = float(np.mean([h["l_pred"] for h in tail])) if tail else float("nan")
        ev["latent_var"] = float(np.mean([h["latent_var"] for h in tail])) if tail else float("nan")
        el = time.time() - t0
        ev["elapsed_h"] = el / 3600
        ev["eta_h"] = (el / max(step, 1)) * (args.steps - step) / 3600
        # Divergence guard. BatchNorm pins each latent dimension to unit
        # variance, so two *uncorrelated* latents can differ by at most ~2.0 in
        # mean-squared terms. identity above that is structurally impossible and
        # means the normalisation has broken down -- pre-BN variance has fallen
        # to the order of eps, so BatchNorm has stopped normalising and started
        # amplifying. The first run hit identity 7.68 at step 85k.
        #
        # This has to be checked explicitly because `skill` does not catch it:
        # skill is a ratio, and an exploding latent inflates numerator and
        # denominator together. During that divergence skill *rose* from +0.35
        # to +0.69 while absolute prediction error got 6x worse.
        ev["diverged"] = bool(ev["identity"] > 2.0 or ev["latent_var"] < 0.85)
        if ev["diverged"]:
            print(f"  *** DIVERGENCE at step {step}: identity {ev['identity']:.3f} "
                  f"(max ~2.0), latent_var {ev['latent_var']:.3f} (want ~1.0) ***",
                  flush=True)
        curve.append(ev)
        args.log.write_text(json.dumps(curve, indent=2))

        # Keep the best model, not just the most recent. The first run
        # overwrote one file every 5k steps, so when it diverged at 75k the
        # healthy step-65k model was already gone.
        healthy = [c for c in curve if not c["diverged"]]
        if healthy and ev is healthy[-1] and ev["skill"] >= max(c["skill"] for c in healthy):
            torch.save({"model": eval_sd, "cfg": cfg, "step": step,
                        "eval": ev, "bn_recalibrated": True},
                       Path(args.ckpt_dir) / "best.pt")
            print(f"  saved best.pt (skill {ev['skill']:+.4f} at step {step})", flush=True)
        # And a second file selected on spatial sensitivity, because the two
        # disagree: the first inverse-dynamics run peaked on mirror_play at step
        # 4000 and on skill at the end, and keeping only the skill-best threw
        # away the checkpoint that was best at the thing the agent needs.
        if healthy and ev is healthy[-1] and ev["mirror_play"] >= max(
                c.get("mirror_play", -1.0) for c in healthy):
            torch.save({"model": eval_sd, "cfg": cfg, "step": step,
                        "eval": ev, "bn_recalibrated": True},
                       Path(args.ckpt_dir) / "best_spatial.pt")
            print(f"  saved best_spatial.pt (mirror_play "
                  f"{ev['mirror_play']:.3f} at step {step})", flush=True)
        print(f"  [eval] step {step:6d} | train {ev['train_pred']:.4f} "
              f"| val {ev['val_pred']:.4f} | identity {ev['identity']:.4f} "
              f"| skill {ev['skill']:+.4f} | AUC {ev['inv_dyn_auc']:.4f} "
              f"| mirror {ev['mirror_play']:.3f}/{ev['mirror_full']:.3f} "
              f"| var {ev['latent_var']:.3f} | {ev['elapsed_h']:.2f}h elapsed, "
              f"{ev['eta_h']:.2f}h left", flush=True)

    train(cfg, ds, steps=args.steps, model=model, verbose=True, log_every=500,
          ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every,
          callback=on_eval, callback_every=args.eval_every)

    save_checkpoint(model, cfg, args.ckpt_dir, args.steps)
    if curve:
        b = max(curve, key=lambda r: r["skill"])
        print(f"\nbest skill {b['skill']:+.4f} at step {b['step']} "
              f"(val {b['val_pred']:.4f}, AUC {b['inv_dyn_auc']:.4f})")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {args.ckpt_dir}")


if __name__ == "__main__":
    main()
