"""Fine-tune the world model so its predictions depend on the action.

    python -m scripts.finetune_action --ckpt /root/ckpt/best.pt --steps 30000

WHY
---
The pretrained model predicts the next latent well (skill +0.864) and is nearly
blind to what the player does: rolling one start state forward under the most
different action sequences available moves the probed health by 0.046 of a bar,
against a probe whose own noise is about 0.13 (scripts/action_sensitivity).
Nothing a policy does is legible above that, so GRPO sat at its random-init
baseline for 800 steps.

The cause is visible in the objective. `prediction_loss` is an MSE against the
true next latent, most of which the current latent already determines -- the
scene carries its own momentum. The action-dependent part of the target is
small, earns little gradient, and a predictor that ignores its conditioning
entirely still scores well. Prediction accuracy and control fidelity are
different properties, and only the first was being optimised.

THE COUNTERFACTUAL OBJECTIVE
----------------------------
Among the true action and several wrong ones, the true action must produce the
prediction closest to what actually happened next:

    d(a)  = || predictor(z, a) - z_next ||^2
    L_cf  = cross entropy over -d(a)/tau, with the true action as the target

Wrong actions are other samples' action sequences from the same batch, so every
candidate is a real thing a player did, just not here.

The number is directly interpretable. A model that ignores its action input
gives every candidate the same distance and scores ln(1 + n_negatives) -- 1.386
at three negatives. Zero is perfect discrimination. The loss reads as "how
action-blind is this model", in nats.

WHY NOT INVERSE DYNAMICS
------------------------
The first attempt asked a head to recover the action from the transition,
`IDM(z_t, zhat_t) -> a_t`, reasoning that a predictor ignoring its action input
produces a zhat carrying no action information. That much is true. The converse
is not: the predictor can satisfy the head by stamping the action into a spare
direction of zhat without modelling anything the action *does*. That is what
happened, and the run says so plainly:

    step    pred     idm_pred   idm_real
     200    0.0079    1.0244     1.0246
     600    0.0180    0.7807     0.8132
    1000    0.0419    0.6431     0.7815
    1400    0.0499    0.5305     0.7849

`idm_real` -- recovering the action from a *real* transition, which is the
genuine physical signal -- stalled around 0.78 by step 600 and stopped
improving. `idm_pred` kept falling well past it while the prediction loss
degraded more than sixfold. The model was buying a decodable watermark with
prediction accuracy, and every nat below 0.78 was decoration.

The counterfactual objective closes that hole. A watermark makes zhat encode
whichever action it was handed, but the real next latent carries no watermark to
match, so the extra component pushes the true and the wrong candidates equally
far from the target and wins nothing. The only way to separate them is for the
action to genuinely inform the prediction.
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
from sokubot.data.soku import build_soku_dataset
from sokubot.losses import prediction_loss
from sokubot.losses.sigreg import sigreg_stepwise
from sokubot.model.world_model import LeWorldModel
from sokubot.train import build_optimizer, enable_fast_math, make_loader, set_seed


def counterfactual_loss(model: LeWorldModel, z: torch.Tensor, act: torch.Tensor,
                        n_neg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (loss, accuracy). Chance is ln(1+n_neg) nats and 1/(1+n_neg).

    `z` arrives detached. This term is about the predictor's use of its
    conditioning; letting it reshape the encoder would hand it a second way to
    cheat, by making latents that are easy to tell apart rather than predictions
    that are accurate.
    """
    B = z.shape[0]
    z_next = z[:, 1:]

    def dist(a: torch.Tensor) -> torch.Tensor:
        zhat = model.predictor(z, model.action_encoder(a))
        return ((zhat[:, :-1] - z_next) ** 2).mean(dim=-1)      # [B, T-1]

    d_true = dist(act)
    negs = []
    for _ in range(n_neg):
        # Roll rather than shuffle: guarantees no sample keeps its own actions.
        shift = int(torch.randint(1, max(2, B), (1,)).item())
        negs.append(dist(act.roll(shift, dims=0)))

    all_d = torch.stack([d_true, *negs], dim=-1)                # [B, T-1, 1+n_neg]
    tau = d_true.detach().mean().clamp_min(1e-8)
    flat = (-all_d / tau).reshape(-1, 1 + n_neg)
    target = torch.zeros(flat.shape[0], dtype=torch.long, device=z.device)
    acc = (flat.argmax(dim=-1) == 0).float().mean()
    return F.cross_entropy(flat, target), acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("/root/ckpt_cf"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="fine-tune, not pretrain: the 2e-4 that trained this "
                         "model from scratch would discard what it knows")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lambda-cf", type=float, default=1.0)
    ap.add_argument("--n-neg", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    cfg.batch_size = a.batch_size
    cfg.num_workers = a.num_workers
    cfg.total_steps = a.steps
    cfg.warmup_steps = a.warmup
    cfg.lr = a.lr
    cfg.seed = a.seed
    cfg.compile = False          # several predictor passes per step; not worth it

    set_seed(a.seed)
    enable_fast_math(cfg)
    device = torch.device(a.device)
    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    chance = float(np.log(1 + a.n_neg))
    print(f"loaded {a.ckpt} (step {blob.get('step','?')})", flush=True)
    print(f"action-blind baseline for cf: {chance:.4f} nats, "
          f"accuracy {1/(1+a.n_neg):.3f}", flush=True)

    opt = build_optimizer(model, cfg)
    ds = build_soku_dataset(cfg, [str(a.corpus / "train")], shuffle_buffer=4096,
                            seed=a.seed)
    loader = make_loader(ds, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))))

    hist, t0, step = [], time.time(), 0
    run = {"pred": 0.0, "sig": 0.0, "cf": 0.0, "acc": 0.0, "n": 0}
    for batch in loader:
        if step >= a.steps:
            break
        obs = batch["obs"].to(device, non_blocking=True)
        act = batch["actions"].to(device, non_blocking=True).float()

        out = model(obs, act)
        z, zhat = out.z, out.zhat
        l_pred = prediction_loss(zhat, z)
        l_sig = sigreg_stepwise(z, cfg)
        l_cf, acc = counterfactual_loss(model, z.detach(), act, a.n_neg)

        loss = l_pred + cfg.lambda_sigreg * l_sig + a.lambda_cf * l_cf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        with torch.no_grad():
            run["pred"] += float(l_pred); run["sig"] += float(l_sig)
            run["cf"] += float(l_cf); run["acc"] += float(acc); run["n"] += 1

        if step % a.log_every == 0:
            n = max(run["n"], 1)
            with torch.no_grad():
                var = float(z.reshape(-1, cfg.latent_dim).var(dim=0).mean())
            el = time.time() - t0
            rec = {"step": step, "l_pred": run["pred"] / n, "sigreg": run["sig"] / n,
                   "cf": run["cf"] / n, "cf_acc": run["acc"] / n,
                   "latent_var": var, "lr": sched.get_last_lr()[0],
                   "elapsed_h": el / 3600,
                   "eta_h": (el / step) * (a.steps - step) / 3600}
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            print(f"step {step:6d} | pred {rec['l_pred']:.4f} | cf {rec['cf']:.4f} "
                  f"(blind {chance:.3f}) | cf_acc {rec['cf_acc']:.3f} | sig "
                  f"{rec['sigreg']:.4f} | var {var:.3f} | {rec['eta_h']:.2f}h left",
                  flush=True)
            run = {k: 0.0 for k in run}

        if step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                       a.out / "sokubot_cf.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
               a.out / "sokubot_cf.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {a.out}")
    print(f"next: python -m scripts.action_sensitivity --ckpt {a.out/'sokubot_cf.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
