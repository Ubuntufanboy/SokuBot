"""Fine-tune the world model so its predictions depend on the action.

    python -m scripts.finetune_idm --ckpt /root/ckpt/best.pt --steps 30000

WHY
---
The pretrained model predicts the next latent well (skill +0.864) and is
nearly blind to what the player does: rolling one start state forward under the
most different action sequences available moves the probed health by 0.046 of a
bar, against a probe whose own noise is about 0.13 (scripts/action_sensitivity).
Nothing a policy does is legible above that, so GRPO sat at its random-init
baseline for 800 steps.

The cause is visible in the objective. `prediction_loss` is an MSE against the
true next latent, most of which is determined by the current latent -- the scene
carries its own momentum. The action-dependent part of the target is small, so
it contributes little to the gradient, and a predictor that ignores its
conditioning entirely still scores well. Prediction accuracy and control
fidelity are simply different things, and only the first was being optimised.

WHAT THIS ADDS
--------------
An inverse-dynamics head that must recover the action from a transition, applied
in two places:

  predicted   IDM(z_t, zhat_t) -> a_t. This is the one that matters. If the
              predictor ignores its action input then zhat_t carries no
              information about a_t and this loss cannot fall, so minimising it
              forces the conditioning pathway to do work.
  real        IDM(z_t, z_t+1) -> a_t. A prerequisite for the above: if the
              encoder throws away the state that distinguishes actions, there is
              nothing for the predictor to reproduce.

One head serves both, so a predicted transition has to look like a real one from
its point of view rather than developing a private code.

The obvious failure is the model satisfying the head by stamping the action into
some spare direction without modelling what the action *does*. Two things push
against it: `prediction_loss` keeps its weight, so zhat must still land on the
true next latent, and the true next latent genuinely does depend on the action
-- so carrying that information is consistent with accuracy rather than traded
against it. It is still worth checking rather than assuming, which is why
success is judged by re-running action_sensitivity and confirming the *direction*
of the effects is sensible, not merely that their magnitude grew.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.soku import build_soku_dataset
from sokubot.losses import prediction_loss
from sokubot.losses.sigreg import sigreg_stepwise
from sokubot.model.world_model import LeWorldModel
from sokubot.train import build_optimizer, enable_fast_math, make_loader, set_seed


class InverseDynamics(nn.Module):
    """(z_t, z_next) -> the action chunk that produced the transition.

    Deliberately small and shallow. A large head could recover the action from
    subtle residue the predictor never has to reproduce, which would let the
    loss fall without the conditioning pathway strengthening at all.
    """

    def __init__(self, latent: int, ticks: int, action_dim: int, width: int = 512):
        super().__init__()
        self.ticks, self.action_dim = ticks, action_dim
        self.net = nn.Sequential(
            nn.Linear(latent * 2, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, ticks * action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, z_next], dim=-1)
        return self.net(x).view(*z_t.shape[:-1], self.ticks, self.action_dim)


def idm_loss(head: InverseDynamics, z_t, z_next, actions, pos_weight) -> torch.Tensor:
    """Binary cross-entropy over every button of every tick in the chunk."""
    logits = head(z_t, z_next)
    return F.binary_cross_entropy_with_logits(logits, actions, pos_weight=pos_weight)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("/root/ckpt_idm"))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="fine-tune, not pretrain: the 2e-4 that trained this "
                         "model from scratch would discard what it knows")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lambda-idm", type=float, default=1.0)
    ap.add_argument("--lambda-idm-real", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
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
    # The IDM head is a separate module, so the compiled graph would have to be
    # stitched across two of them; not worth the startup cost for a fine-tune.
    cfg.compile = False

    set_seed(a.seed)
    enable_fast_math(cfg)
    device = torch.device(a.device)
    model = LeWorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    print(f"loaded {a.ckpt} (step {blob.get('step','?')})", flush=True)

    head = InverseDynamics(cfg.latent_dim, cfg.action_ticks, cfg.action_dim).to(device)
    opt = build_optimizer(model, cfg)
    opt.add_param_group({"params": head.parameters(), "lr": a.lr})

    ds = build_soku_dataset(cfg, [str(a.corpus / "train")], shuffle_buffer=4096,
                            seed=a.seed)
    loader = make_loader(ds, cfg)

    # Buttons are held about a tenth of the time. Unweighted BCE has an easy
    # optimum at "never pressed", which is exactly how the behaviour-cloning head
    # reached 0.910 accuracy while carrying almost no signal (mean AUC 0.564).
    pos_weight = torch.full((cfg.action_dim,), 6.0, device=device)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))))

    hist, t0, step = [], time.time(), 0
    run = {"pred": 0.0, "sig": 0.0, "idm_p": 0.0, "idm_r": 0.0, "n": 0}
    for batch in loader:
        if step >= a.steps:
            break
        obs = batch["obs"].to(device, non_blocking=True)
        act = batch["actions"].to(device, non_blocking=True).float()

        out = model(obs, act)
        z, zhat = out.z, out.zhat
        l_pred = prediction_loss(zhat, z)
        l_sig = sigreg_stepwise(z, cfg)

        # zhat[:, t] is the prediction of z[:, t+1] under act[:, t].
        zt, znext, at = z[:, :-1], z[:, 1:], act[:, :-1]
        l_idm_p = idm_loss(head, zt, zhat[:, :-1], at, pos_weight)
        l_idm_r = idm_loss(head, zt, znext, at, pos_weight)

        loss = (l_pred + cfg.lambda_sigreg * l_sig
                + a.lambda_idm * l_idm_p + a.lambda_idm_real * l_idm_r)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        step += 1

        with torch.no_grad():
            run["pred"] += float(l_pred); run["sig"] += float(l_sig)
            run["idm_p"] += float(l_idm_p); run["idm_r"] += float(l_idm_r)
        run["n"] += 1

        if step % a.log_every == 0:
            n = max(run["n"], 1)
            with torch.no_grad():
                var = float(z.reshape(-1, cfg.latent_dim).var(dim=0).mean())
            el = time.time() - t0
            rec = {"step": step, "l_pred": run["pred"] / n, "sigreg": run["sig"] / n,
                   "idm_pred": run["idm_p"] / n, "idm_real": run["idm_r"] / n,
                   "latent_var": var, "lr": sched.get_last_lr()[0],
                   "elapsed_h": el / 3600,
                   "eta_h": (el / step) * (a.steps - step) / 3600}
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            # idm_pred is the number to watch: it is the one that can only fall
            # if the predictor's action conditioning is doing work.
            print(f"step {step:6d} | pred {rec['l_pred']:.4f} | idm_pred "
                  f"{rec['idm_pred']:.4f} | idm_real {rec['idm_real']:.4f} | "
                  f"sig {rec['sigreg']:.4f} | var {var:.3f} | "
                  f"{rec['eta_h']:.2f}h left", flush=True)
            run = {k: 0.0 for k in run}

        if step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "idm": head.state_dict(),
                        "cfg": cfg, "step": step}, a.out / "sokubot_idm.pt")

    torch.save({"model": model.state_dict(), "idm": head.state_dict(),
                "cfg": cfg, "step": step}, a.out / "sokubot_idm.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {a.out}")
    print("next: python -m scripts.action_sensitivity --ckpt "
          f"{a.out/'sokubot_idm.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
