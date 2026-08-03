"""LeWorldModel training loop.

    L = L_pred + lambda * SIGReg(Z),    lambda = 0.1

No stop-gradient, no EMA, no target encoder, no auxiliary heads. Gradients flow
through every term and all parameters are optimised jointly (LeWM Sec. 3.1).

Run:
    python -m sokubot.train --preset pusht --data-root data/pusht --device cuda
    python -m sokubot.train --preset soku  --data-root /path/to/dataset --device cuda
    python -m sokubot.train --preset tiny  --data-root data/pusht-tiny --steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .config import Config
from .losses import prediction_loss, sigreg_stepwise
from .model.world_model import LeWorldModel


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def effective_rank(z: torch.Tensor) -> float:
    """Entropy-based effective rank of the latent covariance.

    DO NOT USE THIS TO DETECT COLLAPSE. It is logged because it is cheap and
    occasionally suggestive, but measured on this codebase it is actively
    misleading in both directions:

        lambda    R^2 (probe)  latent_var   eff_rank
        0.0            0.097       0.000      59.47   <- fully collapsed
        0.03           0.092       0.000      45.18   <- fully collapsed
        0.1            0.242       0.935       3.45   <- best
        1.0            0.179       0.952      12.27   <- over-regularised

    A collapsed latent scores *near the maximum*. Once the projector's output is
    constant, BatchNorm divides it by ~zero and what survives is float noise --
    which is isotropic, so its covariance spectrum is flat and its entropy is
    high. Meanwhile the best model reads 3.45, because PushT genuinely has about
    five degrees of freedom and a good encoder finds them; above that, rank
    climbs with lambda as SIGReg pushes toward isotropy. The number tracks
    lambda, not quality.

    `latent_var` is the reliable cheap signal (0 means collapsed), and
    `probe.py` is the real measure. See README.

    This is a diagnostic and must never be able to crash training, hence float64
    on CPU, diagonal jitter, and a bare except: eigvalsh can fail to converge on
    an ill-conditioned covariance during a transient loss spike, which is
    exactly when the number is most wanted.
    """
    try:
        with torch.autocast(device_type=z.device.type, enabled=False):
            zc = (z - z.mean(0, keepdim=True)).float()
            cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
            cov = cov.double().cpu()
            cov = cov + 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype)
            ev = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
            p = ev / ev.sum()
            return float(torch.exp(-(p * p.log()).sum()).item())
    except Exception:
        return float("nan")


def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
    return cfg.lr * (0.001 + 0.999 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay switched off for norms, biases and embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in ("pos_embed", "cls_token", "offset")):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def compute_losses(
    model: LeWorldModel, batch: Dict, cfg: Config
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(model.parameters()).device
    obs = batch["obs"].to(device, non_blocking=True)          # [B, T, 3, S, S]
    actions = batch["actions"].to(device, non_blocking=True)  # [B, T, ticks, A]

    out = model(obs, actions)
    l_pred = prediction_loss(out.zhat, out.z)
    l_sig = sigreg_stepwise(out.z, cfg)
    total = l_pred + cfg.lambda_sigreg * l_sig

    with torch.no_grad():
        flat = out.z.reshape(-1, cfg.latent_dim)
        metrics = {
            "loss": float(total.item()),
            "l_pred": float(l_pred.item()),
            "l_sigreg": float(l_sig.item()),
            "latent_var": float(flat.var(dim=0).mean().item()),
            "eff_rank": effective_rank(flat.detach()),
        }
    return total, metrics


def make_loader(dataset: Dataset, cfg: Config) -> DataLoader:
    iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=not iterable,
        drop_last=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=(cfg.device.startswith("cuda")),
    )


def train(
    cfg: Config,
    dataset: Dataset,
    steps: Optional[int] = None,
    log_every: int = 10,
    ckpt_dir: Optional[str | Path] = None,
    ckpt_every: int = 1000,
    model: Optional[LeWorldModel] = None,
    verbose: bool = True,
):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    steps = steps or cfg.total_steps

    model = (model or LeWorldModel(cfg)).to(device)
    loader = make_loader(dataset, cfg)
    opt = build_optimizer(model, cfg)

    use_amp = device.type == "cuda" and cfg.amp_dtype in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and cfg.amp_dtype == "fp16"))

    model.train()
    history = []
    it = iter(loader)
    t0 = time.time()

    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast("cuda", dtype=amp_dtype):
                total, metrics = compute_losses(model, batch, cfg)
        else:
            total, metrics = compute_losses(model, batch, cfg)

        # A single non-finite step must not poison the weights.
        if not torch.isfinite(total):
            if verbose:
                print(f"step {step:6d} | non-finite loss, skipped")
            continue

        if use_amp:
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        metrics["step"] = step
        metrics["lr"] = opt.param_groups[0]["lr"]
        history.append(metrics)

        if ckpt_dir and ckpt_every and step > 0 and step % ckpt_every == 0:
            save_checkpoint(model, cfg, ckpt_dir, step)
        if verbose and (step % log_every == 0 or step == steps - 1):
            print(
                f"step {step:6d} | loss {metrics['loss']:8.4f} "
                f"| pred {metrics['l_pred']:7.4f} | sigreg {metrics['l_sigreg']:7.4f} "
                f"| var {metrics['latent_var']:6.3f} | erank {metrics['eff_rank']:7.2f} "
                f"| {(time.time() - t0) / (step + 1):5.2f}s/it"
            )

    if ckpt_dir:
        save_checkpoint(model, cfg, ckpt_dir, steps)
    return model, history


def save_checkpoint(model: LeWorldModel, cfg: Config, ckpt_dir: str | Path, step: int) -> Path:
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "sokubot.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, path)
    return path


def build_dataset(cfg: Config, preset: str, data_root: str):
    if preset == "soku":
        from .data.soku import build_soku_dataset

        return build_soku_dataset(cfg, [data_root])
    from .data.window import EpisodeWindowDataset

    return EpisodeWindowDataset(cfg, data_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=["pusht", "soku", "tiny"], default="pusht")
    ap.add_argument("--data-root", required=True,
                    help="dir of .npz episodes (pusht/tiny) or of Soku capture manifests")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log-json", default=None, help="write per-step metrics here")
    args = ap.parse_args()

    cfg = {"pusht": Config.pusht, "soku": Config.soku, "tiny": Config.tiny}[args.preset]()
    cfg.device = args.device
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    model = LeWorldModel(cfg)
    rep = model.param_report()
    print("parameters: " + ", ".join(f"{k} {v/1e6:.3f}M" for k, v in rep.items()))

    dataset = build_dataset(cfg, args.preset, args.data_root)
    _, history = train(cfg, dataset, steps=args.steps, ckpt_dir=args.ckpt_dir, model=model)

    if args.log_json:
        Path(args.log_json).write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
