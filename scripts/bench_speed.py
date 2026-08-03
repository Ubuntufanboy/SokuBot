"""Measure training throughput under each speedup, cumulatively.

    python -m scripts.bench_speed --data-root /root/exp/train_10h

Each row adds one lever to the row above, so the delta column is that lever's
contribution in context rather than in isolation. Throughput is reported in
windows/s as well as steps/s, because the batch-size rows are not comparable on
steps/s alone.

Everything here changes numerics. The uint8 loader (measured separately, and
already the default) does not; these do.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import replace

import torch

from sokubot.config import Config
from sokubot.data.soku import build_soku_dataset
from sokubot.model.world_model import LeWorldModel
from sokubot.train import build_optimizer, compute_losses, enable_fast_math, make_loader


def bench(cfg: Config, data_root: str, steps: int = 50, warmup: int = 15,
          label: str = "") -> dict:
    enable_fast_math(cfg)
    torch.manual_seed(0)
    ds = build_soku_dataset(cfg, [data_root], shuffle_buffer=1024)
    model = LeWorldModel(cfg).to(cfg.device)
    opt = build_optimizer(model, cfg)
    step_model = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
    model.train()
    loader = make_loader(ds, cfg)
    it = iter(loader)

    def one():
        b = next(it)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, _ = compute_losses(step_model, b, cfg, want_metrics=False)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    t_warm = time.time()
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    warm_s = time.time() - t_warm

    t0 = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps

    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    del model, step_model, opt, loader, it, ds
    gc.collect()
    torch.cuda.empty_cache()
    return {"label": label, "ms": dt * 1000, "steps_s": 1 / dt,
            "windows_s": cfg.batch_size / dt, "peak_gb": peak, "warmup_s": warm_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="/root/exp/train_10h")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    base = dict(device="cuda", batch_size=128, num_workers=args.workers,
                loader_uint8=True, prefetch_factor=6,
                metrics_every=0, tf32=False, fused_optimizer=False,
                sigreg_batched=False, compile=False)

    rows, cfgs = [], []
    cfgs.append(("baseline (uint8 loader only)", dict(base)))
    step = dict(base, metrics_every=50)
    cfgs.append(("+ metrics every 50 steps", dict(step)))
    step = dict(step, tf32=True)
    cfgs.append(("+ TF32 / cudnn.benchmark", dict(step)))
    step = dict(step, fused_optimizer=True)
    cfgs.append(("+ fused AdamW", dict(step)))
    step = dict(step, sigreg_batched=True)
    cfgs.append(("+ batched SIGReg", dict(step)))
    if not args.skip_compile:
        c = dict(step, compile=True, compile_mode="default")
        cfgs.append(("+ torch.compile", dict(c)))
        c2 = dict(step, compile=True, compile_mode="max-autotune")
        cfgs.append(("+ torch.compile max-autotune", dict(c2)))
        step = c

    for label, kw in cfgs:
        try:
            r = bench(Config.soku(**kw), args.data_root, steps=args.steps,
                      warmup=40 if kw.get("compile") else 15, label=label)
        except Exception as exc:
            print(f"{label:34s} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        rows.append(r)
        d = ""
        if len(rows) > 1:
            d = f"  ({rows[0]['ms'] / r['ms']:.2f}x cumulative)"
        print(f"{label:34s} {r['ms']:7.1f} ms  {r['steps_s']:5.2f} st/s  "
              f"{r['windows_s']:7.0f} win/s  {r['peak_gb']:4.1f} GB{d}", flush=True)

    # Batch size at the best configuration found.
    best = dict(cfgs[-1][1])
    print()
    for bs in (128, 192, 256, 384):
        try:
            r = bench(Config.soku(**dict(best, batch_size=bs)), args.data_root,
                      steps=max(20, args.steps // 2),
                      warmup=40 if best.get("compile") else 15,
                      label=f"batch {bs}")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:<28d} OOM")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            print(f"batch {bs:<28d} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        print(f"batch {bs:<28d} {r['ms']:7.1f} ms  {r['steps_s']:5.2f} st/s  "
              f"{r['windows_s']:7.0f} win/s  {r['peak_gb']:4.1f} GB", flush=True)


if __name__ == "__main__":
    main()
