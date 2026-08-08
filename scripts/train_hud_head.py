"""Train a HUD delta head on frozen trunk features.

    python -m scripts.train_hud_head --ckpt ~/sokubot-art/wm_cf_bnfix.pt \
        --bank ~/bank_hud.npz --out ~/hud_head --steps 15000

WHAT THIS IS FOR
----------------
The reward's KO detector runs at precision 0.003 because the probe reads health
with 0.116 error against a 0.06 threshold. `scripts/anchored_ko_test.py` showed
only ~0.025 of that is a fixed offset -- anchoring the level to `data/hud.py`
fixed the offset exactly and moved KO precision from 0.005 to 0.015, i.e. not at
all. The error is in the probe's per-step **deltas**.

So this learns the deltas directly, from the exact current HUD plus the trunk's
action-conditioned state.

WHAT IT DELIBERATELY DOES NOT DO
---------------------------------
Touch the predictor. Phase 1 measured that fine-tuning it for predictive accuracy
destroys action-awareness: at h=4 the held-out action->return correlation fell
from +0.2622 to +0.1938 (unrolled) and +0.1545 (one-step), and only the untouched
base transfers to unseen starts at all. Here the trunk is read under `no_grad`
and the latent rollout is bit-identical to the base model's -- a property the
smoke test asserts rather than assumes.

THE GATE
--------
Beating copy-forward (rel-err < 1.0) is necessary but not sufficient. The number
that decides whether `win`/`lose` can stay in the reward is KO precision from
`scripts/anchored_ko_test.py --hud-head`, which has to clear roughly 0.5 for the
+-5 term to be paying for outcomes rather than for noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.losses.prediction import rollout_loss
from sokubot.model.augmented import HUD_CHANNELS, HudDeltaHead, HudWorldModel
from sokubot.model.world_model import LeWorldModel
from scripts.finetune_predictor import valid_starts
from scripts.train_grpo import encoder_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--plan", type=int, default=16)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    blob = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]; cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    d = np.load(a.bank)
    if "hud" not in d.files:
        raise SystemExit(f"{a.bank} has no `hud`; use scripts.build_hud_bank")
    Z, A, Hd, ep = d["z"], d["a"], d["hud"], d["ep"]
    chans = tuple(str(x) for x in d["hud_channels"])
    if chans != HUD_CHANNELS:
        raise SystemExit(f"bank channels {chans} != model's {HUD_CHANNELS}")
    bank_enc = str(d["encoder_fingerprint"]) if "encoder_fingerprint" in d.files else None
    if bank_enc and bank_enc != encoder_fingerprint(wm):
        raise SystemExit(
            f"bank encoder {bank_enc} != this checkpoint's "
            f"{encoder_fingerprint(wm)}; latents mean nothing across encoders")

    n_ep = int(ep.max()) + 1
    val_ep = rng.choice(n_ep, size=max(1, int(n_ep * a.val_frac)), replace=False)
    is_val = np.isin(ep, val_ep)
    starts = valid_starts(ep, cfg.history, a.plan)
    tr_starts, va_starts = starts[~is_val[starts]], starts[is_val[starts]]
    print(f"bank {len(Z)} latents, {n_ep} replays | {len(tr_starts)} train / "
          f"{len(va_starts)} val starts", flush=True)

    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)
    Ht = torch.from_numpy(Hd).to(a.device)

    head = HudDeltaHead(cfg, len(HUD_CHANNELS), a.width, a.depth).to(a.device)
    model = HudWorldModel(wm, head).to(a.device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {n_tr/1e6:.3f}M params (hud head only); predictor frozen: "
          f"{not any(p.requires_grad for p in model.predictor.parameters())}",
          flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.steps), eta_min=a.lr * 0.05)

    H, P = cfg.history, a.plan
    off = torch.arange(H, device=a.device) - (H - 1)
    plan_off = torch.arange(P, device=a.device)

    def batch(idx):
        return (Zt[idx[:, None] + off[None, :]].float(),
                Ht[idx[:, None] + off[None, :]].float(),
                At[idx[:, None] + off[None, :-1]].float(),
                At[idx[:, None] + plan_off[None, :]].float(),
                Ht[idx[:, None] + 1 + plan_off[None, :]].float())

    @torch.no_grad()
    def validate() -> dict:
        head.eval()
        rows, n = {}, 0
        for s in range(0, min(len(va_starts), 4096), a.batch):
            idx = torch.from_numpy(va_starts[s : s + a.batch]).to(a.device)
            if len(idx) < 8:
                break
            z_ctx, h_ctx, a_hist, a_plan, h_true = batch(idx)
            _, hh = model.rollout(z_ctx, h_ctx, a_plan, a_hist)
            _, rep = rollout_loss(hh, h_true, h_ctx[:, -1], a.horizons)
            for k, v in rep.items():
                rows[k] = rows.get(k, 0.0) + v
            n += 1
        head.train()
        return {k: v / max(n, 1) for k, v in rows.items()}

    va0 = validate()
    print("before: hud rel-err " +
          " ".join(f"h{k} {v:.4f}" for k, v in sorted(va0.items())) +
          "   (1.0 == copy-forward, which an untrained head reproduces exactly)",
          flush=True)

    hist, best, t0 = [], float("inf"), time.time()
    for step in range(1, a.steps + 1):
        idx = torch.from_numpy(rng.choice(tr_starts, size=a.batch)).to(a.device)
        z_ctx, h_ctx, a_hist, a_plan, h_true = batch(idx)
        _, hh = model.rollout(z_ctx, h_ctx, a_plan, a_hist)
        loss, rep = rollout_loss(hh, h_true, h_ctx[:, -1], a.horizons)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
            sched.step()

        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.detach()),
                   "grad_norm": float(gn),
                   **{f"h{k}": v for k, v in rep.items()},
                   "elapsed_h": (time.time() - t0) / 3600}
            if step % a.eval_every == 0 or step == 1:
                va = validate()
                rec.update({f"val_h{k}": v for k, v in va.items()})
                m = float(np.mean(list(va.values())))
                rec["val_mean"] = m
                if m < best:
                    best = m
                    torch.save({"head": head.state_dict(), "cfg": cfg,
                                "step": step, "val": va, "val_mean": m,
                                "base_ckpt": str(a.ckpt),
                                "hud_channels": list(HUD_CHANNELS),
                                "width": a.width, "depth": a.depth},
                               a.out / "hud_head_best.pt")
                print(f"  [val] step {step:6d} | mean {m:.4f} | " +
                      " ".join(f"h{k} {v:.4f}" for k, v in sorted(va.items())),
                      flush=True)
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            print(f"step {step:6d} | loss {rec['loss']:.4f} | " +
                  " ".join(f"h{k} {rep[k]:.3f}" for k in sorted(rep)) +
                  f" | {rec['elapsed_h']:.2f}h", flush=True)

    va = validate()
    torch.save({"head": head.state_dict(), "cfg": cfg, "step": a.steps, "val": va,
                "base_ckpt": str(a.ckpt), "hud_channels": list(HUD_CHANNELS),
                "width": a.width, "depth": a.depth}, a.out / "hud_head_final.pt")
    print("\nafter: hud rel-err " +
          "  ".join(f"h{k} {va[k]:.4f} (was {va0[k]:.4f})" for k in sorted(va)))
    print(f"\n-> {a.out}")
    print("Gate: python -m scripts.anchored_ko_test --hud-head "
          f"{a.out}/hud_head_best.pt   -- KO precision must clear ~0.5 for "
          "win/lose to be paying for outcomes rather than noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
