"""Train the pixel decoder and the behaviour-cloning policy together.

    python -m scripts.train_heads --corpus /root/corpus --wm /root/ckpt/best.pt

Both read the same frozen world-model latents, so they share one dataloader and
one encoder pass. Video decode dominates the cost here, so running them jointly
is roughly half the wall time of running them in series -- and neither head is
large enough to contend for the GPU.

  decoder  latent -> 224x224 RGB. Visualisation and diagnostics only, exactly as
           in LeWM App. D. It is what makes imagined rollouts inspectable, and
           the planned horizon ablation measures decoded fidelity as a proxy for
           how much noise a rollout has accumulated.

  bc       latent history -> both players' 20 buttons. Supplies the opponent in
           GRPO self-play. It predicts *both* sides so the same weights can play
           either seat; the agent's own seat is masked out at rollout time.

The world model is frozen: these heads must not reshape the representation the
scaling study and the 200 h run produced.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sokubot.config import Config
from sokubot.data.soku import build_soku_dataset
from sokubot.model.layers import Block
from sokubot.model.world_model import LeWorldModel
from sokubot.train import make_loader, set_seed, enable_fast_math

class Decoder(nn.Module):
    """Latent -> image, via learned patch queries cross-attending to the latent."""
    def __init__(self, latent, img=224, patch=16, dim=256, depth=4, heads=8):
        super().__init__()
        self.p, self.n = patch, (img // patch) ** 2
        self.img = img
        self.q = nn.Parameter(torch.zeros(1, self.n, dim)); nn.init.trunc_normal_(self.q, std=.02)
        self.ctx = nn.Linear(latent, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, 4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch * 3)
    def forward(self, z):
        B = z.shape[0]
        x = self.q.expand(B, -1, -1) + self.ctx(z).unsqueeze(1)   # inject latent everywhere
        for b in self.blocks: x = b(x)
        p = self.out(self.norm(x))                                # [B,n,p*p*3]
        s = self.img // self.p
        p = p.view(B, s, s, self.p, self.p, 3).permute(0, 5, 1, 3, 2, 4)
        return p.reshape(B, 3, self.img, self.img)

class BCPolicy(nn.Module):
    """Latent history -> 20 button logits (both players)."""
    def __init__(self, latent, hist, dim=384, depth=3, heads=6, n_act=20):
        super().__init__()
        self.inp = nn.Linear(latent, dim)
        self.pos = nn.Parameter(torch.zeros(1, hist, dim)); nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.ModuleList(Block(dim, heads, 4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, n_act)
    def forward(self, zs):                       # [B,T,latent]
        x = self.inp(zs) + self.pos[:, :zs.shape[1]]
        for b in self.blocks: x = b(x, causal=True)
        return self.head(self.norm(x[:, -1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--wm", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, default=Path("/root/heads"))
    ap.add_argument("--log", type=Path, default=Path("/root/heads_log.json"))
    a = ap.parse_args()

    blob = torch.load(a.wm, map_location="cuda", weights_only=False)
    cfg: Config = blob["cfg"]; cfg.device = "cuda"
    cfg.batch_size = a.batch_size; cfg.num_workers = a.num_workers
    cfg.compile = False
    enable_fast_math(cfg); set_seed(0)

    wm = LeWorldModel(cfg).cuda(); wm.load_state_dict(blob["model"]); wm.eval()
    for p in wm.parameters(): p.requires_grad_(False)

    dec = Decoder(cfg.latent_dim, cfg.image_size).cuda()
    bc = BCPolicy(cfg.latent_dim, cfg.seq_len).cuda()
    nd = sum(p.numel() for p in dec.parameters()); nb = sum(p.numel() for p in bc.parameters())
    print(f"decoder {nd/1e6:.2f}M | bc {nb/1e6:.2f}M | world model frozen "
          f"{sum(p.numel() for p in wm.parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(list(dec.parameters()) + list(bc.parameters()),
                            lr=a.lr, weight_decay=0.05, fused=True)
    ds = build_soku_dataset(cfg, [str(a.corpus / "train")], shuffle_buffer=4096, seed=0)
    loader = make_loader(ds, cfg); it = iter(loader)
    a.out.mkdir(parents=True, exist_ok=True)
    hist, t0 = [], time.time()

    for step in range(a.steps):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        obs = batch["obs"].cuda(non_blocking=True)              # [B,T,3,S,S] uint8
        act = batch["actions"].cuda(non_blocking=True).float()  # [B,T,ticks,20]
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 1000) * (0.5 * (1 + math.cos(
                math.pi * min(1.0, step / a.steps))))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = wm.encode(obs)                                   # [B,T,latent]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tgt = obs[:, -1].float() / 255.0
            rec = dec(z[:, -1].float())
            l_dec = F.l1_loss(rec, tgt)
            # any press during the chunk counts; predict the action at the last step
            y = (act[:, -1].amax(dim=1) > 0.5).float()
            logits = bc(z.float())
            l_bc = F.binary_cross_entropy_with_logits(logits, y)
            loss = l_dec + 0.5 * l_bc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(dec.parameters()) + list(bc.parameters()), 1.0)
        opt.step()

        if step % 200 == 0 or step == a.steps - 1:
            with torch.no_grad():
                acc = (((logits > 0).float() == y).float().mean().item())
                base = max(y.mean().item(), 1 - y.mean().item())
            el = time.time() - t0
            rec_ = {"step": step, "l_dec": float(l_dec), "l_bc": float(l_bc),
                    "bc_acc": acc, "bc_base": base, "elapsed_h": el / 3600,
                    "eta_h": (el / max(step + 1, 1)) * (a.steps - step) / 3600}
            hist.append(rec_); a.log.write_text(json.dumps(hist, indent=2))
            print(f"step {step:6d} | dec L1 {l_dec:.4f} | bc BCE {l_bc:.4f} "
                  f"acc {acc:.3f} (base {base:.3f}) | {rec_['eta_h']:.1f}h left", flush=True)
        if step and step % 5000 == 0:
            torch.save({"decoder": dec.state_dict(), "bc": bc.state_dict(),
                        "cfg": cfg, "step": step}, a.out / "heads.pt")
    torch.save({"decoder": dec.state_dict(), "bc": bc.state_dict(), "cfg": cfg,
                "step": a.steps}, a.out / "heads.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {a.out}/heads.pt")

if __name__ == "__main__": main()
