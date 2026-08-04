"""End-to-end check before RL: does the pipeline produce a usable reward signal?

    python -m scripts.integration_check --corpus /root/corpus --ckpt /root/ckpt/best.pt

Three questions, in order of how badly a "no" would hurt:

1. Does HUD extraction hold up across many replays, not the handful used while
   developing it? Aggregate invariants: KOs per match, damage per match, spirit
   reaching full, the red floor sitting at zero.

2. Do the HUD labels line up with the frames the model sees? The labels are read
   at 60 fps and the model runs at 15 Hz, so an indexing slip here would poison
   every reward silently.

3. **Can a linear probe recover game state from the world model's latent?**
   This is the one that decides whether milestone 4 is possible at all. GRPO
   plans inside imagination, where there are no pixels to read a HUD from -- only
   latents. If health is not linearly decodable from a latent, there is no reward
   signal in imagined rollouts and the whole approach needs rethinking.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from sokubot.config import Config
from sokubot.data.hud import read_trace, damage_events, RED_FLOOR
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import ridge_probe

def decode(path, skip=1):
    vf = [] if skip == 1 else ["-vf", f"select='not(mod(n\\,{skip}))'", "-vsync", "0"]
    p = subprocess.run(["ffmpeg","-v","error","-i",path,*vf,"-f","rawvideo","-pix_fmt","rgb24","-"],
                       capture_output=True)
    n = len(p.stdout)//(480*480*3)
    return np.frombuffer(p.stdout[:n*480*480*3],dtype=np.uint8).reshape(n,480,480,3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--ckpt", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--probe-frames", type=int, default=6000)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (a.corpus/"train"/"manifest.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0); rng.shuffle(rows)
    rows = rows[:a.replays]

    print("=" * 72); print("1. HUD EXTRACTION ACROSS REPLAYS"); print("=" * 72)
    kos, dmg1, dmg2, sp_full, red_floor, rises, fails = [], [], [], [], [], [], 0
    lat_frames, lat_labels = [], []
    cfg = Config.soku()
    for k, r in enumerate(rows):
        try:
            fr = decode(r["video"])
            t = read_trace(fr, smooth=3)
        except Exception as e:
            fails += 1; continue
        kos.append(int(np.diff(t.healing.astype(int)).clip(0).sum()))
        dmg1.append(damage_events(t,1).sum()); dmg2.append(damage_events(t,2).sum())
        sp_full.append(float((t.spirit1 > 0.98).mean()))
        red_floor.append(float(np.median(t.combo1[t.hp1 > 0.1])) if (t.hp1>0.1).any() else 0.0)
        up = np.diff(t.hp1); up = up[(~t.healing[1:]) & (up > 0)]
        rises.append(float(up.sum()))
        if len(lat_frames)*4 < a.probe_frames:          # keep every 4th for the probe
            sel = np.arange(0, len(fr)-1, 4)[:400]
            lat_frames.append(fr[sel])
            lat_labels.append(np.stack([t.hp1[sel], t.hp2[sel], t.spirit1[sel], t.spirit2[sel]],1))
        if (k+1) % 10 == 0: print(f"   {k+1}/{len(rows)} replays", flush=True)

    def st(v): return f"{np.mean(v):.3f} +/- {np.std(v):.3f}  [{np.min(v):.3f}, {np.max(v):.3f}]"
    print(f"   replays ok        : {len(kos)}/{len(rows)}  (failed {fails})")
    print(f"   KOs per replay    : {st(kos)}          expect 2-4")
    print(f"   damage P1 (bars)  : {st(dmg1)}")
    print(f"   damage P2 (bars)  : {st(dmg2)}")
    print(f"   spirit at full    : {st(sp_full)}       expect ~0.5-0.9")
    print(f"   red floor (median): {st(red_floor)}     expect ~0.000")
    print(f"   health rises/match: {st(rises)}         regen, expect small >0")

    print(); print("=" * 72); print("2. LATENT PROBE  (the milestone-4 gate)"); print("=" * 72)
    blob = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    pcfg = blob["cfg"]; pcfg.device = "cuda"
    m = LeWorldModel(pcfg).cuda(); m.load_state_dict(blob["model"]); m.eval()
    F = np.concatenate(lat_frames); L = np.concatenate(lat_labels)
    print(f"   {len(F)} frames from {len(lat_frames)} replays")
    zs = []
    with torch.no_grad():
        for i in range(0, len(F), 128):
            # HUD reading needs native 480x480; the encoder was trained on
            # 224x224 (ffmpeg bilinear). Resize here or the patch grid mismatches.
            x = torch.from_numpy(np.ascontiguousarray(F[i:i+128])).permute(0,3,1,2).float().cuda()
            x = torch.nn.functional.interpolate(x, size=(pcfg.image_size, pcfg.image_size),
                                                mode="bilinear", align_corners=False)
            zs.append(m.encode((x/255.0).unsqueeze(1))[:,0].cpu().numpy())
    Z = np.concatenate(zs)
    res = ridge_probe(Z, L, names=["hp1","hp2","spirit1","spirit2"])
    print(f"   {res}")
    ok = res.r2["hp1"] > 0.5 and res.r2["hp2"] > 0.5
    print()
    print("=" * 72)
    print(f"   VERDICT: health {'IS' if ok else 'IS NOT'} linearly decodable from the latent")
    print(f"   -> rewards in imagined rollouts are {'viable' if ok else 'NOT viable as designed'}")
    print("=" * 72)
    return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
