"""A start-state bank that carries HUD state alongside the latent.

    python -m scripts.build_hud_bank --ckpt ~/sokubot-art/wm_cf_bnfix.pt \
        --corpus ~/corpus --replays 150 --out ~/bank_hud.npz

Same as `scripts/build_bank.py` but with a third array: `hud`, the eight
`data/hud.py` channels at decision-step resolution, read from the **native 480 px
frames** rather than from the 224 px ones the encoder sees.

WHY BOTH ARMS SHARE THIS ONE BANK
----------------------------------
The predictor fine-tune runs as an A/B -- plain unrolled loss against the same
loss over a HUD-augmented state -- and the whole point is to attribute any change
in horizon to the augmentation rather than to the data. So both arms read this
file and arm A simply ignores the `hud` column. Building two banks would leave
"different replays" as a live explanation for any difference between them.

WHY THE HUD IS TRUSTWORTHY ENOUGH TO PUT IN THE STATE
------------------------------------------------------
Because it was checked against a person rather than assumed. 48 blind-annotated
frames, stratified with low health deliberately oversampled, put `hud.py` at
MAE 0.012 for health and 0.025 for spirit -- and at low health specifically, MAE
0.012/0.013, where the *probe* sits at 0.25 noise. The reader is not the weak
link and never was; see `scripts/score_hud_annotation.py`.

`cards1`/`cards2` are carried but were **not** part of that validation, so they
are in the state and must stay out of the reward until they are checked the same
way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.state import STATE_CHANNELS, read_state
from sokubot.model.augmented import HUD_CHANNELS
from sokubot.model.loading import load_world_model
from sokubot.model.world_model import LeWorldModel
from scripts.eval_ckpt import predictor_skill
from scripts.horizon_ablation import TARGETS, capture_paths, encode_all, load_replay
from scripts.train_grpo import encoder_fingerprint, model_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None,
                    help="label cache shared with horizon_ablation; reusing it "
                         "skips the native-resolution decode, which dominates")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cache = a.cache or (a.out.parent / "hud_label_cache")

    # The channel order must match what the augmented predictor expects, or the
    # model learns to predict health in the spirit slot and nothing complains.
    if tuple(TARGETS) != HUD_CHANNELS:
        raise SystemExit(
            f"horizon_ablation.TARGETS {TARGETS} and augmented.HUD_CHANNELS "
            f"{HUD_CHANNELS} disagree. These index the same array positionally; "
            f"letting them drift would silently permute the state.")

    # Through load_world_model, not LeWorldModel(cfg) + strict load. The
    # config in a checkpoint is a PICKLED dataclass, so any field added since
    # it was written falls through to today's class default -- and the fields
    # that gate a head thereby invent heads the weights have no entries for.
    # Building by hand here died with "Missing key(s): hud_head.weight,
    # idm_head..." on a checkpoint that predates both, which is the failure
    # docs/BUGS.md 8 describes and load_world_model exists to prevent.
    wm, cfg, notes = load_world_model(a.ckpt, device=a.device)
    for n in notes:
        print(f"  cfg reconciled: {n}", flush=True)
    wm.eval()
    fp = model_fingerprint(wm)
    print(f"{a.ckpt}  fingerprint {fp}", flush=True)

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    np.random.default_rng(a.seed).shuffle(rows)

    zs, acts, huds, sts, svs, ep, kept = [], [], [], [], [], [], 0
    n_with_state = 0
    for r in rows:
        if kept >= a.replays:
            break
        try:
            video, inputs = capture_paths(r, manifest)
            obs, chunks, labels = load_replay(video, inputs, cfg, a.max_frames,
                                              cache)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        D = min(len(obs), len(chunks), len(labels))
        if D < 64:
            continue
        # Game state, if pipeline/align_sidecar.py has written it beside the
        # video. Decision step d is source frame d*skip, the same mapping the
        # loader uses -- state.csv is indexed by source frame, so this is a
        # gather rather than a slice, and getting it wrong would shift every
        # label by a factor of frame_skip while still producing a full array.
        st = sv = None
        state_path = next((c for c in (video.parent / "state.csv.gz",
                                       video.parent / "state.csv")
                           if c.exists()), None)
        if state_path is not None:
            try:
                arr, valid = read_state(state_path)
                rows_i = np.arange(D) * cfg.frame_skip
                if rows_i[-1] < len(arr):
                    st, sv = arr[rows_i], valid[rows_i]
                    n_with_state += 1
            except (ValueError, OSError) as exc:
                print(f"  state skipped for {r.get('replay_id')}: {exc}",
                      flush=True)

        z = encode_all(wm, obs[:D], a.device)
        zs.append(z.astype(np.float16))
        acts.append(chunks[:D].astype(np.uint8))
        huds.append(labels[:D].astype(np.float16))
        # A replay without labels contributes zeros marked invalid rather than
        # being dropped: the HUD gyms still want it, and `state_valid` is what
        # keeps the mechanic gyms from selecting inside it.
        sts.append(st.astype(np.float16) if st is not None
                   else np.zeros((D, 2, len(STATE_CHANNELS)), np.float16))
        svs.append(sv if sv is not None else np.zeros(D, bool))
        ep.append(np.full(D, kept, dtype=np.int32))
        kept += 1
        del obs, chunks, labels
        if kept % 10 == 0:
            # Peak RSS, because the failure mode here is an OOM kill and those
            # leave no traceback -- the log simply stops. A number that climbs is
            # visible; a process that vanishes is not.
            try:
                rss = int(open("/proc/self/status").read()
                          .split("VmHWM:")[1].split()[0]) / 1e6
                extra = f" | peak rss {rss:.1f} GB"
            except Exception:
                extra = ""
            print(f"   bank {kept}/{a.replays}{extra}", flush=True)

    if kept < 4:
        raise SystemExit(f"only {kept} replays usable")
    Z = np.concatenate(zs); A = np.concatenate(acts)
    Hd = np.concatenate(huds); E = np.concatenate(ep)
    St = np.concatenate(sts); Sv = np.concatenate(svs)
    del zs, acts, huds, sts, svs, ep

    # HUD is a fraction of a gauge, so anything outside [0,1] is a reader bug
    # rather than a rare state, and it would be learned as a target.
    lo, hi = float(np.nanmin(Hd)), float(np.nanmax(Hd))
    if not (-0.01 <= lo and hi <= 1.01):
        raise SystemExit(f"hud values span [{lo:.3f}, {hi:.3f}], outside [0,1]")
    if not np.isfinite(Hd).all():
        raise SystemExit("hud contains non-finite values")

    np.savez(a.out, z=Z, a=A, hud=Hd, ep=E, state=St,
             state_valid=Sv, fingerprint=fp,
             encoder_fingerprint=encoder_fingerprint(wm),
             hud_channels=np.array(HUD_CHANNELS))
    # A receipt: the bank is only meaningful paired with the weights that encoded
    # it, so record what that pairing actually scores.
    sk = predictor_skill(wm, torch.from_numpy(Z).to(a.device),
                         torch.from_numpy(A).to(a.device), E, cfg)
    print(f"\nbank: {len(Z)} latents from {kept} replays -> {a.out}")
    print(f"one-step skill of this (model, bank) pairing: {sk:+.4f}")
    print("hud channel means: " + "  ".join(
        f"{c} {float(Hd[:, i].mean()):.3f}" for i, c in enumerate(HUD_CHANNELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
