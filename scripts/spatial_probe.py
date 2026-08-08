"""Does the encoder represent where the characters are, left to right?

    python -m scripts.spatial_probe --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --corpus ~/corpus --captures 6

THE QUESTION
------------
`scripts/block_effect.py` established that guarding is worth 27% of incoming
damage in real play and that the world model has the sign inverted, because
swapping LEFT and RIGHT moves the predicted latent by 4.6% of a standard
deviation. Blocking is holding *away from the opponent*, so the model cannot
learn it without knowing which side the opponent is on.

That leaves two very different diagnoses:

  **the encoder dropped it**  horizontal arrangement is not in the latent, so no
                              predictor or probe could ever recover it. Then the
                              representation is the problem and grounding it with
                              supervision -- or a different architecture -- is
                              mandatory rather than optional.

  **the predictor ignores it** arrangement is in the latent and the prediction
                              objective simply never had a reason to use it, since
                              a guard pose and a hit pose are similar pixels. Then
                              the world model is salvageable and the fix is the
                              objective.

This decides between them, and it needs no labels at all.

THE TEST
--------
Mirror the **play area only**, horizontally, leaving the HUD untouched, and ask a
linear probe to tell mirrored frames from original ones. Nothing else about the
frame changes: same characters, same health bars in the same places, same stage.
The only thing that moves is who is on the left.

Two controls make the number readable:

  `full frame`   mirror everything including the HUD. The health bars swap ends,
                 so this must be near-trivially detectable. If *this* fails the
                 measurement is broken, not the model.
  `identity`     no mirroring at all, labels assigned at random. Must sit at
                 0.5. This catches a probe that is fitting frame identity rather
                 than content, which is a live risk when both members of a pair
                 come from the same moment.

Held out **by capture**, because the two members of a mirrored pair are otherwise
identical and a random split would put one in train and its twin in test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sokubot.model.loading import load_world_model
from scripts.horizon_ablation import capture_paths, decode_hud, encode_all

# Display-orientation rows. Health bars and names live at 34-74, spirit and cards
# at 428-470 (see data/hud.py). The play area is what is left in between, and
# mirroring only that leaves every HUD cue exactly where it was.
DISP_PLAY_Y = (80, 420)
FRAME = 480


def mirror(frames: np.ndarray, mode: str) -> np.ndarray:
    """frames are stored vertically flipped; horizontal mirroring is unaffected."""
    out = frames.copy()
    if mode == "identity":
        return out
    if mode == "full":
        return out[:, :, ::-1]
    if mode == "play":
        lo, hi = FRAME - DISP_PLAY_Y[1], FRAME - DISP_PLAY_Y[0]   # to stored rows
        out[:, lo:hi] = out[:, lo:hi, ::-1]
        return out
    raise ValueError(mode)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score), np.float64)
    rank[order] = np.arange(1, len(score) + 1)
    npos, nneg = int(label.sum()), int(len(label) - label.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((rank[label > 0].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--captures", type=int, default=6)
    ap.add_argument("--per-capture", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("spatial_probe.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    wm, cfg, _ = load_world_model(a.wm, a.device)
    man = a.corpus / "val" / "manifest.jsonl"
    rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(rows)

    modes = ("play", "full", "identity")
    Z = {m: [] for m in modes}
    grp = []
    kept = 0
    for r in rows:
        if kept >= a.captures:
            break
        try:
            hud = decode_hud(str(capture_paths(r, man)[0]), 0)
        except Exception as exc:
            print(f"  skip {r.get('replay_id')}: {type(exc).__name__}")
            continue
        # Spread the sample across the whole match rather than taking a
        # contiguous block, so one stage or one round cannot dominate.
        sel = np.linspace(0, len(hud) - 1, a.per_capture).astype(int)
        base = np.ascontiguousarray(hud[sel])
        del hud
        for m in modes:
            img = np.ascontiguousarray(mirror(base, m))
            t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
            t = F.interpolate(t, size=(cfg.image_size, cfg.image_size),
                              mode="bilinear", antialias=True, align_corners=False)
            small = t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
            Z[m].append(encode_all(wm, small, a.device))
        grp.append(np.full(len(sel), kept, dtype=np.int32))
        kept += 1
        print(f"  encoded {kept}/{a.captures}", flush=True)
        del base

    if kept < 3:
        raise SystemExit(f"only {kept} captures usable; need at least 3 to hold out")
    g = np.concatenate(grp)
    orig = np.concatenate(Z["identity"])
    print(f"\n{len(orig)} frames from {kept} captures\n")

    res = {"wm": str(a.wm), "n": int(len(orig)), "captures": int(kept), "auc": {}}
    reps = np.unique(g)
    te_rep = set(reps[: max(1, len(reps) // 3)].tolist())
    te = np.array([i for i, x in enumerate(g) if x in te_rep])
    tr = np.array([i for i, x in enumerate(g) if x not in te_rep])

    print("  what is mirrored      held-out AUC   reading")
    for m in modes:
        alt = np.concatenate(Z[m])
        if m == "identity":
            # Same frames on both sides; the label is a coin flip by construction.
            lab_src = rng.random(len(orig)) < 0.5
            X = np.where(lab_src[:, None], alt, orig).astype(np.float32)
            y = lab_src.astype(np.float32)
        else:
            X = np.concatenate([orig, alt]).astype(np.float32)
            y = np.concatenate([np.zeros(len(orig)), np.ones(len(alt))]).astype(np.float32)
        gg = g if m == "identity" else np.concatenate([g, g])
        tr_m = np.array([i for i, x in enumerate(gg) if x not in te_rep])
        te_m = np.array([i for i, x in enumerate(gg) if x in te_rep])

        mu, sd = X[tr_m].mean(0), X[tr_m].std(0) + 1e-6
        Xs = np.concatenate([(X - mu) / sd, np.ones((len(X), 1), np.float32)], axis=1)
        A_ = Xs[tr_m].T @ Xs[tr_m] + a.alpha * np.eye(Xs.shape[1], dtype=np.float32)
        w = np.linalg.solve(A_, Xs[tr_m].T @ y[tr_m])
        v = auc(Xs[te_m] @ w, y[te_m])
        res["auc"][m] = v
        note = {"play": "characters only -- the question",
                "full": "control: must be easy",
                "identity": "control: must be 0.5"}[m]
        print(f"  {m:<20} {v:12.4f}   {note}")

    play, full, ident = res["auc"]["play"], res["auc"]["full"], res["auc"]["identity"]
    print("\n" + "=" * 70)
    if not (0.42 <= ident <= 0.58):
        print(f"CONTROL FAILED: identity reads {ident:.3f}, not 0.5. The probe is "
              f"fitting\nsomething other than content; fix that before reading "
              f"the other rows.")
    elif full < 0.8:
        print(f"CONTROL FAILED: mirroring the whole frame, health bars and all, "
              f"reads only\n{full:.3f}. The measurement is broken, not the model.")
    elif play > 0.9:
        print(f"The encoder DOES represent horizontal arrangement ({play:.3f} with "
              f"the HUD held\nfixed). Relative position is in the latent, so "
              f"blocking is a *predictor*\nproblem -- the objective never had a "
              f"reason to use it. The world model is\nsalvageable and the fix is "
              f"the objective, not the architecture.")
    elif play < 0.65:
        print(f"The encoder does NOT represent horizontal arrangement ({play:.3f}) "
              f"even though\nmirroring the whole frame is trivial ({full:.3f}). "
              f"Position is dropped by the\nencoder, so no probe, predictor or "
              f"reward can recover it. Grounding the\nlatent with supervision is "
              f"mandatory, and if that fails the architecture is\nimplicated.")
    else:
        print(f"Partial: {play:.3f} against a {full:.3f} ceiling. The information "
              f"is present but\nweakly linearly available -- consistent with an "
              f"encoder that keeps position\nonly as far as predicting the next "
              f"frame required.")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
