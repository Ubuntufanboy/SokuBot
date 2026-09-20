"""Does the encoder's positional error track how often it saw that character?

The training corpus has all 20 characters but is badly skewed: Sakuya appears
in 24 of 100 replays and Cirno in 3. The agent plays CIRNO. If positional
accuracy is a function of how much of a character the encoder ever saw, then
the live collapse (`findings/04`: dx +0.839 held-out -> -0.103 live) is a data
coverage problem, and none of the imaging, timing or architecture hypotheses
need to be true.

This groups per-ROW error by the character standing in that row, so each frame
contributes one measurement per player rather than one per frame. Rows are in
the encoder's own left/right order and the labels are reordered to match, which
is the same permutation `train_encoder.build` applies.

The comparison to beat is that error is FLAT across characters -- that would say
coverage does not matter and something else is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sokubot.data.state import CH, STAGE_SPAN, read_state          # noqa: E402
from sokubot.live.visionstate import VisionState                   # noqa: E402
from scripts.train_encoder import SUPERVISED                       # noqa: E402

# Inlined rather than imported: `pipeline.repparse` lives in the
# SokuFrameExtractor repo, and a SokuBot script should not need it on the
# path to name twenty characters. Order is the game's roster index, which is
# what the sidecar and corpus_chars.json both use.
CHARACTER_NAMES = {
    0: "Reimu", 1: "Marisa", 2: "Sakuya", 3: "Alice", 4: "Patchouli",
    5: "Youmu", 6: "Remilia", 7: "Yuyuko", 8: "Yukari", 9: "Suika",
    10: "Reisen", 11: "Aya", 12: "Komachi", 13: "Iku", 14: "Tenshi",
    15: "Sanae", 16: "Cirno", 17: "Meiling", 18: "Utsuho", 19: "Suwako",
}


def _np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--chars", type=Path, required=True)
    ap.add_argument("--train-chars", type=Path, required=True,
                    help="the chars json the ENCODER was trained on; supplies "
                         "the appearance counts the error is compared against")
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--per-replay", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import cv2

    freq: defaultdict = defaultdict(int)
    for v in json.loads(a.train_chars.read_text()).values():
        freq[int(v["p1_char"])] += 1
        freq[int(v["p2_char"])] += 1

    chars = json.loads(a.chars.read_text())
    rng = np.random.default_rng(a.seed)
    dirs = []
    for w in sorted(a.corpus.glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    if not dirs:
        dirs = [d for d in sorted(a.corpus.iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    dirs = [d for d in dirs if (d / "video.mp4").exists()
            and (d / "state.csv.gz").exists() and d.name in chars]
    pick = [dirs[i] for i in rng.choice(len(dirs), min(a.replays, len(dirs)),
                                        replace=False)]

    vss = [VisionState.load(c, a.device) for c in a.ckpt]
    delta, size = vss[0].delta, vss[0].size
    print(f"{len(pick)} replays, delta {delta}, size {size}", flush=True)

    X, S, C = [], [], []
    for n, d in enumerate(pick):
        try:
            st, _p, _act, valid = read_state(d / "state.csv.gz")
        except (ValueError, OSError):
            continue
        hp = st[:, :, CH["hp"]]
        ok = valid & (hp > 0).all(1) & (hp <= 1.001).all(1)
        idx = np.flatnonzero(ok)
        idx = idx[idx >= delta]
        if len(idx) < a.per_replay:
            continue
        rows = np.sort(rng.choice(idx, a.per_replay, replace=False))
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        frames, keep = [], []
        for f in rows:
            pair = []
            for off in (delta, 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - off)
                got, fr = cap.read()
                if not got:
                    break
                fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
                pair.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(pair) != 2:
                continue
            frames.append(np.concatenate(pair, axis=2).transpose(2, 0, 1))
            keep.append(f)
        cap.release()
        if not frames:
            continue
        keep = np.array(keep)
        s_sel = st[keep]
        p1_left = (s_sel[:, 0, CH["dx"]] > 0)
        order = np.where(p1_left[:, None], np.array([0, 1]), np.array([1, 0]))
        take = np.take_along_axis(s_sel, order[:, :, None], axis=1)
        lab = chars[d.name]
        pair_ = np.array([lab["p1_char"], lab["p2_char"]], np.int64)
        C.append(np.where(p1_left[:, None], pair_[None, :], pair_[::-1][None, :]))
        X.append(np.stack(frames))
        S.append(take.astype(np.float32))
        print(f"  [{n}] {d.name}: {len(frames)}", flush=True)

    X = np.concatenate(X); S = np.concatenate(S); C = np.concatenate(C)
    print(f"\n{len(X)} frames, {C.size} character-rows\n", flush=True)

    sup = [CH[n] for n in SUPERVISED]
    Y = S[:, :, sup].reshape(len(S), -1)
    ix, ns = SUPERVISED.index("x"), len(SUPERVISED)

    for vs, cpath in zip(vss, a.ckpt):
        mu = np.asarray(_np(vs.mu), np.float32)
        sd = np.asarray(_np(vs.sd), np.float32)
        preds = []
        with torch.no_grad():
            for j in range(0, len(X), 128):
                xb = torch.from_numpy(X[j:j+128]).to(a.device).float().div(255.0)
                out = vs.net(xb, return_feat=True)
                o = out[0] if isinstance(out, tuple) else out
                preds.append(o[:, :vs.n_state].cpu().numpy())
        pred = np.concatenate(preds) * sd[None, :vs.n_state] + mu[None, :vs.n_state]

        pw = np.stack([pred[:, r * ns + ix] for r in (0, 1)], 1) * STAGE_SPAN
        tw = np.stack([Y[:, r * ns + ix] for r in (0, 1)], 1) * STAGE_SPAN
        err = np.abs(pw - tw)                      # [N, 2] world units

        print(f"=== {cpath.name} " + "=" * 46)
        print(f"{'character':<12} {'train reps':>10} {'rows':>7} {'x err (world u)':>16}")
        fs, es = [], []
        for cid in sorted(set(C.flatten().tolist())):
            m = C == cid
            if m.sum() < 30:
                continue
            e = float(err[m].mean())
            f = freq.get(cid, 0)
            fs.append(f); es.append(e)
            print(f"  {CHARACTER_NAMES[cid]:<10} {f:10d} {int(m.sum()):7d} "
                  f"{e:16.1f}")
        if len(fs) > 2:
            r = float(np.corrcoef(fs, es)[0, 1])
            rs = float(np.corrcoef(np.argsort(np.argsort(fs)),
                                   np.argsort(np.argsort(es)))[0, 1])
            print(f"\n  corr(train appearances, x error)  pearson {r:+.3f}  "
                  f"spearman {rs:+.3f}")
            fs_a, es_a = np.array(fs), np.array(es)
            rare, common = fs_a <= np.median(fs_a), fs_a > np.median(fs_a)
            print(f"  rare half {es_a[rare].mean():.1f} u vs common half "
                  f"{es_a[common].mean():.1f} u  "
                  f"(x{es_a[rare].mean() / max(es_a[common].mean(), 1e-9):.2f})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
