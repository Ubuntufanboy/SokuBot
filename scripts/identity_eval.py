"""Does the character head hold identity through a crossup? Pooled accuracy won't say.

    python -m scripts.identity_eval --ckpt ~/rl/enc_char_s0.pt \
        --corpus ~/corpus --chars ~/corpus_chars.json --replays 20

WHY NOT JUST READ THE VALIDATION NUMBER
---------------------------------------
`train_encoder` reports identity-decision accuracy pooled over frames. That
number is dominated by the ~90% of frames where the two characters are far
apart and telling them apart is easy. The failure this head exists to fix is
the opposite case: the characters CROSS about ten times a minute, and the old
nearest-neighbour tracker was not merely wrong there, it was wrong in an
ABSORBING way -- one bad association at a crossup locked the policy onto the
opponent's row until the next crossing, measured at wrong runs of up to 85
decisions (7.1 s).

So a head that is 95% right overall but wrong for four seconds after every
crossing is worse for play than the number suggests, and a pooled average
cannot tell the two apart. This reports:

  * accuracy stratified by SEPARATION, because that is the variable that makes
    it hard, and
  * the distribution of WRONG-RUN LENGTHS, because duration is what actually
    hurts -- being wrong on scattered single frames costs a decision each,
    being wrong for 85 in a row costs the round.

Ground truth is `p1_left` from the state sidecar, used as an INSTRUMENT only.
The head consumes pixels; nothing here reaches a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.data.state import CH, STAGE_SPAN, read_state
from sokubot.live.visionstate import VisionState


def runs_of_true(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs -- the absorbing-error statistic."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--chars", type=Path, required=True)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--per-replay", type=int, default=300)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N captures, so this can be pointed at "
                         "replays the encoder was not trained on")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import cv2
    vs = VisionState.load(a.ckpt, device=a.device)
    if not vs.n_char:
        raise SystemExit(f"{a.ckpt} has no character head (n_char=0); this "
                         "script has nothing to measure.")
    chars = json.loads(a.chars.read_text())
    dirs = []
    for w in sorted(Path(a.corpus).glob("w*")):
        dirs += [d for d in sorted(w.iterdir()) if d.is_dir()]
    dirs = dirs[a.skip:a.skip + a.replays]

    correct, sep, wrong_runs, n_mirror, n_frames = [], [], [], 0, 0
    for d in dirs:
        lab = chars.get(d.name)
        vid, sc = d / "video.mp4", d / "state.csv.gz"
        if lab is None or not (vid.exists() and sc.exists()):
            continue
        if lab["p1_char"] == lab["p2_char"]:
            n_mirror += 1
            continue                      # no answer from character alone
        st, _pr, _act, valid = read_state(sc)
        idx = np.flatnonzero(valid.astype(bool))
        idx = idx[idx >= vs.delta]
        if len(idx) < 32:
            continue
        pick = idx[np.linspace(0, len(idx) - 1, min(a.per_replay, len(idx))).astype(int)]
        cap = cv2.VideoCapture(str(vid))
        ok_f, sep_f = [], []
        for f in pick:
            pair = []
            for off in (vs.delta, 0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - off)
                got, fr = cap.read()
                if not got:
                    break
                pair.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if len(pair) != 2:
                continue
            frame = np.concatenate(pair, axis=2)
            # p1_left from the sidecar: dx is x_opponent - x_me for player 0.
            p1_left = bool(st[f, 0, CH["dx"]] > 0)
            # The agent plays P1 here purely so the query is well defined; the
            # head is symmetric and the P2 case is the mirror of this one.
            vs.my_char, vs.mirror_match, vs.i_am_left = lab["p1_char"], False, None
            vs.read(frame)
            ok_f.append(vs.i_am_left == p1_left)
            sep_f.append(abs(float(st[f, 0, CH["dx"]])) * STAGE_SPAN)
        cap.release()
        if not ok_f:
            continue
        ok_arr = np.array(ok_f, bool)
        correct.append(ok_arr)
        sep.append(np.array(sep_f))
        wrong_runs += runs_of_true(~ok_arr)
        n_frames += len(ok_f)

    if not correct:
        raise SystemExit("no usable captures")
    ok = np.concatenate(correct)
    sp = np.concatenate(sep)
    print(f"captures {len(correct)} (+{n_mirror} mirror, excluded), "
          f"frames {n_frames}")
    print(f"\npooled identity-decision accuracy: {ok.mean():.4f}")
    print("  references: live nearest-neighbour 0.51, perfect-input 0.64, "
          "chance 0.50")

    print("\nby separation -- the variable that makes it hard:")
    edges = [0, 100, 200, 300, 450, 1e9]
    names = ["0-100", "100-200", "200-300", "300-450", "450+"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (sp >= lo) & (sp < hi)
        if m.sum():
            print(f"  {nm:>8s} u  n={int(m.sum()):6d}  acc {ok[m].mean():.4f}")
    print("  (a character is ~50 u wide; under that the two sprites overlap)")

    print("\nwrong-run lengths -- absorbing error is what cost 7.1 s a time:")
    if wrong_runs:
        r = np.array(wrong_runs)
        print(f"  runs {len(r)}  median {np.median(r):.0f}  p90 "
              f"{np.percentile(r, 90):.0f}  max {r.max()}  "
              f"(frames, sampled at {a.per_replay}/replay)")
    else:
        print("  none -- the head was never wrong on a sampled frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
