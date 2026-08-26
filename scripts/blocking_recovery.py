"""Does the model KNOW that holding toward means no block?

The toward-side gap (model ~0.10, game 0.014) may be my probe's fault rather
than the model's: forcing "toward" across twelve steps against a state history
of a character walking backwards is a pair that cannot occur, and hedging on it
is reasonable. The empirical 0.014 comes from frames where the defender really
was holding toward, so its history agrees with its action.

So: no intervention at all. Select arrival frames by what the defender ACTUALLY
did, feed the real actions, and compare the model's prediction to what happened.
If it predicts near 0.014 on genuine toward frames, the model is right and the
probe is what is wrong.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/anon/SokuBot")
from sokubot.data.state import CH
from sokubot.model.state_dynamics import load_sim

NEAR = 250.0 / 1200.0
d = np.load(sys.argv[1])
S, A, E, P = d["S"], d["A"], d["E"], d["P"]

hb = S[:, 1, CH["hitboxes"]] > 0.01
hb_n = np.zeros(len(S), bool); hb_n[:-1] = hb[1:]
arrives = (~hb) & hb_n
gnow = S[:, 0, CH["guarding"]] > 0.5

for path in sys.argv[2:]:
    model, meta = load_sim(path, "cpu")
    H = int(meta["history"])
    ok = np.ones(len(S), bool); ok[-(H + 2):] = False
    for k in range(1, H + 2):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    t = np.arange(len(S)) + H - 1
    ok[t >= len(S) - 1] = False
    ti = np.clip(t, 0, len(S) - 1)
    ok &= np.abs(S[ti, 0, CH["dx"]]) < NEAR
    ok &= arrives[ti] & ~gnow[ti]

    dx = S[ti, 0, CH["dx"]]
    left = A[ti][:, :, 2].mean(1) > 0.5
    right = A[ti][:, :, 3].mean(1) > 0.5
    one = left ^ right
    away = np.where(dx > 0, left, right)
    print("=== %s ===" % Path(path).parent.name)
    for lab, sel in (("away", ok & one & away), ("toward", ok & one & ~away)):
        idx = np.flatnonzero(sel)
        if len(idx) < 40:
            print("  %-7s too few" % lab); continue
        w = idx[:, None] + np.arange(H)[None, :]
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx), 1500):
                sl = slice(i, i + 1500)
                ns, _ = model(torch.as_tensor(S[w[sl]]).float(),
                              torch.as_tensor(P[w[sl]]).float(),
                              torch.as_tensor(A[w[sl]]).float())
                g = torch.sigmoid(ns[:, -1, 0, CH["guarding"]])
                # NOT `w`: that is the window-index array, and shadowing it
                # made the second batch index a probability tensor instead.
                wb = torch.sigmoid(ns[:, -1, 0, CH["wrongblock"]])
                # P(blocked) = P(clean guard) + P(wrong-height block). The two
                # are disjoint outcomes in the corpus, and mixing wrongblock
                # into the TRUTH while scoring the model on `guarding` alone is
                # what made a 76%-accurate model look like a 47% one.
                preds.append(torch.stack([g, wb, g + wb], 1).numpy())
        pred = np.concatenate(preds)
        gt_g = (S[idx + H, 0, CH["guarding"]] > 0.5)
        gt_w = (S[idx + H, 0, CH["wrongblock"]] > 0.5)
        print("  %-7s n=%5d | guard model %.4f game %.4f | blocked model %.4f game %.4f"
              % (lab, len(idx), pred[:, 0].mean(), gt_g.mean(),
                 pred[:, 2].mean(), (gt_g | gt_w).mean()))
