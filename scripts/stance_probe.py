"""Is the defender's stick already readable from the state history?

If it is, the button input is REDUNDANT rather than confounded, and no amount
of unconfounded data will make a next-state model consult it: the model can
read `guarding` off the animation it can already see. That is a different
disease from the one the proxy modelled -- there `stance` was deliberately
unobserved, so the button was the only route to the answer.

None of the 33 channels is an input channel, so this is not circular. But `vx`
is among them, and walking backward has a velocity signature.

Reported per corpus, with three feature sets, because WHICH part of the state
gives it away is the actionable half:
  full     the whole history the simulator sees
  last     the final step only, to separate "history" from "current pose"
  no-vel   velocity and acceleration removed, to test whether vx is the tell
"""
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/home/anon/SokuBot")
from sokubot.data.state import CH

H = 12
DROP = ("vx", "vy", "ax", "ay", "dx", "dy")


def windows(cache, near=250.0):
    d = np.load(cache)
    S, A, E = d["S"], d["A"], d["E"]
    ok = np.ones(len(S), bool)
    ok[-(H + 2):] = False
    for k in range(1, H + 2):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    ok &= np.abs(S[:, 0, CH["dx"]]) < (near / 1200.0)
    idx = np.flatnonzero(ok)

    left = A[idx][:, :, 2].mean(1) > 0.5
    right = A[idx][:, :, 3].mean(1) > 0.5
    one = left ^ right                      # exactly one horizontal direction
    idx, left = idx[one], left[one]
    dx = S[idx, 0, CH["dx"]]
    y = np.where(dx > 0, left, ~left).astype(np.float32)   # 1 = away = blocking stance
    w = idx[:, None] + np.arange(H)[None, :]
    return S[w].reshape(len(idx), -1), y, S[w]


def auc(score, y):
    o = np.argsort(score)
    r = np.empty(len(score)); r[o] = np.arange(len(score))
    p, n = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - p * (p - 1) / 2) / (p * n)) if p and n else float("nan")


def probe(X, y, tag, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(y); perm = torch.randperm(n, generator=g).numpy()
    X, y = X[perm], y[perm]
    cut = int(n * 0.8)
    mu, sd = X[:cut].mean(0), X[:cut].std(0) + 1e-6
    Xtr = torch.as_tensor((X[:cut] - mu) / sd).float()
    Xte = torch.as_tensor((X[cut:] - mu) / sd).float()
    ytr = torch.as_tensor(y[:cut]); yte = y[cut:]
    m = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(300):
        for i in range(0, cut, 4096):
            opt.zero_grad()
            lossf(m(Xtr[i:i+4096]).squeeze(1), ytr[i:i+4096]).backward()
            opt.step()
    with torch.no_grad():
        s = m(Xte).squeeze(1).numpy()
    print("    %-8s AUC %.4f  acc %.4f  (n=%d, away rate %.3f)"
          % (tag, auc(s, yte), float(((s > 0) == (yte > 0.5)).mean()), n, y.mean()))


for name, cache in (("human", sys.argv[1]), ("self-play", sys.argv[2])):
    Xf, y, Sw = windows(cache)
    print("  %s corpus" % name)
    probe(Xf, y, "full")
    probe(Sw[:, -1].reshape(len(y), -1), y, "last")
    keep = [i for c, i in CH.items() if c not in DROP]
    probe(Sw[:, :, :, keep].reshape(len(y), -1), y, "no-vel")
