"""How much of each corpus is actually an interaction?

If a corpus is mostly two characters walking around, next-state is predictable
from state alone and the buttons earn almost no loss reduction -- the pooled
training objective is dominated by easy frames. That would explain a model with
excellent h1 skill and no action sensitivity.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/anon/SokuBot")
from sokubot.data.state import CH

for name, cache in (("human", sys.argv[1]), ("self-play", sys.argv[2])):
    d = np.load(cache)
    S = d["S"]; n = len(S)
    both = S.reshape(n * 2, S.shape[-1])          # both players, pooled
    def rate(ch, thr=0.5):
        return float((np.abs(both[:, CH[ch]]) > thr).mean())
    hp = S[:, :, CH["hp"]]
    dhp = np.diff(hp, axis=0)
    print("  %-10s n=%d" % (name, n))
    print("     guarding %.4f  wrongblock %.4f  knockdown %.4f  crushed %.4f"
          % (rate("guarding", 0.5), rate("wrongblock", 0.5),
             rate("knockdown", 0.5), rate("crushed", 0.5)))
    print("     hitstop>0 %.4f  untech>0 %.4f  hitboxes>0 %.4f  hurtboxes>0 %.4f"
          % (float((both[:, CH["hitstop"]] > 0.01).mean()),
             float((both[:, CH["untech"]] > 0.01).mean()),
             float((both[:, CH["hitboxes"]] > 0.01).mean()),
             float((both[:, CH["hurtboxes"]] > 0.01).mean()))) 
    print("     hp-drop steps %.4f   proj_n mean %.3f   |dx| median %.4f"
          % (float((dhp < -1e-6).any(-1).mean()),
             float(both[:, CH["proj_n"]].mean()),
             float(np.median(np.abs(S[:, 0, CH["dx"]])))))
    frac = float((np.abs(both[:, CH["hitstop"]]) > 0.01).mean()
                 + rate("guarding", 0.5) + float((both[:, CH["untech"]] > 0.01).mean()))
    print("     crude 'in contact' fraction %.4f" % frac)
