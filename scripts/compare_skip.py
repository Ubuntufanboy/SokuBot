"""Compare simulators at MATCHED REAL-TIME horizons, not matched step counts.

    python -m scripts.compare_skip --sims ~/rl/sim2/sim.pt ~/rl/fs1/best_h1.pt

WHY THIS EXISTS
---------------
`skill` is 1 - mse/identity_mse, and the identity baseline gets *stronger* as
the step shrinks: "nothing changes in 16.7 ms" is a far better prediction than
"nothing changes in 83 ms". So a frame_skip=1 model and a frame_skip=5 model
reporting the same skill at "h4" are not being asked the same question, and
comparing them directly would answer whether the step size helped with a number
that moves for a reason unrelated to the model.

Here every model is asked: given the same real history, where is the state
`ms` milliseconds from now? A skip-5 model takes 1 step to cover 83 ms and a
skip-1 model takes 5. Both are scored against the SAME identity baseline -- the
last observed state, held -- so the comparison is like for like.

The block gate is re-measured the same way: a fixed 500 ms of held input rather
than a fixed number of steps, because guarding takes time to appear and 8 steps
is 667 ms at skip 5 and 133 ms at skip 1.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

from sokubot.config import Config
from sokubot.data.state import CH, STATE_CHANNELS
from sokubot.model.state_dynamics import StateDynamics, rollout
from sokubot.model.state_head import CONTINUOUS
from scripts.train_state_dynamics import load_sequences, BUTTONS

FPS = 60.0

# Channel groups, because one pooled MSE over 28 incommensurable quantities is
# not a measurement -- it is whichever channel happens to have the largest
# units. Measured on the corpus, the raw pooled variance is 57.8%
# `action_frame` and 41.3% `untech`, leaving under 1% for position, velocity,
# health, spirit and every flag combined. Three runs were steered by that
# number before anyone checked what was in it.
GROUPS = {
    # What a rollout is FOR: where the fighters are and where they are going.
    "kinematic": ("dx", "dy", "x", "y", "vx", "vy", "ay"),
    # What the reward reads.
    "resource": ("hp", "spirit", "combo_damage", "combo_hits"),
    # Frame counters. Large-magnitude, mostly-constant, and the reason the
    # pooled number was meaningless.
    "counter": ("action_frame", "untech", "hitstop", "spirit_delay",
                "timestop"),
}


def load_sim(path: Path, device: str):
    b = torch.load(path, map_location=device, weights_only=False)
    ticks = int(b.get("ticks", b["cfg"].frame_skip))
    m = StateDynamics(b["cfg"], b.get("slots", 8), b.get("width", 384),
                      b.get("depth", 6), history=b.get("history", 8),
                      ticks=ticks).to(device)
    m.load_state_dict(b["model"]); m.eval()
    return m, ticks, b.get("step", -1)


def _chan_scale(S, cont_idx):
    """Per-channel std of the VALUE, for standardising the eval.

    The loss standardises and the eval did not, which is the whole reason they
    disagreed: the loss could fall while `skill` collapsed, because `skill` was
    dominated by two counters the loss had already normalised away.
    """
    v = S[:, :, cont_idx].reshape(-1, len(cont_idx))
    return np.maximum(v.std(0), 1e-6).astype(np.float32)


@torch.no_grad()
def at_ms(model, S, P, A, E, H, ticks, ms_list, device, n=256, seed=0):
    steps = {ms: max(1, int(round(ms * FPS / 1000.0 / ticks))) for ms in ms_list}
    span = H + max(steps.values())
    ok = np.ones(len(S), bool); ok[-span:] = False
    for k in range(1, span + 1):
        ok[:len(S) - k] &= (E[k:] == E[:len(S) - k])
    idx = np.random.default_rng(seed).choice(np.flatnonzero(ok),
                                             min(n, int(ok.sum())), replace=False)
    w = idx[:, None] + np.arange(span)[None, :]
    s = torch.as_tensor(S[w]).to(device); p = torch.as_tensor(P[w]).to(device)
    a = torch.as_tensor(A[w]).to(device)
    pred = rollout(model, s[:, :H], p[:, :H], a, max(steps.values()))
    c = torch.tensor(CONTINUOUS, device=device)
    sc = torch.as_tensor(_chan_scale(S, np.array(CONTINUOUS))).to(device)
    names = [STATE_CHANNELS[i] for i in CONTINUOUS]
    gidx = {g: torch.tensor([names.index(x) for x in cols if x in names],
                            device=device)
            for g, cols in GROUPS.items()}
    out = {}
    for ms, k in steps.items():
        tgt = s[:, H + k - 1].index_select(-1, c) / sc
        pr = pred[:, k - 1].index_select(-1, c) / sc
        idn = s[:, H - 1].index_select(-1, c) / sc
        rec = {"steps": k}
        for g, ix in gidx.items():
            if len(ix) == 0:
                continue
            m = float(F.mse_loss(pr.index_select(-1, ix), tgt.index_select(-1, ix)))
            i0 = float(F.mse_loss(idn.index_select(-1, ix), tgt.index_select(-1, ix)))
            rec[g] = 1.0 - m / max(i0, 1e-9)
        rec["all"] = 1.0 - float(F.mse_loss(pr, tgt)) / max(
            float(F.mse_loss(idn, tgt)), 1e-9)
        out[f"{ms}ms"] = rec
    return out


@torch.no_grad()
def block_ms(model, S, P, A, E, H, ticks, device, hold_ms=500, n=512, seed=0):
    k = max(1, int(round(hold_ms * FPS / 1000.0 / ticks)))
    span = H + k
    ok = np.ones(len(S), bool); ok[-span:] = False
    for j in range(1, span + 1):
        ok[:len(S) - j] &= (E[j:] == E[:len(S) - j])
    ok &= np.abs(S[:, 0, CH["dx"]]) < (250.0 / 1200.0)
    pool = np.flatnonzero(ok)
    if len(pool) < 32:
        return {"block_gain": float("nan"), "n": 0, "steps": k}
    idx = np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False)
    w = idx[:, None] + np.arange(span)[None, :]
    s = torch.as_tensor(S[w]).to(device); p = torch.as_tensor(P[w]).to(device)
    a = torch.as_tensor(A[w]).to(device)
    dx = s[:, H - 1, 0, CH["dx"]]
    L, R = BUTTONS.index("left"), BUTTONS.index("right")
    res = {}
    for tag, away in (("away", True), ("toward", False)):
        af = a.clone(); af[:, :, :, L] = 0.0; af[:, :, :, R] = 0.0
        left = (dx > 0) if away else (dx < 0)
        af[left, :, :, L] = 1.0; af[~left, :, :, R] = 1.0
        pr = rollout(model, s[:, :H], p[:, :H], af, k)
        res[tag] = float(torch.sigmoid(pr[:, :, 0, CH["guarding"]]).mean())
    return {"block_away": res["away"], "block_toward": res["toward"],
            "block_gain": res["away"] - res["toward"], "n": len(idx),
            "steps": k, "hold_ms": hold_ms}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sims", type=Path, nargs="+", required=True)
    ap.add_argument("--corpus", type=Path, default=Path("~/corpus").expanduser())
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--ms", type=int, nargs="+", default=[83, 333, 1333])
    ap.add_argument("--hold-ms", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    report = {}
    for sp in a.sims:
        model, ticks, step = load_sim(sp, a.device)
        H = model.history
        print(f"\n{sp}  (step {step}, ticks {ticks} = {ticks*1000/FPS:.1f} ms)")
        S, P, A, E = load_sequences(a.corpus, a.replays, model.slots, ticks)
        r = at_ms(model, S, P, A, E, H, ticks, a.ms, a.device)
        b = block_ms(model, S, P, A, E, H, ticks, a.device, a.hold_ms)
        print(f"   {'horizon':>8} {'steps':>5} {'kinematic':>10} "
              f"{'resource':>9} {'counter':>8} {'all':>7}")
        for ms, v in r.items():
            print(f"   {ms:>8} {v['steps']:>5} {v.get('kinematic',0):>+10.3f} "
                  f"{v.get('resource',0):>+9.3f} {v.get('counter',0):>+8.3f} "
                  f"{v['all']:>+7.3f}")
        print(f"   block ({b['hold_ms']} ms, {b['steps']} steps): gain "
              f"{b['block_gain']:+.4f} (away {b.get('block_away',0):.3f} "
              f"toward {b.get('block_toward',0):.3f}, n={b['n']})")
        report[str(sp)] = {"ticks": ticks, "step": step, "horizons": r, "block": b}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))
    print("\nSkill is against the SAME identity baseline at each real-time "
          "horizon, so these are comparable across frame_skip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
