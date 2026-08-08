"""Is blocking visible in the corpus, and does the world model reproduce it?

    python -m scripts.block_effect --bank ~/bank_hud.npz \
        --wm ~/sokubot-art/wm_cf_bnfix.pt

WHY THIS COMES BEFORE ANY BLOCKING GYM
---------------------------------------
The agent does not block, and in Hisoutensoku that is fatal rather than
suboptimal: a competent attacker converts one opening into the whole health bar,
so a match that should last minutes lasts seconds. Fixing it is the highest-value
thing available.

But a gym is only worth building if the thing it teaches is learnable, and that
needs two facts, in order:

  1. **The effect exists in the data.** When a human holds away during an
     incoming attack, do they take measurably less damage? This is ground truth
     from 200 hours of real play with no model involved. If it does not show up
     here, our labelling of "blocking" is wrong and everything downstream is too.

  2. **The world model reproduces it.** From the same start, does rolling
     "hold away" predict less damage than "hold toward"? Policy gradients consume
     the model's opinion, not the game's. A model blind to guarding cannot teach
     it, and a gym built on one would optimise noise very efficiently.

Both are cheap. Neither has been checked.

THE POSITION PROBLEM, STATED HONESTLY
--------------------------------------
Guarding in Soku is holding *away from the opponent*, and nothing in the corpus
records where the players are. The HUD has health, spirit and combo; it has no
coordinates.

So this uses the round-start prior -- P1 is on the left, so P1 guards with LEFT
and P2 with RIGHT -- which is right most of the time and wrong after a crossup.
That mislabelling is symmetric noise: it moves some genuine blocks into the
"toward" bucket and vice versa, which **shrinks** the measured gap. So an effect
that survives is real and understated, and a null is genuinely ambiguous. The
`--split-by-side` output exists to check that P1 and P2 agree; if the prior were
useless they would disagree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Button layout within one player's 10-wide block.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
HP1, HP2, SPIRIT1, SPIRIT2, COMBO1, COMBO2 = range(6)


def held(actions: np.ndarray, player: int, button: int) -> np.ndarray:
    """[N, ticks, 20] -> [N] fraction of ticks this player held this button."""
    return actions[:, :, player * 10 + button].mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--wm", type=Path, default=None,
                    help="if given, also asks whether the world model predicts "
                         "the same effect from the same starts")
    ap.add_argument("--window", type=int, default=4,
                    help="decision steps over which damage is accumulated; 4 "
                         "matches the training horizon")
    ap.add_argument("--min-hold", type=float, default=0.75,
                    help="fraction of ticks a direction must be held to count "
                         "as holding it")
    ap.add_argument("--attack-min", type=float, default=0.01,
                    help="minimum health the defender must lose over the window "
                         "for it to count as 'under attack'. Below this the "
                         "window is neutral and blocking has nothing to do.")
    ap.add_argument("--probe", type=Path, default=None,
                    help="reads health out of the imagined latents. OPTIONAL: a "
                         "probe is fit against one specific encoder, so a new "
                         "world model has none until horizon_ablation is re-run. "
                         "Without it the health arms are skipped and only the "
                         "latent comparison runs -- which is the decisive half "
                         "anyway, since it is what measured the JEPA encoder at "
                         "4.6% sensitivity to swapping LEFT and RIGHT.")
    ap.add_argument("--starts", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("block_effect.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    b = np.load(a.bank.expanduser())
    A, Hd, E = b["a"].astype(np.float32), b["hud"].astype(np.float32), b["ep"]
    W = a.window
    n = len(A)

    # Windows that stay inside one replay.
    ok = np.ones(n - W, dtype=bool)
    for k in range(1, W + 1):
        ok &= (E[k : n - W + k] == E[: n - W])
    idx_all = np.flatnonzero(ok)
    print(f"bank {n} steps, {int(E.max())+1} replays | {len(idx_all)} windows "
          f"of {W} steps")

    res = {"window": W, "min_hold": a.min_hold, "attack_min": a.attack_min,
            "corpus": {}}
    print(f"\n=== 1. the corpus: does holding away reduce damage taken? ===")
    print("    (P1 guards LEFT, P2 guards RIGHT, from the round-start prior)\n")
    print("  side  n_windows   held away   held toward   neither")
    print("        (attacked)  dmg taken    dmg taken   dmg taken")

    pooled = {}
    for side, (hp_me, guard, expose) in {
        "p1": (HP1, LEFT, RIGHT),
        "p2": (HP2, RIGHT, LEFT),
    }.items():
        player = 0 if side == "p1" else 1
        # Damage taken over the window, as a positive number.
        dmg = Hd[idx_all, hp_me] - Hd[idx_all + W, hp_me]
        attacked = dmg >= a.attack_min
        g = held(A[idx_all], player, guard) >= a.min_hold
        e = held(A[idx_all], player, expose) >= a.min_hold
        neither = ~g & ~e

        row = {}
        for name, m in (("away", g & attacked), ("toward", e & attacked),
                        ("neither", neither & attacked)):
            row[name] = {"n": int(m.sum()),
                         "damage": float(dmg[m].mean()) if m.any() else float("nan")}
        res["corpus"][side] = row
        pooled[side] = row
        print(f"  {side}   {int(attacked.sum()):9d}   "
              f"{row['away']['damage']:9.4f}   {row['toward']['damage']:10.4f}   "
              f"{row['neither']['damage']:8.4f}")
        print(f"        {'':9s}   (n={row['away']['n']:<6d}) (n={row['toward']['n']:<6d})"
              f" (n={row['neither']['n']:<6d})")

    # The headline: away vs toward, averaged over the two chairs so a
    # side-specific quirk in the prior cannot produce the effect on its own.
    aw = np.mean([pooled[s]["away"]["damage"] for s in ("p1", "p2")])
    tw = np.mean([pooled[s]["toward"]["damage"] for s in ("p1", "p2")])
    nt = np.mean([pooled[s]["neither"]["damage"] for s in ("p1", "p2")])
    res["corpus_summary"] = {"away": float(aw), "toward": float(tw),
                             "neither": float(nt),
                             "reduction": float(1.0 - aw / max(tw, 1e-9))}
    print(f"\n  pooled: away {aw:.4f} | toward {tw:.4f} | neither {nt:.4f}")
    print(f"  holding away takes {1 - aw / max(tw, 1e-9):+.1%} the damage of "
          f"holding toward")
    agree = ((pooled["p1"]["away"]["damage"] < pooled["p1"]["toward"]["damage"])
             == (pooled["p2"]["away"]["damage"] < pooled["p2"]["toward"]["damage"]))
    print(f"  P1 and P2 agree on the sign: {agree}"
          + ("" if agree else "  <- the side prior is not carrying signal"))

    if a.wm is None:
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\n-> {a.out}   (pass --wm to ask the world model the same thing)")
        return 0

    # ---- 2. does the world model reproduce it? ----
    from sokubot.model.loading import load_world_model
    from sokubot.probe import LinearProbe
    from sokubot.rl.grpo import GRPOConfig, ImaginedArena, ProbeHead
    from scripts.eval_policy import RCFG

    wm, cfg, _ = load_world_model(a.wm, a.device)
    Z = torch.from_numpy(b["z"]).to(a.device)
    At = torch.from_numpy(b["a"]).to(a.device)

    # Start states that are actually under attack, so the question is live.
    dmg_p1 = Hd[idx_all, HP1] - Hd[idx_all + W, HP1]
    cand = idx_all[dmg_p1 >= a.attack_min]
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(cand, size=min(a.starts, len(cand)), replace=False)
    sel = sel[sel >= cfg.history - 1]
    idx = torch.from_numpy(sel).to(a.device)
    print(f"\n=== 2. the world model, from {len(idx)} under-attack starts ===")

    off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
    z_ctx = Z[idx[:, None] + off[None, :]].float()
    a_hist = At[idx[:, None] + off[None, :-1]].float()

    arena = None
    if a.probe is not None:
        pd_ = np.load(a.probe.expanduser(), allow_pickle=True)
        probe = LinearProbe(zmu=pd_["zmu"], zsd=pd_["zsd"], ymu=pd_["ymu"],
                            ysd=pd_["ysd"], W=pd_["W"],
                            names=[str(x) for x in pd_["names"]])
        arena = ImaginedArena(wm, ProbeHead(probe).to(a.device),
                              GRPOConfig(horizon=W, reward=RCFG),
                              cfg.history, cfg.action_ticks)
    else:
        print("  (no --probe: health arms skipped, latent comparison only)")

    # The corpus half showed "away" and "toward" are indistinguishable, because
    # with no position data the label is a coin flip. So the model is asked a
    # question that needs no label at all: does replaying what the defender
    # ACTUALLY held predict less damage than holding no direction? The corpus
    # says yes by 21% (0.0371 against 0.0473), and that gap is label-free.
    #
    # The opponent replays their true inputs in every arm, so the only thing
    # that varies is the defender's own directional input.
    plan0 = At[idx[:, None] + torch.arange(W, device=a.device)[None, :]].float()

    def forced(mode: str) -> torch.Tensor:
        plan = plan0.clone()
        if mode == "true":
            return plan                                   # what they really did
        if mode == "no_direction":
            plan[..., :4] = 0.0                           # P1's four directions
            return plan
        if mode == "mirrored":
            # Swap LEFT and RIGHT. Whatever "away" was, this is the other one,
            # so it is the right comparison even without knowing which is which.
            left = plan[..., LEFT].clone()
            plan[..., LEFT] = plan[..., RIGHT]
            plan[..., RIGHT] = left
            return plan
        raise ValueError(mode)

    out2 = {}
    if arena is not None:
        with torch.no_grad():
            for name in ("true", "mirrored", "no_direction"):
                z_roll = wm.rollout(z_ctx, forced(name), a_hist)
                st = arena.probe(z_roll)
                dmg = float((st[:, 0, HP1] - st[:, -1, HP1]).mean())
                out2[name] = dmg
                print(f"  defender plays {name:<14} predicted damage {dmg:+.5f}")
    # The probe reads health with a residual of 0.116, and these gaps are ~0.002,
    # so the health comparison alone sits near its own noise floor. The latent
    # test does not depend on the probe at all: if swapping LEFT and RIGHT
    # barely moves the predicted latent, the predictor is not using the
    # distinction, whatever the reward head then reads off it.
    with torch.no_grad():
        z_true = wm.rollout(z_ctx, forced("true"), a_hist)
        z_mirr = wm.rollout(z_ctx, forced("mirrored"), a_hist)
        z_none = wm.rollout(z_ctx, forced("no_direction"), a_hist)
        import torch.nn.functional as _F
        step = z_true.shape[1] - 1
        def cmp(x, y):
            return (float(_F.cosine_similarity(x[:, step], y[:, step], dim=-1).mean()),
                    float((x[:, step] - y[:, step]).norm(dim=-1).mean()))
        c_mirr, l_mirr = cmp(z_true, z_mirr)
        c_none, l_none = cmp(z_true, z_none)
        spread = float(z_true[:, step].std(0).mean())
    res["latent"] = {"cos_true_vs_mirrored": c_mirr, "l2_true_vs_mirrored": l_mirr,
                     "cos_true_vs_nodir": c_none, "l2_true_vs_nodir": l_none,
                     "latent_spread": spread}
    print(f"\n  in latent space, at the last imagined step:")
    print(f"    true vs mirrored     cosine {c_mirr:.5f}  L2 {l_mirr:.4f}")
    print(f"    true vs no-direction cosine {c_none:.5f}  L2 {l_none:.4f}")
    print(f"    (across-batch latent spread for scale: {spread:.4f})")

    res["world_model"] = out2
    # Positive means the model thinks holding a direction helps, which is the
    # direction the corpus points.
    gap = (out2["no_direction"] - out2["true"]) if out2 else float("nan")
    res["world_model_gap"] = gap
    res["world_model_mirror_gap"] = ((out2["mirrored"] - out2["true"])
                                     if out2 else float("nan"))

    print("\n" + "=" * 70)
    corpus_ok = nt > max(aw, tw)          # holding a direction beats holding none
    # With no probe the health arms are absent and the latent test has to
    # decide. The obvious criterion -- mirrored displacement as a fraction of
    # no-direction displacement -- is WRONG, and it handed out a false pass:
    #
    #     JEPA   mirrored 4.6% of spread, no-direction 37%    ratio 0.12
    #     IDM    mirrored 7.1%,           no-direction 11.5%  ratio 0.61
    #
    # The ratio quintupled, but mostly because the denominator collapsed. A
    # model that stopped responding to directional input at all would score a
    # perfect ratio. So the criterion is the *absolute* left/right sensitivity
    # against the spread, with the ratio reported alongside for context.
    #
    # 0.10 of a standard deviation is the bar: JEPA sat at 0.046 while having
    # the sign of guarding inverted, so anything near that is not evidence.
    # This is a proxy either way -- pass --probe and let the health arms answer
    # the question that actually matters.
    mirror_frac = l_mirr / max(spread, 1e-9)
    model_ok = (gap > 0) if out2 else (mirror_frac >= 0.10)
    d_corpus = nt - 0.5 * (aw + tw)
    if corpus_ok and model_ok:
        detail = (f"the world model agrees ({gap:+.5f})" if out2 else
                  f"swapping LEFT and RIGHT moves the prediction "
                  f"{mirror_frac:.1%} of a\nlatent standard deviation "
                  f"(JEPA: 4.6%, with the sign of guarding inverted)")
        print(f"BOTH HOLD. In the corpus, holding a direction takes {d_corpus:.4f} "
              f"less damage\nthan holding none, and {detail}. Defence is in "
              f"the model, so a blocking gym\nhas something real to optimise. "
              f"H=4 is enough: the *decision* is per-step even\nthough the hold "
              f"lasts seconds.")
    elif corpus_ok:
        detail = (f"({gap:+.5f})" if out2 else
                  f"(swapping LEFT and RIGHT moves the prediction only "
                  f"{mirror_frac:.1%} of a latent\nstandard deviation, against "
                  f"{l_none / max(spread, 1e-9):.1%} for removing the direction "
                  f"entirely -- so it\nregisters *whether* a direction is held "
                  f"more than *which*)")
        print(f"The corpus shows defence ({d_corpus:.4f} less damage) and the "
              f"world model does\nNOT {detail}. A gym would optimise the "
              f"model's blind spot, so fix the model\nfirst rather than the "
              f"reward.")
    else:
        print("The corpus does not show the effect even label-free. Check the "
              "window and\nattack-min settings before concluding anything about "
              "the agent.")
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
