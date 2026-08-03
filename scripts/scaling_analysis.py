"""Read a scaling run and fit how held-out loss improves with data.

    python -m scripts.scaling_analysis --results runs/scaling.json

Fits the standard data-scaling form

    L(D) = L_inf + A * D^(-alpha)

where D is footage-hours, `alpha` is the scaling exponent, and `L_inf` is the
irreducible loss the model cannot go below at this capacity and compute -- the
part that more data will never buy.

A caution the output repeats: four points is one degree of freedom against three
parameters. The fit is worth reporting as a trend and an order of magnitude, and
is not worth trusting to two decimal places. `L_inf` in particular is the least
constrained parameter, because nothing in the measured range is close to it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def fit_power_law(D: np.ndarray, L: np.ndarray):
    """Least squares over L = L_inf + A * D^-alpha, by grid search on (alpha, L_inf).

    With four points, a gradient-based joint fit lands in whatever local optimum
    it starts nearest. A coarse-to-fine grid over the two nonlinear parameters
    (A is then linear and solved exactly) is slower and does not have that
    failure mode.
    """
    best = None
    alphas = np.linspace(0.01, 2.0, 400)
    # L_inf cannot exceed the best loss observed, and negative is meaningless.
    linfs = np.linspace(0.0, float(L.min()), 200)
    for alpha in alphas:
        x = D ** (-alpha)
        for linf in linfs:
            y = L - linf
            denom = float((x * x).sum())
            if denom <= 0:
                continue
            A = float((x * y).sum() / denom)
            if A <= 0:
                continue
            resid = float(((linf + A * x - L) ** 2).sum())
            if best is None or resid < best[0]:
                best = (resid, alpha, A, linf)
    resid, alpha, A, linf = best
    ss_tot = float(((L - L.mean()) ** 2).sum())
    r2 = 1.0 - resid / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": alpha, "A": A, "L_inf": linf, "resid": resid, "r2": r2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="runs/scaling.json")
    ap.add_argument("--baselines", default="runs/baselines.json",
                    help="output of scripts.baselines; adds the identity-skill column")
    ap.add_argument("--targets", default="20,50,100,200",
                    help="footage-hours to extrapolate to")
    args = ap.parse_args()

    rows = json.loads(Path(args.results).read_text())
    rows.sort(key=lambda r: r["hours"])
    D = np.array([r["hours"] for r in rows], dtype=float)
    L = np.array([r["val_pred"] for r in rows], dtype=float)
    T = np.array([r["train_pred"] for r in rows], dtype=float)
    AUC = np.array([r["inv_dyn_auc"] for r in rows], dtype=float)

    # Merge in the trivial-predictor baselines. Raw val_pred and correlation
    # both look excellent on 15 Hz video for a model that has learned nothing
    # but pass-through, so `skill` is the column to read.
    bl = {}
    bp = Path(args.baselines)
    if bp.exists():
        bl = {b["hours"]: b for b in json.loads(bp.read_text())}
    for r in rows:
        b = bl.get(r["hours"])
        if b:
            r["identity"] = b["identity"]
            # scripts.baselines calls it skill_vs_identity, scripts.final_eval
            # calls it skill; accept either so the analysis runs against both.
            r["skill"] = b.get("skill", b.get("skill_vs_identity"))

    has_skill = all("skill" in r for r in rows)
    header = (f"{'hours':>7} {'caps':>5} {'train':>8} {'val':>8} {'gap':>8} "
              f"{'corr':>8} {'invdyn':>8}")
    if has_skill:
        header += f" {'identity':>9} {'skill':>8}"
    print(header)
    for r in rows:
        line = (f"{r['hours']:7.0f} {r['captures']:5d} {r['train_pred']:8.4f} "
                f"{r['val_pred']:8.4f} {r['val_pred'] - r['train_pred']:+8.4f} "
                f"{r['val_corr']:+8.4f} {r['inv_dyn_auc']:8.4f}")
        if has_skill:
            line += f" {r['identity']:9.4f} {r['skill']:+8.4f}"
        print(line)
    if not has_skill:
        print("\n  (no baselines.json -- run scripts.baselines first; without it")
        print("   val_pred cannot be told apart from a pass-through predictor)")

    if len(rows) < 3:
        print("\nneed at least 3 sizes to fit a scaling law")
        return

    # Only fit if loss actually falls with data. A power law forced onto a flat
    # or rising curve returns numbers -- an exponent, an extrapolation to 200h --
    # that look like findings and are not. Refusing to fit is the result.
    if L[-1] >= L[0]:
        print(f"\nNO SCALING FIT: held-out loss does not improve with data "
              f"({L[0]:.4f} at {D[0]:.0f}h -> {L[-1]:.4f} at {D[-1]:.0f}h).")
        print("Under fixed compute this is the expected shape once the largest")
        print("dataset stops being revisited: more data is more distribution to")
        print("fit with the same number of gradient steps. Compute, not data, is")
        print("the binding constraint here -- extrapolating to 200h from these")
        print("points would be extrapolating a compute limit.")
    else:
        fit = fit_power_law(D, L)
        print(f"\nfit  L(D) = {fit['L_inf']:.4f} + {fit['A']:.4f} * D^-{fit['alpha']:.3f}"
              f"   (R2 {fit['r2']:.4f} over {len(D)} points, 1 dof)")
        if fit["alpha"] > 0:
            frac = 1 - 2 ** (-fit["alpha"])
            print(f"each doubling removes {frac * 100:.1f}% of the reducible loss "
                  f"(the part above L_inf)")
        print(f"\n{'hours':>8} {'predicted val_pred':>20} {'predicted corr':>16}")
        for t in [float(x) for x in args.targets.split(",")]:
            pred = fit["L_inf"] + fit["A"] * t ** (-fit["alpha"])
            print(f"{t:8.0f} {pred:20.4f} {1 - pred / 2:16.4f}")

    print("\nCaveats")
    print("  * 4 points, 3 parameters -- read the exponent as a trend, not a value.")
    print("  * L_inf is the least constrained: no measured point is near it.")
    print("  * Fixed compute (same steps everywhere), so the large-data runs are")
    print("    undertrained relative to the small ones and this is a lower bound")
    print("    on what more data is worth.")

    if len(AUC) == len(D):
        print(f"\ninverse-dynamics AUC: {AUC[0]:.4f} at {D[0]:.0f}h "
              f"-> {AUC[-1]:.4f} at {D[-1]:.0f}h "
              f"({(AUC[-1] - 0.5) / max(AUC[0] - 0.5, 1e-9):.2f}x the margin over chance)")
    gap = L - T
    print(f"generalisation gap: {gap[0]:+.4f} at {D[0]:.0f}h -> {gap[-1]:+.4f} at {D[-1]:.0f}h")

    if has_skill:
        S = np.array([r["skill"] for r in rows], dtype=float)
        print(f"skill over identity: {S[0]:+.4f} at {D[0]:.0f}h "
              f"-> {S[-1]:+.4f} at {D[-1]:.0f}h")
        # Fit the *skill gap* (1 - skill), the fraction of the pass-through
        # error still unexplained. This is the scaling curve that is not
        # inflated by how similar consecutive video frames happen to be.
        gap_skill = 1.0 - S
        if np.all(gap_skill > 0) and len(S) >= 3:
            fs = fit_power_law(D, gap_skill)
            print(f"fit  (1-skill)(D) = {fs['L_inf']:.4f} + {fs['A']:.4f} * "
                  f"D^-{fs['alpha']:.3f}   (R2 {fs['r2']:.4f})")
            print(f"\n{'hours':>8} {'predicted skill':>17}")
            for t in [float(x) for x in args.targets.split(",")]:
                g = fs["L_inf"] + fs["A"] * t ** (-fs["alpha"])
                print(f"{t:8.0f} {1 - g:17.4f}")


if __name__ == "__main__":
    main()
