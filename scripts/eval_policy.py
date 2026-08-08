"""Score any number of policy checkpoints on one instrument.

    python -m scripts.eval_policy --wm ~/sokubot-art/wm_cf_bnfix.pt \
        --probe ~/gate_base/reward_probe.npz --bank ~/bank_hud.npz \
        --policy grpo=~/sokubot-art/policy_best.pt \
        --policy ac=~/ac/policy_best.pt --horizons 4 16

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
`train_ac.py` and `train_grpo.py` each print a `net` from their own `evaluate()`,
and it is tempting to compare those two numbers directly. They are not
comparable, for a reason that is easy to miss:

    net = mean net damage **per alive step, over a rollout of cfg.horizon steps**

GRPO's recorded +0.00215 was measured at ``--horizon 4``. The actor-critic runs
at horizon 16 by design -- that is the point of the critic. So its `evaluate()`
averages over sixteen steps of a rollout that has had four times as long to blur,
against four. Reading the two side by side would compare a policy change and an
instrument change at once, and attribute the sum to the policy.

This is the same mistake that nearly triggered a full world-model rebuild: the
225k checkpoint was compared against a *recorded* 320k number taken on a
different validation sample, and the ordering flipped once both were measured on
one instrument. The rule that came out of it is the rule here -- **when two
things are compared, measure both, in one run, on one instrument.**

THE INSTRUMENT
--------------
Every arm gets: the same world model, the same probe, the same bank, the same
frozen reference opponent, the same evaluation starts (seed 12345), the same
side-swap, and the same horizon. Only the policy weights differ.

The reference opponent is built here rather than recovered from any training run,
because it cannot be recovered: its non-prior weights come from wherever the RNG
stood when that script happened to construct it. So this builds one reference
from a dedicated seed and uses it for all arms. That makes the absolute numbers
this prints slightly different from the ones in the training logs, and makes the
*differences between arms* exact -- which is the only thing being asked.

THE ZERO POINT
--------------
The first row is always the reference scored against itself. By construction it
should read 0: identical players, sides swapped. Whatever it actually reads is
the instrument's noise floor, and no difference between arms smaller than it
means anything. It is printed rather than assumed because the quantity being
resolved is ~0.002, which is small enough that this genuinely matters.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from sokubot.model.loading import load_world_model
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.reward import KO_BANNER, RewardConfig, banner_ko_masks
from scripts.train_grpo import model_fingerprint, valid_starts

# The reference must not depend on what the caller did to the global RNG before
# calling, so it gets its own seed. Changing this number invalidates comparisons
# against every previously printed table, so it is a constant, not a flag.
REF_SEED = 777
EVAL_SEED = 12345

# Identical to the one both training scripts build. `net` reads only the damage
# terms, but the KO mask that decides which steps are alive is driven from this,
# so it is not inert.
RCFG = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                    flying=0.0015, idle=-0.020)

# Outcome measurement is deliberately independent of what any arm trained on.
# `ko_source` is a *training* choice; this is the ruler, and it always uses the
# better detector so that an arm trained on the health test is not scored by it.
BANNER_EVAL = RewardConfig(ko_source="banner")


def build_reference(bank_actions: np.ndarray, cfg, device: str) -> SokuPolicy:
    """The corpus-prior-initialised policy every arm is scored against."""
    torch.manual_seed(REF_SEED)
    ref = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(device)
    p1 = bank_actions.astype(np.float32).reshape(-1, cfg.action_dim)[:, :10]
    lr_p = np.array([float(((1 - p1[:, 2]) * (1 - p1[:, 3])).mean()),
                     float(p1[:, 2].mean()), float(p1[:, 3].mean())])
    ud_p = np.array([float(((1 - p1[:, 0]) * (1 - p1[:, 1])).mean()),
                     float(p1[:, 0].mean()), float(p1[:, 1].mean())])
    ref.set_action_prior(lr_p / lr_p.sum(), ud_p / ud_p.sum(), p1[:, 4:10].mean(0))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


@torch.no_grad()
def score(arena: ImaginedArena, policy: SokuPolicy, reference: SokuPolicy,
          Zt: torch.Tensor, At: torch.Tensor, eval_idx: torch.Tensor,
          cfg, device: str) -> dict:
    """Net damage per alive step against `reference`, averaged over both chairs.

    The side swap is what cancels the probe's mirrored chair bias: the probe
    reads P1's health slightly differently from P2's, worth about +-0.0007 --
    a third of the effect being measured -- and playing each start from both
    chairs makes that bias appear with both signs and cancel.
    """
    off = torch.arange(cfg.history, device=device) - (cfg.history - 1)
    zc = Zt[eval_idx[:, None] + off[None, :]].float()
    ah = At[eval_idx[:, None] + off[None, :-1]].float()
    out = {}
    wins = losses = 0.0
    n_ko = 0
    measurable = False
    for tag, s0 in (("p1", 0), ("p2", 1)):
        side = torch.full((len(eval_idx),), s0, device=device, dtype=torch.long)
        tr = arena.rollout(zc, ah, side, policy, PolicyOpponent(reference))
        al = tr["alive"]
        n = al.sum().clamp(min=1)
        out[f"{tag}_dealt"] = float((tr["terms"]["dealt"] * al).sum() / n)
        out[f"{tag}_taken"] = float((tr["terms"]["taken"] * al).sum() / n)
        out[f"{tag}_alive"] = float(al.mean())
        # Outcomes, when the probe can see them. `net` is a *damage exchange*,
        # and the +-5 win/lose term rewards finishing -- so a policy that learns
        # to close rounds out can be a better player while looking flat on net.
        # Measured with the banner detector regardless of what any arm trained
        # on, because this is the ruler and the banner is the better instrument
        # (precision 0.803 against the health test's 0.003 on real frames).
        if tr["states"].shape[-1] > KO_BANNER:
            measurable = True
            me, them = banner_ko_masks(tr["states"], side, BANNER_EVAL)
            losses += float(me.any(1).float().mean())
            wins += float(them.any(1).float().mean())
            n_ko += int(me.any(1).sum()) + int(them.any(1).sum())
        if tag == "p1":
            out["press_rate"] = float(tr["mine"].mean())
            out["attack_rate"] = float(tr["mine"][..., 4:8].mean())
    out["net"] = ((out["p1_dealt"] + out["p1_taken"]) +
                  (out["p2_dealt"] + out["p2_taken"])) / 2
    if measurable:
        # Reported even when zero. An absent column and a measured zero look the
        # same to a reader, and at small `--starts` this block silently produced
        # no output at all -- which reads as "the probe cannot see outcomes"
        # rather than "no KO happened in 256 rollouts".
        out["win_rate"] = wins / 2
        out["loss_rate"] = losses / 2
        out["outcome"] = (wins - losses) / 2
        # The event count is the context that decides whether any of the above
        # means anything: these are rates on a ~1% base rate, so a few hundred
        # rollouts contain single-digit KOs and Poisson noise swamps the effect.
        out["ko_events"] = n_ko
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable. Scored in the order given, all against the "
                         "same reference.")
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 16],
                    help="4 is the horizon GRPO's +0.00215 was measured at; 16 is "
                         "the actor-critic's. Both are printed so neither method "
                         "is flattered by the choice.")
    ap.add_argument("--starts", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    wm, cfg, _ = load_world_model(a.wm, a.device)
    fp = model_fingerprint(wm)
    print(f"world model {a.wm} | fingerprint {fp} | image_size {cfg.image_size}",
          flush=True)

    d = np.load(a.probe.expanduser(), allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    probe_fp = str(d["fingerprint"]) if "fingerprint" in d.files else None
    if probe_fp is not None and probe_fp != fp:
        raise SystemExit(f"probe was fit on {probe_fp}, model is {fp}")

    b = np.load(a.bank.expanduser())
    bank_fp = str(b["fingerprint"]) if "fingerprint" in b.files else None
    if bank_fp is not None and bank_fp != fp:
        raise SystemExit(f"bank was encoded by {bank_fp}, model is {fp}")
    Z, A, E = b["z"], b["a"], b["ep"]
    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    # One start set for every horizon. `valid_starts` needs room for the future,
    # so computing it per horizon would hand the short horizon a larger and
    # differently-distributed pool -- an instrument change disguised as a
    # horizon change. Sizing it once at the longest horizon keeps the starts
    # fixed and makes the horizon the only thing that varies.
    Hmax = max(a.horizons)
    starts = valid_starts(E, cfg.history, Hmax)
    eval_idx = torch.from_numpy(
        np.random.default_rng(EVAL_SEED).choice(starts, size=a.starts)
    ).to(a.device)
    print(f"bank {len(Z)} latents, {int(E.max())+1} replays | {len(starts)} valid "
          f"starts at horizon {Hmax} | scoring {a.starts}", flush=True)

    reference = build_reference(A, cfg, a.device)
    arms = [("reference (control)", reference)]
    for spec in a.policy:
        if "=" not in spec:
            raise SystemExit(f"--policy wants NAME=PATH, got {spec!r}")
        name, _, p = spec.partition("=")
        blob = torch.load(Path(p).expanduser(), map_location=a.device,
                          weights_only=False)
        pol = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
        pol.load_state_dict(blob["policy"])
        pol.eval()
        for q in pol.parameters():
            q.requires_grad_(False)
        step = blob.get("step", "?")
        print(f"  arm {name!r}: {p} (step {step}, recorded net "
              f"{blob.get('net', float('nan')):+.5f})", flush=True)
        arms.append((name, pol))

    probe_head = ProbeHead(probe).to(a.device)
    res = {"wm": str(a.wm), "fingerprint": fp, "starts": a.starts, "rows": []}
    for H in a.horizons:
        gcfg = GRPOConfig(horizon=H, reward=RCFG)
        arena = ImaginedArena(wm, probe_head, gcfg, cfg.history, cfg.action_ticks)
        print(f"\n=== horizon {H} ({H * cfg.frame_skip / 60:.2f} s) "
              f"{'=' * 40}", flush=True)
        base = base_oc = None
        for name, pol in arms:
            r = score(arena, pol, reference, Zt, At, eval_idx, cfg, a.device)
            if base is None:
                base = r["net"]
            row = {"horizon": H, "arm": name, **r, "net_over_control": r["net"] - base}
            if "outcome" in r:
                if base_oc is None:
                    base_oc = r["outcome"]
                row["outcome_over_control"] = r["outcome"] - base_oc
            res["rows"].append(row)
            oc = (f" | outcome {r['outcome']:+.3%} "
                  f"(vs control {row['outcome_over_control']:+.3%})"
                  if "outcome" in r else "")
            print(f"  {name:<22} net {r['net']:+.5f} "
                  f"(vs control {row['net_over_control']:+.5f}) | "
                  f"P1 {r['p1_dealt']:+.4f}/{r['p1_taken']:+.4f} "
                  f"P2 {r['p2_dealt']:+.4f}/{r['p2_taken']:+.4f} | "
                  f"press {r['press_rate']:.3f}{oc}", flush=True)

    print("\n" + "=" * 68)
    print("The control row is the reference played against itself with the sides\n"
          "swapped, so it is 0 by construction. Its distance from 0 is this\n"
          "instrument's noise floor: no gap between arms below it is real.")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
