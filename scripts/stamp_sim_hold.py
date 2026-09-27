"""Stamp a simulator checkpoint with the channels its rollout must hold (`state_dynamics.hold_dead`).

Checkpoints written before 2026-09-27 have no "hold" key, so their rollouts feed back the untrained
output of every dead channel. This recomputes the dead set with the trainer's OWN rule
(`delta_scale` over the corpus the simulator was trained on) and writes a COPY with it: the
original's bytes, and so every fingerprint recorded against it, are left alone. A PPO run on the
copy therefore starts fresh instead of resuming a run that rolled a different world.

    python -m scripts.stamp_sim_hold --sim sim-full-h8-raw/sim.pt \\
        --cache sim_corpus_full.npz --out sim-full-h8-raw/sim_hold.pt

Needs about 3x the corpus's S array in memory (the full corpus: run it as a job, not on a login
node).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from scripts.train_state_dynamics import delta_scale
from sokubot.data.state import STATE_CHANNELS
from sokubot.model.state_head import CONTINUOUS


def dead_channels(S: np.ndarray, E: np.ndarray) -> list[str]:
    cont_idx = np.array(CONTINUOUS)
    _, live = delta_scale(S, E, cont_idx)
    return [STATE_CHANNELS[cont_idx[j]] for j in range(len(cont_idx)) if live[j] == 0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True,
                    help="the corpus the simulator was trained on: train_state_dynamics' cache "
                         "or a PPO bank (only S and E are read)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.resolve() == a.sim.resolve():
        raise SystemExit("--out must be a new file: the original's fingerprint is on record")
    d = torch.load(a.sim, map_location="cpu", weights_only=False)
    z = np.load(a.cache)
    dead = dead_channels(z["S"], z["E"])
    if "hold" in d and list(d["hold"]) != dead:
        raise SystemExit(f"{a.sim} already holds {list(d['hold'])}, but this corpus says {dead}: "
                         f"is --cache the corpus it was trained on?")
    d["hold"] = dead
    tmp = a.out.with_name(a.out.name + ".tmp")
    torch.save(d, tmp)
    os.replace(tmp, a.out)
    print(f"{a.out}: hold {dead}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
