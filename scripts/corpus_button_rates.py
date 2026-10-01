"""The corpus's per-button press rates, as train_state_ppo computes them for its ButtonRateFloor.

    python -m scripts.corpus_button_rates --bank ~/sokubot-runs/sim_corpus_full.npz \\
        --out ~/sokubot-runs/button_rates.json

The real-game learner keeps the simulator runs' button-rate floor, which needs these ten numbers;
computing them once here spares every learner start a 29 GB load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_state_ppo import as_buttons, stat_view


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    A = as_buttons(np.load(a.bank)["A"])
    cr = stat_view(A).reshape(-1, 20).astype(np.float32)     # train_state_ppo, line for line
    rates = (cr[:, :10].mean(0) + cr[:, 10:].mean(0)) / 2
    a.out.write_text(json.dumps([float(x) for x in rates]))
    print(f"{a.out}: {np.round(rates, 4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
