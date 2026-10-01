"""Loading a trained state policy, without the live harness.

`load_agent` lived in scripts/play_cheat_match.py, so anything that wanted a policy -- the vision
server, now the vs-COM environment -- imported the whole live client with it (screen capture, the
virtual pad, evdev). It is fifteen lines and depends on two modules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sokubot.rl.policy import SokuPolicy
from sokubot.rl.state_arena import StateObs


def load_agent(path: Path, device="cpu"):
    """Policy + its OWN observation normalisation, from the checkpoint.

    The normalisation travels with the weights rather than being recomputed:
    the policy's input space is defined by the statistics it trained under, and
    a live reader feeding it differently-scaled numbers is asking it to play a
    game it has never seen.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    H, ticks, slots = int(ck["history"]), int(ck["ticks"]), int(ck["slots"])
    obs = StateObs(np.zeros(33, np.float32), np.ones(33, np.float32),
                   np.zeros(7, np.float32), np.ones(7, np.float32), slots)
    obs.load_state_dict(ck["obs"])
    pol = SokuPolicy(obs.dim, H, ticks)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    return pol, obs, H, ticks, slots, ck
