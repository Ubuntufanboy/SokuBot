"""PushT: environment wrapper and offline data collection.

PushT (Chi et al.) is the benchmark both papers lead with, so it is what SokuBot
is smoke-tested on before any Soku data exists. The environment is 2-D, contact
rich, and cheap enough to run on a laptop CPU.

Two things about the native environment shape the code here:

* An action is an **absolute target position** for a PD-controlled agent, in
  [0, 512] pixels -- not a velocity or a force. So a chunk of ``frame_skip``
  actions is a short path, and a sensible behaviour policy emits waypoints
  rather than independent samples.
* Data is collected with "a random policy biased toward interacting with the
  block" (LeWM App. E). Uniform random targets almost never touch the T, and a
  world model trained on them learns that the block is static.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import Config
from .episode import Episode, denormalize_action, normalize_action

ENV_ID = "gym_pusht/PushT-v0"

# Ground-truth state recorded alongside each frame, for latent probing only.
# The T block's orientation enters as (cos, sin) rather than the raw angle so a
# linear probe is not asked to model the 2*pi wrap-around, which it cannot.
STATE_NAMES = ("agent_x", "agent_y", "block_x", "block_y", "block_cos", "block_sin")


def env_state(info: dict, obs: dict) -> np.ndarray:
    agent = np.asarray(info.get("pos_agent", obs["agent_pos"]), dtype=np.float32)
    bx, by, btheta = np.asarray(info["block_pose"], dtype=np.float32)[:3]
    return np.array(
        [agent[0], agent[1], bx, by, np.cos(btheta), np.sin(btheta)], dtype=np.float32
    )


def make_env(cfg: Config, render_mode: str = "rgb_array", **kwargs):
    """Build the PushT env rendering at the model's input resolution."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    return gym.make(
        ENV_ID,
        obs_type="pixels_agent_pos",
        render_mode=render_mode,
        observation_width=cfg.image_size,
        observation_height=cfg.image_size,
        **kwargs,
    )


def _waypoints(start: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation from `start` (exclusive) to `target` (inclusive)."""
    ts = np.linspace(1.0 / n, 1.0, n).reshape(n, 1)
    return start.reshape(1, 2) * (1 - ts) + target.reshape(1, 2) * ts


def _sample_target(rng: np.random.Generator, info: dict, block_bias: float) -> np.ndarray:
    """A waypoint target: mostly near the block, sometimes anywhere."""
    if rng.random() < block_bias:
        block_xy = np.asarray(info["block_pose"][:2], dtype=np.float32)
        # Aim slightly past the block so the agent pushes through it rather than
        # parking on its surface; sigma ~ the block's own scale.
        return np.clip(block_xy + rng.normal(0.0, 60.0, size=2), 0.0, 512.0)
    return rng.uniform(0.0, 512.0, size=2).astype(np.float32)


def collect_episodes(
    cfg: Config,
    out_dir: Path | str,
    n_episodes: int = 64,
    max_steps: int = 60,
    block_bias: float = 0.8,
    seed: int = 0,
    verbose: bool = True,
) -> int:
    """Roll out the biased random policy and write ``episode_*.npz``.

    ``max_steps`` counts *model* steps; each spans ``cfg.frame_skip`` env ticks.
    Returns the number of episodes written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    rng = np.random.default_rng(seed)
    skip = cfg.frame_skip
    written = 0

    for ep_i in range(n_episodes):
        obs, info = env.reset(seed=int(seed * 100_000 + ep_i))
        frames, chunks, states = [], [], []

        for _ in range(max_steps):
            frames.append(obs["pixels"].copy())
            states.append(env_state(info, obs))

            agent = np.asarray(info.get("pos_agent", obs["agent_pos"]), dtype=np.float32)
            target = _sample_target(rng, info, block_bias)
            chunk = _waypoints(agent, target, skip).astype(np.float32)   # [skip, 2]
            chunks.append(normalize_action(chunk, cfg))

            terminated = False
            for tick in range(skip):
                obs, _, terminated, truncated, info = env.step(chunk[tick])
                if terminated or truncated:
                    break
            if terminated:
                break

        if len(frames) <= cfg.seq_len:
            continue    # too short to yield a single training window

        Episode(
            frames=np.stack(frames).astype(np.uint8),
            actions=np.stack(chunks).astype(np.float32),
            states=np.stack(states).astype(np.float32),
            meta={"env": ENV_ID, "frame_skip": skip, "seed": int(seed * 100_000 + ep_i),
                  "state_names": list(STATE_NAMES)},
        ).save(out_dir / f"episode_{ep_i:05d}.npz")
        written += 1
        if verbose and (ep_i + 1) % 10 == 0:
            print(f"  collected {ep_i + 1}/{n_episodes} episodes")

    env.close()
    return written


class PushTRunner:
    """Thin closed-loop wrapper: steps the env with **normalised** action chunks.

    Used by the AdaJEPA evaluation loop so the planner never has to know about
    pixel units.
    """

    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.env = make_env(cfg)
        self.seed = seed
        self.obs: Optional[dict] = None
        self.info: dict = {}

    def reset(self) -> np.ndarray:
        self.obs, self.info = self.env.reset(seed=self.seed)
        return self.obs["pixels"].copy()

    def step(self, chunk_norm: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
        """chunk_norm: [frame_skip, 2] in [-1, 1]. Returns (frame, done, info)."""
        chunk = denormalize_action(chunk_norm, self.cfg)
        done = False
        for tick in range(self.cfg.frame_skip):
            self.obs, _, terminated, truncated, self.info = self.env.step(chunk[tick])
            if terminated or truncated:
                done = True
                break
        return self.obs["pixels"].copy(), done, self.info

    def close(self) -> None:
        self.env.close()
