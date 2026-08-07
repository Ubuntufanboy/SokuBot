"""The inference half of the live loop: frames in, button chunks out.

    python -m scripts.serve_policy \
        --wm ~/sokubot-art/best_bnfix.pt \
        --policy ~/sokubot-art/policy_best.pt \
        --prior ~/sokubot-art/action_prior.npz --threads 6

Runs on whatever machine can hold the model. Measured on the LAN box
(Ryzen 5 4500, CPU only, 6 threads): encoder 19.0 ms, one compensation step
3.7 ms, policy 1.0 ms -- **24.2 ms p50, 24.5 ms p90** against a 66.7 ms budget,
with p90 essentially equal to p50. The jitter that killed the cloud plan is
absent because nothing here crosses the internet.

THREE THINGS THIS DOES THAT ARE NOT OBVIOUS
-------------------------------------------

**1. It drops stale frames.** If two requests are queued, the older one is
already worthless: its slot has passed and the client will refuse the answer
anyway. Answering it also delays the fresh one, so a backlog is self-sustaining.
The server reads whatever is pending and encodes only the newest.

**2. It rolls the latent forward before asking the policy.** The policy was
trained with zero lag. Live, a frame is captured, sent, encoded and answered
while the game keeps running, so acting on the latent as observed means acting
on the past. The predictor is used exactly as `ImaginedArena.rollout` uses it --
`wm.predictor(z_win, wm.action_encoder(a_full))[:, -1]` -- to advance the state
`--steps` decisions, seeded with the chunks already in flight.

This is the best-evidenced use of the world model available:
`scripts/horizon_ablation.py` measures one-step rollout cosine at 0.9963 and
four-step at 0.9717, and `action_effect_test.py` puts the action signal at
r=0.55 at one step falling to r=0.09 at sixteen. **One or two steps is inside
that; four is not.** `--steps` is capped accordingly.

**3. It marginalises over the opponent.** The predictor is conditioned on both
players' twenty buttons and the human's presses are not observable -- the open
issue in `README.md`. The opponent's block is sampled from the corpus prior,
`--opponent-samples` times, and the predicted latents averaged. That is not a
new assumption: `train_grpo.py` rolled every training episode against
`PolicyOpponent(reference)`, a frozen policy initialised to those same
statistics, so this is the distribution the policy was optimised against.

WHAT THIS IS ALLOWED TO SEE
---------------------------
A JPEG of the game window, and which side the agent is playing. No game memory,
ever -- `docs/HANDOFF.md` section 8.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.live.protocol import (DEFAULT_PORT, chunk_bytes, pack_chunk,
                                   pack_reply, read_request)
from sokubot.model.world_model import LeWorldModel
from sokubot.rl.policy import SokuPolicy, to_joint

# The trustworthy horizon, from horizon_ablation.py. Past this the predictor's
# rollout stops being a prediction and starts being a blur.
MAX_STEPS = 2


class Inference:
    """Encoder + predictor + policy, with the rolling latent history."""

    def __init__(self, wm_path: Path, policy_path: Path, prior_path: Path | None,
                 *, steps: int, opponent_samples: int, threads: int,
                 device: str = "cpu"):
        torch.set_num_threads(threads)
        blob = torch.load(wm_path, map_location=device, weights_only=False)
        self.cfg: Config = blob["cfg"]
        self.wm = LeWorldModel(self.cfg).to(device)
        self.wm.load_state_dict(blob["model"])
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad_(False)

        pol_blob = torch.load(policy_path, map_location=device, weights_only=False)
        self.policy = SokuPolicy(self.cfg.latent_dim, self.cfg.history,
                                 self.cfg.action_ticks).to(device)
        self.policy.load_state_dict(pol_blob["policy"])
        self.policy.eval()
        self.trained_step = pol_blob.get("step")

        # The opponent model: a second policy pinned to the corpus statistics.
        # Its trunk is at initialisation, so it is state-independent by
        # construction -- which is what the reference policy in training was.
        self.opponent: SokuPolicy | None = None
        if prior_path is not None:
            pr = np.load(prior_path)
            self.opponent = SokuPolicy(self.cfg.latent_dim, self.cfg.history,
                                       self.cfg.action_ticks).to(device)
            self.opponent.set_action_prior(pr["lr"], pr["ud"], pr["btn"])
            self.opponent.eval()

        self.device = device
        self.steps = max(0, min(steps, MAX_STEPS))
        self.k = max(1, opponent_samples)
        self.reset()

    def reset(self) -> None:
        """Forget the latent history. Call between rounds or after a stall."""
        H, D = self.cfg.history, self.cfg.latent_dim
        self._z = torch.zeros(1, H, D, device=self.device)
        # H-1, not H. The predictor is conditioned on one action per latent in
        # its window, and the action for the *newest* latent is the one being
        # chosen now -- so the stored history is one shorter and the current
        # joint action is concatenated onto it. This matches
        # `ImaginedArena.rollout`, whose signature says `a_hist [B,H-1,ticks,20]`
        # in as many words; getting it wrong raises rather than mispredicting,
        # which is the one mercy here.
        self._a = torch.zeros(1, max(H - 1, 1), self.cfg.action_ticks, 20,
                              device=self.device)
        self._warm = 0

    @torch.inference_mode()
    def act(self, frame480: np.ndarray, side: int) -> tuple[np.ndarray, int]:
        """One 480x480 corpus-orientation frame -> [ticks, 10] for `side`."""
        import cv2
        # Bilinear, because that is the resampler every training sample went
        # through (data/soku.py). Lanczos here would sharpen edges the model
        # learned as soft.
        small = cv2.resize(frame480, (self.cfg.image_size, self.cfg.image_size),
                           interpolation=cv2.INTER_LINEAR)
        x = (torch.from_numpy(np.ascontiguousarray(small))
             .permute(2, 0, 1)[None].float().div_(255.0).to(self.device))
        z = self.wm.encoder(x)                                  # [1, D]

        self._z = torch.cat([self._z[:, 1:], z[:, None]], dim=1)
        self._warm = min(self._warm + 1, self.cfg.history)

        z_win = self._z
        a_win = self._a
        sd = torch.tensor([side], device=self.device, dtype=torch.long)

        # Latency compensation. Each step needs a joint action; ours is the
        # chunk we are about to commit to, which we do not have yet, so the
        # policy is asked first and its own output is fed forward. That is the
        # same order ImaginedArena uses.
        for _ in range(self.steps):
            mine = self.policy(z_win, sd, sample=True).actions
            theirs = self._sample_opponent(z_win, sd)
            zhat = self._predict(z_win, a_win, mine, theirs, sd)
            z_win = torch.cat([z_win[:, 1:], zhat[:, None]], dim=1)
            joint = to_joint(mine, theirs[0], sd)
            a_win = torch.cat([a_win[:, 1:], joint[:, None]], dim=1)

        out = self.policy(z_win, sd, sample=True)
        return out.actions[0].cpu().numpy(), self.steps

    def _sample_opponent(self, z_win, sd) -> torch.Tensor:
        """[k, 1, ticks, 10] samples of what the human might be pressing."""
        if self.opponent is None:
            return torch.zeros(self.k, 1, self.cfg.action_ticks, 10,
                               device=self.device)
        # The opponent's side is the other chair; the reference policy is
        # state-independent so this only matters for its side embedding.
        other = 1 - sd
        return torch.stack([self.opponent(z_win, other, sample=True).actions
                            for _ in range(self.k)])

    def _predict(self, z_win, a_win, mine, theirs, sd) -> torch.Tensor:
        """Average the predicted latent over the opponent samples."""
        zs = []
        for t in theirs:
            joint = to_joint(mine, t, sd)
            a_full = torch.cat([a_win, joint[:, None]], dim=1)
            zs.append(self.wm.predictor(z_win, self.wm.action_encoder(a_full))[:, -1])
        return torch.stack(zs).mean(0)

    def commit(self, chunk: np.ndarray, side: int,
               opponent: np.ndarray | None = None) -> None:
        """Record the chunk actually issued, so the next rollout is seeded right."""
        mine = torch.from_numpy(chunk)[None].to(self.device)
        theirs = (torch.from_numpy(opponent)[None].to(self.device)
                  if opponent is not None
                  else torch.zeros_like(mine))
        sd = torch.tensor([side], device=self.device, dtype=torch.long)
        joint = to_joint(mine, theirs, sd)
        self._a = torch.cat([self._a[:, 1:], joint[:, None]], dim=1)


def serve(inf: Inference, host: str, port: int, quiet: bool) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(2)
    nbytes = chunk_bytes(inf.cfg.action_ticks)
    print(f"serving on {host}:{port}  steps={inf.steps} "
          f"opponent_samples={inf.k} ticks={inf.cfg.action_ticks}", flush=True)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"  client {addr}", flush=True)
        inf.reset()
        n, dropped, t_sum = 0, 0, 0.0
        try:
            while True:
                frame_id, tick, side, jpeg = read_request(conn)
                # Drain anything already queued: only the newest frame is worth
                # answering, and answering an old one delays the new one too.
                conn.setblocking(False)
                try:
                    while True:
                        frame_id, tick, side, jpeg = read_request(conn)
                        dropped += 1
                except (BlockingIOError, OSError):
                    pass
                finally:
                    conn.setblocking(True)

                t0 = time.perf_counter()
                import cv2
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                     cv2.IMREAD_COLOR)[:, :, ::-1]
                chunk, steps = inf.act(np.ascontiguousarray(frame), side)
                inf.commit(chunk, side)
                dt = (time.perf_counter() - t0) * 1000
                t_sum += dt
                n += 1

                start = tick + steps * inf.cfg.frame_skip
                conn.sendall(pack_reply(frame_id, start, steps, pack_chunk(chunk)))
                if not quiet and n % 30 == 0:
                    print(f"    {n} decisions, mean {t_sum / n:5.1f} ms, "
                          f"{dropped} stale dropped", flush=True)
        except Exception as e:
            print(f"  client gone: {type(e).__name__}: {e}", flush=True)
        finally:
            conn.close()
            if n:
                print(f"  session: {n} decisions, mean {t_sum / n:.1f} ms, "
                      f"{dropped} stale dropped", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--prior", type=Path, default=None,
                    help="action_prior.npz; without it the opponent is assumed "
                         "to press nothing, which is a worse assumption than "
                         "the corpus statistics")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--steps", type=int, default=1,
                    help=f"latency compensation, capped at {MAX_STEPS}")
    ap.add_argument("--opponent-samples", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.steps > MAX_STEPS:
        print(f"--steps {a.steps} exceeds the world model's trustworthy "
              f"horizon; clamping to {MAX_STEPS}", file=sys.stderr)
    inf = Inference(a.wm, a.policy, a.prior, steps=a.steps,
                    opponent_samples=a.opponent_samples, threads=a.threads,
                    device=a.device)
    print(f"world model {a.wm.name}, policy {a.policy.name} "
          f"(step {inf.trained_step}), opponent "
          f"{'corpus prior' if inf.opponent else 'NONE'}")
    return serve(inf, a.host, a.port, a.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
