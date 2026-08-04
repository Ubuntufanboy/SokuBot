"""Train the Soku policy with GRPO inside the imagined world.

    python -m scripts.train_grpo --steps 20000 --horizon 24

The world model and the reward probe are frozen; only the policy learns. It
starts from random weights: the imagined environment runs several thousand times
faster than real Hisoutensoku (see scripts/bench_arena.py), so the sample
inefficiency of starting cold costs minutes rather than the days it would cost
against the real game.

Start states are real. Each is a window of encoded gameplay drawn from the
corpus, so the policy is always asked to continue a situation that actually
happened rather than one imagination invented.

HORIZON
-------
scripts/horizon_ablation.py measured how long a rollout stays readable. With the
probe calibrated on predictor outputs, health holds R^2 ~0.83 out to 32 steps
and 0.79 at 48; the combo channel decays much faster, from 0.31 at one step to
0.18 by 24 and roughly nothing by 48. Damage is computed from health, so the
default horizon of 24 steps (1.6 s) sits where the dominant reward term is still
at its ceiling, and the combo weight is small because that term is the one the
ablation says to distrust.

Spirit is not used at all. It is not decodable from the latent even from real
frames (R^2 0.04 / -0.02), which is a property of the representation and not of
the rollout, so guard-crush and spell-cost rewards are switched off rather than
computed from noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sokubot.config import Config
from sokubot.data.soku import decode_frames, read_actions
from sokubot.model.world_model import LeWorldModel
from sokubot.probe import LinearProbe
from sokubot.rl.grpo import (GRPOConfig, ImaginedArena, PolicyOpponent, ProbeHead,
                             ReplayOpponent, SnapshotPool, group_advantages,
                             grpo_loss)
from sokubot.rl.policy import SokuPolicy
from sokubot.rl.reward import RewardConfig
from scripts.horizon_ablation import capture_paths, encode_all


def model_fingerprint(model) -> str:
    """A cheap identity for the weights that produced a set of latents.

    Only the first kilobyte of each tensor is hashed, which is plenty to tell
    two checkpoints apart and fast enough to run on every load.
    """
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes()[:1024])
    return h.hexdigest()[:16]


def build_bank(rows, manifest, model, cfg, device, n_replays, bank_path: Path):
    """Encode gameplay into a bank of start states, cached on disk.

    Stores latents as float16 and buttons as uint8: the bank is read every step
    of training and never differentiated through, so the precision it would cost
    to keep full-width buys nothing.

    The cache records which weights encoded it. Latents only mean anything
    relative to the encoder that produced them, and a bank silently reused across
    a fine-tune would feed the policy start states from a different space --
    wrong in a way that trains happily and reports nothing.
    """
    fp = model_fingerprint(model)
    if bank_path.exists():
        d = np.load(bank_path)
        cached = str(d["fingerprint"]) if "fingerprint" in d else "<unrecorded>"
        if cached == fp:
            print(f"bank: {len(d['z'])} latents from cache", flush=True)
            return d["z"], d["a"], d["ep"]
        print(f"bank: rebuilding, cache was encoded by {cached} and this model "
              f"is {fp}", flush=True)
    zs, acts, ep = [], [], []
    for k, r in enumerate(rows[:n_replays]):
        video, inputs = capture_paths(r, manifest)
        skip = cfg.frame_skip
        obs = list(decode_frames(video, cfg.image_size, skip))
        raw = read_actions(inputs)
        D = min(len(obs), len(raw) // skip)
        if D < 64:
            continue
        z = encode_all(model, np.stack(obs[:D]), device)
        chunks = np.stack([raw[d * skip : d * skip + skip] for d in range(D)])
        ep.append(np.full(D, len(zs), dtype=np.int32))
        zs.append(z.astype(np.float16))
        acts.append(chunks.astype(np.uint8))
        if (k + 1) % 10 == 0:
            print(f"   bank {k+1}/{min(n_replays, len(rows))}", flush=True)
    Z, A, E = np.concatenate(zs), np.concatenate(acts), np.concatenate(ep)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bank_path, z=Z, a=A, ep=E, fingerprint=fp)
    print(f"bank: {len(Z)} latents from {len(zs)} replays -> {bank_path}", flush=True)
    return Z, A, E


def valid_starts(ep: np.ndarray, history: int, horizon: int) -> np.ndarray:
    """Indices whose context and future both stay inside one replay."""
    ok = []
    edges = np.flatnonzero(np.diff(ep)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(ep)]):
        a, b = lo + history - 1, hi - horizon - 1
        if b > a:
            ok.append(np.arange(a, b))
    return np.concatenate(ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=Path("/root/corpus"))
    ap.add_argument("--wm", type=Path, default=Path("/root/ckpt/best.pt"))
    ap.add_argument("--probe", type=Path, default=Path("/root/horizon2/reward_probe.npz"))
    ap.add_argument("--out", type=Path, default=Path("/root/grpo"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--group-size", type=int, default=16,
                    help="the baseline is the group mean, so a larger group "
                         "estimates it better and samples a wider spread of "
                         "behaviour from one start")
    ap.add_argument("--starts", type=int, default=256, help="groups per batch")
    ap.add_argument("--bank-replays", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.5e-4,
                    help="the 3e-4 this defaulted to silently overrode the "
                         "1e-4 in GRPOConfig and drove KL to 54 with 43%% of "
                         "samples outside the trust region")
    ap.add_argument("--replay-share", type=float, default=0.0,
                    help="fraction of rollouts facing recorded human input; it is "
                         "open-loop, so this defaults off")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-starts", type=int, default=1024)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    blob = torch.load(a.wm, map_location=a.device, weights_only=False)
    cfg: Config = blob["cfg"]
    cfg.device = a.device
    wm = LeWorldModel(cfg).to(a.device)
    wm.load_state_dict(blob["model"])
    wm.eval()

    d = np.load(a.probe, allow_pickle=True)
    probe = LinearProbe(zmu=d["zmu"], zsd=d["zsd"], ymu=d["ymu"], ysd=d["ysd"],
                        W=d["W"], names=[str(x) for x in d["names"]])
    print(f"reward probe: alpha {float(d['alpha']):g}, targets {probe.names}",
          flush=True)

    # Spirit is unmeasurable in the latent, so the terms that depend on it are
    # off. Combo survives only at short horizons, hence the reduced weight.
    # Weights checked against a random policy's term breakdown before launching.
    # A damage exchange over one rollout is worth about 0.1, so the shaping terms
    # have to sit well under that or the agent can farm them instead of fighting:
    # flying at 0.005/step was worth 0.12 a rollout, more than the damage it was
    # meant to nudge toward.
    # The idle penalty is set against a specific, measured hazard rather than by
    # feel. The world model credits "P1 presses nothing" with roughly 0.126 of a
    # bar of damage to P2, because in the corpus a silent frame is usually a
    # player in hitstun who cannot input -- the correlation is real and the
    # causality is backwards. At 0.010/step the penalty over a 16-step rollout is
    # 0.16, barely covering that; at 0.020 it is 0.32, so standing still is
    # clearly unprofitable even when the model is wrong about why.
    rcfg = RewardConfig(combo=0.10, crush=0.0, whiff=-0.25, spell_cost_min=1e9,
                        flying=0.0015, idle=-0.020)
    gcfg = GRPOConfig(horizon=a.horizon, group_size=a.group_size,
                      starts_per_batch=a.starts, lr=a.lr, reward=rcfg)

    manifest = a.corpus / "train" / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    rng.shuffle(rows)
    Z, A, E = build_bank(rows, manifest, wm, cfg, a.device, a.bank_replays,
                         a.out / "bank.npz")
    starts = valid_starts(E, cfg.history, a.horizon)
    print(f"{len(starts)} valid start states", flush=True)

    Zt = torch.from_numpy(Z).to(a.device)
    At = torch.from_numpy(A).to(a.device)

    policy = SokuPolicy(cfg.latent_dim, cfg.history, cfg.action_ticks).to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
    arena = ImaginedArena(wm, ProbeHead(probe).to(a.device), gcfg,
                          cfg.history, cfg.action_ticks)
    pool = SnapshotPool(gcfg)
    G, S = a.group_size, a.starts
    B = G * S
    print(f"policy {sum(p.numel() for p in policy.parameters())/1e6:.2f}M | "
          f"{S} groups x {G} = {B} rollouts x {a.horizon} steps", flush=True)

    # ---- a yardstick that self-play cannot provide -------------------------
    # In self-play both chairs hold the same policy, so one side's gain is the
    # other's loss and the mean return is zero however well or badly the policy
    # plays. Training reward is therefore not a progress signal at all. Progress
    # has to be measured against a *fixed* opponent, on *fixed* start states, so
    # the only thing that changes between evaluations is the policy.
    import copy
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    eval_rng = np.random.default_rng(12345)
    eval_idx = torch.from_numpy(eval_rng.choice(starts, size=a.eval_starts)).to(a.device)

    @torch.no_grad()
    def evaluate() -> dict:
        """Net damage of `policy` against the frozen initial policy.

        Each start is played twice with the sides swapped, so a policy cannot
        score by exploiting whichever chair the world model happens to favour.
        """
        off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
        out = {}
        for tag, s0 in (("p1", 0), ("p2", 1)):
            side = torch.full((a.eval_starts,), s0, device=a.device, dtype=torch.long)
            zc = Zt[eval_idx[:, None] + off[None, :]].float()
            ah = At[eval_idx[:, None] + off[None, :-1]].float()
            tr = arena.rollout(zc, ah, side, policy, PolicyOpponent(reference))
            al = tr["alive"]
            n = al.sum().clamp(min=1)
            out[f"{tag}_dealt"] = float((tr["terms"]["dealt"] * al).sum() / n)
            out[f"{tag}_taken"] = float((tr["terms"]["taken"] * al).sum() / n)
            out[f"{tag}_ret"] = float(tr["reward"].sum(1).mean())
        # dealt is positive and taken negative, so their sum is the net exchange.
        out["net"] = ((out["p1_dealt"] + out["p1_taken"]) +
                      (out["p2_dealt"] + out["p2_taken"])) / 2
        return out

    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        # Each group shares one start state and one side.
        idx = torch.from_numpy(rng.choice(starts, size=S)).to(a.device)
        idx = idx.repeat_interleave(G)
        side = torch.randint(0, 2, (S,), device=a.device).repeat_interleave(G)
        off = torch.arange(cfg.history, device=a.device) - (cfg.history - 1)
        z_ctx = Zt[idx[:, None] + off[None, :]].float()
        a_hist = At[idx[:, None] + off[None, :-1]].float()

        if rng.random() < a.replay_share:
            fut = At[idx[:, None] + torch.arange(a.horizon, device=a.device)[None, :]]
            opponent, kind = ReplayOpponent(fut.float()), "replay"
        else:
            snap = pool.sample(rng)
            use_snap = snap is not None and rng.random() < 0.5
            opponent = PolicyOpponent(snap if use_snap else policy)
            kind = "snapshot" if use_snap else "self"

        traj = arena.rollout(z_ctx, a_hist, side, policy, opponent)
        adv = group_advantages(traj["reward"], traj["alive"], G, gcfg.gamma,
                               scale=gcfg.advantage_scale)
        flat_side = side[:, None].expand(B, a.horizon).reshape(-1)
        # Frozen before any update: this is the policy the actions were actually
        # sampled from, which is what makes the ratio mean anything.
        with torch.no_grad():
            old_logp = policy.log_prob_of(
                traj["obs"].reshape(-1, cfg.history, cfg.latent_dim), flat_side,
                traj["mine"].reshape(-1, cfg.action_ticks, 10))[0].view(B, a.horizon)

        skipped = 0
        for _ in range(gcfg.epochs):
            loss, stats = grpo_loss(policy, traj, adv, old_logp, gcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), gcfg.grad_clip)
            # A single non-finite batch otherwise writes NaN into every weight,
            # and from there nothing recovers -- the first run died this way at
            # step 1400. Skipping costs one update.
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
        pool.maybe_add(policy, step)

        if step % a.log_every == 0 or step == 1:
            alive = traj["alive"]
            n = alive.sum().clamp(min=1)
            rec = {"step": step, "loss": float(loss.detach()), "opponent": kind,
                   "reward": float((traj["reward"] * alive).sum() / n),
                   "ret": float(traj["reward"].sum(1).mean()),
                   "grad_norm": float(gn), "alive_frac": float(alive.mean()),
                   "skipped": skipped,
                   **stats,
                   **{f"r_{k}": float((v * alive).sum() / n)
                      for k, v in traj["terms"].items()}}
            # How often the policy actually presses things. The world model
            # associates "nobody is inputting" with a player in hitstun -- in the
            # corpus that is what a silent frame usually means -- so it credits
            # standing still with damaging the opponent. A policy that discovers
            # that will converge on doing nothing, and it would look like a
            # rising reward with no other symptom. These rates make it visible.
            with torch.no_grad():
                m = traj["mine"]                              # [B,T,ticks,10]
                rec["press_rate"] = float(m.mean())           # per button per tick
                rec["attack_rate"] = float(m[..., 4:8].mean())
                rec["move_rate"] = float(m[..., :4].mean())
                # A decision step with nothing held at any tick. This is the one
                # the world model's inverted idle causality would drive upward.
                rec["idle_rate"] = float(
                    (m.amax(dim=2).amax(dim=-1) < 0.5).float().mean())
            if step % a.eval_every == 0 or step == 1:
                ev = evaluate()
                rec.update({f"eval_{k}": v for k, v in ev.items()})
                print(f"  [eval] step {step:6d} | net vs frozen init "
                      f"{ev['net']:+.5f} | as P1 {ev['p1_dealt']:+.4f}/"
                      f"{ev['p1_taken']:+.4f} | as P2 {ev['p2_dealt']:+.4f}/"
                      f"{ev['p2_taken']:+.4f} | idle {rec['idle_rate']:.3f} "
                      f"press {rec['press_rate']:.3f} atk {rec['attack_rate']:.3f}",
                      flush=True)
            # Damage is the term the whole reward is denominated in, so track
            # both halves of the exchange rather than only the net.
            rec["elapsed_h"] = (time.time() - t0) / 3600
            hist.append(rec)
            (a.out / "log.json").write_text(json.dumps(hist, indent=1))
            print(f"step {step:6d} | ret {rec['ret']:+8.4f} | dealt "
                  f"{rec['r_dealt']:+.4f} taken {rec['r_taken']:+.4f} | "
                  f"KL {rec['kl']:.4f} ent {rec['entropy']:.2f} "
                  f"clip {rec['clip_frac']:.3f} | vs {kind} | "
                  f"{rec['elapsed_h']:.2f}h", flush=True)

        if step % a.ckpt_every == 0:
            torch.save({"policy": policy.state_dict(), "cfg": cfg, "gcfg": gcfg,
                        "rcfg": rcfg, "step": step}, a.out / "policy.pt")

    torch.save({"policy": policy.state_dict(), "cfg": cfg, "gcfg": gcfg,
                "rcfg": rcfg, "step": a.steps}, a.out / "policy.pt")
    print(f"done in {(time.time()-t0)/3600:.2f} h -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
