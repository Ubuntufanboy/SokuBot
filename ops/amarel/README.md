# SokuBot on Amarel

Training runs on Rutgers' Amarel cluster (Slurm). The laptop runs the game and the pad; it never
trains. Everything here assumes the VPN tunnel is up and `ssh amarel` works (see the `amarel` skill:
this laptop needs the userspace `ocproxy` tunnel, not a kernel tun device).

## Layout on Amarel

| path | what |
|---|---|
| `~/SokuBot` | this repo, tracked files only, deployed by `ops/amarel/sync.sh`; `DEPLOYED` says which commit |
| `~/venvs/sokubot` | Python 3.12 + torch (CUDA 12.6 build) + numpy + pytest, made with `uv` |
| `~/sokubot-runs/` | submit jobs from here; Slurm logs and run directories land here |
| `~/sfe`, `~/sfe-game`, `~/sfe-rootfs`, `~/sfe-replays` | the capture side, staged from SokuFrameExtractor (`ops/bwrap/`) |
| `/scratch/$USER/statecap/w<k>/` | regenerated state sidecars (purged eventually: keep the bank, not these) |

## The chain

PPO in imagination needs a simulator, and the simulator needs state sidecars. None of those existed
on 2026-09-26 (the box that made the last ones is gone), so the order is:

1. **State sidecars** (CPU, `main` partition) -- SokuFrameExtractor
   `ops/bwrap/amarel-statecap.slurm`, an array over all 12,310 source replays, rootless under bwrap:

       cd ~/sokubot-runs && sbatch --array=0-15 ~/sfe/ops/bwrap/amarel-statecap.slurm

2. **Simulator** (GPU) -- `ops/amarel/sim.slurm` -> `scripts.train_state_dynamics`.

3. **PPO self-play** (GPU) -- `ops/amarel/ppo.slurm` -> `scripts.train_state_ppo`. Resumable:
   time limits requeue the job and it continues where it stopped. Give it the simulator's corpus
   cache as `--bank` (no second parse); the bank stays in host RAM and only batches go to the GPU.

`ops/amarel/smoke.slurm` exercises stage 3 end to end on a GPU against synthetic stand-ins
(`scripts/make_smoke_assets.py`), including a deliberate stop, requeue and resume. Run it after any
change to the trainer, before a real job.

## Rules that cost something to learn

* **Game clones must share a filesystem with their base.** `clone-game.sh` hard-links the ~2 GB of
  `.dat` archives; across filesystems it copies them. vs-COM work lives on /scratch, so make the base
  there once: `cp -a ~/sfe-game /scratch/$USER/sfe-game-base`. Cloning from the home base put ~2 TB
  on /scratch on 2026-10-01 before the quota stopped it; `vscom_play` now refuses a cross-device clone.

* **Deploy before submitting**, and read the `deployed:` line at the top of the job log. A committed
  change that was never synced runs the old code, silently.
* **Two seeds** for anything compared (`--seed 0` and `--seed 1`, separate `--out`).
* **Checkpoints load to CPU.** A resume that loaded to the GPU died in `torch.set_rng_state`; only the
  GPU smoke run could see it.
* Report job state (running, pending, finished) whenever work touches Amarel: an allocation that sits
  idle still spends the group's fair share.
