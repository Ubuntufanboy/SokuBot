#!/bin/bash
#SBATCH --job-name=sokubot-sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH --time=2-00:00:00
#SBATCH --output=%x.%j.out
#
# Stage 2: train the state simulator on the regenerated sidecars. Every argument goes straight to
# scripts.train_state_dynamics:
#
#   cd ~/sokubot-runs && sbatch ~/SokuBot/ops/amarel/sim.slurm \
#       --corpus /scratch/$USER/statecap --replays 0 --cache ~/sokubot-runs/sim_corpus.npz \
#       --out ~/sokubot-runs/sim-h8 --proj-feedback raw --move-dim 0
#
# Two settings are NOT defaults here, and both follow from what the arena can and should roll:
#   --proj-feedback raw   the fixed projectile feedback. Every earlier simulator was trained through
#                         the squashing path that destroys the sign of a bullet's velocity; retraining
#                         from scratch is the moment to stop (09-08 plan, A9).
#   --move-dim 0          the state bank carries no move ids, so the PPO arena cannot roll a
#                         move-identity simulator and refuses one (state_arena.py).
# The trainer has no resume, so the time limit is generous rather than requeued; 96 GB is for the
# parse of ~12k replays (--replays 0 = all of them).
set -uo pipefail
cd ~/SokuBot
echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname)"
echo "== deployed: $(cat DEPLOYED 2>/dev/null || echo 'UNKNOWN (run ops/amarel/sync.sh)')"
# Only the GPU this job was given: the node's others belong to other jobs.
nvidia-smi ${CUDA_VISIBLE_DEVICES:+--id=$CUDA_VISIBLE_DEVICES} --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
srun ~/venvs/sokubot/bin/python -m scripts.train_state_dynamics "$@"
rc=$?
echo "== $(date -Is) trainer exited ${rc}"
exit "$rc"
