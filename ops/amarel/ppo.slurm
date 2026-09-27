#!/bin/bash
#SBATCH --job-name=sokubot-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x.%j.out
#
# PPO self-play in imagination on one GPU. Submit from the directory the log should land in:
#
#   cd ~/sokubot-runs && sbatch ~/SokuBot/ops/amarel/ppo.slurm \
#       --sim ~/sokubot-runs/sim/sim.pt --bank ~/sokubot-runs/bank_state.npz \
#       --out ~/sokubot-runs/ppo-seed0 --seed 0 --steps 20000
#
# Every argument goes straight to scripts.train_state_ppo.
#
# TIME LIMITS ARE ROUTINE, NOT FAILURES. Five minutes before the limit Slurm sends SIGUSR1 to the
# srun step; the trainer finishes its step, writes latest.pt and exits 99, and this script requeues
# the job. The requeued job runs the same command, finds latest.pt in --out and continues exactly
# where it stopped (tests/test_ppo.py checks the result is bit-identical to never stopping). The job
# id, and so the log file, are kept across requeues. MAX_REQUEUES (default 30) bounds a loop.
set -uo pipefail
cd ~/SokuBot
echo "== $(date -Is) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} on $(hostname)"
echo "== deployed: $(cat DEPLOYED 2>/dev/null || echo 'UNKNOWN (run ops/amarel/sync.sh)')"
# Only the GPU this job was given: the node's others belong to other jobs.
nvidia-smi ${CUDA_VISIBLE_DEVICES:+--id=$CUDA_VISIBLE_DEVICES} --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
srun ~/venvs/sokubot/bin/python -m scripts.train_state_ppo "$@"
rc=$?
echo "== $(date -Is) trainer exited ${rc}"
if [ "$rc" -eq 99 ]; then
    if [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_REQUEUES:-30}" ]; then
        scontrol requeue "${SLURM_JOB_ID}" && echo "== requeued"
    else
        echo "== NOT requeued: restart count ${SLURM_RESTART_COUNT} reached MAX_REQUEUES"
    fi
fi
exit "$rc"
