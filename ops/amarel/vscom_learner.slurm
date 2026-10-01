#!/bin/bash
#SBATCH --job-name=sokubot-vscom-learner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x.%j.out
#
# The real-game PPO learner (scripts/train_vscom_ppo.py). Actors (vscom_actors.slurm) find it
# through <run-dir>/learner.addr. Every argument goes to the learner:
#
#   cd ~/sokubot-runs && sbatch --job-name=vscom-learner-s0 ~/SokuBot/ops/amarel/vscom_learner.slurm \
#       --run-dir ~/sokubot-runs/vscom-ppo-s0 --init ~/sokubot-runs/vscom-arms/init-s0.pt \
#       --button-rates ~/sokubot-runs/button_rates.json --seed 0
#
# GPU jobs are preempted: SIGUSR1 (or SIGTERM) -> latest.pt, exit 99 -> requeued here; the new
# process resumes from latest.pt with a new nonce, and the actors reconnect to it.
set -uo pipefail
cd ~/SokuBot
echo "== $(date -Is) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} on $(hostname)"
echo "== deployed: $(cat DEPLOYED 2>/dev/null || echo UNKNOWN)"
nvidia-smi ${CUDA_VISIBLE_DEVICES:+--id=$CUDA_VISIBLE_DEVICES} --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
srun ~/venvs/sokubot/bin/python -m scripts.train_vscom_ppo "$@"
rc=$?
echo "== $(date -Is) learner exited ${rc}"
if [ "$rc" -eq 99 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_REQUEUES:-30}" ]; then
    scontrol requeue "${SLURM_JOB_ID}" && echo "== requeued"
fi
exit "$rc"
