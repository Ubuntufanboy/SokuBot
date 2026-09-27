#!/bin/bash
#SBATCH --job-name=sokubot-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x.%j.out
#
# Exercise the whole PPO path on a GPU node BEFORE any real data exists:
#
#   cd ~/sokubot-runs && sbatch ~/SokuBot/ops/amarel/smoke.slurm
#
# Synthetic simulator and bank (scripts/make_smoke_assets.py -- they model nothing), then a training
# run that deliberately stops itself at step 40 and exits 99, so this script requeues it and the
# second pass must resume from latest.pt and finish. A clean smoke log therefore shows: the unit
# tests passing on the node, a first pass ending "exit 99 ... requeued", and a second pass
# "resumed from ... at step 40" ending "done: 80 steps".
set -uo pipefail
cd ~/SokuBot
OUT=${SMOKE_DIR:-$HOME/sokubot-runs/smoke}
PY=~/venvs/sokubot/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
echo "== $(date -Is) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} on $(hostname)"
echo "== deployed: $(cat DEPLOYED 2>/dev/null || echo UNKNOWN)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
    $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    srun $PY -m pytest -q -p no:cacheprovider tests/test_ppo.py || exit 1
    rm -rf "$OUT"
fi
# Only on the first pass: regenerating would change the simulator's bytes, and the resume refuses a
# run whose simulator fingerprint changed.
[ -f "$OUT/sim.pt" ] || $PY -m scripts.make_smoke_assets --out "$OUT" --replays 40 --steps 400
srun $PY -m scripts.train_state_ppo --sim "$OUT/sim.pt" --bank "$OUT/bank.npz" --out "$OUT/ppo" \
    --steps 80 --stop-at-step 40 --starts 1024 --eval-every 20 --ckpt-every 20 --device cuda
rc=$?
echo "== $(date -Is) trainer exited ${rc}"
if [ "$rc" -eq 99 ] && [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ]; then
    scontrol requeue "${SLURM_JOB_ID}" && echo "== requeued"
fi
exit "$rc"
