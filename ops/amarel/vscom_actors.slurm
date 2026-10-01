#!/bin/bash
#SBATCH --job-name=sokubot-vscom-actors
#SBATCH --partition=main
#SBATCH --time=3-00:00:00
#SBATCH --output=%x.%A_%a.out
#SBATCH --exclude=halk[0001-0159]
#
# Real-game PPO actors: each array task runs PROCS actor processes of GAMES games at CPG cores per
# game, all feeding the learner named in RUN_DIR/learner.addr. Size the allocation to match:
#
#   cd ~/sokubot-runs && RUN_DIR=~/sokubot-runs/vscom-ppo-s0 PROCS=2 GAMES=8 CPG=2 SEED=0 \
#       sbatch --array=0-23 --cpus-per-task=34 --mem=24000 ~/SokuBot/ops/amarel/vscom_actors.slurm
#
# (cpus = PROCS*GAMES*CPG + 2 for the Python processes; ~0.83 GB RAM per game was measured, plus
# ~1 GB per process.) Several processes per node rather than one, so one interpreter's lock is not
# shared by every game on the node. Actors are stateless: a task that dies costs its games, nothing
# else, and resubmitting the array adds them back. They exit when the learner reports DONE.
set -uo pipefail
cd ~/SokuBot
: "${RUN_DIR:?set RUN_DIR}"
P=${PROCS:-2}; G=${GAMES:-8}; C=${CPG:-2}; S=${SEED:-0}
t=${SLURM_ARRAY_TASK_ID:-0}
echo "== $(date -Is) task $t on $(hostname): $P x $G games at $C cores | cpus $(nproc) | $(cat DEPLOYED)"
echo "== sfe $(cat ~/sfe/DEPLOYED) | dll $(sha256sum ~/sfe-vscom-dll/SokuFrameExtractor.dll | cut -c1-16)"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
pids=()
for i in $(seq 0 $((P - 1))); do
    # Game clones (~22 MB each, hundreds of them across a wide run) go to /scratch, not home.
    ~/venvs/sokubot/bin/python -m scripts.vscom_actor --run-dir "$RUN_DIR" --games "$G" \
        --cpus-per-game "$C" --game-offset $((i * G)) --seed $((S * 100000 + t * 100 + i)) \
        --work "/scratch/$USER/vscom-work/$(basename "$RUN_DIR")/${t}_$i" \
        > "$RUN_DIR/actor-${SLURM_ARRAY_JOB_ID:-x}_${t}_$i.log" 2>&1 &
    pids+=($!)
    sleep 20          # staggered: simultaneous first launches exit early more often
done
for p in "${pids[@]}"; do wait "$p"; done
echo "== $(date -Is) task $t done"
