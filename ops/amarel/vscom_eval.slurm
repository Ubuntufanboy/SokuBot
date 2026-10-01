#!/bin/bash
#SBATCH --job-name=sokubot-vscom-eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=10
#SBATCH --mem=8000
#SBATCH --time=04:00:00
#SBATCH --output=%x.%A_%a.out
#SBATCH --exclude=halk[0001-0159]
#
# A fixed evaluation schedule against the COM: one array task per (policy, level, COM character)
# line of $PLAN ("name path level char"), MATCHES matches each, over 2 games. Then
#   python -m scripts.vscom_eval_report $OUT
# for the per-arm tables (Wilson intervals, per character, milestone 2).
#
#   cd ~/sokubot-runs && PLAN=~/sokubot-runs/vscom-eval/plan.txt OUT=~/sokubot-runs/vscom-eval \
#       sbatch --array=1-120 ~/SokuBot/ops/amarel/vscom_eval.slurm
set -uo pipefail
# Clones hard-link the base's .dat archives, which only works on ONE filesystem: the base must
# live on /scratch with the clones (ops/amarel/README.md: make it once with cp -a ~/sfe-game).
[ -f "/scratch/$USER/sfe-game-base/th123.exe" ] || { echo "no /scratch/$USER/sfe-game-base"; exit 2; }
cd ~/SokuBot
: "${PLAN:?}"; : "${OUT:?}"
M=${MATCHES:-10}
read -r name path level char < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAN")
tag="${name}_L${level}_c${char}"
echo "== $(date -Is) $tag on $(hostname) | $(cat DEPLOYED)"
mkdir -p "$OUT"
~/venvs/sokubot/bin/python -m scripts.vscom_play --agent policy --policy "$path" --level "$level" \
    --p2-char "$char" --games 2 --matches "$M" --cpus-per-game 4 \
    --game-base "/scratch/$USER/sfe-game-base" \
    --work "/scratch/$USER/vscom-eval/$tag" --out "$OUT/$tag.json" 2>&1 | tail -3
