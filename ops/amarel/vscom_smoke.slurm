#!/bin/bash
#SBATCH --job-name=sokubot-vscom-smoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=20
#SBATCH --mem=32000
#SBATCH --time=01:30:00
#SBATCH --output=%x.%j.out
#SBATCH --exclude=halk[0001-0159]
#
# The vs-COM environment, end to end on real games, plus the check that moving the capture CSV's
# row writer into a shared function did not change a single byte of the sidecars.
#
#   cd ~/sokubot-runs && sbatch ~/SokuBot/ops/amarel/vscom_smoke.slurm
#
# Needs ~/sfe synced, the vs-COM DLL at ~/sfe-vscom-dll/, ~/sfe-game with the CURRENT (pre-refactor)
# DLL, and ~/SokuBot synced.
#
# PART A -- byte identity. The shortest replay is captured three times: twice with the deployed DLL
# (replay capture's own determinism, never measured before) and once with the new one. A new-vs-old
# difference only means something if old-vs-old has none.
#
# PART B -- real matches against the Lunatic COM, Cirno vs Cirno: the neutral and random floors,
# then the two PPO seeds trained in the stamped simulator (sim_hold.pt) -- the first real-game
# number for any of them.
set -uo pipefail
cd ~/SokuBot
echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname)"
echo "== SokuBot deployed: $(cat DEPLOYED 2>/dev/null || echo UNKNOWN) | sfe: $(cat ~/sfe/DEPLOYED 2>/dev/null || echo UNKNOWN)"
OLD=~/sfe-game/modules/SokuFrameExtractor/SokuFrameExtractor.dll
NEW=~/sfe-vscom-dll/SokuFrameExtractor.dll
echo "== old dll $(sha256sum $OLD | cut -c1-16) | new dll $(sha256sum $NEW | cut -c1-16)"
PY=~/venvs/sokubot/bin/python
W=~/sfe-vscom-work/smoke-${SLURM_JOB_ID}
LOCAL=${TMPDIR:-/tmp}/vscom-smoke-${SLURM_JOB_ID}
mkdir -p "$W" "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT

# ---------------------------------------------------------------- A: byte identity
REP=$(ls -S ~/sfe-replays/*.rep | tail -1)
mkdir -p "$LOCAL/reps" && cp "$REP" "$LOCAL/reps/"
echo "== A: capturing $(basename "$REP") ($(stat -c %s "$REP") bytes) three times"
i=0
for tag in old1 old2 new; do
    g=$W/game-$tag
    ~/sfe/ops/bwrap/clone-game.sh ~/sfe-game "$g" >/dev/null
    [ "$tag" = new ] && cp "$NEW" "$g/modules/SokuFrameExtractor/SokuFrameExtractor.dll"
    SFE_GAME=$g SFE_PREFIX=$LOCAL/prefix-$tag SFE_OUT=$W/cap-$tag SFE_REPLAYS=$LOCAL/reps \
    SFE_CPU_BLOCK=$i SFE_DISPLAY_BASE=$((90 + 10 * i)) \
        ~/sfe/ops/bwrap/sfe-bwrap /app/docker/entrypoint.sh --replay-dir /replays --out /out \
            --shard 0/1 --no-video --gzip-csv --cpus 4 --timeout 900 --min-free-gb 2 --limit 1 \
        > "$W/cap-$tag.log" 2>&1 &
    i=$((i + 1))
done
wait
for tag in old1 old2 new; do
    f=$(find "$W/cap-$tag" -name inputs.csv.gz | head -1)
    echo "   $tag: ${f:-NO CAPTURE} $( [ -n "$f" ] && zcat "$f" | wc -l) lines $( [ -n "$f" ] && zcat "$f" | sha256sum | cut -c1-16)"
done
o1=$(find "$W/cap-old1" -name inputs.csv.gz | head -1); o2=$(find "$W/cap-old2" -name inputs.csv.gz | head -1)
nw=$(find "$W/cap-new" -name inputs.csv.gz | head -1)
if [ -n "$o1" ] && [ -n "$o2" ] && [ -n "$nw" ]; then
    echo "   old1 vs old2 differing lines: $(cmp -l <(zcat "$o1") <(zcat "$o2") 2>/dev/null | wc -l) bytes"
    echo "   old1 vs new  differing lines: $(cmp -l <(zcat "$o1") <(zcat "$nw") 2>/dev/null | wc -l) bytes"
    diff <(zcat "$o1" | head -1) <(zcat "$nw" | head -1) >/dev/null && echo "   headers identical" || echo "   HEADERS DIFFER"
fi

# ---------------------------------------------------------------- B: real matches
run() {   # name, then vscom_play arguments
    local name=$1; shift
    echo "== B: $name"
    srun --exclusive -n1 -c16 $PY -m scripts.vscom_play --work "$W/$name" --level 3 \
        --out "$W/$name.json" "$@" 2>&1 | tail -40
}
run neutral --agent neutral --games 2 --matches 4
run random  --agent random  --games 2 --matches 4
for s in 0 1; do
    run ppo-hold-s$s --agent policy --policy ~/sokubot-runs/ppo-hold-s$s/policy.pt --games 4 --matches 12
done
echo "== $(date -Is) done"
for f in "$W"/*/game*.log; do echo "--- $f"; tail -3 "$f"; done 2>/dev/null | tail -40
for k in 0 1 2 3; do
    l=$W/ppo-hold-s0/game$k/modules/SokuFrameExtractor/SokuFrameExtractor.log
    [ -f "$l" ] && { echo "--- module log game $k"; grep -E "agent|vscom: match|round over|FAILED|ERROR" "$l" | tail -8; }
done
