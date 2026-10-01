#!/bin/bash
#SBATCH --job-name=sokubot-vscom-scale
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=120000
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --exclude=halk[0001-0159]
#
# Before sizing the real-game PPO run: what does a game cost, and do all 20 COM characters work?
#
#   cd ~/sokubot-runs && sbatch ~/SokuBot/ops/amarel/vscom_scale_probe.slurm
#
# 1. chars    random opponent per match (SFE_P2_CHAR=-1), 8 games x 4 cores, 48 matches. Each match's
#             announced COM character is checked against the LIVE character object, with both
#             decks' card counts. A character that crashes or loads no deck shows up here.
# 2. c2g16    16 games x 2 cores, one Python process: decisions/s per game and per core.
# 3. c1g32    32 games x 1 core.
# 4. c2g32    32 games x 2 cores: how one Python process holds up at 32 game threads.
# Peak RAM (sum over every process of this user on the node) is sampled for each run.
set -uo pipefail
cd ~/SokuBot
PY=~/venvs/sokubot/bin/python
W=~/sfe-vscom-work/scale-${SLURM_JOB_ID}
mkdir -p "$W"
echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname) | SokuBot $(cat DEPLOYED) | sfe $(cat ~/sfe/DEPLOYED)"
echo "== dll $(sha256sum ~/sfe-vscom-dll/SokuFrameExtractor.dll | cut -c1-16) | cpus $(nproc)"

run() {   # name, then vscom_play arguments
    local name=$1; shift
    local peak=0
    ( while true; do ps -u "$USER" -o rss= | awk '{s+=$1} END {print s}'; sleep 5; done ) \
        > "$W/$name.rss" &
    local sampler=$!
    local t0=$(date +%s)
    $PY -m scripts.vscom_play --agent random --level 3 --work "$W/$name" --out "$W/$name.json" "$@" \
        > "$W/$name.log" 2>&1
    kill $sampler 2>/dev/null
    peak=$(sort -n "$W/$name.rss" | tail -1)
    $PY - "$W/$name.json" "$name" "$peak" "$(( $(date +%s) - t0 ))" "$@" <<'EOF'
import json, sys
j = json.loads(open(sys.argv[1]).read()); name, peak, secs = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
args = sys.argv[5:]
games = int(args[args.index("--games") + 1]); cpg = int(args[args.index("--cpus-per-game") + 1])
dps = j["decisions_per_s"]
print(f"{name:<8} {games:>3} games x {cpg} cores: {dps:7.1f} decisions/s = {dps / games:5.1f} per game "
      f"= {dps / (games * cpg):5.2f} per core | peak RAM {peak / 2**20:5.1f} GB = "
      f"{peak / 2**20 / games:4.2f} GB per game | {j['matches']} matches in {secs}s | errors {len(j['errors'])}")
EOF
}

run chars  --p2-char -1 --games 8  --cpus-per-game 4 --matches 48
run c2g16  --p2-char -1 --games 16 --cpus-per-game 2 --matches 10000 --seconds 420
run c1g32  --p2-char -1 --games 32 --cpus-per-game 1 --matches 10000 --seconds 420
run c2g32  --p2-char -1 --games 32 --cpus-per-game 2 --matches 10000 --seconds 420

echo "== per-character check (run 'chars'): announced COM char vs the live object, deck sizes, results"
$PY - "$W/chars" <<'EOF'
import json, re, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
start = re.compile(r"vscom: start p1 char (\d+) deck \d+ \((\d+) cards\) vs COM char (\d+) \((\d+) cards\)")
live = re.compile(r"live chars p1 (\d+) p2 (\d+)")
rows = defaultdict(lambda: {"matches": 0, "live_mismatch": 0, "p2_cards": set(), "p1_cards": set()})
for log in sorted(root.glob("game*/modules/SokuFrameExtractor/SokuFrameExtractor.log")):
    want = None
    for line in log.read_text(errors="replace").splitlines():
        m = start.search(line)
        if m:
            want = (int(m.group(1)), int(m.group(3)))
            r = rows[want[1]]; r["matches"] += 1
            r["p1_cards"].add(int(m.group(2))); r["p2_cards"].add(int(m.group(4)))
            continue
        m = live.search(line)
        if m and want is not None:
            if (int(m.group(1)), int(m.group(2))) != want:
                rows[want[1]]["live_mismatch"] += 1
            want = None
rounds = [json.loads(l) for l in open(root.parent / "chars.rounds.jsonl")]
won = defaultdict(lambda: [0, 0])
for r in rounds:
    won[r["opponent"]][0] += r["won"]; won[r["opponent"]][1] += 1
print(f"{'COM char':>8} {'matches':>8} {'live!=announced':>16} {'COM cards':>10} {'P1 cards':>9} {'P1 rounds won':>14}")
for c in sorted(rows):
    r = rows[c]
    print(f"{c:>8} {r['matches']:>8} {r['live_mismatch']:>16} {str(sorted(r['p2_cards'])):>10} "
          f"{str(sorted(r['p1_cards'])):>9} {won[c][0]:>7}/{won[c][1]}")
print(f"characters seen: {len(rows)} of 20")
EOF
echo "== $(date -Is) done"
