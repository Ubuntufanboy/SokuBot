#!/bin/bash
#SBATCH --job-name=sokubot-vscom-compare
#SBATCH --partition=main
#SBATCH --cpus-per-task=52
#SBATCH --mem=48000
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --exclude=halk[0001-0159]
#
# Does simulator training carry into the real game? Three arms against the Lunatic COM (Cirno
# mirror), same harness, two seeds each, run side by side on one node:
#
#   init   the untrained starting policy (the run's frozen `reference`), i.e. the corpus prior
#   full   PPO in the broken simulator (sim.pt: the dead `ax` channel fed back, flags blown up)
#   hold   PPO in the fixed one (sim_hold.pt)
#
#   cd ~/sokubot-runs && MATCHES=12 sbatch ~/SokuBot/ops/amarel/vscom_compare.slurm
set -uo pipefail
cd ~/SokuBot
M=${MATCHES:-12}
PY=~/venvs/sokubot/bin/python
OUT=~/sokubot-runs/vscom-compare-${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "== $(date -Is) job ${SLURM_JOB_ID} on $(hostname) | SokuBot $(cat DEPLOYED) | sfe $(cat ~/sfe/DEPLOYED)"

# The starting policy in the policy.pt format: same observation normalisation, reference weights.
$PY - "$OUT" <<'EOF'
import sys, torch
out = sys.argv[1]
for s in (0, 1):
    pol = torch.load(f"/home/ajt270/sokubot-runs/ppo-hold-s{s}/policy.pt", map_location="cpu", weights_only=False)
    ref_hold = torch.load(f"/home/ajt270/sokubot-runs/ppo-hold-s{s}/latest.pt", map_location="cpu", weights_only=False)["reference"]
    ref_full = torch.load(f"/home/ajt270/sokubot-runs/ppo-full-s{s}/latest.pt", map_location="cpu", weights_only=False)["reference"]
    same = all(torch.equal(ref_hold[k], ref_full[k]) for k in ref_hold)
    print(f"seed {s}: hold and full runs started from the same policy: {same}")
    init = dict(pol)
    init["policy"] = ref_hold
    torch.save(init, f"{out}/init-s{s}.pt")
EOF

arm() {   # name, offset, policy path pattern with {s}
    local name=$1 off=$2 pat=$3
    for s in 0 1; do
        $PY -m scripts.vscom_play --agent policy --policy "${pat//\{s\}/$s}" --games 4 \
            --matches "$M" --level 3 --game-offset "$off" --seed "$s" \
            --work ~/sfe-vscom-work/compare-${SLURM_JOB_ID}/$name-s$s \
            --out "$OUT/$name-s$s.json" > "$OUT/$name-s$s.log" 2>&1
    done
}
arm init 0 "$OUT/init-s{s}.pt" &
arm full 4 "$HOME/sokubot-runs/ppo-full-s{s}/policy.pt" &
arm hold 8 "$HOME/sokubot-runs/ppo-hold-s{s}/policy.pt" &
wait
echo "== $(date -Is) all arms done"
$PY - "$OUT" <<'EOF'
import json, math, sys
from pathlib import Path
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)
out = Path(sys.argv[1])
print(f"{'arm':<6}{'seed':>5}{'rounds':>12}{'matches':>12}")
for arm in ("init", "full", "hold"):
    R = Wr = M = Wm = 0
    for s in (0, 1):
        f = out / f"{arm}-s{s}.json"
        if not f.exists():
            print(f"{arm:<6}{s:>5}   (missing: see {arm}-s{s}.log)"); continue
        j = json.loads(f.read_text())
        print(f"{arm:<6}{s:>5}{j['round_wins']:>6}/{j['rounds']:<5}{j['match_wins']:>6}/{j['matches']:<5}"
              f"{'  errors: ' + str(len(j['errors'])) if j['errors'] else ''}")
        R += j["rounds"]; Wr += j["round_wins"]; M += j["matches"]; Wm += j["match_wins"]
    lo, hi = wilson(Wr, R); mlo, mhi = wilson(Wm, M)
    print(f"{arm:<6} pooled: rounds {Wr}/{R} = {Wr/max(R,1):.3f} [{lo:.3f}, {hi:.3f}] | "
          f"matches {Wm}/{M} = {Wm/max(M,1):.3f} [{mlo:.3f}, {mhi:.3f}]")
EOF
