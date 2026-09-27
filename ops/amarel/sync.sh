#!/usr/bin/env bash
# Deploy this checkout's TRACKED files to Amarel, and stamp exactly what was deployed.
#
#   ops/amarel/sync.sh                 # -> amarel:~/SokuBot, stamp in ~/SokuBot/DEPLOYED
#
# Tracked files only (git ls-files), so no bank, checkpoint, recording or PDF can ride along by
# accident. The stamp is the commit plus how many tracked files were modified: code that is
# committed is not code that is deployed, and every job prints the stamp before it starts.
# Needs the VPN tunnel up (see the amarel skill); `amarel` is the ssh alias for it.
set -euo pipefail
HOST=${AMAREL_HOST:-amarel}
DEST=${AMAREL_DEST:-SokuBot}
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
hash=$(git rev-parse --short=12 HEAD)
dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
git ls-files -z | rsync -a --from0 --files-from=- --delete-missing-args ./ "$HOST:$DEST/"
printf '%s dirty=%s synced=%s\n' "$hash" "$dirty" "$(date -Is)" | ssh "$HOST" "cat > $DEST/DEPLOYED"
echo "deployed $hash (dirty=$dirty) -> $HOST:$DEST"
