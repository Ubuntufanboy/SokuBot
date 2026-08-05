#!/usr/bin/env bash
# Pull the small, expensive-to-regenerate artifacts off a rented GPU box.
#
#   tools/sync_artifacts.sh <host> <port> [dest]
#
# A vast.ai instance disappeared mid-run and took a 225k-step world model, a
# calibrated reward probe, a trained planner and a GRPO policy with it. The
# corpus was safe on HuggingFace and every script was in git, so the only
# genuinely unrecoverable thing was a few hundred megabytes of weights that
# nothing was copying anywhere.
#
# Everything here is tens of MB. Run it after any stage that produces a
# checkpoint, or on a timer:
#
#   watch -n 900 tools/sync_artifacts.sh 1.2.3.4 57358
#
# Deliberately does not pull the corpus (re-downloadable, ~64 GB/shard) or the
# latent banks (derived, ~250 MB, ten minutes to rebuild).

set -euo pipefail

HOST=${1:?usage: sync_artifacts.sh <host> <port> [dest]}
PORT=${2:?usage: sync_artifacts.sh <host> <port> [dest]}
DEST=${3:-$HOME/K0NTR0L-2/artifacts}
KEY=${SOKU_SSH_KEY:-$HOME/.ssh/id_ed25519_ai}

mkdir -p "$DEST"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=15"

# Checkpoints, probes, planners, and the JSON logs that say what each one
# scored -- a checkpoint whose recorded metric is separated from it is most of
# the way to being worthless.
PATTERNS=(
  '/root/ckpt*/best.pt'
  '/root/ckpt*/best_bnfix.pt'
  '/root/ckpt*/log.json'
  '/root/planner*.pt'
  '/root/horizon*/reward_probe*.npz'
  '/root/horizon*/horizon.json'
  '/root/grpo*/policy*.pt'
  '/root/grpo*/log.json'
  '/root/long_*/policy*.pt'
  '/root/long_*/log.json'
  '/root/train_*.json'
  '/root/*_test.json'
)

# One find on the remote, so a missing path is not an error and the transfer is
# a single connection rather than one per pattern.
FILES=$($SSH "root@$HOST" "ls -d ${PATTERNS[*]} 2>/dev/null" || true)
if [ -z "$FILES" ]; then
  echo "nothing to sync from $HOST:$PORT"
  exit 0
fi

echo "$FILES" | while read -r f; do
  [ -n "$f" ] || continue
  rel=${f#/root/}
  mkdir -p "$DEST/$(dirname "$rel")"
  # scp rather than rsync: the rented images do not reliably ship a runnable
  # rsync, and these files are small enough that delta transfer buys nothing.
  if scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no \
        "root@$HOST:$f" "$DEST/$rel"; then
    printf '  %8s  %s\n' "$(du -h "$DEST/$rel" | cut -f1)" "$rel"
  else
    echo "  FAILED  $rel" >&2
  fi
done

echo "synced to $DEST ($(du -sh "$DEST" | cut -f1) total)"
