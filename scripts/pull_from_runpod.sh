#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${1:?usage: scripts/pull_from_runpod.sh <runpod-host> [remote-dir]}"
REMOTE_DIR="${2:-~/Obscura-1}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PORT="${SSH_PORT:-22}"

mkdir -p ml/models ml/results

tar -xzf - -C . < <(
  ssh -p "$SSH_PORT" -i "$SSH_KEY" "$REMOTE_HOST" \
    "cd $REMOTE_DIR && tar -czf - ml/models/nemotron-hybrid-fine-tuned-phone-rtx3080 ml/models/onnx ml/results ml/data"
)

echo "runpod_artifacts_pulled_from:$REMOTE_DIR"