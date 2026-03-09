#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${1:?usage: scripts/push_to_runpod.sh <runpod-host> [remote-dir]}"
REMOTE_DIR="${2:-~/Obscura-1}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PORT="${SSH_PORT:-22}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mkdir -p "$tmpdir/ml/models"

cp -R ml/*.py ml/requirements.txt ml/pyproject.toml ml/tests "$tmpdir/ml/"
mkdir -p "$tmpdir/ml/data"
cp ml/data/nemotron_test.jsonl "$tmpdir/ml/data/"
cp ml/data/hard_negative_business_benchmark.jsonl "$tmpdir/ml/data/"

cp -R ml/models/nemotron-hybrid-fine-tuned "$tmpdir/ml/models/"
rm -rf "$tmpdir/ml/models/nemotron-hybrid-fine-tuned/checkpoints"

cp -R scripts "$tmpdir/"

tar \
  -C "$tmpdir" \
  -czf - . | ssh -p "$SSH_PORT" -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar -xzf -"

echo "repo_synced_to_runpod:$REMOTE_DIR"