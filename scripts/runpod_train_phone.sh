#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-$HOME/Obscura-1}"
MODEL_NAME="${2:-ml/models/nemotron-hybrid-fine-tuned-phone-org-rtx3080}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
NEMOTRON_FRACTION="${NEMOTRON_FRACTION:-0.85}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-8e-6}"
RUN_EVAL="${RUN_EVAL:-0}"
ONTONOTES_ORG_DIR="${ONTONOTES_ORG_DIR:-$HOME/hf-datasets/tner-ontonotes5-parquet}"
ONTONOTES_ORG_TRAIN_LIMIT="${ONTONOTES_ORG_TRAIN_LIMIT:-25000}"
ONTONOTES_ORG_EVAL_LIMIT="${ONTONOTES_ORG_EVAL_LIMIT:-4000}"
TRANSCRIPT_ORG_SAMPLES="${TRANSCRIPT_ORG_SAMPLES:-12000}"

cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt pytest

if command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip uninstall -y onnxruntime || true
  python -m pip install --upgrade onnxruntime-gpu
fi

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='nvidia/Nemotron-PII', repo_type='dataset')
print('nemotron_downloaded')
PY

python ml/generate_synthetic_data.py \
  --source hard-negative \
  --num-samples 2000 \
  --output data/hard_negative_business_benchmark.jsonl

python ml/generate_synthetic_data.py \
  --source hybrid \
  --num-samples "$TRAIN_SAMPLES" \
  --nemotron-fraction "$NEMOTRON_FRACTION" \
  --output data/phone_rich_hybrid_train.jsonl

python ml/generate_synthetic_data.py \
  --source transcript-org \
  --num-samples "$TRANSCRIPT_ORG_SAMPLES" \
  --output data/transcript_org_train.jsonl

python -m pytest ml/tests/test_regex_extensions.py -q

python ml/fine_tune.py \
  --model ml/models/nemotron-hybrid-fine-tuned-phone-rtx3080 \
  --dataset data/phone_rich_hybrid_train.jsonl,data/transcript_org_train.jsonl,fewnerd-org,ontonotes-org \
  --output "$MODEL_NAME" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --train-split 0.9 \
  --fewnerd-org-train-limit 60000 \
  --fewnerd-org-eval-limit 8000 \
  --ontonotes-org-dir "$ONTONOTES_ORG_DIR" \
  --ontonotes-org-train-limit "$ONTONOTES_ORG_TRAIN_LIMIT" \
  --ontonotes-org-eval-limit "$ONTONOTES_ORG_EVAL_LIMIT"

if [[ "$RUN_EVAL" == "1" ]]; then
  python ml/evaluate.py \
    --model "$MODEL_NAME" \
    --dataset data/nemotron_test.jsonl \
    --limit 2000 \
    --mode hybrid \
    --device cuda

  python ml/evaluate.py \
    --model "$MODEL_NAME" \
    --dataset data/hard_negative_business_benchmark.jsonl \
    --mode hybrid \
    --device cuda
else
  echo "Skipping evaluation because RUN_EVAL=$RUN_EVAL"
fi

python ml/export_onnx.py \
  --model "$MODEL_NAME" \
  --output ml/models/onnx \
  --quantize \
  --bundle \
  --version 1.5.0-org

echo "runpod_training_complete"