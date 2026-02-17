# ML Directory — AI/ML Engine

This directory contains the BERT-NER model training, evaluation, and export pipeline.

## Environment
ALWAYS activate the venv before running anything here:
```bash
source .venv/bin/activate
```

## Key Scripts
- `train.py` — Model fine-tuning on NER datasets
- `evaluate.py` — Composite scoring (F1, BERTScore)
- `export_onnx.py` — Convert trained model to optimized ONNX format

## Model Architecture
- Base: `dslim/bert-base-NER` (or clinical variants for PHI)
- Hybrid approach: transformer semantic predictions + deterministic regex for structured PII (SSNs, phone numbers)
- Differential privacy: Laplacian noise injection to defend against membership inference

## Constraints
- Target accuracy: 90%+ F1
- Exported ONNX must be quantized to fit OCI ARM instance RAM
- Inference latency budget: ≤30ms

## Testing
```bash
pytest tests/ -v
```

## When modifying the export format or mapping dictionary schema, flag it — Rainier needs to update the Rust consumer in `/src/inference/`.
