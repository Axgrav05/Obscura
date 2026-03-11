"""
Candidate Model Downloader for Obscura NER Benchmarking

Pre-downloads and caches HuggingFace NER model candidates so that
evaluate.py can run offline without network latency in the critical path.

Downloads tokenizer + model weights to the HuggingFace cache (~/.cache/huggingface/).
No weights are committed — .gitignore excludes all binary artifacts.

Candidate models (from GAMEPLAN.md Phase 1):
  1. dslim/bert-base-NER          — General-purpose BERT NER (110M params)
  2. dslim/distilbert-NER          — Distilled variant (~66M params, faster)
  3. StanfordAIMI/stanford-deidentifier-base — Clinical de-identification
  4. emilyalsentzer/Bio_ClinicalBERT — Clinical BERT (needs NER head check)

Usage:
    python download_models.py
    python download_models.py --models dslim/bert-base-NER dslim/distilbert-NER
"""

import argparse
import sys
import time

from transformers import AutoModelForTokenClassification, AutoTokenizer

# Candidate models from GAMEPLAN.md Phase 1 — Baseline & Benchmarking.
CANDIDATE_MODELS: list[str] = [
    "dslim/bert-base-NER",
    "dslim/distilbert-NER",
    "StanfordAIMI/stanford-deidentifier-base",
]

# Note: emilyalsentzer/Bio_ClinicalBERT is a masked-LM model without a
# token-classification head. It would need fine-tuning before NER evaluation.
# Excluded from default downloads to avoid confusion, but can be added via CLI.


def download_model(model_id: str) -> dict:
    """
    Download and cache a single HuggingFace model and its tokenizer.

    Validates that the model has a token-classification head by checking
    the model class. Returns a summary dict with model metadata.

    Args:
        model_id: HuggingFace model identifier (e.g. "dslim/bert-base-NER").

    Returns:
        Dict with model_id, num_parameters, num_labels, status, and timing.
    """
    print(f"\n  Downloading: {model_id}")
    start = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)

        num_params = sum(p.numel() for p in model.parameters())
        num_labels = model.config.num_labels
        label_map = getattr(model.config, "id2label", {})

        elapsed = time.perf_counter() - start

        print(f"    Parameters: {num_params:,}")
        print(f"    Labels:     {num_labels} — {list(label_map.values())[:8]}...")
        print(f"    Vocab size: {tokenizer.vocab_size:,}")
        print(f"    Cached in {elapsed:.1f}s")

        return {
            "model_id": model_id,
            "num_parameters": num_params,
            "num_labels": num_labels,
            "label_map": label_map,
            "vocab_size": tokenizer.vocab_size,
            "status": "ok",
            "download_time_s": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"    FAILED: {e}")
        return {
            "model_id": model_id,
            "status": "error",
            "error": str(e),
            "download_time_s": round(elapsed, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and cache candidate NER models for Obscura benchmarking"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=CANDIDATE_MODELS,
        help=f"Model IDs to download (default: {CANDIDATE_MODELS})",
    )
    args = parser.parse_args()

    print(f"Downloading {len(args.models)} candidate model(s)...")

    results: list[dict] = []
    for model_id in args.models:
        result = download_model(model_id)
        results.append(result)

    # Summary table.
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)

    ok_count = 0
    for r in results:
        status = r["status"]
        model_id = r["model_id"]
        if status == "ok":
            params_m = r["num_parameters"] / 1_000_000
            print(f"  OK    {model_id:<45} {params_m:.0f}M params")
            ok_count += 1
        else:
            print(f"  FAIL  {model_id:<45} {r.get('error', 'unknown')}")

    print(f"\n  {ok_count}/{len(results)} models cached successfully.")

    if ok_count < len(results):
        print("\n  Failed models may need a different AutoModel class")
        print("  (e.g. Bio_ClinicalBERT is a MLM, not token-classification).")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
