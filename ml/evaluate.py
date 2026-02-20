"""
Obscura NER Model Evaluation Harness

Loads a HuggingFace NER model, runs inference on synthetic data, and
computes per-entity F1/precision/recall using seqeval plus latency
statistics.

Usage:
    python evaluate.py --model dslim/bert-base-NER
    python evaluate.py --model dslim/bert-base-NER \\
        --dataset data/synthetic.jsonl --limit 50

Output:
    - Human-readable results table to stdout
    - JSON report saved to ml/results/<model>_<timestamp>.json

Note: BERT NER models detect PER/LOC/ORG/MISC. Structured PII types
(SSN, PHONE, EMAIL, MRN, DOB) are handled by the regex layer and are
excluded from BERT-only evaluation. Hybrid pipeline metrics are tracked
separately once the regex layer is integrated.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)

# Entity types that BERT NER models can detect. Others are regex-only.
BERT_ENTITY_TYPES: set[str] = {"PER", "ORG", "LOC", "MISC"}


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    """
    Load synthetic JSONL dataset.

    Args:
        path: Path to the .jsonl file.
        limit: Max number of samples to load (None = all).

    Returns:
        List of dicts with 'tokens' and 'ner_tag_labels' keys.
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            samples.append(json.loads(line))
    return samples


def filter_to_bert_tags(tags: list[str]) -> list[str]:
    """
    Replace non-BERT entity tags (SSN, PHONE, etc.) with 'O'.

    BERT was never trained on these types — they're regex-only.
    Keeping them as ground truth would artificially lower recall.

    Args:
        tags: Full BIO tag list from the synthetic dataset.

    Returns:
        Filtered tag list with only BERT-detectable types.
    """
    filtered = []
    for tag in tags:
        if tag == "O":
            filtered.append("O")
        else:
            # Extract entity type: B-PER -> PER, I-ORG -> ORG
            entity_type = tag.split("-", 1)[1] if "-" in tag else tag
            if entity_type in BERT_ENTITY_TYPES:
                filtered.append(tag)
            else:
                filtered.append("O")
    return filtered


def align_predictions_to_words(
    tokens: list[str],
    ner_pipeline: TokenClassificationPipeline,
) -> list[str]:
    """
    Run the NER pipeline on a token list and align predictions back
    to the original word boundaries.

    The HuggingFace pipeline operates on raw text, so we join tokens,
    run inference, then map character-level entity spans back to the
    word-level token indices using character offsets.

    Args:
        tokens: Original whitespace-tokenized word list.
        ner_pipeline: Loaded HuggingFace NER pipeline.

    Returns:
        BIO tag list aligned to the input tokens.
    """
    text = " ".join(tokens)

    # Build a mapping: for each token index, its (start, end) char offsets.
    word_offsets: list[tuple[int, int]] = []
    pos = 0
    for token in tokens:
        start = text.index(token, pos)
        end = start + len(token)
        word_offsets.append((start, end))
        pos = end

    # Run NER inference.
    raw_entities = ner_pipeline(text)

    # Initialize all tags as O.
    pred_tags = ["O"] * len(tokens)

    for ent in raw_entities:
        ent_start = int(ent["start"])
        ent_end = int(ent["end"])
        ent_label = ent["entity_group"]

        # Find which word tokens overlap with this entity span.
        for idx, (ws, we) in enumerate(word_offsets):
            # Check for overlap.
            if ws < ent_end and we > ent_start:
                if pred_tags[idx] == "O":
                    # Check if this is the start of the entity.
                    if ws >= ent_start or idx == 0:
                        pred_tags[idx] = f"B-{ent_label}"
                    else:
                        pred_tags[idx] = f"I-{ent_label}"

    return pred_tags


def run_evaluation(
    model_name: str,
    dataset_path: Path,
    limit: int | None = None,
) -> dict:
    """
    Run full evaluation: load model, infer on dataset, compute metrics.

    Args:
        model_name: HuggingFace model ID or local path.
        dataset_path: Path to synthetic JSONL dataset.
        limit: Max samples to evaluate.

    Returns:
        Results dictionary with metrics and metadata.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU — matches production target
    )

    print(f"Loading dataset: {dataset_path}")
    samples = load_dataset(dataset_path, limit=limit)
    print(f"Evaluating {len(samples)} samples...\n")

    all_true_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []
    latencies_ms: list[float] = []

    for i, sample in enumerate(samples):
        tokens = sample["tokens"]
        true_tags = filter_to_bert_tags(sample["ner_tag_labels"])

        # Measure inference latency.
        start_ns = time.perf_counter_ns()
        pred_tags = align_predictions_to_words(tokens, ner_pipeline)
        end_ns = time.perf_counter_ns()

        latency_ms = (end_ns - start_ns) / 1_000_000
        latencies_ms.append(latency_ms)

        all_true_tags.append(true_tags)
        all_pred_tags.append(pred_tags)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    # Compute metrics via seqeval.
    macro_f1 = f1_score(all_true_tags, all_pred_tags, average="macro", zero_division=0)
    macro_precision = precision_score(
        all_true_tags, all_pred_tags, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        all_true_tags, all_pred_tags, average="macro", zero_division=0
    )
    report_str = classification_report(all_true_tags, all_pred_tags, zero_division=0)

    # Latency statistics.
    lat_array = np.array(latencies_ms)
    latency_stats = {
        "p50_ms": round(float(np.percentile(lat_array, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_array, 95)), 2),
        "p99_ms": round(float(np.percentile(lat_array, 99)), 2),
        "mean_ms": round(float(np.mean(lat_array)), 2),
        "max_ms": round(float(np.max(lat_array)), 2),
    }

    results = {
        "model": model_name,
        "dataset": str(dataset_path),
        "num_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
        },
        "latency": latency_stats,
        "classification_report": report_str,
        "note": (
            "BERT-only evaluation. SSN/PHONE/EMAIL/MRN/DOB ground truth "
            "tags are filtered to O (handled by regex layer, not BERT). "
            "Hybrid pipeline metrics tracked separately."
        ),
    }

    return results


def print_results(results: dict) -> None:
    """Print results in a clean table format."""
    print("=" * 60)
    print(f"  Model:    {results['model']}")
    print(f"  Samples:  {results['num_samples']}")
    print(f"  Date:     {results['timestamp']}")
    print("=" * 60)

    m = results["metrics"]
    print(f"\n  Macro F1:        {m['macro_f1']:.4f}")
    print(f"  Macro Precision: {m['macro_precision']:.4f}")
    print(f"  Macro Recall:    {m['macro_recall']:.4f}")

    lat = results["latency"]
    print(f"\n  Latency p50:  {lat['p50_ms']:.1f} ms")
    print(f"  Latency p95:  {lat['p95_ms']:.1f} ms")
    print(f"  Latency p99:  {lat['p99_ms']:.1f} ms")
    print(f"  Latency mean: {lat['mean_ms']:.1f} ms")
    print(f"  Latency max:  {lat['max_ms']:.1f} ms")

    print("\n  Per-entity breakdown:")
    print(results["classification_report"])

    print(f"  Note: {results['note']}")
    print("=" * 60)


def save_results(results: dict, output_dir: Path) -> Path:
    """Save results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = results["model"].replace("/", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{model_slug}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NER model on synthetic Obscura dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dslim/bert-base-NER",
        help="HuggingFace model ID or local path (default: dslim/bert-base-NER)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic.jsonl",
        help="Path to synthetic JSONL dataset (default: data/synthetic.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run generate_synthetic_data.py first.")
        raise SystemExit(1)

    results = run_evaluation(args.model, dataset_path, limit=args.limit)
    print_results(results)

    results_dir = Path(__file__).parent / "results"
    saved_path = save_results(results, results_dir)
    print(f"\nResults saved to: {saved_path}")


if __name__ == "__main__":
    main()
