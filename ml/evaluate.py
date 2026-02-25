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

Supports two evaluation modes:
  - bert:   BERT-only (structured tags filtered to O). Original behavior.
  - hybrid: Full BERT + regex pipeline, SSN ground truth evaluated.

Note: BERT NER models detect PER/LOC/ORG/MISC. In hybrid mode, the regex
layer additionally detects SSN. Other structured types (PHONE, EMAIL, MRN,
DOB) are filtered to O until their regex patterns are implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.pii_engine import PIIEngine  # noqa: F811

import numpy as np
import psutil
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

# Entity types detectable by the full hybrid pipeline (BERT + regex SSN).
HYBRID_ENTITY_TYPES: set[str] = BERT_ENTITY_TYPES | {"SSN"}


def _get_memory_mb() -> float:
    """Return current RSS (Resident Set Size) of this process in MB.

    RSS measures the actual physical RAM consumed, which is what matters
    for the EC2 t3.medium 4GB constraint. Uses psutil for cross-platform
    accuracy.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


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


def filter_to_hybrid_tags(tags: list[str]) -> list[str]:
    """Filter ground-truth tags to types detectable by the hybrid pipeline.

    In hybrid mode, BERT detects PER/LOC/ORG/MISC and regex detects SSN.
    Other structured types (PHONE, EMAIL, MRN, DOB) that don't yet have
    regex patterns are filtered to O to avoid artificially lowering recall.

    Args:
        tags: Full BIO tag list from the synthetic dataset.

    Returns:
        Filtered tag list with only hybrid-detectable types.
    """
    filtered = []
    for tag in tags:
        if tag == "O":
            filtered.append("O")
        else:
            entity_type = tag.split("-", 1)[1] if "-" in tag else tag
            if entity_type in HYBRID_ENTITY_TYPES:
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


def align_hybrid_predictions_to_words(
    tokens: list[str],
    engine: PIIEngine,
) -> list[str]:
    """Run the hybrid PIIEngine (BERT + regex) and align to word-level BIO tags.

    Unlike align_predictions_to_words (BERT-only), this uses PIIEngine.detect()
    which includes regex detection and conflict resolution.

    Args:
        tokens: Original whitespace-tokenized word list.
        engine: Loaded PIIEngine instance with regex enabled.

    Returns:
        BIO tag list aligned to input tokens.
    """
    text = " ".join(tokens)

    # Build word offset mapping (same logic as BERT-only version).
    word_offsets: list[tuple[int, int]] = []
    pos = 0
    for token in tokens:
        start = text.index(token, pos)
        end = start + len(token)
        word_offsets.append((start, end))
        pos = end

    # Run hybrid detection.
    entities = engine.detect(text)

    # Reverse map from Presidio-style entity types back to BIO labels.
    entity_to_bio: dict[str, str] = {
        "PERSON": "PER",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "MISC": "MISC",
        "SSN": "SSN",
    }

    pred_tags = ["O"] * len(tokens)

    for ent in entities:
        bio_label = entity_to_bio.get(ent.entity_type, ent.entity_type)
        for idx, (ws, we) in enumerate(word_offsets):
            if ws < ent.end and we > ent.start:
                if pred_tags[idx] == "O":
                    if ws >= ent.start or idx == 0:
                        pred_tags[idx] = f"B-{bio_label}"
                    else:
                        pred_tags[idx] = f"I-{bio_label}"

    return pred_tags


def run_evaluation(
    model_name: str,
    dataset_path: Path,
    limit: int | None = None,
    mode: str = "hybrid",
) -> dict:
    """Run full evaluation: load model, infer on dataset, compute metrics.

    Args:
        model_name: HuggingFace model ID or local path.
        dataset_path: Path to synthetic JSONL dataset.
        limit: Max samples to evaluate.
        mode: "bert" for BERT-only, "hybrid" for BERT + regex SSN.

    Returns:
        Results dictionary with metrics and metadata.
    """
    # Memory baseline before model load.
    rss_before_mb = _get_memory_mb()
    tracemalloc.start()

    print(f"Loading model: {model_name} (mode={mode})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU — matches production target
    )

    # For hybrid mode, wrap the pipeline in a PIIEngine with regex enabled.
    engine = None
    if mode == "hybrid":
        try:
            from ml.pii_engine import PIIEngine
        except ModuleNotFoundError:
            from pii_engine import PIIEngine  # type: ignore[no-redef]

        engine = PIIEngine(model_id=model_name, enable_regex=True)
        # Reuse the already-loaded pipeline to avoid double-loading.
        engine._pipeline = ner_pipeline

    rss_after_load_mb = _get_memory_mb()
    model_rss_mb = rss_after_load_mb - rss_before_mb
    print(f"  Model RSS delta: {model_rss_mb:.1f} MB")

    print(f"Loading dataset: {dataset_path}")
    samples = load_dataset(dataset_path, limit=limit)
    print(f"Evaluating {len(samples)} samples...\n")

    all_true_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []
    latencies_ms: list[float] = []

    for i, sample in enumerate(samples):
        tokens = sample["tokens"]

        if mode == "bert":
            true_tags = filter_to_bert_tags(sample["ner_tag_labels"])
        else:
            true_tags = filter_to_hybrid_tags(sample["ner_tag_labels"])

        # Measure inference latency.
        start_ns = time.perf_counter_ns()
        if mode == "bert":
            pred_tags = align_predictions_to_words(tokens, ner_pipeline)
        else:
            pred_tags = align_hybrid_predictions_to_words(tokens, engine)
        end_ns = time.perf_counter_ns()

        latency_ms = (end_ns - start_ns) / 1_000_000
        latencies_ms.append(latency_ms)

        all_true_tags.append(true_tags)
        all_pred_tags.append(pred_tags)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")

    # Capture peak memory after inference loop.
    rss_peak_mb = _get_memory_mb()
    _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc_peak_mb = tracemalloc_peak_bytes / (1024 * 1024)

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

    if mode == "bert":
        note = (
            "BERT-only evaluation. SSN/PHONE/EMAIL/MRN/DOB ground truth "
            "tags are filtered to O (handled by regex layer, not BERT)."
        )
    else:
        note = (
            "Hybrid evaluation (BERT + regex SSN). PHONE/EMAIL/MRN/DOB "
            "ground truth tags filtered to O (regex patterns not yet added)."
        )

    results = {
        "model": model_name,
        "mode": mode,
        "dataset": str(dataset_path),
        "num_samples": len(samples),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
        },
        "latency": latency_stats,
        "memory": {
            "rss_before_model_mb": round(rss_before_mb, 1),
            "rss_after_model_load_mb": round(rss_after_load_mb, 1),
            "model_rss_delta_mb": round(model_rss_mb, 1),
            "rss_peak_mb": round(rss_peak_mb, 1),
            "tracemalloc_peak_mb": round(tracemalloc_peak_mb, 1),
        },
        "classification_report": report_str,
        "note": note,
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

    if "memory" in results:
        mem = results["memory"]
        print(f"\n  RAM (RSS) before model:  {mem['rss_before_model_mb']:.1f} MB")
        print(f"  RAM (RSS) after load:    {mem['rss_after_model_load_mb']:.1f} MB")
        print(f"  Model RSS delta:         {mem['model_rss_delta_mb']:.1f} MB")
        print(f"  RAM (RSS) peak:          {mem['rss_peak_mb']:.1f} MB")
        print(f"  tracemalloc peak:        {mem['tracemalloc_peak_mb']:.1f} MB")

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


def print_comparison_table(all_results: list[dict]) -> None:
    """Print a side-by-side comparison table of multiple model evaluations.

    Columns: Model, Macro F1, Precision, Recall, Latency p50, Latency p95,
    RSS Peak (MB), tracemalloc Peak (MB). This is the primary output for
    OBS-3 benchmarking — lets us compare candidates at a glance.
    """
    header = (
        f"{'Model':<42} {'F1':>6} {'Prec':>6} {'Rec':>6} "
        f"{'p50ms':>7} {'p95ms':>7} {'RSS_MB':>7} {'tmPeak':>7}"
    )
    print("\n" + "=" * len(header))
    print("  MODEL COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        m = r["metrics"]
        lat = r["latency"]
        mem = r.get("memory", {})
        rss = mem.get("rss_peak_mb", 0)
        tmp = mem.get("tracemalloc_peak_mb", 0)
        print(
            f"{r['model']:<42} {m['macro_f1']:>6.4f} {m['macro_precision']:>6.4f} "
            f"{m['macro_recall']:>6.4f} {lat['p50_ms']:>7.1f} {lat['p95_ms']:>7.1f} "
            f"{rss:>7.1f} {tmp:>7.1f}"
        )

    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NER model on synthetic Obscura dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["dslim/bert-base-NER"],
        help=(
            "HuggingFace model ID(s). Pass multiple for comparison table. "
            "(default: dslim/bert-base-NER)"
        ),
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bert", "hybrid"],
        default="hybrid",
        help=(
            "Evaluation mode. 'bert': BERT-only (structured tags filtered "
            "to O). 'hybrid': full BERT + regex pipeline, SSN tags "
            "evaluated. (default: hybrid)"
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run generate_synthetic_data.py first.")
        raise SystemExit(1)

    results_dir = Path(__file__).parent / "results"
    all_results: list[dict] = []

    for model_name in args.model:
        results = run_evaluation(
            model_name, dataset_path, limit=args.limit, mode=args.mode
        )
        print_results(results)
        saved_path = save_results(results, results_dir)
        print(f"\nResults saved to: {saved_path}\n")
        all_results.append(results)

    if len(all_results) > 1:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
