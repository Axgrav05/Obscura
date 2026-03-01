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
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.pii_engine import PIIEngine

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

# Entity types detectable by the full hybrid pipeline (BERT + regex).
HYBRID_ENTITY_TYPES: set[str] = BERT_ENTITY_TYPES | {"SSN", "PHONE", "EMAIL", "MRN"}

# Label adapters for models with non-IOB2 label schemes.
# Maps model-specific entity_group values to CoNLL-2003 equivalents.
LABEL_ADAPTERS: dict[str, dict[str, str]] = {
    "StanfordAIMI/stanford-deidentifier-base": {
        "PATIENT": "PER",
        "HCW": "PER",
        "HOSPITAL": "ORG",
        "VENDOR": "ORG",
        "DATE": "MISC",
        "ID": "MISC",
        "PHONE": "MISC",
    },
}


def _detect_label_adapter(
    model_name: str, model_config: object
) -> dict[str, str] | None:
    """Auto-detect if a model needs a label adapter.

    Checks LABEL_ADAPTERS by name first, then inspects id2label for
    non-IOB2 labels (no B-/I- prefixes).
    """
    if model_name in LABEL_ADAPTERS:
        return LABEL_ADAPTERS[model_name]

    id2label = getattr(model_config, "id2label", {})
    non_o_labels = [v for v in id2label.values() if v != "O"]
    if non_o_labels and not any(
        lab.startswith("B-") or lab.startswith("I-") for lab in non_o_labels
    ):
        # Non-IOB2 model without a known adapter — log warning
        print(
            f"  WARNING: {model_name} uses non-IOB2 labels {non_o_labels} "
            f"with no adapter defined. Results may be inaccurate."
        )
    return None


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
    label_adapter: dict[str, str] | None = None,
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
        label_adapter: Optional mapping from model-specific labels to
            CoNLL-2003 equivalents (e.g. PATIENT -> PER).

    Returns:
        BIO tag list aligned to the input tokens.
    """
    text = " ".join(tokens)

    # Build a mapping: for each token index, its (start, end) char offsets.
    # Uses deterministic cumulative offsets from " ".join() rather than
    # text.index() which can misalign on duplicate substrings.
    word_offsets: list[tuple[int, int]] = []
    pos = 0
    for token in tokens:
        word_offsets.append((pos, pos + len(token)))
        pos += len(token) + 1  # +1 for space separator

    # Run NER inference.
    raw_entities = ner_pipeline(text)

    # Initialize all tags as O.
    pred_tags = ["O"] * len(tokens)

    for ent in raw_entities:
        ent_start = int(ent["start"])
        ent_end = int(ent["end"])
        ent_label = ent["entity_group"]

        # Remap non-IOB2 labels if adapter is provided.
        if label_adapter:
            ent_label = label_adapter.get(ent_label, ent_label)

        # Find which word tokens overlap with this entity span.
        first_tagged = False
        for idx, (ws, we) in enumerate(word_offsets):
            if ws < ent_end and we > ent_start:
                if pred_tags[idx] == "O":
                    if not first_tagged:
                        pred_tags[idx] = f"B-{ent_label}"
                        first_tagged = True
                    else:
                        pred_tags[idx] = f"I-{ent_label}"

    return pred_tags


def align_hybrid_predictions_to_words(
    tokens: list[str],
    engine: PIIEngine,
) -> list[str]:
    """Run the hybrid PIIEngine (BERT + regex) and align to word-level BIO tags.

    Unlike align_predictions_to_words (BERT-only), this uses PIIEngine.detect()
    which includes regex detection and conflict resolution. The label adapter
    is applied inside PIIEngine._detect_bert(), so no remapping needed here.

    Args:
        tokens: Original whitespace-tokenized word list.
        engine: Loaded PIIEngine instance with regex enabled.

    Returns:
        BIO tag list aligned to input tokens.
    """
    text = " ".join(tokens)

    # Build word offset mapping — deterministic cumulative offsets from
    # " ".join() rather than text.index() which misaligns on duplicates.
    word_offsets: list[tuple[int, int]] = []
    pos = 0
    for token in tokens:
        word_offsets.append((pos, pos + len(token)))
        pos += len(token) + 1  # +1 for space separator

    # Run hybrid detection.
    entities = engine.detect(text)

    # Reverse map from Presidio-style entity types back to BIO labels.
    entity_to_bio: dict[str, str] = {
        "PERSON": "PER",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "MISC": "MISC",
        "SSN": "SSN",
        "PHONE": "PHONE",
        "EMAIL": "EMAIL",
        "MRN": "MRN",
    }

    pred_tags = ["O"] * len(tokens)

    for ent in entities:
        bio_label = entity_to_bio.get(ent.entity_type, ent.entity_type)
        first_tagged = False
        for idx, (ws, we) in enumerate(word_offsets):
            if ws < ent.end and we > ent.start:
                if pred_tags[idx] == "O":
                    if not first_tagged:
                        pred_tags[idx] = f"B-{bio_label}"
                        first_tagged = True
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
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,  # CPU — matches production target
    )

    # Auto-detect label adapter for non-IOB2 models (e.g. StanfordAIMI).
    label_adapter = _detect_label_adapter(model_name, model.config)
    if label_adapter:
        print(f"  Label adapter active: {label_adapter}")

    # For hybrid mode, wrap the pipeline in a PIIEngine with regex enabled.
    engine = None
    if mode == "hybrid":
        from ml.pii_engine import PIIEngine

        engine = PIIEngine(
            model_id=model_name,
            enable_regex=True,
            label_adapter=label_adapter,
        )
        # Reuse the already-loaded pipeline to avoid double-loading.
        engine._pipeline = ner_pipe

    rss_after_load_mb = _get_memory_mb()
    model_rss_mb = rss_after_load_mb - rss_before_mb
    print(f"  Model RSS delta: {model_rss_mb:.1f} MB")

    print(f"Loading dataset: {dataset_path}")
    samples = load_dataset(dataset_path, limit=limit)
    print(f"Evaluating {len(samples)} samples...")

    # Warmup: 5 untimed inferences to stabilize JIT and caches.
    num_warmup = min(5, len(samples))
    print(f"  Running {num_warmup} warmup inferences...")
    for i in range(num_warmup):
        warmup_tokens = samples[i]["tokens"]
        if mode == "bert":
            align_predictions_to_words(warmup_tokens, ner_pipe, label_adapter)
        else:
            align_hybrid_predictions_to_words(warmup_tokens, engine)
    print()

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
            pred_tags = align_predictions_to_words(tokens, ner_pipe, label_adapter)
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
            "Hybrid evaluation (BERT + regex SSN/PHONE/EMAIL/MRN). DOB "
            "ground truth tags filtered to O (regex pattern not yet added)."
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
    # Ensure the repository root is on sys.path for script-mode invocation.
    # `python ml/evaluate.py` prepends ml/ to sys.path[0], but ml.* imports
    # require the repo root (parent of ml/) on the path.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    main()
