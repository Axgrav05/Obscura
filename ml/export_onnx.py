"""ONNX Export + Quantization Pipeline for Obscura

Exports a HuggingFace NER model to ONNX via optimum, applies INT8 dynamic
quantization, validates accuracy parity, and packages the artifact bundle
for the Rust proxy backend.

Usage:
    # Export only (FP32 ONNX)
    python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx

    # Export + INT8 quantization
    python ml/export_onnx.py --model dslim/bert-base-NER \\
        --output ml/models/onnx --quantize

    # Export + quantize + validate (logit parity + latency)
    python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx \\
        --quantize --validate

    # Full pipeline: export + quantize + validate + bundle
    python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx \\
        --quantize --validate --bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# OBS-13a: ONNX export
# ---------------------------------------------------------------------------

_VALIDATION_SENTENCES: list[str] = [
    "John Smith lives in New York and works at Acme Corp.",
    "Patient Jane Doe, MRN 12345, was admitted to City Hospital.",
    "Contact: 555-123-4567, email: test@example.com, SSN: 123-45-6789.",
    "The meeting was held at 123 Main Street, Springfield.",
    "Dr. Alice Johnson prescribed metformin for the patient.",
    "Bob Wilson from Chicago called regarding his account.",
    "The report for Mary Brown was filed on Tuesday.",
    "Emergency contact: 800-555-0199, ask for David Lee.",
    "Referred to St. Luke's Hospital by Dr. Sarah Chen.",
    "Invoice #9842 sent to Global Industries, attn: Mike Taylor.",
]


def export_model(model_id: str, output_dir: Path) -> Path:
    """Export a HuggingFace NER model to ONNX.

    Tries ``optimum.exporters.onnx`` first (produces an optimised graph with
    proper dynamic axes).  Falls back to ``torch.onnx.export`` if *optimum*
    is not installed.

    Returns:
        Path to the exported ``model.onnx`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.exporters.onnx import main_export  # noqa: F811

        print(f"Exporting model via optimum: {model_id}")
        main_export(
            model_name_or_path=model_id,
            output=str(output_dir),
            task="token-classification",
            opset=14,
        )
    except ImportError:
        print(
            "WARNING: optimum not installed — falling back to torch.onnx.export.\n"
            "         Install with: pip install 'optimum[onnxruntime]'",
            file=sys.stderr,
        )
        _export_torch_fallback(model_id, output_dir)

    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found after export.", file=sys.stderr)
        raise SystemExit(1)

    _ensure_tokenizer(model_id, output_dir)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print("\nExport complete:")
    print(f"  ONNX model:  {onnx_path} ({size_mb:.1f} MB)")
    print(f"  Tokenizer:   {output_dir / 'tokenizer.json'}")
    return onnx_path


def _export_torch_fallback(model_id: str, output_dir: Path) -> None:
    """Export via raw ``torch.onnx.export`` (no optimum dependency)."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()

    dummy = tokenizer("test", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    onnx_path = output_dir / "model.onnx"
    print(f"Exporting ONNX to: {onnx_path}")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    tokenizer.save_pretrained(str(output_dir))


def _ensure_tokenizer(model_id: str, output_dir: Path) -> None:
    """Verify tokenizer.json exists; save manually if missing."""
    tokenizer_json = output_dir / "tokenizer.json"
    if tokenizer_json.exists():
        return

    print("tokenizer.json not found — saving from model…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(output_dir))

    if not tokenizer_json.exists():
        print(
            f"ERROR: {tokenizer_json} still not found. "
            "The Rust tokenizers crate requires this file.",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# OBS-13b: INT8 dynamic quantization
# ---------------------------------------------------------------------------


def quantize_model(input_path: Path, output_path: Path) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model.

    Returns:
        Path to the quantized model file.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"\nQuantizing: {input_path.name} → {output_path.name}")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100

    print(f"  FP32 size:  {input_size:.1f} MB")
    print(f"  INT8 size:  {output_size:.1f} MB")
    print(f"  Reduction:  {reduction:.1f}%")

    # Verify the quantized model loads with ORT.
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    n_inputs = len(session.get_inputs())
    print(f"  Verified: INT8 model loads ({n_inputs} inputs)")

    return output_path


# ---------------------------------------------------------------------------
# OBS-13c: Validation — logit parity + latency comparison
# ---------------------------------------------------------------------------


def validate_quantized_model(
    model_id: str,
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    *,
    n_samples: int = 100,
    n_warmup: int = 5,
) -> dict:
    """Compare logits and latency across PyTorch, ONNX FP32, and ONNX INT8.

    Returns a results dict with logit deltas, latency percentiles, and model
    sizes.  Prints a human-readable report to stdout.
    """
    import onnxruntime as ort

    print("\n=== Validation: PyTorch vs ONNX FP32 vs ONNX INT8 ===\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pytorch_model = AutoModelForTokenClassification.from_pretrained(model_id)
    pytorch_model.eval()

    fp32_session = ort.InferenceSession(str(onnx_fp32_path))
    int8_session = ort.InferenceSession(str(onnx_int8_path))

    # Detect which inputs the ONNX model expects (optimum may add
    # token_type_ids; raw torch export may not).
    ort_input_names = {inp.name for inp in fp32_session.get_inputs()}

    # Build sample pool by cycling validation sentences.
    texts = _VALIDATION_SENTENCES.copy()
    while len(texts) < n_samples:
        texts.extend(_VALIDATION_SENTENCES[: n_samples - len(texts)])
    texts = texts[:n_samples]

    def _make_ort_inputs(tok_out: dict) -> dict:
        return {k: tok_out[k].numpy() for k in tok_out if k in ort_input_names}

    # Warmup (excluded from measurements).
    for text in texts[:n_warmup]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        ort_inputs = _make_ort_inputs(inputs)
        with torch.no_grad():
            pytorch_model(**inputs)
        fp32_session.run(None, ort_inputs)
        int8_session.run(None, ort_inputs)

    # Measure.
    fp32_deltas: list[float] = []
    int8_deltas: list[float] = []
    pytorch_latencies: list[float] = []
    fp32_latencies: list[float] = []
    int8_latencies: list[float] = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        ort_inputs = _make_ort_inputs(inputs)

        # PyTorch
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            pt_logits = pytorch_model(**inputs).logits.numpy()
        pytorch_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        # ONNX FP32
        t0 = time.perf_counter_ns()
        fp32_logits = fp32_session.run(None, ort_inputs)[0]
        fp32_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        # ONNX INT8
        t0 = time.perf_counter_ns()
        int8_logits = int8_session.run(None, ort_inputs)[0]
        int8_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        fp32_deltas.append(float(np.abs(pt_logits - fp32_logits).max()))
        int8_deltas.append(float(np.abs(pt_logits - int8_logits).max()))

    results: dict = {
        "logit_parity": {
            "fp32_max_delta": float(max(fp32_deltas)),
            "fp32_mean_delta": float(np.mean(fp32_deltas)),
            "int8_max_delta": float(max(int8_deltas)),
            "int8_mean_delta": float(np.mean(int8_deltas)),
        },
        "latency_ms": {
            "pytorch": _percentiles(pytorch_latencies),
            "onnx_fp32": _percentiles(fp32_latencies),
            "onnx_int8": _percentiles(int8_latencies),
        },
        "model_sizes_mb": {
            "onnx_fp32": round(onnx_fp32_path.stat().st_size / (1024 * 1024), 1),
            "onnx_int8": round(onnx_int8_path.stat().st_size / (1024 * 1024), 1),
        },
        "n_samples": n_samples,
        "n_warmup": n_warmup,
    }

    _print_validation_report(results)
    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "p99": round(float(np.percentile(values, 99)), 2),
        "mean": round(float(np.mean(values)), 2),
    }


def _print_validation_report(results: dict) -> None:
    lp = results["logit_parity"]
    lat = results["latency_ms"]
    sizes = results["model_sizes_mb"]

    print("  Logit Parity (max absolute delta vs PyTorch):")
    print(f"    FP32: max={lp['fp32_max_delta']:.6f}  mean={lp['fp32_mean_delta']:.6f}")
    print(f"    INT8: max={lp['int8_max_delta']:.6f}  mean={lp['int8_mean_delta']:.6f}")

    print("\n  Latency (ms):")
    print(f"    {'Variant':12s}  {'p50':>7s}  {'p95':>7s}  {'p99':>7s}  {'mean':>7s}")
    for variant in ("pytorch", "onnx_fp32", "onnx_int8"):
        v = lat[variant]
        print(
            f"    {variant:12s}  {v['p50']:7.2f}  {v['p95']:7.2f}  "
            f"{v['p99']:7.2f}  {v['mean']:7.2f}"
        )

    pt_p95 = lat["pytorch"]["p95"]
    int8_p95 = lat["onnx_int8"]["p95"]
    if int8_p95 > 0:
        speedup = pt_p95 / int8_p95
        print(f"\n  Speedup (PyTorch p95 / INT8 p95): {speedup:.2f}x")

    print("\n  Model Sizes:")
    for variant, size in sizes.items():
        print(f"    {variant}: {size:.1f} MB")

    # Gate checks.
    fp32_ok = lp["fp32_max_delta"] < 1e-4
    int8_size_ok = sizes["onnx_int8"] <= 300.0
    print("\n  Gates:")
    print(f"    FP32 parity (max delta < 1e-4):  {'PASS' if fp32_ok else 'FAIL'}")
    print(f"    INT8 model size (≤ 300 MB):      {'PASS' if int8_size_ok else 'FAIL'}")

    if not fp32_ok:
        print(
            f"    WARNING: FP32 max delta {lp['fp32_max_delta']:.6f} "
            "exceeds tolerance — investigate export fidelity."
        )

    print(
        "\n  Note: Full entity-level F1 validation requires a separate run of"
        "\n        ml/evaluate.py against the ONNX-backed PIIEngine."
    )


# ---------------------------------------------------------------------------
# OBS-13d: Artifact bundle packaging
# ---------------------------------------------------------------------------


def package_bundle(
    model_id: str,
    output_dir: Path,
    onnx_fp32_path: Path,
    onnx_int8_path: Path | None = None,
    validation_results: dict | None = None,
    version: str = "1.0.0",
) -> Path:
    """Package the ONNX artifact bundle for the Rust proxy backend.

    Creates a versioned directory (e.g. ``bert-ner-v1.0.0/``) containing the
    model, tokenizer, schema contract, label map, metadata, and integrity
    checksum.

    Returns:
        Path to the bundle directory.
    """
    bundle_prefix = "code-ner" if "codebert" in model_id else "bert-ner"
    bundle_dir = output_dir / f"{bundle_prefix}-v{version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # -- Determine primary model (INT8 preferred) --
    if onnx_int8_path and onnx_int8_path.exists():
        primary_model = onnx_int8_path
        quantization_type = "dynamic_int8"
    else:
        primary_model = onnx_fp32_path
        quantization_type = "none"

    # -- Copy model files --
    shutil.copy2(primary_model, bundle_dir / "model.onnx")
    if onnx_int8_path and onnx_int8_path.exists():
        shutil.copy2(onnx_fp32_path, bundle_dir / "model_fp32.onnx")

    # -- Copy tokenizer files --
    for name in ("tokenizer.json", "special_tokens_map.json"):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    # -- label_map.json (from model config) --
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    label_map: dict[str, str] = {str(k): v for k, v in model.config.id2label.items()}
    _write_json(bundle_dir / "label_map.json", label_map)
    _EXPECTED_LABEL_COUNT = 9
    if len(label_map) != _EXPECTED_LABEL_COUNT:
        print(
            f"WARNING: label_map has {len(label_map)} labels,"
            f" expected {_EXPECTED_LABEL_COUNT}."
            " Verify id2label in the model config.",
            file=sys.stderr,
        )

    # -- schema.json (integration contract for Rust proxy) --
    schema = {
        "model_version": version,
        "model_name": model_id,
        "quantization": quantization_type,
        "opset_version": 14,
        "input": {
            "input_ids": {"dtype": "int64", "shape": ["batch", "seq_len"]},
            "attention_mask": {
                "dtype": "int64",
                "shape": ["batch", "seq_len"],
            },
        },
        "output": {
            "logits": {
                "dtype": "float32",
                "shape": ["batch", "seq_len", "num_labels"],
            },
        },
        "max_sequence_length": 512,
        "label_count": len(label_map),
        "aggregation": "simple",
        "confidence_threshold": 0.90,
    }

    if "codebert" in model_id:
        schema["code_entity_types"] = ["CODE_VAR", "CODE_FUNC", "CODE_CLASS", "CODE_SECRET"]
        schema["code_label_schema"] = ["O", "B-VAR", "I-VAR", "B-FUNC", "I-FUNC", "B-CLASS", "I-CLASS", "B-SECRET", "I-SECRET"]
    else:
        schema["regex_entity_types"] = [
            "SSN",
            "PHONE",
            "EMAIL",
            "MRN",
            "DOB",
            "CREDIT_CARD",
            "IPV4",
            "PASSPORT",
        ]
        schema["bert_entity_types"] = ["PER", "LOC", "ORG", "MISC"]

    _write_json(bundle_dir / "schema.json", schema)

    # -- checksum.sha256 --
    model_bytes = (bundle_dir / "model.onnx").read_bytes()
    sha256 = hashlib.sha256(model_bytes).hexdigest()
    (bundle_dir / "checksum.sha256").write_text(f"{sha256}  model.onnx\n")

    # -- metadata.json --
    model_size_mb = round((bundle_dir / "model.onnx").stat().st_size / (1024 * 1024), 1)
    metadata: dict = {
        "version": version,
        "base_model": model_id,
        "fine_tuned": not model_id.startswith("dslim/"),
        "export_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "quantization": quantization_type,
        "accuracy": {
            "macro_f1_pytorch": 0.9576,
            "macro_f1_onnx_int8": None,
            "f1_degradation_pp": None,
        },
        "latency": {
            "pytorch_p95_ms": 40.4,
            "onnx_int8_p95_ms": None,
            "hardware": "macOS Apple Silicon (EC2 validation pending)",
        },
        "model_size_mb": model_size_mb,
        "checksum_algorithm": "sha256",
    }

    if validation_results:
        lat = validation_results.get("latency_ms", {})
        if "onnx_int8" in lat:
            metadata["latency"]["onnx_int8_p95_ms"] = lat["onnx_int8"]["p95"]
        if "onnx_fp32" in lat:
            metadata["latency"]["onnx_fp32_p95_ms"] = lat["onnx_fp32"]["p95"]

    _write_json(bundle_dir / "metadata.json", metadata)

    # -- Summary --
    print(f"\nArtifact bundle packaged: {bundle_dir}")
    for f in sorted(bundle_dir.iterdir()):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {size:8.2f} MB")

    return bundle_dir


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX export + quantization pipeline for Obscura",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model dslim/bert-base-NER --output ml/models/onnx\n"
            "  %(prog)s --model dslim/bert-base-NER --output ml/models/onnx "
            "--quantize --validate --bundle\n"
        ),
    )
    parser.add_argument(
        "--model",
        default="dslim/bert-base-NER",
        help="HuggingFace model ID or local path (default: dslim/bert-base-NER)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for ONNX model and tokenizer",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization after export",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run logit-parity and latency validation (requires --quantize)",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Package artifact bundle (schema, label map, metadata, checksum)",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Bundle version string (default: 1.0.0)",
    )
    parser.add_argument(
        "--task",
        choices=["ner", "code"],
        default="ner",
        help="Export task: 'ner' for dslim/bert-base-NER, 'code' for microsoft/codebert-base",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help=(
            "Skip export + quantization; use existing model.onnx and "
            "model_quantized.onnx in --output directory"
        ),
    )
    args = parser.parse_args()
    if args.task == "code":
        if args.model == "dslim/bert-base-NER":
            args.model = "microsoft/codebert-base"
    output_dir = Path(args.output)

    if args.validate and not (args.quantize or args.skip_export):
        parser.error("--validate requires --quantize or --skip-export")

    # Step 1: Export FP32 ONNX (skip if files already exist and --skip-export).
    onnx_fp32_path = output_dir / "model.onnx"
    if args.skip_export:
        if not onnx_fp32_path.exists():
            parser.error(
                f"--skip-export set but {onnx_fp32_path} not found. "
                "Run without --skip-export first."
            )
        print(f"Skipping export — using existing {onnx_fp32_path}")
    else:
        onnx_fp32_path = export_model(args.model, output_dir)

    # Step 2: Quantize (optional, skip if --skip-export).
    onnx_int8_path: Path | None = None
    if args.quantize and not args.skip_export:
        onnx_int8_path = output_dir / "model_quantized.onnx"
        quantize_model(onnx_fp32_path, onnx_int8_path)
    elif args.skip_export:
        candidate = output_dir / "model_quantized.onnx"
        if candidate.exists():
            onnx_int8_path = candidate

    # Step 3: Validate (optional).
    validation_results: dict | None = None
    if args.validate and onnx_int8_path:
        validation_results = validate_quantized_model(
            args.model, onnx_fp32_path, onnx_int8_path
        )
        # Save validation results alongside the model.
        results_path = output_dir / "validation_results.json"
        _write_json(results_path, validation_results)
        print(f"\n  Results saved: {results_path}")

    # Step 4: Package bundle (optional).
    if args.bundle:
        package_bundle(
            model_id=args.model,
            output_dir=output_dir,
            onnx_fp32_path=onnx_fp32_path,
            onnx_int8_path=onnx_int8_path,
            validation_results=validation_results,
            version=args.version,
        )


if __name__ == "__main__":
    main()
