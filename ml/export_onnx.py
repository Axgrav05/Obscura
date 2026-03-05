"""BYOM (Bring-Your-Own-Model) ONNX Exporter for Obscura

Converts a HuggingFace NER model from PyTorch to ONNX format for the
Rust proxy backend. Exports both the ONNX weight file and the
tokenizer.json required by the Rust ``tokenizers`` crate.

Usage:
    python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def export_model(model_id: str, output_dir: Path) -> None:
    """Download a HuggingFace NER model and export to ONNX.

    Args:
        model_id: HuggingFace model identifier (e.g. ``dslim/bert-base-NER``).
        output_dir: Directory to write ``model.onnx`` and ``tokenizer.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()

    # Create dummy input for tracing.
    dummy = tokenizer("test", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    # Export to ONNX with dynamic batch and sequence axes.
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting ONNX to: {onnx_path}")

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )

    # Save tokenizer files — the Rust backend requires tokenizer.json.
    print(f"Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(str(output_dir))

    tokenizer_json = output_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        print(
            f"ERROR: {tokenizer_json} not found. "
            "The Rust tokenizers crate requires this file.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nExport complete:")
    print(f"  ONNX model:  {onnx_path}")
    print(f"  Tokenizer:   {tokenizer_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a HuggingFace NER model to ONNX for Obscura"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dslim/bert-base-NER",
        help="HuggingFace model ID (default: dslim/bert-base-NER)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for ONNX model and tokenizer",
    )
    args = parser.parse_args()

    export_model(args.model, Path(args.output))


if __name__ == "__main__":
    main()
