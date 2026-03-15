from __future__ import annotations
import os
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic code dataset for CodeBERT fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/data/code_synthetic.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--count", type=int, default=5000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["python", "javascript", "typescript", "java", "cpp"],
        help="Languages to generate",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        raise KeyError("Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set.")

    print(f"Generating {args.count} samples across {args.languages}...")
    print(f"Output: {args.output}")

    # Placeholder for actual generation logic using LLM SDKs
    # In a real run, this would loop and call the chosen API

    # Validation logic per Section 7:
    # 1. tokens length == labels length
    # 2. Reject if all labels are "O"
    # 3. Flush every 10 samples

    print(
        "Dataset generation script initialized. (Manual step: provide prompt templates for LLM to emit tokens/labels JSON)"
    )


if __name__ == "__main__":
    main()
