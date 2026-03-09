"""
Fine-tune dslim/bert-base-NER on Obscura synthetic data.

Trains the BERT token classification model on domain-specific enterprise
and clinical text to improve ORG/LOC/MISC/PHONE detection. Some structured
PII types still fall back to regex in production, but PHONE is trained as a
semantic class to improve recall on varied formats.

Usage:
    python ml/fine_tune.py
    python ml/fine_tune.py --epochs 5 --lr 3e-5 --output ml/models/fine-tuned
    python ml/fine_tune.py --dataset nemotron --train-limit 15000 --eval-limit 3000

Output:
    Saves the fine-tuned model and tokenizer to ml/models/fine-tuned/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from seqeval.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ml.external_ner_datasets import (
    load_conll2003_samples,
    load_fewnerd_samples,
    load_multinerd_en_samples,
    load_ontonotes5_samples,
    load_wnut17_samples,
)
from ml.fewnerd_data import load_fewnerd_org_samples, resolve_fewnerd_org_snapshot
from ml.nemotron_data import load_nemotron_samples, resolve_nemotron_snapshot
from ml.ontonotes_data import load_ontonotes_org_samples, resolve_ontonotes_org_snapshot

# BIO labels for BERT-detectable entity types.
# Regex still handles SSN/EMAIL/MRN/DOB and remains authoritative for PHONE
# on exact overlaps, but we train PHONE to improve recall on varied formats.
LABELS: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-PHONE",
    "I-PHONE",
    "B-MISC",
    "I-MISC",
]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(LABELS)}

# Entity types handled by BERT.
BERT_TYPES: frozenset[str] = frozenset({"PER", "ORG", "LOC", "PHONE", "MISC"})


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL dataset and map non-BERT tags to O."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            tokens = sample["tokens"]

            # Map regex-only entity tags to O for BERT training.
            tags: list[int] = []
            for tag_str in sample["ner_tag_labels"]:
                if tag_str == "O":
                    tags.append(LABEL2ID["O"])
                else:
                    parts = tag_str.split("-", 1)
                    if len(parts) == 2 and parts[1] in BERT_TYPES:
                        tags.append(LABEL2ID.get(tag_str, 0))
                    else:
                        tags.append(LABEL2ID["O"])

            records.append({"tokens": tokens, "ner_tags": tags})
    return records


def resolve_repo_or_ml_path(raw_path: str, ml_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    repo_root = ml_dir.parent
    repo_candidate = repo_root / path
    if repo_candidate.exists() or raw_path.startswith("ml/"):
        return repo_candidate

    return ml_dir / path


def _load_single_dataset(
    dataset_spec: str,
    args: argparse.Namespace,
    ml_dir: Path,
) -> tuple[Dataset, Dataset, str]:
    dataset_spec = dataset_spec.strip()

    if dataset_spec == "nemotron":
        snapshot = resolve_nemotron_snapshot(args.nemotron_dir)
        if snapshot is None:
            raise SystemExit(
                "Nemotron-PII dataset not found. Download it with "
                "`hf download nvidia/Nemotron-PII --repo-type=dataset` or pass --nemotron-dir."
            )

        train_records = load_nemotron_samples(
            snapshot,
            split="train",
            limit=args.train_limit,
            seed=args.seed,
        )
        eval_records = load_nemotron_samples(
            snapshot,
            split="test",
            limit=args.eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), str(snapshot)

    if dataset_spec == "fewnerd-org":
        snapshot = resolve_fewnerd_org_snapshot(args.fewnerd_org_dir)
        train_records = load_fewnerd_org_samples(
            snapshot,
            split="train",
            limit=args.fewnerd_org_train_limit,
            seed=args.seed,
        )
        eval_records = load_fewnerd_org_samples(
            snapshot,
            split="validation",
            limit=args.fewnerd_org_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), str(snapshot)

    if dataset_spec == "ontonotes-org":
        snapshot = resolve_ontonotes_org_snapshot(args.ontonotes_org_dir)
        train_records = load_ontonotes_org_samples(
            snapshot,
            split="train",
            limit=args.ontonotes_org_train_limit,
            seed=args.seed,
        )
        eval_records = load_ontonotes_org_samples(
            snapshot,
            split="validation",
            limit=args.ontonotes_org_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), str(snapshot)

    if dataset_spec == "ontonotes5":
        train_records = load_ontonotes5_samples(
            args.ontonotes5_dir,
            split="train",
            limit=args.ontonotes5_train_limit,
            seed=args.seed,
        )
        eval_records = load_ontonotes5_samples(
            args.ontonotes5_dir,
            split="validation",
            limit=args.ontonotes5_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), "tner/ontonotes5"

    if dataset_spec == "fewnerd":
        train_records = load_fewnerd_samples(
            args.fewnerd_dir,
            split="train",
            limit=args.fewnerd_train_limit,
            seed=args.seed,
        )
        eval_records = load_fewnerd_samples(
            args.fewnerd_dir,
            split="validation",
            limit=args.fewnerd_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), "DFKI-SLT/few-nerd"

    if dataset_spec == "conll2003":
        train_records = load_conll2003_samples(
            args.conll2003_dir,
            split="train",
            limit=args.conll2003_train_limit,
            seed=args.seed,
        )
        eval_records = load_conll2003_samples(
            args.conll2003_dir,
            split="validation",
            limit=args.conll2003_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), "conll2003"

    if dataset_spec == "wnut17":
        train_records = load_wnut17_samples(
            args.wnut17_dir,
            split="train",
            limit=args.wnut17_train_limit,
            seed=args.seed,
        )
        eval_records = load_wnut17_samples(
            args.wnut17_dir,
            split="validation",
            limit=args.wnut17_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), "wnut_17"

    if dataset_spec == "multinerd-en":
        train_records = load_multinerd_en_samples(
            args.multinerd_dir,
            split="train",
            limit=args.multinerd_train_limit,
            seed=args.seed,
        )
        eval_records = load_multinerd_en_samples(
            args.multinerd_dir,
            split="validation",
            limit=args.multinerd_eval_limit,
            seed=args.seed,
        )
        return Dataset.from_list(train_records), Dataset.from_list(eval_records), "Babelscape/multinerd:en"

    dataset_path = resolve_repo_or_ml_path(dataset_spec, ml_dir)

    records = load_jsonl(dataset_path)
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1.0 - args.train_split, seed=args.seed)
    return split["train"], split["test"], str(dataset_path)


def load_training_datasets(args: argparse.Namespace, ml_dir: Path) -> tuple[Dataset, Dataset, str]:
    dataset_specs = [spec.strip() for spec in args.dataset.split(",") if spec.strip()]
    if not dataset_specs:
        raise SystemExit("No dataset sources were provided.")

    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []
    descriptions: list[str] = []

    for dataset_spec in dataset_specs:
        train_ds, eval_ds, description = _load_single_dataset(dataset_spec, args, ml_dir)
        train_parts.append(train_ds)
        eval_parts.append(eval_ds)
        descriptions.append(description)

    train_ds = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts).shuffle(seed=args.seed)
    eval_ds = eval_parts[0] if len(eval_parts) == 1 else concatenate_datasets(eval_parts).shuffle(seed=args.seed)
    return train_ds, eval_ds, ", ".join(descriptions)


def tokenize_and_align(
    examples: dict,
    tokenizer: AutoTokenizer,
) -> dict:
    """Tokenize and align BIO labels with subword tokens.

    For subword continuations of the same word, B- labels are converted
    to I- labels. Special tokens ([CLS], [SEP], padding) get label -100
    so they're ignored by the loss function.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
    )

    all_labels: list[list[int]] = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_row: list[int] = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                label_row.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a new word — use original label.
                label_row.append(label_ids[word_idx])
            else:
                # Subword continuation — convert B- to I-.
                original_label = label_ids[word_idx]
                original_name = ID2LABEL.get(original_label, "O")
                if original_name.startswith("B-"):
                    i_name = "I-" + original_name[2:]
                    label_row.append(LABEL2ID.get(i_name, original_label))
                else:
                    label_row.append(original_label)
            previous_word_idx = word_idx

        all_labels.append(label_row)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_preds: tuple) -> dict:
    """Compute seqeval metrics for the Trainer callback."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Convert IDs back to label strings, skipping -100 padding.
    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_row: list[str] = []
        pred_row: list[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_row.append(ID2LABEL.get(int(label_id), "O"))
            pred_row.append(ID2LABEL.get(int(pred_id), "O"))
        true_labels.append(true_row)
        pred_labels.append(pred_row)

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT NER on Obscura data")
    parser.add_argument(
        "--model",
        type=str,
        default="dslim/bert-base-NER",
        help="Base model to fine-tune (default: dslim/bert-base-NER)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nemotron",
        help=(
            "Training dataset path or dataset key. Supported keys include "
            "'nemotron', 'ontonotes5', 'fewnerd', 'conll2003', 'wnut17', "
            "'multinerd-en', plus the legacy 'fewnerd-org' and 'ontonotes-org'. "
            "You can pass a comma-separated combination."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/fine-tuned",
        help="Output directory for fine-tuned model (default: models/fine-tuned)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--nemotron-dir",
        type=str,
        default=None,
        help="Optional path to a downloaded Nemotron-PII snapshot",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional cap on training rows when using Nemotron",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional cap on eval rows when using Nemotron",
    )
    parser.add_argument(
        "--fewnerd-org-dir",
        type=str,
        default=None,
        help="Optional path to a downloaded nbroad/fewnerd-organizations snapshot",
    )
    parser.add_argument(
        "--fewnerd-org-train-limit",
        type=int,
        default=None,
        help="Optional cap on FewNERD organization training rows",
    )
    parser.add_argument(
        "--fewnerd-org-eval-limit",
        type=int,
        default=None,
        help="Optional cap on FewNERD organization eval rows",
    )
    parser.add_argument(
        "--ontonotes-org-dir",
        type=str,
        default=None,
        help="Optional path to a downloaded tner/ontonotes5 parquet snapshot",
    )
    parser.add_argument(
        "--ontonotes-org-train-limit",
        type=int,
        default=None,
        help="Optional cap on OntoNotes organization training rows",
    )
    parser.add_argument(
        "--ontonotes-org-eval-limit",
        type=int,
        default=None,
        help="Optional cap on OntoNotes organization eval rows",
    )
    parser.add_argument(
        "--ontonotes5-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory for tner/ontonotes5",
    )
    parser.add_argument(
        "--ontonotes5-train-limit",
        type=int,
        default=None,
        help="Optional cap on tner/ontonotes5 training rows",
    )
    parser.add_argument(
        "--ontonotes5-eval-limit",
        type=int,
        default=None,
        help="Optional cap on tner/ontonotes5 eval rows",
    )
    parser.add_argument(
        "--fewnerd-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory for DFKI-SLT/few-nerd",
    )
    parser.add_argument(
        "--fewnerd-train-limit",
        type=int,
        default=None,
        help="Optional cap on DFKI-SLT/few-nerd training rows",
    )
    parser.add_argument(
        "--fewnerd-eval-limit",
        type=int,
        default=None,
        help="Optional cap on DFKI-SLT/few-nerd eval rows",
    )
    parser.add_argument(
        "--conll2003-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory for conll2003",
    )
    parser.add_argument(
        "--conll2003-train-limit",
        type=int,
        default=None,
        help="Optional cap on conll2003 training rows",
    )
    parser.add_argument(
        "--conll2003-eval-limit",
        type=int,
        default=None,
        help="Optional cap on conll2003 eval rows",
    )
    parser.add_argument(
        "--wnut17-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory for wnut_17",
    )
    parser.add_argument(
        "--wnut17-train-limit",
        type=int,
        default=None,
        help="Optional cap on wnut_17 training rows",
    )
    parser.add_argument(
        "--wnut17-eval-limit",
        type=int,
        default=None,
        help="Optional cap on wnut_17 eval rows",
    )
    parser.add_argument(
        "--multinerd-dir",
        type=str,
        default=None,
        help="Optional datasets cache directory for Babelscape/multinerd",
    )
    parser.add_argument(
        "--multinerd-train-limit",
        type=int,
        default=None,
        help="Optional cap on English MultiNERD training rows",
    )
    parser.add_argument(
        "--multinerd-eval-limit",
        type=int,
        default=None,
        help="Optional cap on English MultiNERD eval rows",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve paths relative to ml/ directory.
    ml_dir = Path(__file__).parent
    output_dir = resolve_repo_or_ml_path(args.output, ml_dir)

    print(f"Base model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Seed: {args.seed}")

    train_ds, eval_ds, dataset_description = load_training_datasets(args, ml_dir)
    print(f"Dataset: {dataset_description}")
    print(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # Load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # Classifier head size changes (9 vs 4)
    )

    # Tokenize datasets.
    train_tokenized = train_ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tokenized = eval_ds.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    use_cuda_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
        fp16=use_cuda_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    # Save best model.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to {output_dir}")

    # Final evaluation.
    print("\nFinal evaluation on held-out set:")
    metrics = trainer.evaluate()
    print(f"  Macro F1: {metrics.get('eval_macro_f1', 'N/A')}")
    print(f"  Loss: {metrics.get('eval_loss', 'N/A')}")


if __name__ == "__main__":
    main()
