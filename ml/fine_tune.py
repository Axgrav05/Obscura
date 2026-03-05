"""
Fine-tune dslim/bert-base-NER on Obscura synthetic data.

Trains the BERT token classification model on domain-specific enterprise
and clinical text to improve ORG/LOC/MISC detection. Structured PII types
(SSN, PHONE, EMAIL, MRN) are mapped to O since they are handled by the
regex layer in production.

Usage:
    python ml/fine_tune.py
    python ml/fine_tune.py --epochs 5 --lr 3e-5 --output ml/models/fine-tuned

Output:
    Saves the fine-tuned model and tokenizer to ml/models/fine-tuned/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# BIO labels for BERT-detectable entity types only.
# Regex-only types (SSN, PHONE, EMAIL, MRN, DOB) are mapped to O.
LABELS: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(LABELS)}

# Entity types handled by BERT (not regex).
BERT_TYPES: frozenset[str] = frozenset({"PER", "ORG", "LOC", "MISC"})


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
        default="data/synthetic.jsonl",
        help="Training dataset (default: data/synthetic.jsonl)",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve paths relative to ml/ directory.
    ml_dir = Path(__file__).parent
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ml_dir / dataset_path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ml_dir / output_dir

    print(f"Base model: {args.model}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Seed: {args.seed}")

    # Load and split data.
    records = load_jsonl(dataset_path)
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1.0 - args.train_split, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
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

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    main()
