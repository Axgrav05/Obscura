from __future__ import annotations

import argparse
import json
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

from ml.code_engine import CODE_ID2LABEL, CODE_LABEL2ID, CODE_LABELS

DEFAULT_MODEL = "microsoft/codebert-base"
DEFAULT_DATASET = "ml/data/code_synthetic.jsonl"
DEFAULT_OUTPUT = "ml/models/codebert-finetuned"
VALID_LABEL_ROOTS = frozenset({"VAR", "FUNC", "CLASS", "SECRET"})


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return []

    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            tokens = sample["tokens"]
            ner_tags = [
                CODE_LABEL2ID.get(tag, CODE_LABEL2ID["O"])
                for tag in sample["ner_tag_labels"]
            ]
            records.append({"tokens": tokens, "ner_tags": ner_tags})
    return records


def tokenize_and_align(examples: dict, tokenizer: AutoTokenizer) -> dict:
    # Verbatim copy from fine_tune.py as per §8 instructions
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
                label_row.append(-100)
            elif word_idx != previous_word_idx:
                label_row.append(label_ids[word_idx])
            else:
                original_label = label_ids[word_idx]
                original_name = CODE_ID2LABEL.get(original_label, "O")
                if original_name.startswith("B-"):
                    i_name = "I-" + original_name[2:]
                    label_row.append(CODE_LABEL2ID.get(i_name, original_label))
                else:
                    label_row.append(original_label)
            previous_word_idx = word_idx

        all_labels.append(label_row)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_preds: tuple) -> dict:
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_row: list[str] = []
        pred_row: list[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_row.append(CODE_ID2LABEL.get(int(label_id), "O"))
            pred_row.append(CODE_ID2LABEL.get(int(pred_id), "O"))
        true_labels.append(true_row)
        pred_labels.append(pred_row)

    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeBERT for code redaction"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

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

    records = load_jsonl(dataset_path)
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=1.0 - args.train_split, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(CODE_LABELS),
        id2label=CODE_ID2LABEL,
        label2id=CODE_LABEL2ID,
        ignore_mismatched_sizes=True,
    )

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
        fp16=True,
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

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to {output_dir}")

    print("\nFinal evaluation on held-out set:")
    metrics = trainer.evaluate()
    print(f"  Macro F1: {metrics.get('eval_macro_f1', 'N/A')}")
    print(f"  Loss: {metrics.get('eval_loss', 'N/A')}")


if __name__ == "__main__":
    main()
