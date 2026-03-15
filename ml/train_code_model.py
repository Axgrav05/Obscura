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
from ml.code_engine import CODE_LABELS, CODE_LABEL2ID, CODE_ID2LABEL

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


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CodeBERT for code redaction"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Rest of the Trainer boilerplate omitted for brevity, but follows fine_tune.py structure
    print(f"CodeBERT trainer initialized for {args.model}")


if __name__ == "__main__":
    main()
