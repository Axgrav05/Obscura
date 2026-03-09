from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

CANONICAL_LABELS: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-SSN",
    "I-SSN",
    "B-PHONE",
    "I-PHONE",
    "B-EMAIL",
    "I-EMAIL",
    "B-MRN",
    "I-MRN",
    "B-DOB",
    "I-DOB",
    "B-MISC",
    "I-MISC",
    "B-CREDIT_CARD",
    "I-CREDIT_CARD",
    "B-IPV4",
    "I-IPV4",
    "B-PASSPORT",
    "I-PASSPORT",
]
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(CANONICAL_LABELS)}

_PERSON_LABELS = frozenset({"first_name", "last_name", "full_name", "person_name"})
_ORG_LABELS = frozenset({"company_name"})
_LOC_LABELS = frozenset(
    {"street_address", "city", "state", "country", "county", "postcode", "coordinate"}
)
_MISC_LABELS = frozenset(
    {
        "race_ethnicity",
        "religious_belief",
        "blood_type",
        "gender",
        "age",
        "political_view",
        "sexuality",
        "date",
        "time",
        "date_time",
        "url",
    }
)
_STRUCTURED_LABELS: dict[str, str] = {
    "ssn": "SSN",
    "phone_number": "PHONE",
    "fax_number": "PHONE",
    "email": "EMAIL",
    "medical_record_number": "MRN",
    "health_plan_beneficiary_number": "MRN",
    "date_of_birth": "DOB",
    "credit_debit_card": "CREDIT_CARD",
    "ipv4": "IPV4",
}


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron dataset support requires pandas and pyarrow. "
            "Install the ml dependencies before using this path."
        ) from exc
    return pd


def resolve_nemotron_snapshot(dataset_dir: str | Path | None = None) -> Path | None:
    if dataset_dir:
        candidate = Path(dataset_dir).expanduser()
        if (candidate / "data").is_dir():
            return candidate
        if candidate.is_dir() and list(candidate.glob("train-*.parquet")):
            return candidate

    snapshots_dir = (
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--nvidia--Nemotron-PII" / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _resolve_split_path(dataset_dir: str | Path | None, split: str) -> Path:
    snapshot = resolve_nemotron_snapshot(dataset_dir)
    if snapshot is None:
        raise FileNotFoundError(
            "Nemotron-PII dataset not found. Download it with "
            "`hf download nvidia/Nemotron-PII --repo-type=dataset`."
        )

    data_dir = snapshot / "data" if (snapshot / "data").is_dir() else snapshot
    matches = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet file found for split '{split}' in {data_dir}")
    return matches[0]


def parse_spans(raw_spans: Any) -> list[dict[str, Any]]:
    if isinstance(raw_spans, list):
        return raw_spans
    if raw_spans is None:
        return []
    if isinstance(raw_spans, str):
        return ast.literal_eval(raw_spans) if raw_spans else []
    return list(raw_spans)


def canonicalize_label(nemotron_label: str) -> str | None:
    if nemotron_label in _PERSON_LABELS:
        return "PER"
    if nemotron_label in _ORG_LABELS:
        return "ORG"
    if nemotron_label in _LOC_LABELS:
        return "LOC"
    if nemotron_label in _MISC_LABELS:
        return "MISC"
    return _STRUCTURED_LABELS.get(nemotron_label)


def text_to_bio_sample(text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    char_labels: list[str] = ["O"] * len(text)

    mapped_spans: list[tuple[int, int, str]] = []
    for span in spans:
        canonical = canonicalize_label(str(span["label"]))
        if canonical is None:
            continue
        start = int(span["start"])
        end = int(span["end"])
        if start < 0 or end > len(text) or start >= end:
            continue
        mapped_spans.append((start, end, canonical))

    mapped_spans.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    for start, end, canonical in mapped_spans:
        if any(label != "O" for label in char_labels[start:end]):
            continue
        for idx in range(start, end):
            char_labels[idx] = canonical

    tokens: list[str] = []
    ner_tag_labels: list[str] = []
    previous_entity_type: str | None = None

    for match in re.finditer(r"\S+", text):
        token = match.group()
        token_labels = [label for label in char_labels[match.start() : match.end()] if label != "O"]
        if not token_labels:
            entity_type = "O"
        else:
            entity_type = token_labels[0]

        if entity_type == "O":
            ner_tag_labels.append("O")
            previous_entity_type = None
        else:
            prefix = "I" if previous_entity_type == entity_type else "B"
            ner_tag_labels.append(f"{prefix}-{entity_type}")
            previous_entity_type = entity_type
        tokens.append(token)

    return {
        "tokens": tokens,
        "ner_tags": [LABEL_TO_ID.get(tag, 0) for tag in ner_tag_labels],
        "ner_tag_labels": ner_tag_labels,
    }


def load_nemotron_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    pd = _require_pandas()
    split_path = _resolve_split_path(dataset_dir, split)
    frame = pd.read_parquet(split_path, columns=["uid", "domain", "locale", "text", "spans"])
    if limit is not None and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)

    samples: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        sample = text_to_bio_sample(str(row.text), parse_spans(row.spans))
        sample["source"] = "nemotron"
        sample["uid"] = str(row.uid)
        sample["domain"] = str(row.domain)
        sample["locale"] = str(row.locale)
        samples.append(sample)
    return samples


def load_nemotron_demo_texts(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int = 250,
    seed: int = 42,
) -> list[str]:
    pd = _require_pandas()
    split_path = _resolve_split_path(dataset_dir, split)
    frame = pd.read_parquet(split_path, columns=["text"])
    if len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)
    return [str(text) for text in frame["text"].tolist()]


def build_nemotron_jsonl(
    output_path: Path,
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> Counter[str]:
    samples = load_nemotron_samples(dataset_dir, split=split, limit=limit, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    with open(output_path, "w") as handle:
        for sample in samples:
            for tag in sample["ner_tag_labels"]:
                if tag != "O":
                    counts[tag.split("-", 1)[1]] += 1
            handle.write(json.dumps(sample) + "\n")
    return counts


def choose_demo_text(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    seed: int | None = None,
) -> str | None:
    texts = load_nemotron_demo_texts(dataset_dir, split=split, limit=250, seed=seed or 42)
    if not texts:
        return None
    rng = random.Random(seed)
    return rng.choice(texts)