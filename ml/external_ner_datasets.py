from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable


LABEL2ID: dict[str, int] = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-PHONE": 7,
    "I-PHONE": 8,
    "B-MISC": 9,
    "I-MISC": 10,
}


def _require_load_dataset() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "External NER dataset support requires the datasets package."
        ) from exc
    return load_dataset


def _require_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "External NER dataset support requires huggingface_hub."
        ) from exc
    return snapshot_download


def _cache_dir(dataset_dir: str | Path | None) -> str | None:
    if dataset_dir is None:
        return None
    return str(Path(dataset_dir).expanduser())


def _resolve_parquet_snapshot(
    repo_id: str,
    patterns: list[str],
    dataset_dir: str | Path | None = None,
) -> Path:
    if dataset_dir is not None:
        candidate = Path(dataset_dir).expanduser()
        if candidate.is_dir() and any(candidate.glob(pattern) for pattern in patterns):
            return candidate

    snapshot_download = _require_snapshot_download()
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=patterns + [".gitattributes"],
    )
    return Path(snapshot_path)


def _load_parquet_split(
    repo_id: str,
    split_glob: str,
    dataset_dir: str | Path | None = None,
) -> Any:
    load_dataset = _require_load_dataset()
    snapshot = _resolve_parquet_snapshot(repo_id, [split_glob], dataset_dir)
    parquet_files = sorted(str(path) for path in snapshot.glob(split_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {repo_id} split glob {split_glob}")
    return load_dataset("parquet", data_files=parquet_files, split="train")


def _normalize_label_name(label_name: str, mapping: dict[str, str]) -> str:
    return mapping.get(label_name, "O")


def _project_label_ids(labels: Iterable[str]) -> list[int]:
    return [LABEL2ID.get(label, 0) for label in labels]


def _sample_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if limit is None or len(records) <= limit:
        return records
    rng = random.Random(seed)
    return rng.sample(records, limit)


def _build_samples(
    rows: Iterable[dict[str, Any]],
    *,
    tokens_key: str,
    label_names: list[str],
    label_key: str,
    mapping: dict[str, str],
    source: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        tokens = [str(token) for token in row[tokens_key]]
        raw_ids = row[label_key]
        ner_tag_labels = [_normalize_label_name(label_names[int(tag)], mapping) for tag in raw_ids]
        if not any(label != "O" for label in ner_tag_labels):
            continue
        samples.append(
            {
                "tokens": tokens,
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": source,
            }
        )
    return samples


ONTONOTES_MAPPING: dict[str, str] = {
    "PERSON": "B-PER",
    "ORG": "B-ORG",
    "GPE": "B-LOC",
    "LOC": "B-LOC",
    "FAC": "B-LOC",
    "DATE": "B-MISC",
    "TIME": "B-MISC",
    "EVENT": "B-MISC",
    "PRODUCT": "B-MISC",
    "WORK_OF_ART": "B-MISC",
    "LAW": "B-MISC",
    "LANGUAGE": "B-MISC",
    "NORP": "B-MISC",
    "CARDINAL": "B-MISC",
    "ORDINAL": "B-MISC",
    "MONEY": "B-MISC",
    "PERCENT": "B-MISC",
    "QUANTITY": "B-MISC",
}

CONLL2003_MAPPING: dict[str, str] = {
    "PER": "B-PER",
    "ORG": "B-ORG",
    "LOC": "B-LOC",
    "MISC": "B-MISC",
}

WNUT17_MAPPING: dict[str, str] = {
    "person": "B-PER",
    "location": "B-LOC",
    "corporation": "B-ORG",
    "group": "B-ORG",
    "creative-work": "B-MISC",
    "product": "B-MISC",
}

FEWNERD_MAPPING: dict[str, str] = {
    "person": "B-PER",
    "organization": "B-ORG",
    "location": "B-LOC",
    "building": "B-LOC",
    "art": "B-MISC",
    "event": "B-MISC",
    "product": "B-MISC",
    "other": "B-MISC",
}

MULTINERD_MAPPING: dict[str, str] = {
    "PER": "B-PER",
    "ORG": "B-ORG",
    "LOC": "B-LOC",
    "TIME": "B-MISC",
    "ANIM": "B-MISC",
    "BIO": "B-MISC",
    "CEL": "B-MISC",
    "DIS": "B-MISC",
    "EVE": "B-MISC",
    "FOOD": "B-MISC",
    "INST": "B-MISC",
    "MEDIA": "B-MISC",
    "MYTH": "B-MISC",
    "PLANT": "B-MISC",
    "VEHI": "B-MISC",
}

ONTONOTES5_LABEL_NAMES: list[str] = [
    "O",
    "B-CARDINAL",
    "B-DATE",
    "I-DATE",
    "B-PERSON",
    "I-PERSON",
    "B-NORP",
    "B-GPE",
    "I-GPE",
    "B-LAW",
    "I-LAW",
    "B-ORG",
    "I-ORG",
    "B-PERCENT",
    "I-PERCENT",
    "B-ORDINAL",
    "B-MONEY",
    "I-MONEY",
    "B-WORK_OF_ART",
    "I-WORK_OF_ART",
    "B-FAC",
    "B-TIME",
    "I-CARDINAL",
    "B-LOC",
    "B-QUANTITY",
    "I-QUANTITY",
    "I-NORP",
    "I-LOC",
    "B-PRODUCT",
    "I-TIME",
    "B-EVENT",
    "I-EVENT",
    "I-FAC",
    "B-LANGUAGE",
    "I-PRODUCT",
    "I-ORDINAL",
    "I-LANGUAGE",
]

MULTINERD_LABEL_NAMES: list[str] = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-ANIM",
    "I-ANIM",
    "B-BIO",
    "I-BIO",
    "B-CEL",
    "I-CEL",
    "B-DIS",
    "I-DIS",
    "B-EVE",
    "I-EVE",
    "B-FOOD",
    "I-FOOD",
    "B-INST",
    "I-INST",
    "B-MEDIA",
    "I-MEDIA",
    "B-MYTH",
    "I-MYTH",
    "B-PLANT",
    "I-PLANT",
    "B-TIME",
    "I-TIME",
    "B-VEHI",
    "I-VEHI",
]


def _expand_bio(label_name: str, mapped_base: str) -> str:
    if mapped_base == "O" or label_name == "O":
        return "O"
    prefix = "I-" if label_name.startswith("I-") else "B-"
    entity_type = mapped_base.split("-", 1)[1]
    return f"{prefix}{entity_type}"


def _map_labels(label_names: list[str], raw_ids: Iterable[int], mapping: dict[str, str]) -> list[str]:
    mapped: list[str] = []
    for raw_id in raw_ids:
        raw_label = label_names[int(raw_id)]
        if raw_label == "O":
            mapped.append("O")
            continue

        raw_type = raw_label.split("-", 1)[1] if "-" in raw_label else raw_label
        base = _normalize_label_name(raw_type, mapping)
        mapped.append(_expand_bio(raw_label, base))
    return mapped


def _map_io_sequence(label_names: list[str], raw_ids: Iterable[int], mapping: dict[str, str]) -> list[str]:
    mapped: list[str] = []
    previous_type: str | None = None
    for raw_id in raw_ids:
        raw_label = label_names[int(raw_id)]
        if raw_label == "O":
            mapped.append("O")
            previous_type = None
            continue

        mapped_base = _normalize_label_name(raw_label, mapping)
        if mapped_base == "O":
            mapped.append("O")
            previous_type = None
            continue

        entity_type = mapped_base.split("-", 1)[1]
        prefix = "I" if previous_type == entity_type else "B"
        mapped.append(f"{prefix}-{entity_type}")
        previous_type = entity_type
    return mapped


def load_ontonotes5_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset = _load_parquet_split("tner/ontonotes5", f"ontonotes5/{split}/*.parquet", dataset_dir)
    label_names = ONTONOTES5_LABEL_NAMES

    records: list[dict[str, Any]] = []
    for row in dataset:
        ner_tag_labels = _map_labels(label_names, row["tags"], ONTONOTES_MAPPING)
        if not any(label != "O" for label in ner_tag_labels):
            continue
        records.append(
            {
                "tokens": [str(token) for token in row["tokens"]],
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": "ontonotes5",
            }
        )
    return _sample_records(records, limit=limit, seed=seed)


def load_conll2003_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset = _load_parquet_split("conll2003", f"conll2003/{split}/*.parquet", dataset_dir)
    label_names = list(dataset.features["ner_tags"].feature.names)

    records: list[dict[str, Any]] = []
    for row in dataset:
        ner_tag_labels = _map_labels(label_names, row["ner_tags"], CONLL2003_MAPPING)
        if not any(label != "O" for label in ner_tag_labels):
            continue
        records.append(
            {
                "tokens": [str(token) for token in row["tokens"]],
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": "conll2003",
            }
        )
    return _sample_records(records, limit=limit, seed=seed)


def load_wnut17_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset = _load_parquet_split("wnut_17", f"wnut_17/{split}/*.parquet", dataset_dir)
    label_names = list(dataset.features["ner_tags"].feature.names)

    records: list[dict[str, Any]] = []
    for row in dataset:
        ner_tag_labels = _map_labels(label_names, row["ner_tags"], WNUT17_MAPPING)
        if not any(label != "O" for label in ner_tag_labels):
            continue
        records.append(
            {
                "tokens": [str(token) for token in row["tokens"]],
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": "wnut17",
            }
        )
    return _sample_records(records, limit=limit, seed=seed)


def load_fewnerd_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset = _load_parquet_split(
        "DFKI-SLT/few-nerd",
        f"supervised/{split}/*.parquet",
        dataset_dir,
    )
    label_names = list(dataset.features["ner_tags"].feature.names)

    records: list[dict[str, Any]] = []
    for row in dataset:
        ner_tag_labels = _map_io_sequence(label_names, row["ner_tags"], FEWNERD_MAPPING)
        if not any(label != "O" for label in ner_tag_labels):
            continue
        records.append(
            {
                "tokens": [str(token) for token in row["tokens"]],
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": "fewnerd",
            }
        )
    return _sample_records(records, limit=limit, seed=seed)


def load_multinerd_en_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    dataset = _load_parquet_split("Babelscape/multinerd", f"default/{split}/*.parquet", dataset_dir)
    label_names = MULTINERD_LABEL_NAMES

    records: list[dict[str, Any]] = []
    for row in dataset:
        if row.get("lang") != "en":
            continue
        ner_tag_labels = _map_labels(label_names, row["ner_tags"], MULTINERD_MAPPING)
        if not any(label != "O" for label in ner_tag_labels):
            continue
        records.append(
            {
                "tokens": [str(token) for token in row["tokens"]],
                "ner_tags": _project_label_ids(ner_tag_labels),
                "ner_tag_labels": ner_tag_labels,
                "source": "multinerd-en",
            }
        )

    return _sample_records(records, limit=limit, seed=seed)


__all__ = [
    "load_conll2003_samples",
    "load_fewnerd_samples",
    "load_multinerd_en_samples",
    "load_ontonotes5_samples",
    "load_wnut17_samples",
]