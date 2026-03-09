from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "OntoNotes dataset support requires pandas and pyarrow. "
            "Install the ml dependencies before using this path."
        ) from exc
    return pd


def _require_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "OntoNotes dataset support requires huggingface_hub."
        ) from exc
    return snapshot_download


ONTONOTES_TAGS: dict[int, str] = {
    0: "O",
    4: "B-PER",
    5: "I-PER",
    7: "B-LOC",
    8: "I-LOC",
    11: "B-ORG",
    12: "I-ORG",
    23: "B-LOC",
    27: "I-LOC",
}


def resolve_ontonotes_org_snapshot(dataset_dir: str | Path | None = None) -> Path:
    if dataset_dir:
        candidate = Path(dataset_dir).expanduser()
        if candidate.is_dir() and list(candidate.glob("ontonotes5/train/*.parquet")):
            return candidate

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--tner--ontonotes5"
        / "snapshots"
    )
    if snapshots_dir.exists():
        snapshots = sorted(
            (
                path
                for path in snapshots_dir.iterdir()
                if path.is_dir() and list(path.glob("ontonotes5/train/*.parquet"))
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]

    snapshot_download = _require_snapshot_download()
    snapshot_path = snapshot_download(
        repo_id="tner/ontonotes5",
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=["ontonotes5/*.parquet", "ontonotes5/*/*.parquet"],
    )
    return Path(snapshot_path)


def _resolve_split_path(dataset_dir: str | Path | None, split: str) -> Path:
    snapshot = resolve_ontonotes_org_snapshot(dataset_dir)
    matches = sorted(snapshot.glob(f"ontonotes5/{split}/*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet file found for split '{split}' in {snapshot / 'ontonotes5' / split}"
        )
    return matches[0]


def load_ontonotes_org_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    pd = _require_pandas()
    split_path = _resolve_split_path(dataset_dir, split)
    frame = pd.read_parquet(split_path, columns=["tokens", "tags"])

    samples: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        tokens = [str(token) for token in row.tokens]
        ner_tag_labels = [ONTONOTES_TAGS.get(int(tag), "O") for tag in row.tags]
        if not any(label.endswith("ORG") for label in ner_tag_labels):
            continue

        samples.append(
            {
                "tokens": tokens,
                "ner_tags": [
                    0
                    if label == "O"
                    else (1 if label == "B-PER" else 2)
                    if label.endswith("PER")
                    else (3 if label == "B-ORG" else 4)
                    if label.endswith("ORG")
                    else (5 if label == "B-LOC" else 6)
                    for label in ner_tag_labels
                ],
                "ner_tag_labels": ner_tag_labels,
                "source": "ontonotes-org",
            }
        )

    if limit is not None and len(samples) > limit:
        import random

        rng = random.Random(seed)
        samples = rng.sample(samples, limit)

    return samples