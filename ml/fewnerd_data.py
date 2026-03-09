from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "FewNERD dataset support requires pandas and pyarrow. "
            "Install the ml dependencies before using this path."
        ) from exc
    return pd


def _require_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "FewNERD dataset support requires huggingface_hub."
        ) from exc
    return snapshot_download


FEWNERD_ORG_TAGS: dict[int, str] = {
    0: "O",
    1: "B-ORG",
    2: "I-ORG",
}


def resolve_fewnerd_org_snapshot(dataset_dir: str | Path | None = None) -> Path:
    if dataset_dir:
        candidate = Path(dataset_dir).expanduser()
        if candidate.is_dir() and list(candidate.glob("data/*.parquet")):
            return candidate

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--nbroad--fewnerd-organizations"
        / "snapshots"
    )
    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0]

    snapshot_download = _require_snapshot_download()
    snapshot_path = snapshot_download(
        repo_id="nbroad/fewnerd-organizations",
        repo_type="dataset",
        allow_patterns=["data/*.parquet", "README.md"],
    )
    return Path(snapshot_path)


def _resolve_split_path(dataset_dir: str | Path | None, split: str) -> Path:
    snapshot = resolve_fewnerd_org_snapshot(dataset_dir)
    matches = sorted(snapshot.glob(f"data/{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet file found for split '{split}' in {snapshot / 'data'}"
        )
    return matches[0]


def load_fewnerd_org_samples(
    dataset_dir: str | Path | None = None,
    *,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    pd = _require_pandas()
    split_path = _resolve_split_path(dataset_dir, split)
    frame = pd.read_parquet(split_path, columns=["tokens", "ner_tags"])
    if limit is not None and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)

    samples: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        tokens = [str(token) for token in row.tokens]
        ner_tag_labels = [FEWNERD_ORG_TAGS.get(int(tag), "O") for tag in row.ner_tags]
        samples.append(
            {
                "tokens": tokens,
                "ner_tags": [0 if label == "O" else (3 if label == "B-ORG" else 4) for label in ner_tag_labels],
                "ner_tag_labels": ner_tag_labels,
                "source": "fewnerd-org",
            }
        )
    return samples