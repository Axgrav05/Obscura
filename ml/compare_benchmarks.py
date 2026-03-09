"""Generate a benchmark PNG and example comparison for two Obscura models.

Usage:
    python ml/compare_benchmarks.py \
        --baseline-result ml/results/old.json \
        --candidate-result ml/results/new.json \
        --baseline-model ml/models/baselines/current-fine-tuned-v2 \
        --candidate-model ml/models/nemotron-hybrid-fine-tuned
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from seqeval.metrics import f1_score

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ml.evaluate import (
    align_hybrid_predictions_to_words,
    filter_to_hybrid_tags,
    load_dataset,
)
from ml.pii_engine import PIIEngine

PNG_WIDTH = 1600
PNG_HEIGHT = 980
BG = "#f6f3ec"
TEXT = "#1e1b18"
MUTED = "#6b6259"
CARD = "#fffaf2"
CARD_BORDER = "#ddd2c2"
BASELINE = "#c7694f"
CANDIDATE = "#2d7f64"
ACCENT = "#c8b786"


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def parse_classification_report(report: str) -> dict[str, dict[str, float]]:
    pattern = re.compile(
        r"^\s*([A-Z_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$"
    )
    rows: dict[str, dict[str, float]] = {}
    for line in report.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        label, precision, recall, f1_value, support = match.groups()
        rows[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_value),
            "support": int(support),
        }
    return rows


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    if bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        ] + candidates

    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=24, fill=CARD, outline=CARD_BORDER, width=2)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.ImageFont, fill: str = TEXT) -> None:
    draw.text(xy, value, font=font, fill=fill)


def fmt_delta(delta: float, suffix: str = "") -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}{suffix}"


def draw_metric_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    baseline_value: float,
    candidate_value: float,
    suffix: str = "",
    delta_suffix: str = "",
) -> None:
    draw_card(draw, box)
    x1, y1, x2, y2 = box
    width = x2 - x1
    title_font = load_font(24, bold=True)
    value_font = load_font(42, bold=True)
    meta_font = load_font(20)
    delta_font = load_font(22, bold=True)

    text(draw, (x1 + 28, y1 + 24), title, title_font, MUTED)
    text(draw, (x1 + 28, y1 + 78), f"Old  {baseline_value:.4f}{suffix}", meta_font, BASELINE)
    text(draw, (x1 + 28, y1 + 112), f"New {candidate_value:.4f}{suffix}", meta_font, CANDIDATE)

    delta = candidate_value - baseline_value
    delta_color = CANDIDATE if delta >= 0 else BASELINE
    text(draw, (x1 + 28, y1 + 152), fmt_delta(delta, delta_suffix), delta_font, delta_color)

    bar_left = x1 + 28
    bar_right = x1 + width - 28
    bar_top = y2 - 46
    bar_bottom = y2 - 24
    draw.rounded_rectangle((bar_left, bar_top, bar_right, bar_bottom), radius=10, fill="#eee2d5")

    max_value = max(baseline_value, candidate_value, 0.0001)
    inner_width = bar_right - bar_left
    old_width = int(inner_width * (baseline_value / max_value))
    new_width = int(inner_width * (candidate_value / max_value))
    draw.rounded_rectangle((bar_left, bar_top, bar_left + old_width, bar_bottom), radius=10, fill=BASELINE)
    draw.rounded_rectangle((bar_left, bar_top + 10, bar_left + new_width, bar_bottom - 10), radius=8, fill=CANDIDATE)
    text(draw, (x2 - 180, y1 + 78), "benchmark", meta_font, MUTED)
    text(draw, (x2 - 180, y1 + 112), "comparison", meta_font, MUTED)


def draw_entity_gains(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    baseline_entities: dict[str, dict[str, float]],
    candidate_entities: dict[str, dict[str, float]],
) -> None:
    draw_card(draw, box)
    x1, y1, x2, y2 = box
    title_font = load_font(28, bold=True)
    label_font = load_font(21, bold=True)
    meta_font = load_font(18)
    text(draw, (x1 + 28, y1 + 24), "Per-entity F1 gains", title_font)

    rows = []
    for label, metrics in candidate_entities.items():
        if label not in baseline_entities:
            continue
        support = metrics["support"]
        if support <= 0:
            continue
        delta = metrics["f1"] - baseline_entities[label]["f1"]
        rows.append((delta, label, baseline_entities[label]["f1"], metrics["f1"], support))

    rows.sort(reverse=True)
    rows = rows[:6]
    max_gain = max(max(delta for delta, *_ in rows), 0.01) if rows else 0.01

    top = y1 + 82
    row_height = 72
    bar_left = x1 + 220
    bar_right = x2 - 30
    bar_width = bar_right - bar_left

    for index, (delta, label, old_f1, new_f1, support) in enumerate(rows):
        y = top + index * row_height
        text(draw, (x1 + 28, y + 6), label, label_font)
        text(draw, (x1 + 28, y + 34), f"old {old_f1:.2f}  new {new_f1:.2f}  n={support}", meta_font, MUTED)

        draw.rounded_rectangle((bar_left, y + 18, bar_right, y + 44), radius=10, fill="#eee2d5")
        gain_width = int(bar_width * (delta / max_gain)) if max_gain else 0
        draw.rounded_rectangle((bar_left, y + 18, bar_left + gain_width, y + 44), radius=10, fill=CANDIDATE)
        text(draw, (bar_right - 92, y + 18), fmt_delta(delta), meta_font, TEXT)


def draw_summary(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], baseline: dict, candidate: dict) -> None:
    draw_card(draw, box)
    x1, y1, _, _ = box
    title_font = load_font(28, bold=True)
    body_font = load_font(21)
    strong_font = load_font(22, bold=True)

    macro_gain = candidate["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"]
    p95_delta = candidate["latency"]["p95_ms"] - baseline["latency"]["p95_ms"]
    rss_delta = candidate["memory"]["rss_peak_mb"] - baseline["memory"]["rss_peak_mb"]

    text(draw, (x1 + 28, y1 + 24), "Readout", title_font)
    text(draw, (x1 + 28, y1 + 82), f"Macro F1 moved from {baseline['metrics']['macro_f1']:.4f} to {candidate['metrics']['macro_f1']:.4f}.", body_font)
    text(draw, (x1 + 28, y1 + 118), f"That is a {macro_gain:.4f} absolute gain, or {macro_gain / baseline['metrics']['macro_f1'] * 100:.1f}% relative.", strong_font, CANDIDATE)
    text(draw, (x1 + 28, y1 + 164), f"Latency stayed nearly flat: p95 {baseline['latency']['p95_ms']:.2f} ms to {candidate['latency']['p95_ms']:.2f} ms ({p95_delta:+.2f} ms).", body_font)
    text(draw, (x1 + 28, y1 + 200), f"Peak RSS moved by {rss_delta:+.1f} MB on the evaluated CPU path.", body_font)
    text(draw, (x1 + 28, y1 + 246), f"Benchmark: {baseline['num_samples']} Nemotron test samples in hybrid mode.", body_font, MUTED)


def tag_entities(tokens: list[str], tags: list[str]) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    current_type: str | None = None
    current_tokens: list[str] = []
    start_index = 0

    def flush(end_index: int) -> None:
        nonlocal current_type, current_tokens, start_index
        if current_type is None:
            return
        entities.append(
            {
                "label": current_type,
                "start": start_index,
                "end": end_index,
                "text": " ".join(current_tokens),
            }
        )
        current_type = None
        current_tokens = []

    for index, (token, tag) in enumerate(zip(tokens, tags, strict=False)):
        if tag == "O":
            flush(index)
            continue

        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            flush(index)
            current_type = entity_type
            current_tokens = [token]
            start_index = index
        else:
            current_tokens.append(token)

    flush(len(tokens))
    return entities


def entity_key(entity: dict[str, object]) -> tuple[str, int, int, str]:
    return (
        str(entity["label"]),
        int(entity["start"]),
        int(entity["end"]),
        str(entity["text"]),
    )


def format_entities(entities: list[dict[str, object]]) -> str:
    if not entities:
        return "none"
    return ", ".join(f"{entity['text']} [{entity['label']}]" for entity in entities)


def extract_examples(
    dataset_path: Path,
    baseline_model: str,
    candidate_model: str,
    num_examples: int,
    sample_limit: int,
) -> list[dict[str, object]]:
    samples = load_dataset(dataset_path, limit=sample_limit)

    baseline_engine = PIIEngine(model_id=baseline_model, enable_regex=True)
    baseline_engine.load()

    candidate_engine = PIIEngine(model_id=candidate_model, enable_regex=True)
    candidate_engine.load()

    ranked_examples: list[dict[str, object]] = []

    for sample in samples:
        tokens = sample["tokens"]
        true_tags = filter_to_hybrid_tags(sample["ner_tag_labels"])
        baseline_tags = align_hybrid_predictions_to_words(tokens, baseline_engine)
        candidate_tags = align_hybrid_predictions_to_words(tokens, candidate_engine)

        baseline_f1 = float(f1_score([true_tags], [baseline_tags], average="macro", zero_division=0))
        candidate_f1 = float(f1_score([true_tags], [candidate_tags], average="macro", zero_division=0))
        delta = candidate_f1 - baseline_f1
        if delta <= 0:
            continue

        truth_entities = tag_entities(tokens, true_tags)
        baseline_entities = tag_entities(tokens, baseline_tags)
        candidate_entities = tag_entities(tokens, candidate_tags)

        truth_set = {entity_key(entity) for entity in truth_entities}
        baseline_set = {entity_key(entity) for entity in baseline_entities}
        candidate_set = {entity_key(entity) for entity in candidate_entities}

        gained = sorted(candidate_set & truth_set - baseline_set)
        if not gained:
            continue

        ranked_examples.append(
            {
                "text": " ".join(tokens),
                "baseline_f1": round(baseline_f1, 4),
                "candidate_f1": round(candidate_f1, 4),
                "delta_f1": round(delta, 4),
                "ground_truth": truth_entities,
                "baseline_entities": baseline_entities,
                "candidate_entities": candidate_entities,
                "newly_correct": [
                    {"label": label, "start": start, "end": end, "text": text_value}
                    for label, start, end, text_value in gained
                ],
            }
        )

    ranked_examples.sort(
        key=lambda example: (
            example["delta_f1"],
            len(example["newly_correct"]),
            example["candidate_f1"],
        ),
        reverse=True,
    )
    return ranked_examples[:num_examples]


def save_examples_markdown(
    output_path: Path,
    baseline: dict,
    candidate: dict,
    examples: list[dict[str, object]],
) -> None:
    lines = [
        "# Model comparison examples",
        "",
        f"Old model: {baseline['model']}",
        f"New model: {candidate['model']}",
        f"Benchmark dataset: {baseline['dataset']}",
        "",
        "## Aggregate metrics",
        "",
        f"- Macro F1: {baseline['metrics']['macro_f1']:.4f} -> {candidate['metrics']['macro_f1']:.4f}",
        f"- Macro Precision: {baseline['metrics']['macro_precision']:.4f} -> {candidate['metrics']['macro_precision']:.4f}",
        f"- Macro Recall: {baseline['metrics']['macro_recall']:.4f} -> {candidate['metrics']['macro_recall']:.4f}",
        "",
        "## Example slices",
        "",
    ]

    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"### Example {index}",
                "",
                f"Text: {example['text']}",
                "",
                f"- Ground truth: {format_entities(example['ground_truth'])}",
                f"- Old model: {format_entities(example['baseline_entities'])}",
                f"- New model: {format_entities(example['candidate_entities'])}",
                f"- Newly correct in new model: {format_entities(example['newly_correct'])}",
                f"- Sample F1: {example['baseline_f1']:.4f} -> {example['candidate_f1']:.4f}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))


def render_png(
    output_path: Path,
    baseline: dict,
    candidate: dict,
    baseline_entities: dict[str, dict[str, float]],
    candidate_entities: dict[str, dict[str, float]],
) -> None:
    image = Image.new("RGB", (PNG_WIDTH, PNG_HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, bold=True)
    subtitle_font = load_font(24)

    text(draw, (56, 42), "Obscura benchmark gain report", title_font)
    text(
        draw,
        (58, 102),
        f"Old: {Path(baseline['model']).name}    New: {Path(candidate['model']).name}",
        subtitle_font,
        MUTED,
    )

    draw_metric_card(
        draw,
        (56, 152, 416, 382),
        "Macro F1",
        baseline["metrics"]["macro_f1"],
        candidate["metrics"]["macro_f1"],
    )
    draw_metric_card(
        draw,
        (436, 152, 796, 382),
        "Macro precision",
        baseline["metrics"]["macro_precision"],
        candidate["metrics"]["macro_precision"],
    )
    draw_metric_card(
        draw,
        (816, 152, 1176, 382),
        "Macro recall",
        baseline["metrics"]["macro_recall"],
        candidate["metrics"]["macro_recall"],
    )
    draw_metric_card(
        draw,
        (1196, 152, 1544, 382),
        "Latency p95 ms",
        baseline["latency"]["p95_ms"],
        candidate["latency"]["p95_ms"],
        delta_suffix=" ms",
    )

    draw_entity_gains(draw, (56, 412, 920, 908), baseline_entities, candidate_entities)
    draw_summary(draw, (944, 412, 1544, 676), baseline, candidate)

    draw_card(draw, (944, 700, 1544, 908))
    small_title = load_font(28, bold=True)
    body_font = load_font(22)
    text(draw, (972, 726), "What stayed weak", small_title)
    weak = []
    for label, metrics in candidate_entities.items():
        if metrics["support"] <= 0:
            continue
        if metrics["f1"] < 0.2:
            weak.append((metrics["f1"], label, metrics["support"]))
    weak.sort()
    lines = weak[:3] or [(0.0, "none", 0)]
    for idx, (f1_value, label, support) in enumerate(lines):
        text(draw, (972, 786 + idx * 36), f"{label}: F1 {f1_value:.2f} on support {support}", body_font, MUTED)

    footer = load_font(18)
    text(
        draw,
        (56, 930),
        f"Dataset: {Path(baseline['dataset']).name}   Samples: {baseline['num_samples']}   Generated on CPU hybrid evaluation path",
        footer,
        MUTED,
    )

    image.save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Obscura benchmark comparison artifacts")
    parser.add_argument("--baseline-result", required=True, help="Path to old-model evaluation JSON")
    parser.add_argument("--candidate-result", required=True, help="Path to new-model evaluation JSON")
    parser.add_argument("--baseline-model", required=True, help="Path or model id for the old model")
    parser.add_argument("--candidate-model", required=True, help="Path or model id for the new model")
    parser.add_argument("--dataset", default=None, help="Optional override dataset path")
    parser.add_argument("--output-dir", default="ml/results/comparison", help="Directory for PNG and examples")
    parser.add_argument("--num-examples", type=int, default=3, help="How many side-by-side examples to save")
    parser.add_argument("--example-scan-limit", type=int, default=300, help="How many benchmark rows to scan for examples")
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline_result))
    candidate = load_json(Path(args.candidate_result))
    dataset_path = Path(args.dataset or baseline["dataset"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_entities = parse_classification_report(baseline["classification_report"])
    candidate_entities = parse_classification_report(candidate["classification_report"])

    png_path = output_dir / "nemotron_vs_baseline_benchmark.png"
    render_png(png_path, baseline, candidate, baseline_entities, candidate_entities)

    examples = extract_examples(
        dataset_path=dataset_path,
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        num_examples=args.num_examples,
        sample_limit=args.example_scan_limit,
    )

    examples_json_path = output_dir / "nemotron_vs_baseline_examples.json"
    examples_json_path.write_text(json.dumps(examples, indent=2))

    examples_md_path = output_dir / "nemotron_vs_baseline_examples.md"
    save_examples_markdown(examples_md_path, baseline, candidate, examples)

    print(f"PNG saved to: {png_path}")
    print(f"Examples saved to: {examples_md_path}")
    print(f"Examples JSON saved to: {examples_json_path}")


if __name__ == "__main__":
    main()