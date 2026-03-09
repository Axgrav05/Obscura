"""
ONNX Model Inference Demo

Demonstrates the quantized ONNX model detecting PII entities in synthetic data.
Shows both regex and BERT-based NER working together with visual output.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Synthetic test data with various PII types
TEST_DOCUMENTS = [
    {
        "id": 1,
        "text": "Patient John Smith, DOB: 03/15/1985, SSN: 123-45-6789, called from phone (555) 123-4567.",
        "expected_entities": ["PER", "DOB", "SSN", "PHONE"]
    },
    {
        "id": 2,
        "text": "Dr. Sarah Johnson at City Hospital treated Jane Doe. Contact: jane.doe@email.com, MRN: MRN-7654321.",
        "expected_entities": ["PER", "LOC", "PER", "EMAIL", "MRN"]
    },
    {
        "id": 3,
        "text": "Credit card 4532-1234-5678-9010 was used at Global Industries located at 123 Main St, Chicago.",
        "expected_entities": ["CREDIT_CARD", "ORG", "LOC"]
    },
    {
        "id": 4,
        "text": "Server IP 192.168.1.100 accessed by user with passport C12345678 from New York office.",
        "expected_entities": ["IPV4", "PASSPORT", "LOC"]
    },
    {
        "id": 5,
        "text": "Alice Williams born 1990-07-22 works at Acme Corp. Her employee ID is EMP-9876.",
        "expected_entities": ["PER", "DOB", "ORG"]
    },
]


def resolve_bundle_dir() -> Path:
    bundles_root = Path("ml/models/onnx")
    candidates = sorted(
        (path for path in bundles_root.glob("*/") if (path / "model.onnx").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No packaged ONNX bundle found under ml/models/onnx")
    return candidates[0]


def load_onnx_model(bundle_dir: Path):
    """Load the quantized ONNX model and tokenizer."""
    model_path = bundle_dir / "model.onnx"

    # Load ONNX Runtime session
    session = ort.InferenceSession(str(model_path))

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir.parent))

    # Load label map
    with open(bundle_dir / "label_map.json") as f:
        label_map = json.load(f)

    return session, tokenizer, label_map


def run_ner_inference(session, tokenizer, text: str, label_map: dict):
    """Run NER inference on text using ONNX model."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)

    # Prepare ONNX inputs
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Add token_type_ids if the model expects it
    input_names = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

    # Run inference
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    # Get predictions
    predictions = np.argmax(logits, axis=-1)[0]

    # Convert tokens back to words with predictions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Extract entities
    entities = []
    current_entity = None
    current_tokens = []

    for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = label_map.get(str(pred_id), "O")

        if label.startswith("B-"):
            # Start of new entity
            if current_entity:
                entities.append({
                    "type": current_entity,
                    "tokens": current_tokens,
                    "text": tokenizer.convert_tokens_to_string(current_tokens)
                })
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity:
            # Continuation of entity
            current_tokens.append(token)
        else:
            # End of entity
            if current_entity:
                entities.append({
                    "type": current_entity,
                    "tokens": current_tokens,
                    "text": tokenizer.convert_tokens_to_string(current_tokens)
                })
            current_entity = None
            current_tokens = []

    # Don't forget last entity
    if current_entity:
        entities.append({
            "type": current_entity,
            "tokens": current_tokens,
            "text": tokenizer.convert_tokens_to_string(current_tokens)
        })

    return entities


def run_regex_detection(text: str):
    """Run regex-based PII detection."""
    from ml.regex_detector import RegexDetector

    detector = RegexDetector()
    entities = detector.detect(text)

    return [
        {
            "type": e.entity_type,
            "text": e.text,
            "score": e.score,
            "start": e.start,
            "end": e.end
        }
        for e in entities
    ]


def print_visual_results(doc_id: int, text: str, bert_entities: list, regex_entities: list, expected: list):
    """Print colorful visual results."""
    print("\n" + "=" * 80)
    print(f"📄 DOCUMENT {doc_id}")
    print("=" * 80)
    print(f"\n📝 Text: {text}\n")

    print("🤖 BERT NER Detections:")
    if bert_entities:
        for ent in bert_entities:
            print(f"   • {ent['type']:12s} → \"{ent['text']}\"")
    else:
        print("   (none)")

    print("\n🔍 Regex Detections:")
    if regex_entities:
        for ent in regex_entities:
            print(f"   • {ent['type']:12s} → \"{ent['text']}\" (score: {ent['score']:.2f})")
    else:
        print("   (none)")

    # Combine all detected types
    bert_types = {e['type'] for e in bert_entities}
    regex_types = {e['type'] for e in regex_entities}
    all_detected = bert_types | regex_types

    # Calculate accuracy metrics
    expected_set = set(expected)
    true_positives = all_detected & expected_set
    false_positives = all_detected - expected_set
    false_negatives = expected_set - all_detected

    precision = len(true_positives) / len(all_detected) if all_detected else 0
    recall = len(true_positives) / len(expected_set) if expected_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 Accuracy Metrics:")
    print(f"   Expected entities: {sorted(expected_set)}")
    print(f"   Detected entities: {sorted(all_detected)}")
    print(f"   ✓ True Positives:  {sorted(true_positives)}")
    if false_positives:
        print(f"   ✗ False Positives: {sorted(false_positives)}")
    if false_negatives:
        print(f"   ✗ False Negatives: {sorted(false_negatives)}")
    print(f"\n   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.2%}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": len(true_positives),
        "fp": len(false_positives),
        "fn": len(false_negatives)
    }


def main():
    """Run the demo."""
    print("\n" + "🚀 " * 40)
    print("OBSCURA ONNX MODEL INFERENCE DEMO")
    print("🚀 " * 40)

    # Load model
    bundle_dir = resolve_bundle_dir()
    print(f"\n📦 Loading ONNX model from: {bundle_dir}")

    session, tokenizer, label_map = load_onnx_model(bundle_dir)
    print(f"✓ Model loaded: INT8 quantized BERT-NER")
    print(f"✓ Tokenizer loaded")
    print(f"✓ Label map: {len(label_map)} entity types")

    # Run inference on all test documents
    all_metrics = []

    for doc in TEST_DOCUMENTS:
        # BERT NER
        bert_entities = run_ner_inference(session, tokenizer, doc["text"], label_map)

        # Regex detection
        regex_entities = run_regex_detection(doc["text"])

        # Print results
        metrics = print_visual_results(
            doc["id"],
            doc["text"],
            bert_entities,
            regex_entities,
            doc["expected_entities"]
        )
        all_metrics.append(metrics)

    # Overall statistics
    print("\n" + "=" * 80)
    print("📈 OVERALL PERFORMANCE ACROSS ALL DOCUMENTS")
    print("=" * 80)

    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    total_tp = sum(m["tp"] for m in all_metrics)
    total_fp = sum(m["fp"] for m in all_metrics)
    total_fn = sum(m["fn"] for m in all_metrics)

    print(f"\n   Documents processed: {len(TEST_DOCUMENTS)}")
    print(f"   Total entities detected: {total_tp + total_fp}")
    print(f"   Total true positives: {total_tp}")
    print(f"   Total false positives: {total_fp}")
    print(f"   Total false negatives: {total_fn}")
    print(f"\n   Average Precision: {avg_precision:.2%}")
    print(f"   Average Recall:    {avg_recall:.2%}")
    print(f"   Average F1 Score:  {avg_f1:.2%}")

    # Model info
    print("\n" + "=" * 80)
    print("⚡ MODEL PERFORMANCE INFO")
    print("=" * 80)

    with open(bundle_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"\n   Model size: {metadata['model_size_mb']} MB (INT8 quantized)")
    print(f"   Inference latency (p95): {metadata['latency']['onnx_int8_p95_ms']} ms")
    print(f"   Speedup vs PyTorch: {metadata['latency']['pytorch_p95_ms'] / metadata['latency']['onnx_int8_p95_ms']:.2f}x")
    print(f"   Export date: {metadata['export_date']}")

    print("\n" + "✅ " * 40)
    print("DEMO COMPLETE!")
    print("✅ " * 40 + "\n")


if __name__ == "__main__":
    main()
