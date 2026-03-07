# OBS-13: ONNX Export and Quantization Pipeline — Execution Plan

> **Date:** 2026-03-06
> **Ticket:** OBS-13 (subtasks: OBS-13a through OBS-13d)
> **Author:** Arjun (AI/ML Lead)
> **Branch:** TBD (e.g., `feat/ml/OBS-13-onnx-quantization-pipeline`)
> **Baseline:** Hybrid macro F1 = 0.9576 | Latency p95 = 40.4ms (macOS) | RSS Peak = 802 MB

---

## 1. Prerequisite Gap Analysis

Two gaps from the prior sprint (OBS-8) must be addressed before or alongside this pipeline.

### 1.1 Latency Over Budget

**Current state:**
- PyTorch hybrid pipeline p95 = **40.4ms** on macOS Apple Silicon
- Target: **p95 ≤ 30ms** on EC2 t3.medium (Intel Xeon x86_64, 2 vCPUs)
- GAMEPLAN Phase 3 designates ONNX INT8 dynamic quantization as the solution path

**Why OBS-13 addresses this directly:**
INT8 dynamic quantization provides ~1.5–2x inference speedup on x86_64 CPUs via AVX2/VNNI vectorized integer instructions. This is the primary lever for closing the 40ms → 30ms gap:

| Scenario | Expected p95 | Basis |
|----------|-------------|-------|
| PyTorch FP32 (current, macOS) | 40.4ms | OBS-8 benchmark |
| ONNX FP32 (no quantization) | ~30–35ms | ONNX Runtime graph optimizations (operator fusion, constant folding) typically yield 15–20% speedup over PyTorch eager |
| ONNX INT8 dynamic (target) | ~18–25ms | INT8 adds ~1.3–1.5x on top of ONNX FP32 via VNNI |

**Contingency:** If INT8 on EC2 still exceeds 30ms p95:
1. Reduce `max_sequence_length` from 512 → 256 (halves computation for most enterprise inputs)
2. Apply ONNX Runtime graph-level optimizations (`SessionOptions.graph_optimization_level = ORT_ENABLE_ALL`)
3. Fallback to `dslim/distilbert-NER` (p50 = 19ms, 43% faster) — requires re-validation of accuracy

**Action items embedded in OBS-13:**
- OBS-13a: ONNX export with optimum (graph optimizations baked in)
- OBS-13b: INT8 dynamic quantization
- OBS-13c: Latency benchmark included in validation (PyTorch vs ONNX FP32 vs ONNX INT8)

### 1.2 DOB Context-Aware Analysis

**Current state:**
- DOB regex matches **all** valid dates in MM/DD/YYYY or YYYY-MM-DD format (entity_registry.md: "HIGH" false-positive risk)
- No context-aware scoring — unlike SSN dashless, which uses trigger/negative word disambiguation
- BENCHMARKS.md lists "Add DOB regex" as future work; entity_registry.md lists this as Known Limitation #1

**Why address now:**
The ONNX artifact bundle (OBS-13d) freezes the model + regex configuration for consumption by the Rust proxy. Any DOB improvements should land *before* the bundle is packaged, so the Rust backend receives the final regex behavior.

**Proposed DOB context scoring (mirroring SSN dashless pattern):**

```python
# Trigger words that indicate a date is a birth date
DOB_TRIGGER_WORDS: frozenset[str] = frozenset({
    "dob", "birth", "born", "birthday", "birthdate",
    "date of birth", "age", "newborn", "neonatal",
})

# Negative context words — these indicate non-DOB dates
DOB_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "meeting", "appointment", "scheduled", "deadline",
    "due", "expires", "expiration", "created", "updated",
    "filed", "issued", "effective", "invoice", "report",
})
```

Scoring model:
- Base score: 0.50 (down from current flat 0.95)
- Trigger word found: +0.40 → 0.90
- Trigger phrase found (e.g., "date of birth"): +0.10 → 1.00
- Negative word found: −0.35 → suppressed below threshold
- Threshold: 0.70 (same as SSN dashless)
- Dates with no context clues stay at 0.50 → filtered out (not emitted)

**Impact on F1:**
- DOB currently contributes F1 = 1.00 in benchmarks because synthetic data always places DOB in obvious context. Context scoring should preserve recall on these labeled samples while reducing false positives on unlabeled date strings.
- We will validate that macro F1 remains ≥ 0.95 after the change.

### 1.3 Credit Card Context-Aware Analysis (Undashed Overlap)

**Current state:**
- The `CREDIT_CARD` regex matches 16-digit numeric strings exactly. While padded with word boundaries `(?<!\w)` and `(?!\w)`, an undashed 16-digit string in isolation will always trigger a match.
- This was identified as a Known Limitation (#2) in the OBS-8 verification report: "Credit Card undashed (16-digit) overlap with long numeric strings."
- Although flagged as a low risk, it must be properly addressed before freezing the final ONNX artifact bundle.

**Proposed CC context scoring (mirroring SSN and DOB pattern):**

```python
# Trigger words that indicate a 16-digit string is a Credit Card
CC_TRIGGER_WORDS: frozenset[str] = frozenset({
    "visa", "mastercard", "amex", "discover", "card",
    "credit", "debit", "cc", "pan", "expiration", "cvv",
})

# Negative context words
CC_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "id", "identifier", "tracking", "order", "receipt",
    "account", "routing", "invoice", "transaction",
})
```

Scoring model:
- Applies *only* to undashed 16-digit numerical strings. Formatted strings (e.g. `1234-5678-9012-3456` or `1234 5678 9012 3456`) bypass this and receive full scoring immediately.
- Undashed Base score: 0.50
- CC Trigger word found in context (e.g., "Visa:"): +0.45 → 0.95
- Negative word found: -0.35 → suppressed
- Threshold: 0.70

**Files to modify:**
- `ml/regex_detector.py` — Add trigger/negative words, scoring methods (for both DOB and CC).
- `ml/tests/` — Add CC and DOB context tests.

---

## 2. Execution Plan

### Pre-work: DOB and Credit Card Context-Aware Scoring

**File:** `ml/regex_detector.py`

1. Add `DOB_TRIGGER_WORDS`, `DOB_NEGATIVE_WORDS`, `CC_TRIGGER_WORDS`, and `CC_NEGATIVE_WORDS` frozensets (module-level, following SSN convention).
2. Add `_score_dob_context()` and `_score_cc_context()` methods to `RegexDetector` — mirrors `_score_dashless_context()` architecture:
   - Extract context window (default 10 words each side)
   - Strip punctuation, lowercase
   - Calculate contextual score based on trigger/negative combinations defined in the gap analysis.
3. Modify `_detect_dob()` and `_detect_credit_card()` to leverage context thresholds. CC context scoring should *only* evaluate when a string matches the 16-digit undashed variant.
4. Add `dob_base_score`, `dob_threshold`, `cc_undashed_base_score` and `cc_threshold` dataclass fields as needed.

**Tests:** `ml/tests/test_regex_extensions.py`
- DOB with trigger context ("DOB: 01/15/1990") → detected, score ≥ 0.90
- DOB with negative context ("meeting scheduled 01/15/2026") → suppressed
- DOB with neutral context ("the date 01/15/1990 was noted") → suppressed (below 0.70)
- ISO format DOB ("born 1990-01-15") → detected
- Undashed CC with trigger context ("Visa 1234567890123456") → detected, score ≥ 0.90
- Undashed CC with negative context ("Tracking ID 1234567890123456") → suppressed
- Undashed CC with neutral context ("Value 1234567890123456") → suppressed (below threshold)
- Dashed/spaced CC ("1234-5678-9012-3456") → bypassed context, full score immediately

**Validation:** `pytest ml/tests/ -v` — all existing tests pass + new DOB/CC tests pass. `ruff check ml/ && black --check ml/`.

---

### OBS-13a: ONNX Export with HuggingFace optimum

**Goal:** Replace the current `ml/export_onnx.py` (raw `torch.onnx.export`) with an optimum-based exporter that produces a well-optimized ONNX graph.

**Why optimum over raw torch.onnx.export:**
- `optimum` handles attention mask edge cases, dynamic axes, and opset selection automatically
- Produces pre-validated ONNX graphs that are tested against ONNX Runtime
- Integrates directly with the quantization step (OBS-13b)
- Our existing `export_onnx.py` uses `torch.onnx.export` with manual dynamic_axes — functional but lacks graph optimizations

**File:** `ml/export_onnx.py` (rewrite)

**Implementation:**

```python
"""ONNX Export + Quantization Pipeline for Obscura

Exports a HuggingFace NER model to ONNX via optimum, applies INT8 dynamic
quantization, and packages the artifact bundle for the Rust proxy backend.

Usage:
    python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx
    python ml/export_onnx.py --model ml/models/fine-tuned --output ml/models/onnx --quantize
"""
```

Steps:
1. Accept CLI args: `--model` (HF model ID or local path), `--output` (directory), `--quantize` (flag), `--validate` (flag)
2. Use `optimum.exporters.onnx` to export:
   - `from optimum.exporters.onnx import main_export`
   - `main_export(model_name_or_path, output, task="token-classification", opset=14)`
   - This handles dynamic axes, attention masks, and graph optimization automatically
3. Verify `model.onnx` and `tokenizer.json` exist in output directory
4. If `--quantize` is passed, proceed to OBS-13b logic (in same script)
5. If `--validate` is passed, proceed to OBS-13c logic (in same script)

**Dependencies:**
- `optimum[onnxruntime]` — add to dev requirements
- `onnxruntime` — already implied by optimum

**Backward compatibility:**
The existing `export_model()` function signature changes. Since no other module imports it (CLI-only), this is safe. The script remains a standalone CLI tool.

---

### OBS-13b: INT8 Dynamic Quantization

**Goal:** Apply dynamic INT8 quantization to the exported ONNX model using `onnxruntime.quantization`.

**Why dynamic over static:**
- No calibration dataset required (simpler pipeline)
- x86_64 Intel Xeon CPUs benefit from INT8 via AVX2/VNNI
- GAMEPLAN Phase 3.1 explicitly chooses dynamic quantization
- Static quantization would give ~5% better latency but adds calibration complexity

**Implementation (within `ml/export_onnx.py`):**

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path: Path, output_path: Path) -> None:
    """Apply INT8 dynamic quantization to an ONNX model."""
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
```

Steps:
1. Input: `model.onnx` (FP32 from OBS-13a)
2. Output: `model_quantized.onnx` (INT8 dynamic)
3. Log file sizes for both (expect ~2x reduction: ~440MB → ~220MB for bert-base)
4. Verify output file loads with `onnxruntime.InferenceSession`

**Quantization targets:** All linear layers (MatMul ops) quantized to INT8. Embedding layers and LayerNorm remain FP32 (standard for NER models — these layers are numerically sensitive).

---

### OBS-13c: Accuracy Validation (Quantized vs. Original)

**Goal:** Verify that INT8 quantized F1 stays within 1% (1 percentage point) of the PyTorch original.

**Acceptance criterion:** `|F1_pytorch - F1_quantized| < 0.01` (i.e., < 1pp degradation)

Note: The GAMEPLAN specifies < 0.5pp (stricter). The ticket says 1%. We implement the 1% gate per the ticket but flag if degradation exceeds 0.5pp per GAMEPLAN.

**Implementation (new function in `ml/export_onnx.py` or standalone `ml/validate_onnx.py`):**

```python
def validate_quantized_model(
    pytorch_model_id: str,
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    dataset_path: Path,
    limit: int = 100,
) -> dict:
    """Compare F1 scores across PyTorch, ONNX FP32, and ONNX INT8."""
```

**Validation methodology:**

#### Step 1: Logit-Level Parity Check
- Run inference on a 100-sample golden set with PyTorch, ONNX FP32, and ONNX INT8
- Compare raw logit outputs: `max |logits_pytorch - logits_onnx_fp32| < 1e-4`
- For INT8: `max |logits_pytorch - logits_int8|` — expect larger delta but check distribution
- Report: mean absolute error, max absolute error, per-layer statistics

#### Step 2: Entity-Level F1 Comparison
- Run full hybrid evaluation (BERT + regex) using `ml/evaluate.py` logic on:
  1. **PyTorch baseline** — `PIIEngine` with PyTorch model (existing infrastructure)
  2. **ONNX FP32** — `PIIEngine` with ONNX Runtime session replacing PyTorch forward pass
  3. **ONNX INT8** — `PIIEngine` with quantized ONNX Runtime session
- Compute per-entity F1 and macro F1 for each
- Assert: `|macro_f1_pytorch - macro_f1_int8| < 0.01`
- Warn: `|macro_f1_pytorch - macro_f1_int8| > 0.005` (GAMEPLAN stricter threshold)

#### Step 3: Latency Comparison
- Benchmark all three variants on the same hardware (500 samples, 5 warmup):
  - PyTorch FP32 (CPU)
  - ONNX FP32 (ORT, CPU)
  - ONNX INT8 (ORT, CPU)
- Report: p50, p95, p99 for each
- This establishes the speedup factor before EC2 validation

#### Step 4: Edge Case Regression
- Run the full `pytest ml/tests/ -v` suite with the ONNX INT8 model backing PIIEngine
- Verify: SSN context scoring, regex extensions, redaction config all pass identically
- Any test that relies on exact logit values may need tolerance adjustment

**Output:** Validation report JSON + human-readable table:

```
╔══════════════════════════════════════════════════════════════╗
║  OBS-13 Validation Report                                    ║
╠══════════════════╦══════════╦══════════╦═════════════════════╣
║  Metric          ║ PyTorch  ║ ONNX FP32║ ONNX INT8           ║
╠══════════════════╬══════════╬══════════╬═════════════════════╣
║  Macro F1        ║ 0.9576   ║ TBD      ║ TBD (must ≥ 0.9476)║
║  Latency p95 (ms)║ 40.4     ║ TBD      ║ TBD (target ≤ 30)  ║
║  Model Size (MB) ║ ~440     ║ ~440     ║ TBD (~110-220)      ║
║  RSS Peak (MB)   ║ 802      ║ TBD      ║ TBD                 ║
╚══════════════════╩══════════╩══════════╩═════════════════════╝
  Gate: |F1_pytorch - F1_int8| < 0.01 → PASS/FAIL
```

**Fallback if INT8 fails validation:**
1. Investigate per-entity breakdown — which entity types degraded?
2. If clinical/contextual entities (LOC, ORG) degraded but structured (SSN, EMAIL) held: acceptable since regex handles structured types
3. If degradation > 1pp: try FP16 quantization (`QuantType.QUInt8` with `op_types_to_quantize` restricted to MatMul only)
4. If FP16 also fails: ship ONNX FP32 (no quantization) — larger model but guaranteed accuracy parity

---

### OBS-13d: Artifact Bundle with Schema JSON

**Goal:** Package the complete artifact bundle consumed by the Rust proxy's inference crate.

**Bundle structure** (per GAMEPLAN Phase 3.3):

```
ml/models/onnx/bert-ner-v1.0.0/
├── model.onnx                  # INT8 quantized (or FP32 if INT8 fails validation)
├── model_fp32.onnx             # FP32 reference (kept for validation, not deployed)
├── tokenizer.json              # HuggingFace fast tokenizer (required by Rust tokenizers crate)
├── special_tokens_map.json     # [CLS], [SEP], [PAD], [MASK] definitions
├── schema.json                 # Input/output contract for Rust inference crate
├── label_map.json              # {0: "O", 1: "B-PER", 2: "I-PER", ...}
├── checksum.sha256             # SHA-256 of model.onnx
└── metadata.json               # Version, training date, F1, latency, quantization info
```

**schema.json:**
```json
{
  "model_version": "1.0.0",
  "model_name": "dslim/bert-base-NER",
  "quantization": "dynamic_int8",
  "opset_version": 14,
  "input": {
    "input_ids": {"dtype": "int64", "shape": ["batch", "seq_len"]},
    "attention_mask": {"dtype": "int64", "shape": ["batch", "seq_len"]}
  },
  "output": {
    "logits": {"dtype": "float32", "shape": ["batch", "seq_len", "num_labels"]}
  },
  "max_sequence_length": 512,
  "label_count": 9,
  "aggregation": "simple",
  "confidence_threshold": 0.90,
  "regex_entity_types": ["SSN", "PHONE", "EMAIL", "MRN", "DOB", "CREDIT_CARD", "IPV4", "PASSPORT"],
  "bert_entity_types": ["PER", "LOC", "ORG", "MISC"]
}
```

**metadata.json:**
```json
{
  "version": "1.0.0",
  "base_model": "dslim/bert-base-NER",
  "fine_tuned": true,
  "training_samples": 400,
  "training_epochs": 5,
  "export_date": "2026-03-XX",
  "quantization": "dynamic_int8",
  "accuracy": {
    "macro_f1_pytorch": 0.9576,
    "macro_f1_onnx_int8": null,
    "f1_degradation_pp": null
  },
  "latency": {
    "pytorch_p95_ms": 40.4,
    "onnx_int8_p95_ms": null,
    "hardware": "macOS Apple Silicon (EC2 validation pending)"
  },
  "model_size_mb": null,
  "checksum_algorithm": "sha256"
}
```

**label_map.json:** Extracted from model's `config.id2label`:
```json
{"0": "O", "1": "B-PER", "2": "I-PER", "3": "B-ORG", "4": "I-ORG", "5": "B-LOC", "6": "I-LOC", "7": "B-MISC", "8": "I-MISC"}
```

**checksum.sha256:** Generated via `hashlib.sha256` over the `model.onnx` binary. Provides integrity verification before the Rust backend loads the model.

**Implementation:** Add a `package_bundle()` function to `ml/export_onnx.py` that generates all JSON files, computes the checksum, and copies artifacts into the versioned directory.

---

## 3. Validation Strategy

### 3.1 Quantized F1 Within 1% — Formal Methodology

**Primary metric:** Macro F1 (seqeval, average="macro", zero_division=0)

**Test set:** `ml/data/synthetic.jsonl` — 500 samples (50% enterprise, 35% clinical, 15% negative). Same set used for OBS-8 benchmarks to enable direct comparison.

**Protocol:**

1. **Establish PyTorch baseline** (already known: macro F1 = 0.9576)
   - Re-run to confirm reproducibility on the same hardware
   - Record per-entity F1 breakdown as reference

2. **ONNX FP32 parity check**
   - Export via optimum → load with ORT InferenceSession
   - Run evaluation: expect F1 = 0.9576 ± 0.0001 (FP32 export should be numerically equivalent)
   - If delta > 0.001: investigate export fidelity issue before proceeding to INT8

3. **ONNX INT8 validation**
   - Apply dynamic quantization → load with ORT InferenceSession
   - Run full evaluation on 500 samples
   - **Gate:** `F1_pytorch - F1_int8 < 0.01` (1 percentage point)
   - **Warning:** `F1_pytorch - F1_int8 > 0.005` (flag for GAMEPLAN compliance)
   - Record per-entity breakdown — identify which types (if any) degraded

4. **Per-entity analysis**
   - For each of the 8 evaluated entity types (PER, LOC, ORG, MISC, SSN, PHONE, EMAIL, MRN):
     - Compute `delta_f1 = F1_pytorch - F1_int8`
     - Flag any type with delta > 0.02 (2pp per-entity degradation)
   - Regex-authoritative types (SSN, PHONE, EMAIL, MRN, DOB, CREDIT_CARD, IPV4, PASSPORT) should show **zero degradation** since regex detection is independent of the ONNX model
   - BERT types (PER, LOC, ORG, MISC) are the only types at risk from quantization

5. **Logit-level sanity check**
   - On a 100-sample golden subset:
     - Run PyTorch and ONNX INT8 inference
     - Compute mean absolute logit difference per token position
     - Report: mean, std, max
   - This catches silent numerical issues that might not yet affect F1 but could degrade at different input distributions

6. **Latency validation**
   - All three variants benchmarked on same hardware, same conditions
   - 500 samples, 5 warmup inferences excluded
   - Report: p50, p95, p99, mean, max
   - Compute speedup factor: `latency_pytorch / latency_int8`

### 3.2 Pass/Fail Criteria (Go/No-Go Gate)

| Gate | Threshold | Source | Blocking? |
|------|-----------|--------|-----------|
| Macro F1 degradation | < 1pp (0.01) | OBS-13 ticket | Yes |
| Macro F1 degradation | < 0.5pp (0.005) | GAMEPLAN Phase 3 | Warn (flag in report) |
| Per-entity F1 degradation | < 2pp per entity | Engineering judgment | Warn |
| ONNX FP32 parity | < 0.001 F1 delta | Sanity check | Yes (blocks INT8 if FP32 is wrong) |
| Regex entity F1 change | == 0 | Regex is model-independent | Yes |
| Model size (ONNX INT8) | ≤ 300MB | GAMEPLAN | Yes |
| tokenizer.json exists | Present in bundle | Rust backend requirement | Yes |
| schema.json valid | Parseable, all fields present | Integration contract | Yes |
| All existing tests pass | 66/66 (pytest) | Regression gate | Yes |
| Lint clean | ruff + black zero errors | Code quality | Yes |

### 3.3 Failure Handling

| Failure Scenario | Action |
|-----------------|--------|
| INT8 F1 degrades > 1pp | Try FP16 quantization; if FP16 also fails, ship ONNX FP32 |
| INT8 F1 degrades > 0.5pp but < 1pp | Ship INT8 but log warning in metadata.json; discuss with team |
| ONNX FP32 F1 differs from PyTorch by > 0.001 | Export bug — investigate optimum export, try raw torch.onnx.export as fallback |
| Latency still > 30ms p95 after INT8 | Reduce max_sequence_length to 256; enable ORT graph optimizations; consider distilbert fallback |
| tokenizer.json missing after export | Use `tokenizer.backend_tokenizer.save(str(path))` as direct save fallback |

---

## 4. Implementation Order

```
1. [Pre-work]  DOB and Credit Card context-aware scoring
   └─ regex_detector.py + tests → validate F1 ≥ 0.95

2. [OBS-13a]   Rewrite export_onnx.py with optimum
   └─ Export FP32 ONNX → verify loads with ORT

3. [OBS-13b]   Add quantize_model() function
   └─ Apply INT8 dynamic → verify loads with ORT → log file sizes

4. [OBS-13c]   Implement validation pipeline
   └─ Run 3-way comparison (PyTorch vs FP32 vs INT8) → generate report
   └─ Gate: F1 delta < 0.01

5. [OBS-13d]   Package artifact bundle
   └─ Generate schema.json, label_map.json, metadata.json, checksum
   └─ Verify directory structure matches GAMEPLAN spec

6. [Verification] 6-phase gate
   └─ ruff + black → pytest (66 existing + new CC/DOB tests) → security → diff → docs
```

**Estimated new/modified files:**

| File | Action |
|------|--------|
| `ml/regex_detector.py` | Modify — DOB/CC context scoring |
| `ml/export_onnx.py` | Rewrite — optimum export + quantization + validation + packaging |
| `ml/tests/test_regex_extensions.py` | Modify — add DOB/CC context tests |
| `ml/tests/test_onnx_validation.py` | Create — ONNX parity + quantization tests |
| `verification/OBS-13_verification_report.md` | Create — final verification report |

**Dependencies to add (dev only):**
- `optimum[onnxruntime]`
- `onnxruntime` (if not already pulled in by optimum)
- `onnx` (already in use by current export_onnx.py)
