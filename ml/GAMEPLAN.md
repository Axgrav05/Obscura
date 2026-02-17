# Obscura NER Engine — Improvement & Testing Game Plan

> **Author:** Arjun (AI/ML Lead) | **Date:** 2026-02-16
> **Jira Epics:** OBS-3 (Benchmarking), OBS-7 (Integration), OBS-12 (Hardening)
> **Hardware Target:** OCI Always Free ARM Ampere A1 (4 OCPUs, 24GB shared RAM)
> **Latency Budget:** ≤30ms inference (within 60ms total proxy overhead)
> **Accuracy Target:** 90%+ macro F1 across enterprise + medical entity types

---

## Design Rationale: Why This Ordering

The phases are sequenced by dependency and risk reduction:

1. **Baseline first** because you cannot improve what you haven't measured. Every subsequent decision (model choice, quantization strategy, privacy budget) depends on knowing where we start.
2. **Improvement before optimization** because it's wasteful to optimize a model we'll replace. Get the best accuracy first, then compress.
3. **Export/optimization before privacy** because differential privacy adds noise that degrades accuracy — we need to know our post-optimization ceiling before we can calculate how much degradation we can tolerate.
4. **Privacy before integration testing** because the privacy layer changes model behavior, so end-to-end tests must run against the final artifact.
5. **CI last** because automated gates only make sense once we know the exact metrics, thresholds, and artifact format.

---

## Phase 1: Baseline & Benchmarking (OBS-3)

### 1.1 Model Candidates

| Model | Params | Size (MB) | Why Evaluate |
|-------|--------|-----------|-------------|
| `dslim/bert-base-NER` | 110M | ~440 | Current baseline. CoNLL-2003 trained. Strong general PER/LOC/ORG. |
| `dslim/distilbert-NER` | 66M | ~260 | 40% smaller, ~60% faster. If F1 holds within 2-3 points, this is the production pick. |
| `emilyalsentzer/Bio_ClinicalBERT` | 110M | ~440 | Pre-trained on MIMIC-III clinical notes. Needs NER fine-tuning but understands medical context. Evaluate as the base for PHI detection. |
| `StanfordAIMI/stanford-deidentifier-base` | 110M | ~440 | Trained specifically on clinical de-identification. Closest to our PHI use case out of the box. |

**Reasoning:** We evaluate four models, not more. Two general-purpose (one full, one distilled) and two medical. The distilled model is critical — if it meets 90% F1, it halves our latency and RAM problems in one move. The clinical models tell us whether off-the-shelf medical NER is viable or if we need Phase 2 fine-tuning.

**What I'm unsure about:** Whether `StanfordAIMI/stanford-deidentifier-base` produces IOB2 tags compatible with our `NER_LABEL_TO_ENTITY` mapping, or if it uses a different label scheme. We'll discover this during evaluation and adapt.

### 1.2 Benchmark Datasets

| Dataset | Entity Types | Purpose | Access |
|---------|-------------|---------|--------|
| CoNLL-2003 (eng) | PER, LOC, ORG, MISC | Standard NER benchmark. Establishes comparable numbers. | Public (HuggingFace `conll2003`) |
| OntoNotes 5.0 | 18 types incl. DATE, MONEY, NORP | Broader entity coverage, tests generalization. | LDC license — use the subset available via HuggingFace or request access. |
| Synthetic-Enterprise | PER, SSN, PHONE, EMAIL, ORG, ADDRESS | **We generate this.** Realistic enterprise LLM prompts containing structured PII. | Generated locally (see 1.3). |
| Synthetic-Clinical | PER, MRN, DATE, MEDICATION, CONDITION | **We generate this.** Simulates clinical notes with PHI. | Generated locally (see 1.3). |

**Trade-off:** We intentionally skip i2b2 2014 (the gold standard for clinical de-identification) because its Data Use Agreement restricts redistribution and CI integration. Synthetic data avoids the licensing issue and the HIPAA risk of having real PHI in our dev environment — even de-identified data carries re-identification risk.

### 1.3 Synthetic Data Generation

Use `Faker` (with medical provider) + template-based generation:

```
Templates (enterprise):
  "Please summarize the case for {PERSON} (SSN: {SSN}), who lives at {ADDRESS} and can be reached at {PHONE}."
  "The meeting between {PERSON_1} from {ORG} and {PERSON_2} was held in {LOCATION} on {DATE}."

Templates (clinical):
  "Patient {PERSON}, MRN {MRN}, presents with {CONDITION}. Current medications: {MEDICATION}. Follow-up scheduled {DATE}."
  "Dr. {PERSON} at {ORG} diagnosed {CONDITION} and prescribed {MEDICATION} {DOSAGE}."
```

Generate 2,000 samples per domain (4,000 total). Each sample includes ground-truth IOB2 labels computed from the known entity spans. This gives us a labeled evaluation set with zero real PII.

**Script dependency:** Requires `Faker` (add to dev dependencies). The generation script outputs CoNLL-format `.bio` files compatible with `seqeval`.

### 1.4 Metrics

Track all of the following for every model candidate:

| Metric | Target | Why |
|--------|--------|-----|
| **Per-entity F1** (PER, LOC, ORG, SSN, MRN, etc.) | ≥85% each | Identifies weak spots per category. A model with 95% macro F1 but 60% on MRN is unacceptable. |
| **Macro F1** | ≥90% | The headline number. Weighted by entity type frequency in real traffic (estimated). |
| **Precision** | ≥92% | For a redaction system, false positives (over-masking) are more tolerable than false negatives (PII leakage). But excessive false positives degrade LLM output quality. |
| **Recall** | ≥88% | Non-negotiable floor. Missed PII is a compliance violation. |
| **Latency p50/p95/p99 (ms)** | p95 ≤30ms | Measured on ARM or simulated with throttled CPU. p99 matters for tail latency SLOs. |
| **Model size (MB)** | ≤300MB ONNX | Must fit in OCI ARM instance memory alongside the Rust proxy and OS. |
| **Peak RAM (MB)** | ≤1.5GB inference | Ampere A1 free tier is 24GB shared across all pods. Budget ~2GB for the inference container. |

### 1.5 Evaluation Script Outline

```
ml/evaluate.py

1. Load model (from HF hub or local path)
2. Load dataset (CoNLL-format .bio file or HF dataset)
3. For each sample:
   a. Tokenize → run inference → collect predicted IOB2 tags
   b. Record wall-clock time (time.perf_counter_ns)
4. Compute metrics:
   - seqeval.metrics.classification_report (per-entity P/R/F1)
   - seqeval.metrics.f1_score (macro)
   - numpy percentiles on latency array (p50, p95, p99)
5. Compute model metadata:
   - os.path.getsize for model file
   - torch.cuda.max_memory_allocated or psutil for RAM
6. Output: JSON report + human-readable table to stdout
7. Save report to ml/reports/{model_name}_{dataset}_{timestamp}.json
```

**Dependencies to add:** `seqeval`, `Faker` (dev only — not needed in production image).

---

## Phase 2: Model Improvement

### 2.1 Strategies Ranked by Impact

| # | Strategy | Expected F1 Lift | Latency Impact | Effort | Priority |
|---|----------|-----------------|----------------|--------|----------|
| 1 | **Regex layer for structured PII** | +5-10% on SSN/PHONE/EMAIL (these are 0% on BERT alone) | +0.1ms (negligible) | ~4h | **Do first** |
| 2 | **Fine-tune on synthetic enterprise data** | +3-5% macro | None (same model) | ~8h (data prep + training) | High |
| 3 | **Fine-tune on synthetic clinical data** | +5-8% on medical entities | None (same model) | ~8h | High |
| 4 | **DistilBERT swap** (if baseline shows it's close) | -1-3% (trade accuracy for speed) | -40% latency | ~2h | High (if latency is tight) |
| 5 | **Data augmentation** (entity replacement, synonym sub) | +1-2% macro | None | ~4h | Medium |
| 6 | **Confidence threshold tuning** | +1-2% (tuning precision/recall) | None | ~1h | Medium |
| 7 | **Multi-model ensemble** | +2-4% | +100% latency (two models) | ~16h | **Skip** — violates latency budget |

**Reasoning for ordering:** Strategy #1 (regex) is the single highest-impact move because BERT literally cannot detect SSNs or phone numbers — it was never trained on them. A handful of regex patterns immediately cover entity types that represent real compliance risk. Strategy #7 (ensemble) is explicitly rejected because running two BERT models doubles latency past our 30ms budget.

### 2.2 Medical Entity Strategy: Recommendation

**Recommended approach: Fine-tune `dslim/bert-base-NER` on synthetic clinical data.**

Not scispaCy as a runtime component. Here's why:

- **Latency:** scispaCy's `en_core_sci_lg` pipeline adds ~50-80ms per call. Combined with BERT, we'd blow the 30ms budget. Running them in parallel is possible but doubles RAM and adds orchestration complexity.
- **Entity overlap:** scispaCy detects diseases, chemicals, genes — most of which aren't PII. We need PERSON, MRN, DATE, MEDICATION in clinical context. Fine-tuning BERT on these specific types is more targeted.
- **Single model advantage:** One ONNX file, one inference path, one set of latency characteristics. Rainier consumes one artifact, not two.
- **scispaCy's role:** Use it as a **data augmentation tool during training** — leverage its medical entity recognition to auto-label synthetic clinical data — but don't deploy it at inference time.

**Trade-off acknowledged:** Fine-tuning on synthetic data won't match a model trained on real clinical notes (like i2b2). If Phase 1 benchmarks show the clinical BERT variants significantly outperform our fine-tuned model on medical entities, we should revisit this decision and consider using `StanfordAIMI/stanford-deidentifier-base` as the base model instead.

### 2.3 Hybrid Conflict Resolution (Regex + Transformer)

When both the regex layer and the transformer detect an entity at the same or overlapping span, we need deterministic resolution:

```
Priority rules:
1. EXACT OVERLAP (same start/end): regex wins for structured types (SSN, PHONE, EMAIL, MRN),
   transformer wins for semantic types (PERSON, ORGANIZATION, LOCATION).
   Reasoning: regex is deterministic and 100% precise on structured patterns.

2. PARTIAL OVERLAP: take the longer span.
   Reasoning: "Dr. John Smith" (transformer: PERSON) partially overlaps with
   "John" (regex name list match). The longer span is more complete.

3. NESTED SPANS: keep outer span, discard inner.
   Reasoning: "John Smith at Acme Corp" should not produce both
   PERSON("John Smith") and PERSON("John").

4. NO OVERLAP: keep both.
```

Implementation: run both detectors, merge results into a single sorted span list, apply conflict resolution as a post-processing pass before token replacement. This is a ~50-line function in `pii_engine.py`.

---

## Phase 3: Optimization & Export (OBS-7)

### 3.1 ONNX Conversion Pipeline

```
Step 1: Export PyTorch → ONNX
  - Use optimum.exporters.onnx (handles attention masks, dynamic axes)
  - Set opset_version=14 (broadest ort compatibility)
  - Enable dynamic batch + sequence length axes

Step 2: Quantize
  - Strategy: Dynamic INT8 quantization
  - Why dynamic over static: no calibration dataset needed, simpler pipeline,
    and ARM CPUs benefit from INT8 via NEON SIMD but don't have native FP16
    compute units (unlike GPUs). Static quantization would give ~5% better
    latency but requires a representative calibration set and more tooling.
  - Why INT8 over FP16: ARM Ampere A1 doesn't have FP16 ALUs for server
    workloads. INT8 is the sweet spot — ~2x smaller model, ~1.5x faster
    inference via NEON vectorization.
  - Tool: onnxruntime.quantization.quantize_dynamic

Step 3: Validate
  - Run the full evaluation suite (Phase 1) on both PyTorch and ONNX models
  - Assert: max |F1_pytorch - F1_onnx| < 0.5 percentage points
  - Assert: max absolute logit difference < 1e-4 on a golden 100-sample set
  - If validation fails, fall back to FP16 or investigate which layers
    are numerically sensitive to quantization

Step 4: Package
  - Output the artifact bundle (see 3.3)
```

**Uncertainty:** I'm not 100% confident that dynamic INT8 quantization will stay within 0.5 F1 points on clinical NER. Medical entity detection often relies on subtle contextual cues that quantization can blur. We'll know after Phase 1 benchmarks. If degradation exceeds tolerance, we fall back to FP16 (larger model but lossless accuracy).

### 3.2 Latency Profiling Methodology

Measuring "30ms" correctly requires care:

```
What to measure:
  - Tokenization time (tokenizer encode)
  - Model inference time (forward pass only)
  - Post-processing time (IOB2 decoding + entity grouping)
  - Regex layer time
  - Conflict resolution + token replacement time
  Total = sum of above = end-to-end PIIEngine.redact() wall clock

How to measure:
  - Use time.perf_counter_ns (not time.time — too coarse)
  - Warm up with 10 throwaway inferences (first call loads model into cache)
  - Run 1,000 inferences on representative inputs (varying length: 50-500 tokens)
  - Report: p50, p95, p99, max
  - For ARM simulation on dev machines: use CPU frequency throttling (if on macOS)
    or benchmark directly on an OCI A1 instance via SSH

What NOT to do:
  - Don't measure with PyTorch — measure ONNX Runtime only (that's what runs in prod)
  - Don't benchmark with batch_size > 1 (proxy processes one request at a time)
  - Don't include model load time (one-time cost at container startup)
```

### 3.3 Exported Artifact Bundle

The bundle that Rainier's Rust inference crate consumes:

```
models/
├── bert-ner-v{VERSION}/
│   ├── model.onnx              # Quantized INT8 ONNX model
│   ├── tokenizer.json          # HuggingFace fast tokenizer (single file)
│   ├── special_tokens_map.json # [CLS], [SEP], [PAD], [MASK] definitions
│   ├── schema.json             # Input/output contract (see below)
│   ├── label_map.json          # {0: "O", 1: "B-PER", 2: "I-PER", ...}
│   ├── checksum.sha256         # SHA-256 of model.onnx (verify before loading)
│   └── metadata.json           # Version, training date, F1 score, latency p95
```

**schema.json contract:**
```json
{
  "model_version": "1.0.0",
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
  "confidence_threshold": 0.85,
  "mapping_dictionary_format": {
    "description": "Token-to-original-text mapping for restoration",
    "example": {"[PERSON_1]": "John Smith", "[SSN_1]": "123-45-6789"},
    "serialization": "JSON object, keys are tokens, values are original text"
  }
}
```

**Flag for Rainier:** This schema is the integration contract. Any changes to field names, dtypes, shapes, or the mapping dictionary format must be communicated before merge. Suggest a shared `contracts/` directory or a Confluence page both sides reference.

---

## Phase 4: Privacy Hardening (OBS-12)

### 4.1 Laplacian Noise Injection

**Where to inject:** At the model's output logits, *before* softmax/argmax classification.

```
Forward pass produces logits: shape (seq_len, num_labels)
↓
Add Laplacian noise: logits_noisy = logits + Lap(0, sensitivity/epsilon)
↓
Apply softmax → argmax → IOB2 labels
```

**Why logits and not embeddings:**
- Embedding-level noise (earlier in the network) distorts the model's internal representations and causes severe accuracy degradation. It's theoretically stronger privacy but practically unusable.
- Logit-level noise is the standard approach for output perturbation in DP-NER. It preserves the model's learned representations while making individual predictions plausible for neighboring inputs.
- This is easier to implement, tune, and reason about.

**Sensitivity calibration:** The sensitivity (Δf) for token classification is 1.0 (changing one input token can flip at most one output label). So noise scale = 1.0 / epsilon.

**Implementation:** A wrapper around the model's forward method, applied *after* ONNX inference in the Python evaluation pipeline. In production (Rust), this would be applied by the inference crate post-model-call. **Flag for Rainier:** Rust side needs to implement Laplacian sampling over the logits tensor.

### 4.2 Accuracy-Privacy Trade-off

Sweep epsilon values and measure F1:

```
Epsilon values to test: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, ∞ (no noise)]

For each epsilon:
  1. Run evaluation suite on synthetic test set with noise enabled
  2. Record macro F1, per-entity F1, precision, recall
  3. Plot: F1 vs epsilon (utility curve)
  4. Plot: per-entity breakdown (some types are more noise-sensitive)
```

**Expected behavior:** F1 should plateau around epsilon=5-10 (nearly identical to no noise) and degrade sharply below epsilon=1.0. The "knee" of the curve is our operating point.

### 4.3 Acceptable Degradation Threshold

**Proposed threshold:** Maximum 3 percentage points of macro F1 degradation from the non-private baseline.

Reasoning:
- If baseline is 92% F1, the private model must be ≥89% — still within our 90% target if we have margin.
- If baseline is exactly 90%, we have zero room and must either improve the base model first or accept weaker privacy (higher epsilon).
- **This is a product decision, not purely technical.** Recommend discussing with the team: is 3pp acceptable? Some compliance frameworks may require stricter DP guarantees (lower epsilon) regardless of accuracy cost.

### 4.4 Membership Inference Attack (MIA) Testing

```
Method: Shadow Model Attack (Shokri et al., 2017)

1. Train N shadow models on subsets of the training data
2. For each shadow model, record its confidence scores on:
   - Data it WAS trained on (members)
   - Data it was NOT trained on (non-members)
3. Train a binary classifier (attack model) to distinguish members from non-members
   based on the confidence score distribution
4. Apply the attack model to the real model's outputs
5. Metric: Attack AUC (area under ROC curve)
   - AUC ≈ 0.5: model leaks no membership information (ideal)
   - AUC > 0.6: concerning
   - AUC > 0.7: unacceptable — increase DP noise

Run with noise OFF and noise ON to quantify the privacy improvement.
```

**Practical concern:** Shadow model training is expensive (N=10 minimum, each is a full BERT fine-tune). Budget ~8-12 GPU-hours. Consider running this on a cloud GPU instance (not on ARM) as a one-time validation step.

---

## Phase 5: Integration Testing & CI

### 5.1 End-to-End Round-Trip Test

```
test_round_trip():
  input_text = "Patient John Smith, MRN 12345, SSN 123-45-6789, takes Metformin."

  result = engine.redact(input_text)

  # Verify all PII is masked
  assert "John Smith" not in result.masked_text
  assert "12345" not in result.masked_text
  assert "123-45-6789" not in result.masked_text

  # Verify tokens are present
  assert "[PERSON_1]" in result.masked_text
  assert "[SSN_1]" in result.masked_text

  # Verify round-trip restoration
  restored = PIIEngine.restore(result.masked_text, result.mapping)
  assert restored == input_text

  # Verify mapping is complete
  assert len(result.mapping) == len(result.entities)

  # Verify no PII in logs (check that our logging doesn't contain raw text)
```

### 5.2 Regression Test Suite

Run on every PR to `/ml/**` or `/src/inference/**`:

| Test Category | What It Checks | Failure = Block PR? |
|---------------|---------------|---------------------|
| **Unit: entity detection** | Known inputs produce expected entity spans | Yes |
| **Unit: masking/restoration** | Round-trip identity on 50+ test cases | Yes |
| **Unit: regex patterns** | Each regex pattern matches/rejects known samples | Yes |
| **Unit: conflict resolution** | Overlapping regex+transformer spans resolve correctly | Yes |
| **Integration: ONNX parity** | ONNX output matches PyTorch output (< 1e-4 diff) | Yes |
| **Benchmark: F1** | Macro F1 ≥ 90% on synthetic test set | Yes |
| **Benchmark: latency** | p95 ≤ 30ms on 500-sample run (CPU) | Yes (warn at 25ms) |
| **Benchmark: model size** | ONNX file ≤ 300MB | Yes |
| **Smoke: no PII in logs** | Grep test output for known PII strings → must find zero | Yes |

### 5.3 GitHub Actions CI Integration

```yaml
# .github/workflows/ml-ci.yml

Triggers: push/PR to ml/**, src/inference/**, models/**

Jobs:
  lint:
    - ruff check ml/
    - black --check ml/

  unit-tests:
    - pytest ml/tests/ -v --tb=short

  benchmark:
    - python ml/evaluate.py --model models/current --dataset ml/data/synthetic_test.bio
    - Parse JSON report
    - Assert F1 >= 0.90 (fail PR if not)
    - Assert latency_p95_ms <= 30 (fail PR if not)
    - Post results as PR comment (F1, latency, model size)

  onnx-parity:
    - python ml/validate_onnx.py --pytorch models/current_pytorch --onnx models/current.onnx
    - Assert max logit diff < 1e-4

  security-scan:
    - Grep all test output and logs for known synthetic PII patterns
    - Fail if any raw PII appears outside of test input fixtures
```

**Note:** The benchmark job needs a cached model download (HuggingFace or OCI Object Storage) to avoid re-downloading 400MB on every CI run. Use GitHub Actions cache or a self-hosted runner with persistent storage.

### 5.4 Go/No-Go Criteria

Before merging any model change into the develop branch:

| Gate | Threshold | Non-Negotiable? |
|------|-----------|----------------|
| Macro F1 | ≥ 90% | Yes |
| Per-entity F1 (every type) | ≥ 85% | Yes |
| Recall (macro) | ≥ 88% | Yes — missed PII is a compliance risk |
| Latency p95 | ≤ 30ms | Yes |
| Latency p99 | ≤ 45ms | Soft — investigate but don't block |
| ONNX parity | < 0.5pp F1 diff vs PyTorch | Yes |
| Model size | ≤ 300MB | Yes |
| Round-trip tests | 100% pass | Yes |
| PII-in-logs scan | 0 findings | Yes |
| Ruff + Black | 0 errors | Yes |

---

## Jira Subtask Checklists

### OBS-3: Benchmarking

```
[ ] OBS-3.1  Generate synthetic enterprise test set (2,000 samples, CoNLL format)
[ ] OBS-3.2  Generate synthetic clinical test set (2,000 samples, CoNLL format)
[ ] OBS-3.3  Write evaluate.py (model-agnostic evaluation script with seqeval)
[ ] OBS-3.4  Benchmark dslim/bert-base-NER on CoNLL-2003
[ ] OBS-3.5  Benchmark dslim/distilbert-NER on CoNLL-2003
[ ] OBS-3.6  Benchmark StanfordAIMI/stanford-deidentifier-base on synthetic clinical set
[ ] OBS-3.7  Benchmark Bio_ClinicalBERT (fine-tuned) on synthetic clinical set
[ ] OBS-3.8  Measure latency p50/p95/p99 for all candidates (CPU, batch=1)
[ ] OBS-3.9  Measure model size + peak RAM for all candidates
[ ] OBS-3.10 Write baseline report (JSON + markdown summary)
[ ] OBS-3.11 Select primary model candidate based on F1/latency/size Pareto front
[ ] OBS-3.12 Implement regex layer for SSN, PHONE, EMAIL, MRN patterns
[ ] OBS-3.13 Implement hybrid conflict resolution (regex + transformer merge)
[ ] OBS-3.14 Re-benchmark with hybrid pipeline enabled
```

### OBS-7: Integration & Export

```
[ ] OBS-7.1  Write export_onnx.py using optimum (dynamic axes, opset 14)
[ ] OBS-7.2  Implement dynamic INT8 quantization step
[ ] OBS-7.3  Write validate_onnx.py (PyTorch vs ONNX parity check)
[ ] OBS-7.4  Run full evaluation suite on ONNX model, compare to PyTorch baseline
[ ] OBS-7.5  If INT8 degrades F1 > 0.5pp on clinical entities, test FP16 fallback
[ ] OBS-7.6  Define and document schema.json (input/output contract)
[ ] OBS-7.7  Define and document mapping dictionary JSON spec
[ ] OBS-7.8  Package artifact bundle (model, tokenizer, schema, checksums, metadata)
[ ] OBS-7.9  Write latency profiler script (1,000 inferences, varying lengths)
[ ] OBS-7.10 Profile on OCI ARM instance (SSH benchmark) — confirm ≤30ms p95
[ ] OBS-7.11 Handoff artifact bundle to Rainier + walk through schema.json
[ ] OBS-7.12 Integration smoke test: Rust ort loads model, runs one inference, outputs match
```

### OBS-12: Privacy Hardening

```
[ ] OBS-12.1  Implement Laplacian noise injection on output logits
[ ] OBS-12.2  Run epsilon sweep [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] — record F1 for each
[ ] OBS-12.3  Plot utility curve (F1 vs epsilon) and per-entity breakdown
[ ] OBS-12.4  Select operating epsilon (max 3pp F1 degradation from baseline)
[ ] OBS-12.5  Coordinate with Rainier: Rust-side Laplacian sampling implementation
[ ] OBS-12.6  Train 10 shadow models for membership inference attack test
[ ] OBS-12.7  Run MIA attack, record AUC with and without DP noise
[ ] OBS-12.8  If MIA AUC > 0.6 with noise, decrease epsilon and re-test
[ ] OBS-12.9  Document privacy parameters in metadata.json (epsilon, sensitivity, noise layer)
[ ] OBS-12.10 Final go/no-go: F1 ≥ 90%, MIA AUC ≤ 0.6, latency ≤ 30ms — all simultaneously
```

---

## Risks & Unknowns

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | **INT8 quantization degrades clinical NER accuracy beyond tolerance** | Medium | High — could blow F1 below 90% | Fallback to FP16. If FP16 is too large for ARM RAM, try mixed quantization (INT8 body, FP16 classification head). |
| 2 | **30ms latency not achievable on ARM A1** | Medium | High — fundamentally blocks the architecture | DistilBERT reduces latency ~40%. If still over budget, investigate ONNX Runtime graph optimizations (operator fusion, constant folding) or reduce max_sequence_length from 512 to 256. |
| 3 | **Synthetic test data doesn't represent real-world distribution** | High | Medium — benchmarks look good but production accuracy is lower | After deployment, add a human-in-the-loop review on a sample of redacted outputs (with PII already masked). Use findings to improve synthetic data generation. |
| 4 | **Differential privacy degrades medical entity detection disproportionately** | Medium | Medium — clinical entities may be more noise-sensitive | Per-entity DP analysis (Phase 4.2). If medical F1 drops below 85%, consider applying noise only to non-medical entity types (trade weaker privacy on names for stronger on MRNs). |
| 5 | **Shadow model MIA testing is too expensive** | Low | Low — delays OBS-12 but doesn't block deployment | Reduce to N=5 shadow models (less statistical power but still directionally correct). Or use simpler MIA methods (threshold attack on confidence scores). |
| 6 | **ONNX Runtime on ARM has limited operator support** | Low | High — model won't load at all | Test ONNX export early (Phase 3, not after Phase 2). Use `onnxruntime` profiling to identify unsupported ops. The `ort` Rust crate tracks the same backend. |
| 7 | **Integration contract changes cause Rust-side rework** | Medium | Medium — delays OBS-7 | Freeze schema.json as early as possible (end of Phase 1). Communicate all changes via PR reviews that tag Rainier. |

---

## Questions Before Implementation

1. **Model selection scope:** Should we evaluate any other model candidates beyond the four listed? I excluded multilingual models (XLM-R) and large models (bert-large) — the former because Obscura's scope appears English-only, the latter because it won't fit the ARM latency budget. Confirm?

2. **DistilBERT as production default:** If DistilBERT-NER benchmarks within 2-3 F1 points of bert-base-NER and meets latency, are you comfortable making it the production model? This is the single biggest lever for hitting 30ms.

3. **Privacy budget ownership:** Who decides the epsilon value — the ML team, the security team, or the customer? This determines whether we hardcode it, make it configurable per-deployment, or expose it as an API parameter.

4. **OCI ARM access for benchmarking:** Do we have an ARM instance provisioned where I can SSH in and run latency benchmarks? Simulating on macOS (Apple Silicon) won't give accurate numbers for Ampere A1.

5. **Rainier sync cadence:** How often should I sync with Rainier on the integration contract? I'd suggest a brief check-in after Phase 1 (model selected, schema draft) and a formal handoff at end of Phase 3 (artifact bundle ready).

6. **CI compute budget:** The benchmark job needs ~2-3 minutes of CPU time per run (1,000 inferences). Is this acceptable in GitHub Actions, or should we use a self-hosted runner? The shadow model training (OBS-12) definitely needs a GPU — where should that run?
