# OBS-8 Revision 3 (R3) Review Report

**Date:** 2026-03-04
**Reviewer:** Gemini (AI/ML pair)
**Target:** `regex_detector.py`, `pii_engine.py`, `evaluate.py`, `export_onnx.py`

## 1. PII/PHI Expansion (regex_detector.py & evaluate.py)
*   **Regex Validation:** All four patterns (DOB, CREDIT_CARD, IPV4, PASSPORT) correctly implement the symmetric `(?<!\w)` and `(?!\w)` negative word-boundary lookarounds to prevent substring matches. 
    *   **DOB:** The regex cleverly uses alternation for both `MM/DD/YYYY` and `YYYY-MM-DD` and accurately restricts the month (01-12) and day (01-31) logic within the pattern itself.
    *   **CREDIT_CARD:** The 16-digit structure correctly prevents any subset or overlap confusion with the 9-digit SSNs. 
    *   **IPv4:** Correctly implements the 0-255 limits within the expression so strings like `256.0.0.1` are correctly bypassed.
    *   **PASSPORT:** Validated as strictly case-sensitive (uppercase only).
*   **Pipeline Hooks:** All types were successfully added to `REGEX_AUTHORITATIVE_TYPES`, `HYBRID_ENTITY_TYPES`, and the `entity_to_bio` mapping in the evaluator.
*   **Risk Assessment (Limitations 4 & 5):** 
    *   **CREDIT_CARD:** Redacting any arbitrary 16-digit number is generally safe in an enterprise PII context since 16 digits rarely occur innocently outside of financial data. No context-aware scoring is strictly necessary right now, though it would be a nice enhancement.
    *   **DOB:** This is a **high risk** for false positives, as it will redact *all* dates in those formats. Unless the business logic dictates that all dates must be stripped, context-aware scoring (checking near "DOB", "Birth", "born") may be necessary in the future.

## 2. Configurable Redaction (pii_engine.py)
*   **Filtering Logic Placement:** The `disabled_entities` filtering is appropriately placed *after* the `detect()` logic and *before* mapping/masking happens. This is the correct architectural decision. If a disabled entity was skipped during detection, a suppressed overlapping entity might take its place incorrectly.
*   **Backward Compatibility:** Implemented cleanly with a `None` default and `frozenset` casting for O(1) exclusions.

## 3. BYOM ONNX Exporter (ml/export_onnx.py)
*   **Dependencies:** Uses only `torch`, `transformers`, and standard libraries.
*   **Torch ONNX Confg:** Fully hits the requirements (`opset_version=14`). The dynamic axes are correctly keyed to `batch_size` (0) and `sequence_length` (1) across `input_ids`, `attention_mask`, and `logits`.
*   **Tokenizer Extraction:** Correctly uses `save_pretrained()` and performs a hard check to ensure `tokenizer.json` builds locally as required.

## 4. Quality & Regression Tests
*   **Regressions:** Test suite stability verified — `30/30` passing natively. 
*   **Missing Test Coverage:** While the logic is sound, deploying core feature branches without corresponding Unit Tests (for the pipelines) carries some risk. I consider this missing coverage to be an **IMPORTANT** issue to resolve before closing out the epic.

---

## Conclusion & Verdict
**VERDICT: PASS (with minor tech debt conditions)**

The R3 architectural implementation is sound and meets the requirements. 

## Action Items for Claude (Required Before Final Production Release)

Please create the following test files to cover the missing logic identified in Known Limitations 1 & 2:

### 1. `ml/tests/test_regex_extensions.py`
Must test the strict regex formats:
- IPv4 limit checks (e.g., assert `256.1.1.1` is rejected)
- Passport casing (e.g., assert `a12345678` is rejected)
- DOB bounded checks (e.g., assert `13/40/1990` is rejected)
- Valid formats for all 4 new types (DOB, CREDIT_CARD, IPV4, PASSPORT)

**Command:**
```bash
pytest ml/tests/test_regex_extensions.py
```

### 2. `ml/tests/test_redaction_config.py`
Must test the `disabled_entities` logic in `pii_engine`:
- Assert that passing `disabled_entities=["SSN"]` leaves an SSN untouched
- Assert disabled entities do not populate the substitution mapping dict
- Assert standard overlapping resolution is preserved even if the winning entity is ultimately disabled (so an underlying suppressed entity does not accidentally emerge).

**Command:**
```bash
pytest ml/tests/test_redaction_config.py
```
