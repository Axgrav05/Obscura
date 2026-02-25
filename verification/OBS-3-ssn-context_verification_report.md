# OBS-3 SSN Context-Aware Detection — Verification Report

**Ticket:** OBS-3 — Benchmarking (Phase 2.1: Regex Layer + Phase 2.3: Hybrid Conflict Resolution)
**Branch:** `feat/ml/OBS-3-7-bert-ner-engine`
**Date:** 2026-02-25
**Author:** Claude Opus 4.6 (AI pair, co-authoring with Arjun Agravat)

---

## Problem Statement

The Obscura redaction pipeline had **0% recall on SSN entities**. BERT NER models (dslim/bert-base-NER) are trained on CoNLL-2003 which has no SSN entity type. Dashed SSNs (123-45-6789) are detectable by regex, but dashless SSNs (123456789) are syntactically identical to phone numbers, order IDs, and serial numbers. A naive `\d{9}` pattern would produce unacceptable false positives.

## Solution: Hybrid BERT + Context-Aware Regex

Implemented a two-stage cascade:
1. **Regex for recall** — catch all 9-digit candidates (dashed and dashless)
2. **Context-aware scoring for precision** — trigger word analysis within a 10-word window disambiguates dashless SSNs from non-SSN 9-digit numbers
3. **Conflict resolution** — merge BERT and regex entities with 4-rule resolution per GAMEPLAN.md Section 2.3

---

## Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `ml/schemas.py` | **Created** | Extracted `DetectedEntity` + `RedactionResult` dataclasses (avoids circular imports and stdlib `types` shadow) |
| `ml/regex_detector.py` | **Created** | `RegexDetector` class: dashed/dashless SSN patterns, IRS validation, context-aware scoring |
| `ml/pii_engine.py` | **Modified** | Integrated regex layer, added `merge_entities()` with 4-rule conflict resolution |
| `ml/generate_synthetic_data.py` | **Modified** | IRS-valid SSN generation, dashless SSN templates, negative sample generator |
| `ml/tests/__init__.py` | **Created** | Test package init |
| `ml/tests/test_ssn_context.py` | **Created** | 21 gold-standard tests (dashed, dashless context, merge logic) |
| `ml/evaluate.py` | **Modified** | Added `--mode hybrid` with SSN ground truth evaluation |

---

## Context-Aware Scoring Design

### Scoring Formula (dashless 9-digit numbers)

```
score = base (0.40)
      + 0.35 if trigger word found (ssn, social, tax, w-2, ss#, ...)
      + 0.20 if trigger phrase found ("social security", "tax id", ...)
      - 0.35 if negative word found (phone, call, tracking, serial, ...)
```

Clamped to [0.0, 1.0]. Threshold: **0.70** (only matches scoring ≥0.70 are emitted as SSN).

### Dashed SSN Scoring
- Dashed format (XXX-XX-XXXX) gets fixed score **0.99** — format is near-unique to SSNs
- IRS structural validation rejects: area 000/666/900-999, group 00, serial 0000

### Context Window
- 10 words on each side of the candidate match
- Punctuation stripped before matching (e.g., "SSN:" → "ssn")

---

## IRS SSN Validation (SSA Publication 4557)

Invalid SSN structures rejected by `_is_valid_ssn_parts()`:

| Component | Invalid Values | Reason |
|-----------|---------------|--------|
| Area (first 3 digits) | 000 | Not assigned |
| Area | 666 | Excluded by IRS |
| Area | 900-999 | Reserved for ITIN |
| Group (middle 2 digits) | 00 | Not assigned |
| Serial (last 4 digits) | 0000 | Not assigned |

---

## Conflict Resolution Rules (GAMEPLAN.md 2.3)

| Rule | Condition | Winner |
|------|-----------|--------|
| 1 | Exact overlap | Regex wins for structured types (SSN, PHONE, EMAIL, MRN); BERT wins for semantic types (PERSON, ORG, LOC) |
| 2 | Partial overlap | Longer span wins |
| 3 | Nested spans | Outer (longer) span wins |
| 4 | No overlap | Keep both |

---

## Test Results

### Unit Tests: 21/21 PASSED

```
ml/tests/test_ssn_context.py::TestDashedSSN::test_basic_dashed_ssn PASSED
ml/tests/test_ssn_context.py::TestDashedSSN::test_invalid_area_666 PASSED
ml/tests/test_ssn_context.py::TestDashedSSN::test_invalid_area_900 PASSED
ml/tests/test_ssn_context.py::TestDashedSSN::test_invalid_group_00 PASSED
ml/tests/test_ssn_context.py::TestDashedSSN::test_invalid_serial_0000 PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_trigger_word_ssn PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_phone_context_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_social_security_phrase PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_tracking_context_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_tax_id_trigger PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_zip_context_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_background_check_trigger PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_serial_context_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_ss_hash_trigger PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_invoice_context_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_no_context_rejected PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_no_overlap_keeps_both PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_exact_overlap_regex_wins_structured PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_exact_overlap_bert_wins_semantic PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_nested_keeps_outer PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_empty_inputs PASSED

21 passed in 3.96s
```

### Gold Standard Test Strings (10 cases)

| # | Input | Expected | Result | Score |
|---|-------|----------|--------|-------|
| 1 | "Applicant's SSN is 123456789, verified via W-2." | SSN detected | PASS | ≥0.70 |
| 2 | "Call the vendor at 941555012 to confirm delivery." | Rejected | PASS | <0.70 |
| 3 | "Social security number on file: 234567890." | SSN detected | PASS | ≥0.85 |
| 4 | "Tracking number 345678901 shipped via FedEx." | Rejected | PASS | <0.70 |
| 5 | "Employee tax ID 456789012 must be updated on the W-9." | SSN detected | PASS | ≥0.70 |
| 6 | "Enter ZIP code 100234567 for extended postal lookup." | Rejected | PASS | <0.70 |
| 7 | "Background check confirmed identity 567890123 for Jane Doe." | SSN detected | PASS | ≥0.70 |
| 8 | "Serial number 678901234 on the replacement device." | Rejected | PASS | <0.70 |
| 9 | "Patient SS# 789012345 requires HIPAA verification." | SSN detected | PASS | ≥0.70 |
| 10 | "Invoice 890123456 was processed for account closure." | Rejected | PASS | <0.70 |

### Synthetic Data Regeneration

```
Generating 500 synthetic NER samples...
Wrote 500 samples to ml/data/synthetic.jsonl

Entity distribution:
       PER: 500
       ORG: 325
       SSN: 201
       MRN: 175
     PHONE: 115
       DOB:  86
     EMAIL:  83
       LOC:  49
     TOTAL: 1534

Sanity check passed: token/tag alignment OK.
```

- **201 SSN entities** (mix of dashed and dashless with trigger context)
- **75 negative samples** (9-digit numbers in phone/tracking/serial/account context, NOT labeled as SSN)
- Split: 50% enterprise, 35% clinical, 15% negative

### Hybrid Evaluation Smoke Test (20 samples)

| Mode | Macro F1 | SSN F1 | ORG F1 | PER F1 |
|------|----------|--------|--------|--------|
| **hybrid** | 0.3125 | **1.00** | 0.25 | 0.00 |
| **bert** | 0.0976 | N/A | 0.29 | 0.00 |

**Key takeaway:** SSN detection went from **0.00 → 1.00 F1** with the regex layer. Hybrid macro F1 is 3.2x higher than BERT-only due to the SSN contribution. PER shows 0.00 on this small slice due to whitespace tokenization misalignment (known issue — BERT's WordPiece vs. our ground truth tokenization).

### Lint

```
$ ruff check ml/
All checks passed!

$ black --check ml/
All done! 9 files would be left unchanged.
```

---

## Bugs Found and Fixed During Implementation

1. **`merge_entities()` short-circuit bug**: When only one source had entities (e.g., only BERT, empty regex), the function returned the list without deduplication. This meant nested BERT spans (e.g., "John Smith Jr" and "John Smith") were both kept instead of resolving to the outer span. **Fix:** Removed early-return short-circuits; always run the full merge loop.

2. **`types.py` stdlib shadow**: Naming our shared types module `ml/types.py` shadowed Python's stdlib `types` module, causing `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'`. **Fix:** Renamed to `ml/schemas.py`.

3. **Script-mode import resolution**: Running `python ml/evaluate.py` as a script adds `ml/` to `sys.path`, making `from ml.pii_engine import PIIEngine` fail (there's no `ml/ml/` directory). **Fix:** Added try/except fallbacks for all cross-module imports so both `python ml/evaluate.py` (script mode) and `from ml.xxx import` (package mode via pytest) work.

---

## Architecture Diagram

```
Input Text
    │
    ├──────────────────────────┐
    ▼                          ▼
┌─────────────┐      ┌──────────────────┐
│  BERT NER   │      │  RegexDetector   │
│  Pipeline   │      │  (SSN patterns)  │
│             │      │                  │
│ PER/ORG/LOC │      │ Dashed: 0.99     │
│ MISC        │      │ Dashless: scored │
└─────┬───────┘      └────────┬─────────┘
      │                       │
      └───────────┬───────────┘
                  ▼
         ┌──────────────┐
         │merge_entities│
         │              │
         │ Rule 1: Exact│
         │ Rule 2: Part.│
         │ Rule 3: Nest │
         │ Rule 4: None │
         └──────┬───────┘
                ▼
        Merged Entities
```

---

## Known Limitations

1. **PHONE, EMAIL, MRN regex not yet implemented** — only SSN has a regex pattern. Other structured types are filtered to O in evaluation.
2. **Whitespace tokenization mismatch** — synthetic data uses whitespace tokenization while BERT uses WordPiece. This causes alignment issues for multi-word entities, depressing PER/ORG F1 on the synthetic benchmark.
3. **Small smoke test sample** — 20-sample evaluation is for smoke testing only. Full 500-sample runs needed for production metrics.
4. **No adversarial testing** — current test suite covers common patterns but not adversarial inputs (e.g., SSNs embedded in URLs, SSNs split across lines).

---

## Gemini Verification Prompt

```
You are a senior ML engineer reviewing a hybrid BERT + regex NER pipeline for PII/PHI redaction. The following files implement context-aware SSN detection with dashless disambiguation. Please review for:

1. **Correctness**: Are the IRS SSN validation rules (Publication 4557) correctly implemented? Check area 000/666/900-999, group 00, serial 0000 rejection.

2. **Context scoring logic**: Review the trigger word / negative word scoring in `regex_detector.py`. Are the weights (base 0.40, +0.35 trigger, +0.20 phrase, -0.35 negative) sensible? Could a valid SSN be missed? Could a non-SSN be false-positived?

3. **Conflict resolution**: Review `merge_entities()` in `pii_engine.py`. Does the greedy merge correctly implement all 4 rules (exact overlap, partial overlap, nested, no overlap)? Are there edge cases where the greedy approach fails?

4. **Test coverage**: Review `test_ssn_context.py`. Are the 10 gold standard strings sufficient? What edge cases are missing (e.g., SSN at start/end of text, multiple SSNs, SSN adjacent to another entity)?

5. **Import structure**: The codebase uses try/except fallbacks for `ml.xxx` vs bare `xxx` imports to support both script mode and package mode. Is this a maintainable pattern or should we use a proper package install instead?

6. **Security**: Does the regex detector inadvertently log or expose raw SSN values? Review the DetectedEntity dataclass — does it store the raw SSN text, and is that acceptable for HIPAA compliance?

Files to review:
- ml/schemas.py (DetectedEntity, RedactionResult dataclasses)
- ml/regex_detector.py (RegexDetector with context scoring)
- ml/pii_engine.py (merge_entities, PIIEngine hybrid orchestration)
- ml/generate_synthetic_data.py (synthetic data with dashless SSN + negative samples)
- ml/tests/test_ssn_context.py (21 gold standard unit tests)
- ml/evaluate.py (hybrid evaluation mode)

Key design decisions to validate:
- Threshold 0.70 for dashless SSN (is this the right tradeoff?)
- 10-word context window (sufficient for disambiguation?)
- Asymmetric weights (single negative word can suppress even when trigger present)
- Greedy merge vs. optimal assignment for conflict resolution
```
