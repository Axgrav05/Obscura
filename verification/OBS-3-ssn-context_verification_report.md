# OBS-3 SSN Context-Aware Detection — Verification Report

**Ticket:** OBS-3 — Benchmarking (Phase 2.1: Regex Layer + Phase 2.3: Hybrid Conflict Resolution)
**Branch:** `feat/ml/OBS-3-7-bert-ner-engine`
**Date:** 2026-02-25 (initial), 2026-02-26 (revision after Gemini review)
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
| `ml/schemas.py` | **Created** | Extracted `DetectedEntity` + `RedactionResult` dataclasses with PII-safe `__repr__`/`__str__` |
| `ml/regex_detector.py` | **Created** | `RegexDetector` class: dashed/dashless SSN patterns with word-boundary guards, IRS validation, context-aware scoring |
| `ml/pii_engine.py` | **Modified** | Integrated regex layer, added `merge_entities()` with multi-overlap-aware 4-rule conflict resolution |
| `ml/generate_synthetic_data.py` | **Modified** | IRS-valid SSN generation, dashless SSN templates, negative sample generator |
| `ml/tests/__init__.py` | **Created** | Test package init |
| `ml/tests/test_ssn_context.py` | **Created** | 27 tests (dashed, dashless context, boundary edge cases, multi-overlap merge) |
| `ml/evaluate.py` | **Modified** | Added `--mode hybrid` with SSN ground truth evaluation |
| `ml/pyproject.toml` | **Modified** | Added `[build-system]` for editable install (`pip install -e .`) |

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

### Design Tradeoff: Asymmetric Weights

The negative word penalty (-0.35) can suppress a dashless SSN even when a trigger word is present. Example: "The SSN matches tracking number 123456789" scores `0.40 + 0.35 - 0.35 = 0.40` (below threshold). This is an intentional V1 design choice — for a HIPAA-regulated redaction pipeline, **false negatives on ambiguous mixed-context sentences are preferable to false positives that redact non-SSN identifiers** and corrupt downstream data. This tradeoff is documented and will be revisited in V2 with a learned context classifier.

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

### Multi-Overlap Resolution

When a candidate entity overlaps with **multiple** already-accepted entities, the merge algorithm now:
1. Collects ALL conflicting indices (not just the first)
2. Resolves the candidate against each conflict — candidate must **win all** to be accepted
3. If candidate wins all: removes all losers and inserts candidate
4. If candidate loses any: discards candidate, all accepted entities survive

This prevents the prior bug where a long span replacing only the first conflicting entity left a second overlapping entity in the output.

---

## Test Results

### Unit Tests: 27/27 PASSED

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
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_multiple_ssns_in_one_string PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_alpha_adjacent_dashed_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_alpha_adjacent_dashless_rejected PASSED
ml/tests/test_ssn_context.py::TestDashlessSSNContext::test_ssn_after_colon_accepted PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_no_overlap_keeps_both PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_exact_overlap_regex_wins_structured PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_exact_overlap_bert_wins_semantic PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_nested_keeps_outer PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_three_way_overlap_long_span_wins PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_three_way_overlap_short_wins PASSED
ml/tests/test_ssn_context.py::TestMergeEntities::test_empty_inputs PASSED

27 passed in 3.56s
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

3. **Script-mode import resolution**: Running `python ml/evaluate.py` as a script adds `ml/` to `sys.path`, making `from ml.pii_engine import PIIEngine` fail (there's no `ml/ml/` directory). **Initial fix:** try/except fallbacks. **Revised fix (post-review):** proper editable install (see below).

---

## Gemini Review Findings and Fixes

A senior ML engineering review (see `verification/OBS-3-ssn-context_verification_review.md`) identified six findings across correctness, security, and code quality. Five required code changes; one was acknowledged as a documented design tradeoff.

### Finding 1: IRS SSN Validation — PASS (no change needed)

> **Verdict:** The IRS validation rules from Publication 4557 are implemented accurately.

No action required. `_is_valid_ssn_parts()` correctly rejects area 000/666/900-999, group 00, serial 0000.

### Finding 2: Context Scoring Logic — Acknowledged, documented tradeoff

> **Verdict:** Reasonable heuristics for V1, but high false-negative risk when sentences conflate identifiers.

The reviewer identified that mixed-context sentences (e.g., "The SSN matches tracking number 123456789") produce `0.40 + 0.35 - 0.35 = 0.40`, failing the 0.70 threshold. This is an intentional conservative bias: for a HIPAA redaction pipeline, we prefer to miss ambiguous cases rather than corrupt non-SSN data. A V2 learned classifier can address mixed-context disambiguation.

**Action:** Documented the tradeoff in the "Context-Aware Scoring Design" section above. No code change.

### Finding 3: Conflict Resolution Multi-Overlap Bug — FIXED

> **Verdict:** Buggy for overlapping entities. It should check overlap with all existing merged entities and remove all overridden ones.

The reviewer correctly identified that `merge_entities()` broke after resolving the first conflict, leaving a second overlapping entity in the output.

**Fix:** Rewrote the merge loop to:
1. Collect ALL indices of accepted entities overlapping with the candidate
2. Resolve each conflict — candidate must win all to be accepted
3. If candidate wins all: remove all losers (in reverse index order), add candidate
4. If candidate loses any: discard candidate

**Before (buggy):**
```python
for i, accepted in enumerate(merged):
    overlap = _classify_overlap(accepted, candidate)
    if overlap == "none":
        continue
    conflict_found = True
    if overlap == "exact":
        winner = _resolve_exact_overlap(accepted, candidate)
    else:
        winner = _resolve_by_length(accepted, candidate)
    if winner is candidate:
        merged[i] = candidate
    break  # BUG: only resolves first conflict
```

**After (fixed):**
```python
conflict_indices: list[int] = []
for i, accepted in enumerate(merged):
    overlap = _classify_overlap(accepted, candidate)
    if overlap != "none":
        conflict_indices.append(i)

if not conflict_indices:
    merged.append(candidate)
    continue

candidate_wins_all = True
for i in conflict_indices:
    accepted = merged[i]
    overlap = _classify_overlap(accepted, candidate)
    if overlap == "exact":
        winner = _resolve_exact_overlap(accepted, candidate)
    else:
        winner = _resolve_by_length(accepted, candidate)
    if winner is not candidate:
        candidate_wins_all = False
        break

if candidate_wins_all:
    for i in reversed(conflict_indices):
        merged.pop(i)
    merged.append(candidate)
```

**Verified by:** `test_three_way_overlap_long_span_wins` and `test_three_way_overlap_short_wins` (both passing).

### Finding 4: Test Coverage Gaps — FIXED

> **Verdict:** Insufficient coverage for text boundary edge-cases and multi-overlap conflict resolution.

The reviewer identified three missing edge cases. All now have tests:

| Missing Case | Test Added | Result |
|---|---|---|
| Multiple SSNs in a single string | `test_multiple_ssns_in_one_string` | PASS — both detected |
| Alpha-adjacent dashed SSN (`A123-45-6789B`) | `test_alpha_adjacent_dashed_rejected` | PASS — rejected |
| Alpha-adjacent dashless SSN (`REF123456789X`) | `test_alpha_adjacent_dashless_rejected` | PASS — rejected |
| Punctuation-adjacent SSN (`SSN:123-45-6789`) | `test_ssn_after_colon_accepted` | PASS — accepted |
| Three-way overlap (long span beats two shorts) | `test_three_way_overlap_long_span_wins` | PASS — 1 entity |
| Three-way overlap (short span loses to accepted) | `test_three_way_overlap_short_wins` | PASS — 1 entity |

**Regex boundary fix:** Changed `(?<!\d)` / `(?!\d)` to `(?<!\w)` / `(?!\w)` on both SSN patterns. This rejects alpha-adjacent matches (`A123-45-6789B`) while still accepting punctuation-adjacent ones (`SSN:123-45-6789`), since `:` is not a word character.

Test count: **21 → 27** (6 new tests, all passing).

### Finding 5: Import Structure — FIXED

> **Verdict:** Should be removed. We should transition to a proper setup tool configuration.

**Fix:**
1. Added `[build-system]` and `[tool.setuptools.packages.find]` to `ml/pyproject.toml`
2. Ran `pip install -e .` to install the `ml` package in editable mode
3. Removed all `try/except ModuleNotFoundError` fallbacks from `regex_detector.py`, `pii_engine.py`, and `evaluate.py`
4. All imports now use consistent `from ml.* import` absolute paths
5. `evaluate.py` uses `TYPE_CHECKING` guard + `from __future__ import annotations` for the lazy `PIIEngine` import (only loaded at runtime in hybrid mode)

### Finding 6: Security / HIPAA Compliance — FIXED

> **Verdict:** `DetectedEntity` should override `__repr__` and `__str__` to obfuscate the `text` field.

**Fix:** Added `__repr__` and `__str__` overrides to both `DetectedEntity` and `RedactionResult` in `ml/schemas.py`:

- **`DetectedEntity.__repr__`**: Shows `text='***'` instead of raw PII. Entity type, offsets, score, token, and source are visible.
- **`DetectedEntity.__str__`**: Compact format `[regex:SSN (10:21) score=0.99]` with no raw text.
- **`RedactionResult.__repr__`**: Replaces all mapping values with `'***'`. Masked text and session ID are visible.

**Verification:**
```python
>>> e = DetectedEntity('123-45-6789', 'SSN', 10, 21, 0.99, '[SSN_1]', 'regex')
>>> repr(e)
"DetectedEntity(text='***', entity_type='SSN', start=10, end=21, score=0.99, token='[SSN_1]', source='regex')"
>>> str(e)
'[regex:SSN (10:21) score=0.99]'

>>> r = RedactionResult('File [SSN_1]', [e], {'[SSN_1]': '123-45-6789'}, 'abc123')
>>> repr(r)
"RedactionResult(masked_text='File [SSN_1]', entities=[DetectedEntity(text='***', ...)], mapping={'[SSN_1]': '***'}, session_id='abc123')"
```

Raw PII (`123-45-6789`) never appears in `repr()` or `str()` output. The `.text` attribute is still accessible for intentional programmatic use (e.g., the masking logic in `PIIEngine.redact()`), but accidental logging, debug printing, or JSON serialization of the dataclass representation is safe.

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
         ┌──────────────────┐
         │  merge_entities  │
         │  (multi-overlap) │
         │                  │
         │ Rule 1: Exact    │
         │ Rule 2: Partial  │
         │ Rule 3: Nested   │
         │ Rule 4: None     │
         └────────┬─────────┘
                  ▼
          Merged Entities
                  │
                  ▼
         ┌──────────────────┐
         │  DetectedEntity  │
         │  (PII-safe repr) │
         │  text='***'      │
         └──────────────────┘
```

---

## Known Limitations

1. **PHONE, EMAIL, MRN regex not yet implemented** — only SSN has a regex pattern. Other structured types are filtered to O in evaluation.
2. **Whitespace tokenization mismatch** — synthetic data uses whitespace tokenization while BERT uses WordPiece. This causes alignment issues for multi-word entities, depressing PER/ORG F1 on the synthetic benchmark.
3. **Small smoke test sample** — 20-sample evaluation is for smoke testing only. Full 500-sample runs needed for production metrics.
4. **Mixed-context false negatives** — sentences containing both SSN trigger words and negative context words (e.g., "SSN matches tracking number 123456789") are conservatively rejected. Acknowledged V1 tradeoff; V2 will use a learned classifier.
5. **No adversarial testing** — current test suite covers common patterns but not adversarial inputs (e.g., SSNs embedded in URLs, SSNs split across lines).

---

## Commit History

| Commit | Description |
|--------|-------------|
| `bd390dc` | `feat(ML): OBS-3 add context-aware SSN detection with hybrid BERT+regex pipeline` |
| `51269da` | `fix(ML): OBS-3 address Gemini review findings for SSN context detection` |

---

## Gemini Verification Prompt (Round 2)

```
You are a senior ML engineer performing a SECOND review of a hybrid BERT + regex
NER pipeline for PII/PHI redaction. A previous review identified six findings.
Five have been addressed with code changes; one was documented as an intentional
design tradeoff. Please verify that each fix is correct and complete, and
identify any remaining issues.

PREVIOUS FINDINGS AND CLAIMED FIXES:

1. IRS SSN Validation — No change needed (confirmed correct in first review).

2. Context Scoring Logic — Acknowledged as V1 tradeoff. Mixed-context sentences
   (both trigger and negative words present) are conservatively rejected.
   VERIFY: Is this tradeoff adequately documented? Are there additional
   mixed-context scenarios beyond the one identified that could be problematic?

3. Conflict Resolution Multi-Overlap Bug — FIXED. merge_entities() now collects
   ALL overlapping indices, resolves candidate against each, and only accepts
   candidate if it wins all conflicts. Losers are removed in reverse index order.
   VERIFY: Read the new merge_entities() implementation in pii_engine.py. Does
   the fix correctly handle: (a) candidate overlapping 3+ accepted entities,
   (b) candidate losing to one accepted but winning against another, (c) empty
   conflict_indices path? Check that reverse-order pop() preserves valid indices.

4. Test Coverage — FIXED. Added 6 new tests:
   - test_multiple_ssns_in_one_string
   - test_alpha_adjacent_dashed_rejected
   - test_alpha_adjacent_dashless_rejected
   - test_ssn_after_colon_accepted
   - test_three_way_overlap_long_span_wins
   - test_three_way_overlap_short_wins
   VERIFY: Are these tests sufficient to cover the gaps identified in round 1?
   Are there still missing edge cases (e.g., SSN at very start/end of text with
   no surrounding context, underscore-adjacent like ID_123456789, or SSN inside
   a larger numeric string like 1234567890)?

5. Import Structure — FIXED. Replaced try/except with editable install:
   - Added [build-system] and [tool.setuptools.packages.find] to pyproject.toml
   - pip install -e . installs ml as a package
   - All imports use consistent from ml.* paths
   - evaluate.py uses TYPE_CHECKING + from __future__ import annotations
   VERIFY: Is the pyproject.toml configuration correct? Does
   where=[".."], include=["ml", "ml.*"] correctly discover the package from
   the ml/ directory? Could this break if the repo is cloned to a different
   path or if another ml/ package exists in the environment?

6. Security / HIPAA — FIXED. Added __repr__ and __str__ to DetectedEntity and
   RedactionResult that mask raw PII with '***'.
   VERIFY: Read schemas.py. Does the __repr__ override cover all PII-containing
   fields? Is there a path where raw PII could still leak (e.g., dataclasses
   asdict(), JSON serialization via dataclasses.asdict() + json.dumps, or
   direct attribute access in f-strings)? Should we also add __format__?

FILES TO REVIEW:
- ml/schemas.py — DetectedEntity/RedactionResult with __repr__/__str__ overrides
- ml/regex_detector.py — word-boundary guards (?<!\w)/(?!\w) on SSN patterns
- ml/pii_engine.py — rewritten merge_entities() with multi-overlap resolution
- ml/tests/test_ssn_context.py — 27 tests (6 new edge cases)
- ml/evaluate.py — clean imports with TYPE_CHECKING guard
- ml/pyproject.toml — [build-system] and package discovery config

ADDITIONAL REVIEW POINTS:
- The regex patterns now use (?<!\w) and (?!\w) instead of (?<!\d) and (?!\d).
  Does this correctly reject A123-45-6789B while accepting SSN:123-45-6789?
  Are there legitimate SSN contexts where the number might follow a word
  character (e.g., "SSN#123-45-6789" where # is stripped but digits are not)?
- The merge algorithm uses `winner is not candidate` (identity check, not
  equality). Is this correct given that _resolve_by_length and
  _resolve_exact_overlap return one of the two input objects by reference?
- With the editable install, does `python ml/evaluate.py` still work as a
  standalone script, or must it now be invoked as `python -m ml.evaluate`?
```
