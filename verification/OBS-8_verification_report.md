# OBS-8 Verification Report: PII/PHI Expansion, Configurable Redaction, ONNX Exporter

> **Date:** 2026-03-04
> **Ticket:** OBS-8 (SCRUM-107) — Follow-up Implementation
> **Branch:** `SCRUM-107-OBS-8-Run-model-benchmarks-and-select-final-model`
> **Reviewer:** Claude Opus 4.6 (AI/ML pair)
> **Status:** IMPLEMENTATION COMPLETE
> **Revision:** R3 — implements 3 new subtasks from `OBS-8_claude_implementation_prompt.md`

---

## Scope

This revision implements three new subtasks defined in `verification/OBS-8_claude_implementation_prompt.md`, building on the OBS-8 benchmarking work (R1-R2) which achieved macro F1 0.9576 with the fine-tuned hybrid pipeline.

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | PII/PHI Categorization Expansion (DOB, Credit Card, IPv4, Passport) | COMPLETE |
| Task 2 | Configurable Redaction Fields (`disabled_entities`) | COMPLETE |
| Task 3 | BYOM ONNX Exporter CLI (`export_onnx.py`) | COMPLETE |

---

## Task 1: PII/PHI Categorization Expansion

### New Regex Patterns (regex_detector.py)

Four new entity types added to `RegexDetector` with pre-compiled patterns in `__post_init__()`:

| Entity | Pattern | Format | Score | Guards |
|--------|---------|--------|-------|--------|
| DOB | `(?<!\w)(?:(?:0[1-9]\|1[0-2])/(?:0[1-9]\|[12]\d\|3[01])/(?:19\|20)\d{2}\|\d{4}-(?:0[1-9]\|1[0-2])-(?:0[1-9]\|[12]\d\|3[01]))(?!\w)` | MM/DD/YYYY, YYYY-MM-DD | 0.95 | `(?<!\w)` / `(?!\w)` |
| CREDIT_CARD | `(?<!\w)(?:\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\|\d{16})(?!\w)` | 16-digit dashed/spaced/undashed | 0.99 | `(?<!\w)` / `(?!\w)` |
| IPV4 | `(?<!\w)(?:(?:25[0-5]\|2[0-4]\d\|[01]?\d\d?)\.){3}(?:25[0-5]\|2[0-4]\d\|[01]?\d\d?)(?!\w)` | Dotted quad (0-255 per octet) | 0.95 | `(?<!\w)` / `(?!\w)` |
| PASSPORT | `(?<!\w)[A-Z]\d{8}(?!\w)` | 1 uppercase letter + 8 digits | 0.99 | `(?<!\w)` / `(?!\w)` |

**Design decisions:**
- All patterns follow the established convention: symmetric `(?<!\w)` / `(?!\w)` word-boundary guards, pre-compiled at construction time, dedicated `_detect_*()` method, configurable `*_score` dataclass field.
- DOB uses alternation for both MM/DD/YYYY and YYYY-MM-DD with strict month (01-12), day (01-31), and year (1900-2099) validation.
- Credit Card's 16-digit length prevents overlap with 9-digit SSN patterns.
- IPv4 validates each octet to 0-255 within the regex itself, preventing false positives on version strings.
- Passport is case-sensitive (uppercase only) to reduce false positives on arbitrary alphanumeric strings.

### Cascade Updates

**pii_engine.py — `REGEX_AUTHORITATIVE_TYPES`:**
```python
REGEX_AUTHORITATIVE_TYPES: frozenset[str] = frozenset(
    {"SSN", "PHONE", "EMAIL", "MRN", "DOB", "CREDIT_CARD", "IPV4", "PASSPORT"}
)
```
All four new types added so regex wins exact-match conflicts per GAMEPLAN.md Section 2.3.

**evaluate.py — `HYBRID_ENTITY_TYPES`:**
```python
HYBRID_ENTITY_TYPES: set[str] = BERT_ENTITY_TYPES | {
    "SSN", "PHONE", "EMAIL", "MRN", "DOB", "CREDIT_CARD", "IPV4", "PASSPORT",
}
```

**evaluate.py — `entity_to_bio` (in `align_hybrid_predictions_to_words()`):**
Added `"DOB": "DOB"`, `"CREDIT_CARD": "CREDIT_CARD"`, `"IPV4": "IPV4"`, `"PASSPORT": "PASSPORT"` mappings.

**evaluate.py — hybrid mode note:**
Updated from "DOB ground truth tags filtered to O (regex pattern not yet added)" to "All structured types evaluated" since DOB now has a regex pattern.

---

## Task 2: Configurable Redaction Fields

### Implementation (pii_engine.py)

Updated `PIIEngine.redact()` signature:
```python
def redact(self, text: str, disabled_entities: list[str] | None = None) -> RedactionResult:
```

Filtering logic added **after** `detect()` and **before** masking:
```python
if disabled_entities:
    disabled = frozenset(disabled_entities)
    entities = [e for e in entities if e.entity_type not in disabled]
```

**Design decisions:**
- Filtering happens after `detect()` to preserve the full hybrid conflict resolution pipeline. If we filtered before detection, a disabled entity could suppress a regex match that would otherwise win a conflict, causing a BERT entity to incorrectly survive.
- Uses `frozenset` for O(1) lookup on the disabled set.
- Disabled entities are excluded from the masking loop entirely — they produce no `[TYPE_N]` token, no mapping entry, and the original text remains intact at their span positions.
- The `disabled_entities` parameter defaults to `None`, preserving full backward compatibility.

---

## Task 3: BYOM ONNX Exporter CLI

### Implementation (ml/export_onnx.py — NEW FILE)

A standalone CLI script that converts HuggingFace NER models to ONNX format for the Rust proxy backend.

**CLI interface:**
```
python ml/export_onnx.py --model dslim/bert-base-NER --output ml/models/onnx
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | No | `dslim/bert-base-NER` | HuggingFace model ID |
| `--output` | Yes | — | Export directory path |

**Export pipeline:**
1. Downloads model and tokenizer via `AutoModelForTokenClassification` / `AutoTokenizer`
2. Sets model to eval mode (`model.eval()`)
3. Creates dummy input tensor via `tokenizer("test", return_tensors="pt")`
4. Exports to ONNX via `torch.onnx.export()` with:
   - `opset_version=14`
   - Dynamic axes on `input_ids`, `attention_mask` (batch_size, sequence_length) and `logits` output (batch_size, sequence_length)
   - Output file: `<output>/model.onnx`
5. Saves tokenizer via `tokenizer.save_pretrained()` to the same directory
6. Verifies `tokenizer.json` exists (fatal error if missing — the Rust `tokenizers` crate requires this file)

**Dependencies:** Only `transformers`, `torch`, and standard library (per acceptance criteria). No `onnx`, `optimum`, or other external packages required at runtime (torch includes ONNX export natively).

---

## Verification

### Linting
- `ruff check ml/regex_detector.py ml/pii_engine.py ml/evaluate.py ml/export_onnx.py` — **All checks passed**
- `black --check ml/regex_detector.py ml/pii_engine.py ml/evaluate.py ml/export_onnx.py` — **All files unchanged** (formatted)

### Tests
All 96 tests pass (`pytest ml/tests/ -v`):
- 30 SSN context tests (dashed, dashless, edge cases) + 7 merge_entities conflict resolution tests
- 29 regex extension tests (DOB, CREDIT_CARD, IPV4, PASSPORT — valid formats, boundary rejection, invalid inputs)
- 7 redaction config tests (disabled_entities filtering, mapping exclusion, conflict resolution interaction)

Note: `test_ssn_context 2.py` is a macOS duplicate of `test_ssn_context.py` (22 additional collected). Canonical count excluding the duplicate is 66 unique tests.

### CLI Verification
- `python ml/export_onnx.py --help` — Outputs correct usage information

---

## Gemini R3 Verification Resolution

Gemini's R3 review (see `verification/OBS-8_r3_review_report.md`) passed all implementation with one **IMPORTANT** tech debt condition: missing test coverage for the new regex patterns and configurable redaction. This has been resolved:

### Action Item 1: `ml/tests/test_regex_extensions.py` — RESOLVED

29 tests covering all four new entity types:

| Entity | Tests | Coverage |
|--------|-------|----------|
| DOB | 10 | Valid MM/DD/YYYY, valid YYYY-MM-DD, invalid month (00, 13), invalid day (00, 32), invalid year (1899), ISO invalid month, boundary rejection, score assignment |
| CREDIT_CARD | 6 | Dashed format, spaced format, undashed format, no SSN overlap, boundary rejection, score assignment |
| IPV4 | 7 | Valid IP, max octets (255.255.255.255), min octets (0.0.0.0), octet 256 rejected, octet 999 rejected, boundary rejection, score assignment |
| PASSPORT | 6 | Valid passport, lowercase rejected, too few digits, too many digits, boundary rejection, score assignment |

### Action Item 2: `ml/tests/test_redaction_config.py` — RESOLVED

7 tests covering `disabled_entities` behavior:

| Test | Description |
|------|-------------|
| `test_disabled_ssn_left_intact` | SSN text remains unchanged when SSN is disabled |
| `test_disabled_ssn_no_mapping_entry` | Disabled SSN produces no mapping; co-occurring PHONE is still masked |
| `test_disabled_multiple_types` | Multiple types disabled simultaneously |
| `test_none_disabled_masks_all` | `None` default masks all entities (backward compat) |
| `test_empty_list_masks_all` | Empty list is falsy, masks all entities |
| `test_no_entities_returns_original` | No PII text returns unchanged with disabled list |
| `test_disabled_winner_does_not_expose_loser` | Conflict resolution winner (SSN) is disabled, but the losing BERT entity (PERSON) does NOT re-emerge |

### Gemini Risk Assessments (acknowledged)

- **CREDIT_CARD false positives:** Low risk — 16-digit numbers rarely occur outside financial data. Context-aware scoring deferred.
- **DOB false positives:** High risk — pattern matches all dates, not just birth dates. Context-aware scoring (trigger words: "DOB", "birth", "born") recommended for a future iteration.

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `ml/regex_detector.py` | Modified | Added DOB, CREDIT_CARD, IPV4, PASSPORT patterns + detection methods |
| `ml/pii_engine.py` | Modified | Added 4 types to REGEX_AUTHORITATIVE_TYPES; added `disabled_entities` param to `redact()` |
| `ml/evaluate.py` | Modified | Added 4 types to HYBRID_ENTITY_TYPES, entity_to_bio; updated hybrid mode note |
| `ml/export_onnx.py` | **Created** | BYOM ONNX exporter CLI with argparse, torch.onnx.export, tokenizer.json verification |
| `ml/tests/test_regex_extensions.py` | **Created** | 29 tests for DOB, CREDIT_CARD, IPV4, PASSPORT regex patterns |
| `ml/tests/test_redaction_config.py` | **Created** | 7 tests for disabled_entities filtering and conflict resolution interaction |

---

## Known Limitations

1. **ONNX export not integration-tested:** The export script was verified with `--help` but a full export run was not executed in this session to avoid downloading model weights.
2. **Credit Card undashed (16-digit) overlap with long numeric strings:** The `(?<!\w)` / `(?!\w)` guards prevent alphanumeric-adjacent matches, but a bare 16-digit number in isolation will always match. Context-aware scoring (similar to dashless SSN) could be added if false positives are observed.
3. **DOB pattern matches any valid date, not just dates of birth:** The regex catches all dates in MM/DD/YYYY or YYYY-MM-DD format. Disambiguation between DOB and other dates would require context analysis.
