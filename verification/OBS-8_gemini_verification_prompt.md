# Gemini Verification Prompt for OBS-8 (R4)

Copy the prompt below into Gemini, along with the file contents listed in the "Files to Provide" section.

This is the **fourth round** of verification. R3 implemented three new subtasks (PII/PHI expansion, configurable redaction, ONNX exporter). Gemini's R3 review passed the implementation but flagged missing test coverage as IMPORTANT tech debt. This round verifies the new test files resolve that tech debt.

---

## Files to Provide

Attach or paste the contents of these files when running this prompt:

1. `ml/regex_detector.py`
2. `ml/pii_engine.py`
3. `ml/evaluate.py`
4. `ml/export_onnx.py`
5. `ml/tests/test_regex_extensions.py`
6. `ml/tests/test_redaction_config.py`
7. `verification/OBS-8_verification_report.md`
8. `verification/OBS-8_r3_review_report.md`

---

## Prompt

```
You are performing a FOURTH ROUND review of OBS-8 for Project Obscura, a zero-trust security proxy that redacts PII/PHI using a hybrid BERT + regex NER pipeline.

The R3 review (see OBS-8_r3_review_report.md) passed the implementation of three new subtasks but flagged one IMPORTANT tech debt condition: missing test coverage for the new regex patterns and the disabled_entities configurable redaction feature.

Two new test files have been created to resolve this:
1. `ml/tests/test_regex_extensions.py` — 29 tests for DOB, CREDIT_CARD, IPV4, PASSPORT
2. `ml/tests/test_redaction_config.py` — 7 tests for disabled_entities filtering

Full test suite: 96/96 pass (includes 30 original SSN/merge tests + 29 regex extension + 7 redaction config + 30 from a macOS duplicate file).

Please verify:

### 1. Test Coverage — Regex Extensions (test_regex_extensions.py)

For each entity type (DOB, CREDIT_CARD, IPV4, PASSPORT), verify:
- At least one test for valid format detection (positive case)
- At least one test for invalid/boundary input rejection (negative case)
- Word-boundary guard testing (alphanumeric-adjacent rejection)
- Score and source assignment verification

Specific checks requested by R3:
- IPv4: Confirm `256.1.1.1` is rejected (octet > 255)
- Passport: Confirm `a12345678` is rejected (lowercase)
- DOB: Confirm `13/40/1990` is rejected (invalid month/day)
- DOB: Confirm both MM/DD/YYYY and YYYY-MM-DD formats are tested

### 2. Test Coverage — Redaction Config (test_redaction_config.py)

Verify the three scenarios specified in R3:
- Disabled entity (SSN) leaves the original text intact
- Disabled entities do not populate the substitution mapping dict
- Conflict resolution is preserved when the winning entity is disabled (suppressed entity does NOT re-emerge)

Additional checks:
- Backward compatibility: `None` and empty list both result in all entities masked
- Multiple disabled types work simultaneously
- Mock strategy: confirm tests use a mocked BERT pipeline (no model downloads in CI)

### 3. Test Quality

- Are the tests independent and deterministic?
- Do they follow the project's existing test conventions (pytest fixtures, class grouping)?
- Are edge cases adequately covered?
- Any missing scenarios that should be added?

### 4. Regression Check

- Confirm all 96 tests pass (per the verification report)
- Confirm no changes were made to the implementation files (regex_detector.py, pii_engine.py, evaluate.py, export_onnx.py) — only test files were added

Please provide:
1. Confirmation that both R3 action items are resolved
2. Any gaps in test coverage, categorized as CRITICAL, IMPORTANT, or MINOR
3. An overall PASS/FAIL verdict on the tech debt resolution
4. Any remaining recommendations
```
