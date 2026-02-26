# OBS-3 SSN Context-Aware Detection — Verification Review

This document contains the senior ML engineering review of the hybrid BERT + regex NER pipeline implemented for PII/PHI redaction, addressing the key validation points.

## 1. Correctness (IRS SSN Validation)
**Implementation:** `_is_valid_ssn_parts()` in `regex_detector.py` correctly checks for invalid regions.
- `area == 0` correctly rejects 000.
- `area == 666` correctly rejects 666.
- `area >= 900` correctly rejects 900-999.
- `group == 0` rejects 00.
- `serial == 0` rejects 0000.
**Verdict:** The IRS validation rules from Publication 4557 are implemented accurately.

## 2. Context Scoring Logic
**Implementation:** Dashless SSNs start at 0.40, trigger words add +0.35, phrases +0.20, negative words -0.35, threshold 0.70.
**Analysis:** 
- **Missed Valid SSN (False Negatives):** Yes. The asymmetric weights (-0.35 for negative vs +0.35 for trigger) mean that if a text contains *both* a trigger word and a negative word (e.g., "The SSN matches tracking number 123456789"), the score will drop to `0.40(base) + 0.35(trigger) - 0.35(negative) = 0.40`, failing the 0.70 threshold. 
- **False Positives:** Also possible. If a non-SSN 9-digit number matches a phrase like "tax ID" (e.g., "Tax ID background check returned application 123456789"), it might score 0.95 and be erroneously flagged.
**Verdict:** Reasonable heuristics for V1, but high false-negative risk when sentences conflate identifiers (common in noisy enterprise logs).

## 3. Conflict Resolution
**Implementation:** `merge_entities()` in `pii_engine.py` greedily iterates and merges entities based on overlap rules.
**Analysis:** The greedy conflict resolution has a **major flaw in Rule 2 & 3 (Nested/Partial Overlap)**. The algorithm evaluates `candidate` against `merged` items and `break`s after resolving the first conflict:
```python
        for i, accepted in enumerate(merged):
            ...
            if winner is candidate:
                merged[i] = candidate
            break
```
**Edge Case:** If `candidate` is a long span that overlaps with *two distinct* `accepted` entities, it will replace only the *first* one and then break. The second overlapped `accepted` entity will remain in the resultant list, leading to an output with overlapping entities. 
**Verdict:** Buggy for overlapping entities. It should check overlap with all existing merged entities and remove all overridden ones, rather than simply replacing one and breaking.

## 4. Test Coverage
**Implementation:** `test_ssn_context.py` has 10 gold standard strings and checks boundaries and overlap.
**Analysis:** Missing critical edge cases:
- Multiple SSNs in a single string.
- SSNs embedded flush with alphabetic characters (e.g., `A123-45-6789B`). The regex `(?<!\d)` only checks for adjacent digits, meaning letters adjacent to digits would parse as valid SSNs.
- Conflict tests involving three or more overlapping entities (which would trigger the bug mentioned in Section 3).
**Verdict:** Insufficient coverage for text boundary edge-cases and multi-overlap conflict resolution.

## 5. Import Structure
**Implementation:** The code uses `try...except` ModuleNotFoundError blocks.
**Analysis:** This is a poor practice and an anti-pattern in Python. It is brittle and confuses linters/type-checkers.
**Verdict:** Should be removed. We should transition to a proper setup tool configuration (e.g., `pip install -e .` with `pyproject.toml`) and use consistent absolute or relative imports.

## 6. Security
**Implementation:** `DetectedEntity` dataclass stores the raw PII strings (SSNs) in the `text` attribute. `RedactionResult` directly contains both `entities` and a `mapping` dictionary which maps tokens to the raw text.
**Analysis:** Storing raw SSNs in plain dataclasses introduces significant compliance risk. If these dataclasses are carelessly logged, dumped to JSON for debugging, or cached natively by application wrappers, the plaintext SSN is exposed.
**Verdict:** For HIPAA/GDPR compliance, `DetectedEntity` should ideally not store raw values if they are being persisted, or at a minimum, it should override `__repr__` and `__str__` to obfuscate the `text` field and prevent accidental logging of the exact text payload.
