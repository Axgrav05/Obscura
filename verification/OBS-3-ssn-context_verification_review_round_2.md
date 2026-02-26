# OBS-3 SSN Context-Aware Detection — Verification Review (Round 2)

This document contains the Round 2 senior ML engineering review of the hybrid BERT + regex NER pipeline implemented for PII/PHI redaction.

## 1. IRS SSN Validation
**Fix Verify:** No changes were required; logic remains structurally correct.

## 2. Context Scoring Logic
**Fix Verify:** The intentional tradeoff (rejecting mixed-context inputs) is properly documented. 
**Remaining Issues:** As requested, additional problematic mixed contexts exist. Any sentence with both a negative and a positive phrase but overall clear intent will fail. Example: "Do not fax this Social Security number 123456789" (`fax` is -0.35, `social security` is +0.35 + 0.20, translating to a score of 0.60 < 0.70). This is acceptable as a V1 tradeoff but emphasizes the need for a V2 ML-based context classifier.

## 3. Conflict Resolution Multi-Overlap Bug
**Fix Verify:** FIXED. The new logic collects all intersecting entities into `conflict_indices`, runs resolutions against each, and only replaces them *if* the `candidate` span wins all conflicts. The use of `for i in reversed(conflict_indices): merged.pop(i)` correctly maintains internal list indexing integrity when deleting multiple elements.
**Remaining Issues:** None. The object identity check (`winner is not candidate`) works securely because the helper functions return the original objects.

## 4. Test Coverage
**Fix Verify:** FIXED. Six new robust edge-cases were provided.
**Remaining Issues:** 
- The tests don't explicitly test the boundaries of string literals (e.g., `detector.detect("123456789")` purely on its own without context, though this should be rejected).
- Since `\w` is `[a-zA-Z0-9_]`, `1234567890` is correctly rejected, but `SSN_123-45-6789` will also be rejected because `_` is considered a word-character.

## 5. Import Structure
**Fix Verify:** Addressed via an editable install (`pip install -e .`) and `from ml.*` statements.
**Remaining Issues:** 
- **Script Invocation Failure:** Running `python ml/evaluate.py` directly as a standalone script from the `Obscura/` root directory will likely throw a `ModuleNotFoundError`. This occurs because Python prepends the script's directory (`Obscura/ml/`) to `sys.path`. When `evaluate.py` executes `from ml.pii_engine import PIIEngine`, it strictly looks for a directory sequence of `Obscura/ml/ml/pii_engine.py`, which does not exist. Users must invoke it as a module: `python -m ml.evaluate`.
- **Namespace Collision:** Using a top-level package name of simply `ml` is risky if this environment is shared with any other general machine learning packages or folders, but it is acceptable in a dedicated internal virtualenv.

## 6. Security / HIPAA
**Fix Verify:** FIXED (Partially). Overriding `__repr__` and `__str__` prevents accidental leakage via standard `print()` or logging statements.
**Remaining Issues:**
- **Dataclass Serialization Leak:** `dataclasses.asdict(entity)` bypasses `__repr__` and directly extracts the raw structure. If the application forwards this dataclass to a JSON serializer via `json.dumps(dataclasses.asdict(...))`, the unmasked `text` will leak into the output payload. For true HIPAA compliance, either provide a custom `asdict()` serialization method or avoid retaining the raw text unless technically mandatory.

---
**Verdict:** The codebase has fundamentally improved with these bug fixes and security guards in place. The outstanding issues—chiefly the module invocation path, `_` boundary rules, and dataclass serialization vulnerability—should be fixed before full production merge.
