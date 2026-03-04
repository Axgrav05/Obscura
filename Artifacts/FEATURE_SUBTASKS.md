# PII/PHI Redaction Feature Subtasks

Based on the architectural feasibility analysis, here are the refined subtasks that should be added to the current Jira ticket. **Note:** The proposal to implement a real-time feedback loop has been fully rejected due to the strict "Zero-Knowledge" requirements and the excessive operational overhead of creating a sanitized local review system for clients.

## 1. PII/PHI Categorization Expansion
**Description:** Implement deterministic regex fallbacks for new PII/PHI categories missing from the current pipeline (Dates, Financial accounts, Network Identifiers, IDs). Ensure the evaluation scripts can correctly benchmark them.

**Acceptance Criteria:**
* Add regex patterns with strict word boundaries to `ml/regex_detector.py` for Dates/DOB, Credit Card Numbers, IP/MAC Addresses, and US Government IDs (e.g. Passports).
* The new entities must be added to the conflict resolution mapping in `ml/pii_engine.py`.
* `ml/evaluate.py` must be updated to track benchmark metrics for these new categories.

## 2. Configurable Redaction Fields
**Description:** Enable the Rust proxy or clients to specify which detected entity types should be skipped during the redaction phase, allowing context-dependent masking.

**Acceptance Criteria:**
* Update `PIIEngine.redact()` or `PIIEngine.detect()` in `ml/pii_engine.py` to accept an optional list of `disabled_entities`.
* When an entity type is in the `disabled_entities` list, it should NOT be replaced by a token (e.g., `[PERSON_1]`) and must remain in its original text form.
* Write a unit test validating that passing `disabled_entities=["PERSON"]` leaves person names unmasked while still redacting SSNs.

## 3. Model Interchangeability (Bring-Your-Own-Model Export CLI)
**Description:** Create an automated script to generate an evaluation-ready "BYOM Bundle". The tokenizers crate in the backend requires a `tokenizer.json` to process strings in addition to the `.onnx` weight file. Ensure clients can seamlessly port their HuggingFace models.

**Acceptance Criteria:**
* Create a Python CLI script (e.g., `ml/export_onnx.py`) that accepts a HuggingFace Model ID.
* The script must download the model and execute the PyTorch-to-ONNX conversion using `onnxruntime` or `transformers.onnx`.
* The script must save both the converted `model.onnx` file and the model's `tokenizer.json` into a single, specified output directory.
* Add a `README` usage command demonstrating how to run the script.
