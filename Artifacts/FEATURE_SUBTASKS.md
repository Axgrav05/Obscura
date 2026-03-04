# PII/PHI Redaction Feature Subtasks & Research

Based on the architectural feasibility analysis, here is the breakdown of the research into specific subtasks that can be added to the current Jira ticket.

## 1. PII/PHI Categorization Expansion
**Research:**
By cross-analyzing Obscura's current supported types (PER, ORG, LOC, MISC, SSN, PHONE, EMAIL, MRN) against regulatory frameworks (HIPAA's 18 identifiers, GDPR, PCI-DSS), several critical gaps emerge. For instance, dates (e.g., DOBs), health plan beneficiary numbers, financial accounts (CCN/IBAN), and device identifiers (IP/MAC addresses) are missing.

**Subtasks:**
* [ ] **Regex: Dates & DOB** - Implement pattern for catching generic dates and dates of birth.
* [ ] **Regex: Financial Data** - Implement patterns for Credit Card Numbers (PCI-DSS compliant) and IBANs.
* [ ] **Regex: Network Identifiers** - Implement patterns for IP Addresses (IPv4/v6) and MAC Addresses.
* [ ] **Regex: Government IDs** - Implement patterns for Passport Numbers and US Driver's License formats.
* [ ] **Regex: Healthcare/Account IDs** - Implement patterns for Health Plan Beneficiary Numbers and standard alphanumeric Account IDs.
* [ ] **Evaluation Integration** - Update the evaluation harness (`evaluate.py`) and synthetic data generator (`generate_synthetic_data.py`) to encompass all newly supported regex entity types for benchmarking.

## 2. Feedback Loop Integration
**Research & Validation:**
The assertion is correct: a continuous feedback loop is highly complex and caters to a niche audience (enterprises with dedicated ML teams operating in isolated VPCs). It requires the client to build proprietary internal tooling to manually review logs (which inherently risks violating Zero-Knowledge principles if not perfectly sanitized), label false positives/negatives, and run custom retraining pipelines. For the core open-source proxy, the ROI of building this infrastructure is inherently low compared to simply shipping better baseline models.
**Subtasks:**
* [ ] **Documentation Update** - Record the decision to deprioritize real-time feedback loop integration due to the high operational toll and strict privacy constraints. Shift resources toward expanding baseline categorization (Categorization Expansion) and BYOM (Task 4) support.

## 3. Configurable Redaction Fields
**Subtasks:**
* [ ] **Configuration Schema** - Define JSON/TOML schema for `obscura.toml` to accept a globally disabled redaction entities list (e.g., `disabled_entities = ["PERSON", "LOCATION"]`).
* [ ] **Dynamic Overrides** - Add an HTTP Header interceptor in the Rust proxy to parse `X-Obscura-Skip-Redaction` for per-request masking overrides.
* [ ] **Engine Filtering** - Implement the filtering logic inside the Rust engine to conditionally skip string replacement for disabled entity types before forwarding the payload to the downstream LLM.

## 4. Model Interchangeability (Bring-Your-Own-Model)
**Research:**
An `.onnx` weight file alone is insufficient for text generation or NER via the `tokenizers` library; the engine *must* have the corresponding `tokenizer.json` to map raw text strings into the integer token IDs matching the model's vocabulary. If a client only has a standard HuggingFace PyTorch model, they would need an automated way to export this bundle. The most straightforward approach is to provide a Python CLI tool inside the `/ml` directory that automatically downloads the tokenizer, converts the `.pt` or `.bin` model to ONNX, and packages them together in a mounting directory.

**Subtasks:**
* [ ] **BYOM Export CLI** - Create a user-facing Python script (e.g., `python -m ml.export_onnx <model_id>`) that accepts a HuggingFace model ID or a local PyTorch path.
* [ ] **Bundle Generation** - Inside the script, automate the ONNX conversion process and serialize both the `.onnx` weight file and the associated `tokenizer.json` into a unified export bundle directory.
* [ ] **Backend Dynamic Load** - Update the Rust backend initialization block to dynamically load the model and tokenizer from the volume mounts dictated by the `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` environment variables.
* [ ] **BYOM Documentation** - Document the Bring-Your-Own-Model workflow and mount procedures in the project README for end-users.
