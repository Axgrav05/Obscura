# Prompt for Claude: Obscura PII/PHI Feature Implementation (OBS-8 Follow-up)

**Context:**
Obscura is a privacy-preserving middleware proxy deployed as a Kubernetes sidecar. It intercepts HTTP requests to upstream LLMs, redacts PII/PHI using a hybrid BERT + Regex NER pipeline, and then restores the sensitive data upon the LLM's response. The ML components (Python) are packaged and exported for the Rust proxy backend to consume.

We have identified three new major subtask requirements to expand our PII/PHI coverage and add enterprise functionality. Please implement the following features by directly modifying the appropriate Python files in the `/ml` directory.

## Task 1: PII/PHI Categorization Expansion
**Goal:** Expand the deterministic fallback engine to catch missing PII/PHI identifiers.
**Instructions:**
1. Open `ml/regex_detector.py` and implement highly confident, standard regex patterns for:
   * **Dates / DOB:** (e.g., MM/DD/YYYY, YYYY-MM-DD). Use strict boundaries.
   * **Financial Data:** Credit Card Numbers (standard 16-digit dashed or un-dashed formatting).
   * **Network Identifiers:** IPv4 Addresses.
   * **Government IDs:** Standard 9-alphanumeric US Passport formats.
2. Incorporate these into the `detect()` method and assign appropriate detection scores.
3. Open `ml/pii_engine.py` and add these new entity types to the `REGEX_AUTHORITATIVE_TYPES` constant so they win exact-match conflicts.
4. Open `ml/evaluate.py` and update the `HYBRID_ENTITY_TYPES` to include these new entities so they are properly benchmarked.

## Task 2: Configurable Redaction Fields
**Goal:** Allow developers to selectively ignore specific entities from being masked.
**Instructions:**
1. Open `ml/pii_engine.py`.
2. Modify the `redact(self, text: str, disabled_entities: list[str] = None)` method footprint to accept an optional list of string entity types (e.g., `["PERSON", "LOCATION"]`).
3. Ensure that any entity matching a type in the `disabled_entities` list is completely ignored during string masking and does NOT yield a mapping token. The original text should remain intact.
4. Ensure the session mapping dictionary successfully skips registering keys for disabled entities.

## Task 3: BYOM (Bring-Your-Own-Model) Exporter CLI
**Goal:** Build a script that converts PyTorch HuggingFace NER models into the compatible bundle our Rust proxy requires.
**Instructions:**
1. Create a new file: `ml/export_onnx.py`.
2. Write a Python CLI using `argparse` with two arguments: `--model` (HuggingFace model ID, default: `dslim/bert-base-NER`) and `--output` (Export directory path).
3. The script must safely download the model and convert it from PyTorch to ONNX format.
4. The script MUST extract and save the `tokenizer.json` file in the same output directory as the ONNX weight file (the Rust backend `tokenizers` crate will crash without this).
5. Add a simple docstring at the top demonstrating the terminal command to invoke this script.

**Acceptance Verification:**
* Do NOT run any complex fine-tuning operations.
* Ensure all code strictly passes `ruff` and `black`.
* Do not utilize external dependencies outside of `transformers`, `torch`, `onnx`, and standard Python libraries.
