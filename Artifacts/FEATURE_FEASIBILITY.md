# Architectural Analysis: PII/PHI Redaction Features

This document evaluates the technical feasibility and architectural implications of four proposed features for the Obscura PII/PHI redaction proxy.

## 1. PII/PHI Categorization
**Feasibility:** High
**Rationale:** 
Identifying a comprehensive list of PII/PHI categories is well-supported by established regulatory frameworks (e.g., the 18 HIPAA Safe Harbor identifiers, GDPR personal data categories, PCI-DSS for financial data). From a technical standpoint, expanding coverage is highly feasible given Obscura's hybrid architecture:
* **Structured Data:** Identifiers like Credit Card Numbers, IP Addresses, VINs, and Dates of Birth can be fully supported by extending the deterministic regex engine (`regex_detector.py`) with new pre-compiled patterns and word-boundary conditions.
* **Semantic Data:** Entities requiring context (e.g., specific medical conditions, diverse physical addresses) can be integrated by expanding the NER taxonomy and augmenting the training templates in `generate_synthetic_data.py`. 
* **Architectural Impact:** Minimal. It solely requires configuration additions and model/regex expansion without altering the Rust proxy or ONNX execution path.

## 2. Feedback Loop Integration
**Feasibility:** Medium (Requires strict privacy guardrails)
**Rationale:** 
A continuous improvement feedback loop typically requires collecting false positives and false negatives to retrain models. However, Obscura operates under a strict "Zero-Knowledge Logging" posture where no raw PII/PHI can be transmitted or persisted. 
* **Implementation Strategy:** To maintain compliance, the feedback mechanism cannot automatically capture raw request payloads. Instead, it must rely on local, on-premise telemetry where an enterprise administrator can review edge-cases within their secure VPC. 
* **Improvement Cycle:** The system can support an opt-in or synthetic feedback loop where administrators use an internal tooling framework to generate *synthetic analogs* of the failed redactions (leveraging the existing `generate_synthetic_data.py` paradigm) to retrain the BERT model offline. This ensures no actual patient data enters the CI/CD training pipeline while still capturing the structural nuances of the missed entity.

## 3. Configurable Redaction Fields
**Feasibility:** High
**Rationale:** 
Enterprises often have varying compliance requirements depending on the downstream LLM or the specific use case (e.g., masking SSNs but leaving Person Names intact for generation personalization). 
* **Implementation Strategy:** The masking module currently maps detected entities to a generic token (e.g., `[PERSON_1]`). Implementing toggles is a straightforward filtering operation that can be applied immediately after the `PIIEngine` completes detection. 
* **Architectural Impact:** The Rust proxy can accept either global granular toggles via the `obscura.toml` configuration file, or dynamic per-request overrides via custom HTTP headers (e.g., `X-Obscura-Mask-Config: "SSN,EMAIL"`). The proxy simply filters the detected entities against this active configuration before applying the string replacement, imposing virtually zero latency overhead.

## 4. Model Interchangeability (Bring-Your-Own-Model)
**Feasibility:** High
**Rationale:** 
"Bring-your-own-model" (BYOM) capabilities are highly feasible due to Obscura's cleanly decoupled architecture and adherence to the ONNX standard. 
* **Implementation Strategy:** Python is strictly used for offline training and export, while the Rust backend uses the `ort` crate to execute the resulting `.onnx` graph independently. As long as an enterprise's custom model adheres to the expected input/output tensor shapes (token IDs and attention masks in; sequential logits out) and provides a compatible `tokenizer.json`, the Rust proxy can hot-swap it.
* **Architectural Impact:** This adds massive value for enterprise adoption with minimal architectural friction. It allows healthcare or financial institutions to replace the default baseline model with their own highly specialized, proprietary NER models simply by updating the `NER_MODEL_PATH` environment variable and providing the `.onnx` weight file.
