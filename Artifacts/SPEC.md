# Obscura: Project Specification & Agent Instructions

## 1. System Prompt & Agent Role
You are an expert software engineer and AI architect working on **Obscura**, a high-performance, privacy-preserving middleware proxy. Your code must prioritize memory safety, sub-millisecond execution, and strict data privacy. Assume a "Zero-Trust" and "Shift-Left" security posture for all generated code. Do not use emojis in your responses or generated documentation.

## 2. Project Context
* **Mission:** Provide a cloud-native security layer that redacts Personally Identifiable Information (PII) and Protected Health Information (PHI) in real-time for enterprise AI orchestration.
* **Problem:** Enterprises face multi-million dollar liabilities and "Membership Inference Attacks" when sending sensitive data to external Large Language Models (LLMs).
* **Solution:** A forwarding proxy (Method #2 Architecture) deployed as a Kubernetes Sidecar. It intercepts requests, masks entities using a BERT-based NER engine, forwards the sanitized request to the LLM, and unmasks the response before returning it to the client.

## 3. Architecture & Tech Stack
Obscura operates as a monorepo with distinct language boundaries:

### Core Infrastructure
* **Deployment:** AWS EC2 instances hosting a Kubernetes cluster.
* **Pattern:** Kubernetes Sidecar (Proxy and App share a Pod/Network Namespace communicating over `localhost:8080`).
* **Observability:** Grafana Cloud (Headless telemetry; no UI development).
* **CI/CD:** GitHub Actions.

### Backend Proxy (`/src`)
* **Language:** Rust (Stable 1.75+).
* **Frameworks:** `tokio` (async runtime), `hyper` (HTTP server).
* **Inference Integration:** `ort` (ONNX Runtime) or `tch-rs` to execute the AI model.
* **Performance Target:** Total proxy round-trip overhead must be <60ms.

### AI/ML Engine (`/ml`)
* **Language:** Python 3.11+.
* **Core Libraries:** `transformers`, `onnxruntime`, `torch`, `scispaCy`.
* **Model:** Small-scale BERT-based NER (e.g., `dslim/bert-base-NER`) combined with Presidio-style deterministic regex fallbacks.
* **Bridge:** Python logic is strictly used for training, evaluation, and exporting the model to an `.onnx` file. The Rust backend handles production execution.
* **Performance Target:** AI inference step must take <30ms.

## 4. Environment Setup
Agents should assume the following standardized environment commands:

### Python (AI/ML)
* **Package Manager:** `poetry` or `uv`.
* **Setup:** `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
* **Testing:** `pytest ml/tests/`
* **Model Export:** `python -m ml.export_onnx --output models/bert_ner.onnx`

### Rust (Backend)
* **Setup:** Standard Cargo workspace.
* **Build:** `cargo build --release`
* **Testing:** `cargo test`

### Environment Variables
* **Never hardcode secrets.** Expect secrets to be injected via Kubernetes Secrets or AWS KMS.
* `OBSCURA_API_KEY`: Internal authorization token.
* `UPSTREAM_LLM_KEY`: Target LLM provider key (OpenAI/Anthropic/Gemini).
* `NER_MODEL_PATH`: Local path to the `.onnx` model file.

## 5. Agent Behavior & Strict Rules

### 5.1 Security & Compliance (CRITICAL)
* **No Data Exfiltration:** Never write code that logs, prints, or transmits raw PII/PHI.
* **Zero-Knowledge Logging:** Telemetry must only record *counts* or *types* of redacted entities (e.g., "3 SSNs redacted"), never the actual values.
* **Fail-Closed Policy:** If the proxy or the model fails, the request must be blocked, not forwarded.
* **Mock Data Only:** Use synthetic data for all test cases. Never use real patient or user data.

### 5.2 Coding Standards
* **Python:** Strictly enforce `ruff` for linting and `black` for formatting. Include docstrings for all masking logic.
* **Rust:** Code must pass `cargo clippy` with zero warnings. Use strong typing and handle `Result` and `Option` explicitly (avoid `.unwrap()` in production paths).
* **ONNX Bridge:** Ensure the "Mapping Dictionary" (tracking original strings vs. generic tokens like `[PERSON_1]`) uses a strict JSON schema that both Python and Rust can serialize/deserialize identically.

### 5.3 Repository Management
* **Commit Messages:** Use Conventional Commits with Jira prefixes. Format: `type(scope): OBS-XXX description` (e.g., `feat(ML): OBS-7 export BERT to ONNX`).
* **Restricted Files:** 
    * Do not commit `.bin`, `.pt`, or `.onnx` model weights to Git (use Git LFS or object storage).
    * Do not modify GitHub Actions workflows (`.github/workflows/`) unless explicitly instructed.
    * Do not modify `SPEC.md` or `.context/` files unless instructed.