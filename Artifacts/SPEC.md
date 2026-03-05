# Obscura: Project Specification & Cognitive Architecture

## 1. System Prompt, Agent Role, & Behavioral Contract
You are an expert software engineer and AI architect working on **Obscura**, a high-performance, privacy-preserving middleware proxy. Your code must prioritize memory safety, sub-millisecond execution, and strict data privacy. Assume a "Zero-Trust" and "Shift-Left" security posture for all generated code. Do not use emojis in your responses or generated documentation.

**Behavioral Contract (Hard Boundaries):**
- **ALWAYS** fail-closed. If authentication, model loading, or redaction processes fail, block the request entirely. Never forward raw text on error.
- **NEVER** write code that logs, prints, or transmits raw PII/PHI. Telemetry must strictly be zero-knowledge (e.g., counts of entities redacted, not the entity text).
- **NEVER** use real patient or user data for testing; exclusively use synthetic data.
- **NEVER** construct dynamic queries or use `.unwrap()` in production Rust paths; handle `Result` and `Option` safely.

## 2. Project Context & Objectives
* **Mission:** Provide a cloud-native security layer that redacts Personally Identifiable Information (PII) and Protected Health Information (PHI) in real-time for enterprise AI orchestration.
* **Problem:** Enterprises face multi-million dollar liabilities and "Membership Inference Attacks" when sending sensitive data to external Large Language Models (LLMs).
* **Solution:** A forwarding proxy (Method #2 Architecture) deployed as a Kubernetes Sidecar. It intercepts HTTP requests, masks sensitive entities using a dual-engine (BERT NER + Regex) pipeline, forwards the sanitized request to the upstream target LLM, and unmasks the response before returning it to the original client.

## 3. High-Level Architecture & Request Flow
The system acts as a transparent, bidirectional string mutation proxy.

**Step-by-Step Request Pipeline:**
1. **Authentication (Tower Middleware):** Validates the inbound `Authorization: Bearer <OBSCURA_API_KEY>` header using constant-time comparison. Rejects unauthorized requests (`401/403/429`) in <1ms without touching the proxy or ML engine.
2. **Interception & Extraction:** The proxy parses the request body and HTTP headers. Pluggable adapters (e.g., `OpenAIAdapter`) extract the target input text, handling both flat strings and multimodal JSON arrays. It parses `X-Obscura-Skip-Redaction` headers to dynamically inject bypass rules.
3. **Inference (AI Masking):** The text is passed to the ONNX runtime. A hybrid pipeline detects entities:
    - **Regex Engine:** Authoritative exact-matches for structured data (SSN, IPv4, Credit Cards, arbitrary Dates of Birth).
    - **BERT Engine:** Contextual predictions for semantic data (Names, Locations, Organizations).
    - **Conflict Resolution:** Regex always overrides overlapping BERT spans.
4. **Dictionary Mapping:** Detected entities are swapped for generic tokens (e.g., `"John Doe"` -> `"[PERSON_1]"`). The mapping is recorded in a highly strict, bidirectional serialization state dictionary.
5. **Upstream Forwarding:** The sanitized text is repackaged into the upstream provider’s native schema and forwarded using an asynchronous client.
6. **Response Unmasking:** Upon receiving the upstream LLM text (or streaming deltas), the proxy uses the saved mapping dictionary to swap the generic tokens back to their original states and returns the HTTP response to the client.

## 4. Codebase Mapping & Module Boundaries
Obscura operates as a monorepo with distinct language boundaries:

### Backend Proxy (`/src`, `/inference`, `/config`, `/proxy`)
* **Language:** Rust (Stable 1.75+).
* **`proxy` crate:** Core Hyper HTTP server handling `tokio` (async network I/O). Manages the routing, API adapters (in `src/providers`), and Tower authentication middleware.
* **`inference` crate:** The bridging layer. Wraps `ort` (ONNX Runtime) and HuggingFace `tokenizers` to execute the AI model directly in Rust memory without Python. Handles the `MappingDictionary` state.
* **`config` crate:** Parses `obscura.toml` and environment variables.
* **Target:** Total proxy network overhead must be <60ms.

### AI/ML Engine (`/ml`)
* **Language:** Python 3.11+.
* **Core Libraries:** `transformers`, `torch`, `seqeval`.
* **Purpose:** Strictly used offline for dataset generation (`generate_synthetic_data.py`), model training (`fine_tune.py`), and evaluation (`evaluate.py`).
* **Deployment Bridging:** The CLI script (`export_onnx.py`) exports the PyTorch BERT model to an `.onnx` weight file with `opset_version=14` and dynamic sequence axes. It packages `tokenizer.json` to hand off to the Rust backend.
* **Target:** ML inference inside Rust must execute in <30ms.

## 5. Technology Stack & Dependencies
- **Core Infrastructure:** AWS EC2 Kubernetes Cluster (Sidecar deployment via `localhost:8080`).
- **Observability:** Prometheus metrics (`/metrics` endpoint via `prometheus` and `once_cell` crates). Grafana Cloud telemetry.
- **Rust Stack:** `tokio` (runtime), `hyper` v1.0, `reqwest` (upstream client), `serde_json`, `ort` (ML execution).
- **Security:** Keys injected via K8s Secrets/AWS KMS. Never hardcoded. Configured via `OBSCURA_API_KEY` and `UPSTREAM_LLM_KEY`.

## 6. Environment & Developer Setup
To contribute, agents should standardize to the following toolchains:

### Python (AI/ML)
* **Setup:** `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
* **Testing:** Run validation suites via `pytest ml/tests/` (verifies regex bounds, SSN contexts, and `disabled_entities` filtering).
* **Format/Lint:** `ruff check` and `black`.

### Rust (Backend)
* **Setup:** Standard Cargo workspace.
* **Build/Run:** `cargo build --release` / `cargo run`
* **Testing:** `cargo test` (ensures adapter multimodal extraction and metrics counters function correctly).
* **Format/Lint:** Must pass `cargo clippy -- -D warnings` and `cargo fmt -- --check`.

## 7. Repository Management Guidelines
* **Commit Messages:** Use Conventional Commits with Jira prefixes. Format: `type(scope): OBS-XXX description` (e.g., `feat(ML): OBS-7 export BERT to ONNX`).
* **Restricted Action Items:** 
    * Do not commit `.bin`, `.pt`, `.safetensors`, or `.onnx` model weights to Git (use Git LFS or external bucket storage).
    * Do not modify GitHub Actions workflows (`.github/workflows/`) without explicit instruction.
    * Maintain `SPEC.md` dynamically. Any new architectural module must be documented here heavily enough to inform continuous AI development context.