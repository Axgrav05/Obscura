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
2. **Interception & Extraction:** The proxy parses the request body and HTTP headers. Pluggable adapters (e.g., `OpenAIAdapter`) extract the target input text, handling both flat strings and multimodal JSON content arrays (see §14). It parses the `X-Obscura-Skip-Redaction` header — a comma-separated list of entity type strings (e.g., `"PERSON,EMAIL"`) — to dynamically extend the disabled entity list beyond the `obscura.toml` defaults.
3. **Inference (AI Masking):** The text is passed to the NER pipeline. For texts exceeding 1500 characters, chunking occurs first (see §12). A hybrid pipeline detects 13 entity types across two engines:
    - **BERT Engine** (`dslim/bert-base-NER`): Contextual predictions for semantic data — `PERSON`, `ORGANIZATION`, `LOCATION`, `MISC`.
    - **Regex Engine:** Deterministic exact-match detection for structured data — `SSN`, `PHONE`, `EMAIL`, `MRN`, `DOB`, `CREDIT_CARD`, `IPV4`, `IPV6`, `PASSPORT`.
    - **Conflict Resolution:** Four-rule merge algorithm (see §9). Regex wins exact-span overlaps for all 9 structured types. Both engines' results are merged into a single deduplicated list.
4. **Dictionary Mapping:** Detected entities are swapped for indexed tokens using the format `[ENTITY_TYPE_N]` (e.g., `"John Doe"` → `"[PERSON_1]"`). The counter resets per-type each call and is assigned in descending position order to preserve character offsets during replacement. The mapping is recorded in a bidirectional `MappingDictionary` for downstream restoration.
5. **Upstream Forwarding:** The sanitized text is repackaged into the upstream provider's native schema and forwarded using an asynchronous `reqwest` client. All original headers except `Host` are forwarded.
6. **Response Unmasking:** Upon receiving the upstream LLM response (full body or streaming deltas), the proxy uses the saved `MappingDictionary` to swap generic tokens back to their original values. Two streaming delta formats are supported: the OpenAI Responses API (`{"type":"response.output_text.delta","delta":"..."}`) and Chat Completions (`{"choices":[{"delta":{"content":"..."}}]}`).

## 4. Codebase Mapping & Module Boundaries
Obscura operates as a monorepo with distinct language boundaries:

### Backend Proxy (`/src`, `/inference`, `/config`, `/proxy`)
* **Language:** Rust (Stable 1.75+).
* **`proxy` crate** (`proxy/src/main.rs`): Core Hyper v1.0 HTTP server on `tokio`. Implements the `/health` endpoint, config loading, `X-Obscura-Skip-Redaction` header parsing, and pass-through forwarding to `upstream_url`. Loads `ModelEnvironment` at startup (fail-closed).
* **`inference` crate** (`inference/src/lib.rs`, `inference/src/mapping.rs`): Bridging layer for AI execution. `ModelEnvironment` reads `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` from environment at startup. `MappingDictionary` is a `HashMap<String, String>` keyed by token (e.g., `"[PERSON_1]"`) with original PII text as value.
* **`config` crate** (`config/src/lib.rs`): Parses `obscura.toml` via `serde`/`toml`. Exposes `Config { upstream_url: String, disabled_entities: Vec<String> }`.
* **`src/` (legacy metrics server)** (`src/main.rs`): Standalone Hyper server serving `GET /metrics` (Prometheus text format). Contains `src/providers/` with the `ProviderAdapter` trait and `OpenAIAdapter`.
* **Target:** Total proxy network overhead must be <60ms.

### AI/ML Engine (`/ml`)
* **Language:** Python 3.11+.
* **Core Libraries:** `transformers`, `torch`, `optimum`, `onnxruntime`, `seqeval`, `scispacy`, `datasets`, `pandas`, `numpy`.
* **Dev Libraries:** `faker`, `ruff`, `black`, `pytest`.
* **Base Model:** `dslim/bert-base-NER` (HuggingFace Hub). Fine-tuned on IOB2-tagged NER data.
* **Purpose:** Strictly used offline for dataset generation, model training (`fine_tune.py`), and evaluation (`evaluate.py`).
* **Deployment Bridging:** `export_onnx.py` exports the PyTorch BERT model to an `.onnx` weight file with `opset_version=14` and dynamic sequence axes. It packages `tokenizer.json` for the Rust `tokenizers` crate.
* **Target:** ML inference inside Rust must execute in <30ms.

## 5. Technology Stack & Dependencies
- **Core Infrastructure:** AWS EC2 Kubernetes Cluster (Sidecar deployment via `0.0.0.0:8080`).
- **Observability:** Prometheus metrics (`/metrics` endpoint via `prometheus` and `once_cell` crates). Grafana Cloud telemetry.
- **Rust Stack:** `tokio` (async runtime), `hyper` v1.0 + `hyper-util`, `reqwest 0.13` with `stream` + `json` features, `serde` + `serde_json`, `http-body-util`, `bytes`, `anyhow`, `tracing` + `tracing-subscriber`, `toml`, `ort` (ONNX Runtime execution), HuggingFace `tokenizers` (tokenizer.json parsing), `prometheus`, `once_cell`.
- **Python ML Stack:** `transformers`, `torch`, `optimum`, `onnxruntime`, `seqeval`, `scispacy`, `datasets`, `pandas`, `numpy`, `faker` (dev), `ruff` (lint), `black` (format), `pytest` (tests).
- **Security:** Keys injected via K8s Secrets/AWS KMS. Never hardcoded. Configured via `OBSCURA_API_KEY` and `UPSTREAM_LLM_KEY` (proxy auth/forwarding); `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` (model loading).

## 6. Environment & Developer Setup
To contribute, agents should standardize to the following toolchains:

### Python (AI/ML)
* **Setup:** `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
* **Testing:** Run validation suites via `pytest ml/tests/` (verifies regex bounds, SSN contexts, and `disabled_entities` filtering).
* **Format/Lint:** `ruff check ml/ && black --check ml/`

### Rust (Backend)
* **Setup:** Standard Cargo workspace. Binary: `proxy` crate (entry point).
* **Build/Run:** `cargo build --release` / `cargo run --bin proxy`
* **Testing:** `cargo test` (ensures adapter multimodal extraction and metrics counters function correctly).
* **Format/Lint:** Must pass `cargo clippy -- -D warnings` and `cargo fmt -- --check`.

### Required Environment Variables
| Variable | Required By | Purpose |
|---|---|---|
| `OBSCURA_API_KEY` | proxy auth middleware | Inbound bearer token validation |
| `UPSTREAM_LLM_KEY` | proxy forwarding | API key forwarded to upstream LLM |
| `NER_MODEL_PATH` | `inference` crate | Filesystem path to the `.onnx` model file |
| `NER_TOKENIZER_PATH` | `inference` crate | Filesystem path to `tokenizer.json` |

The proxy binds to `0.0.0.0:8080`. Both `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` are validated at startup — the server refuses to start if either is missing or points to a non-existent path (fail-closed).

## 7. Repository Management Guidelines
* **Commit Messages:** Use Conventional Commits with Jira prefixes. Format: `type(scope): OBS-XXX description` (e.g., `feat(ML): OBS-7 export BERT to ONNX`).
* **Restricted Action Items:**
    * Do not commit `.bin`, `.pt`, `.safetensors`, or `.onnx` model weights to Git (use Git LFS or external bucket storage).
    * Do not modify GitHub Actions workflows (`.github/workflows/`) without explicit instruction.
    * Maintain `SPEC.md` dynamically. Any new architectural module must be documented here heavily enough to inform continuous AI development context.

---

## 8. Entity Type Registry
All 13 canonical entity types recognized by Obscura, their detection source, and base confidence scores.

### BERT-Only Types (semantic/contextual)
| Entity Type | Description | Source |
|---|---|---|
| `PERSON` | Human names (first, last, full) | BERT (`PER` label) |
| `ORGANIZATION` | Company, institution, agency names | BERT (`ORG` label) |
| `LOCATION` | Cities, countries, addresses, geographic names | BERT (`LOC` label) |
| `MISC` | Miscellaneous named entities (events, nationalities, products) | BERT (`MISC` label) |

The BERT model produces IOB2-prefixed labels (`B-PER`, `I-PER`, etc.). The `aggregation_strategy="simple"` pipeline collapses these into spans. The `NER_LABEL_TO_ENTITY` map translates model-specific short labels (`PER`, `LOC`, `ORG`, `MISC`) to canonical Obscura types.

### Regex-Only Types (structured/deterministic)
| Entity Type | Format | Base Score | Context-Scored |
|---|---|---|---|
| `SSN` | `XXX-XX-XXXX` (dashed) or `XXXXXXXXX` (dashless) | 0.99 (dashed) / 0.40 base (dashless) | Dashless only |
| `PHONE` | US `(NNN) NNN-NNNN` or international E.164 `+CC...` | 0.95 | No |
| `EMAIL` | `user@domain.tld` | 0.99 | No |
| `MRN` | `MRN-XXXXXXX` (case-insensitive prefix, 7-digit suffix) | 0.99 | No |
| `DOB` | `MM/DD/YYYY` or `YYYY-MM-DD` | 0.50 base | Yes |
| `CREDIT_CARD` | `NNNN-NNNN-NNNN-NNNN`, `NNNN NNNN NNNN NNNN`, or `NNNNNNNNNNNNNNNN` | 0.99 (dashed/spaced) / 0.50 base (undashed) | Undashed only |
| `IPV4` | Dotted quad with 0-255 validated octets | 0.95 | No |
| `IPV6` | Full, zero-compressed (`::1`), IPv4-mapped (`::ffff:a.b.c.d`) | 0.95 | No |
| `PASSPORT` | US format: 1 uppercase letter + 8 digits | 0.99 | No |

All regex patterns use symmetric `(?<!\w)` / `(?!\w)` word-boundary guards. Patterns are pre-compiled once at `RegexDetector.__post_init__()`. Each entity type has its own `_detect_*()` method.

**SSN IRS-rule validation** (`_is_valid_ssn_parts`): rejects area codes 000, 666, 900-999; group 00; serial 0000 — per SSA Publication 4557.

## 9. Conflict Resolution Algorithm
When BERT and regex produce overlapping spans, the `merge_entities()` function in `pii_engine.py` applies four rules in priority order. Candidates are processed sorted by `(start, -length)`:

**Rule 1 — Exact overlap** (same `start` and `end`):
- If either entity is from `regex` and its type is in `REGEX_AUTHORITATIVE_TYPES`, the regex entity wins.
- Otherwise, higher confidence score wins.

**Rule 2 — Partial overlap** (spans share some characters but neither contains the other):
- Longer span wins.

**Rule 3 — Nested spans** (one span fully contains the other):
- Outer (longer) span wins; inner span is discarded.

**Rule 4 — No overlap**:
- Both entities are retained.

`REGEX_AUTHORITATIVE_TYPES` is the complete set of all 9 regex entity types:
```python
REGEX_AUTHORITATIVE_TYPES: frozenset[str] = frozenset(
    {"SSN", "PHONE", "EMAIL", "MRN", "DOB", "CREDIT_CARD", "IPV4", "IPV6", "PASSPORT"}
)
```

The merge algorithm is reused for two purposes: (1) merging BERT vs. regex results, and (2) deduplicating stride-overlapped chunks during long-text processing.

## 10. Token Format & Mapping
**Token format:** `[{ENTITY_TYPE}_{N}]` where `N` is a per-type counter starting at 1.

Examples: `[PERSON_1]`, `[SSN_2]`, `[EMAIL_1]`.

**Counter behavior:**
- Counters are initialized to zero and increment per entity type within a single `redact()` call.
- Counters reset on every new `redact()` call — they are not persisted across sessions.
- A session UUID (`session_id`) is generated per call for downstream correlation.

**Replacement order:**
- Entities are sorted in **descending** character position order before substitution.
- This preserves earlier character offsets — replacing a later span does not shift the positions of earlier spans.
- Counter assignment follows this descending order, so `[PERSON_1]` may appear later in the text than `[PERSON_2]`.

**Mapping dictionary:**
- A `dict[str, str]` mapping `token → original_text` is returned in `RedactionResult.mapping`.
- This dictionary is the bridge artifact consumed by the Rust `MappingDictionary` for restoration.
- The `mapping` values are raw PII and must never be logged. `RedactionResult.__repr__()` and `to_dict()` mask mapping values with `"***"`.

## 11. disabled_entities Contract
`PIIEngine.redact(text, disabled_entities=["PERSON", "EMAIL"])` accepts an optional list of entity type strings to exclude from masking.

**Critical ordering constraint:** Filtering happens **after** `detect()` completes and the full conflict-resolution merge has run. Disabled types are removed from the merged entity list immediately before token substitution.

**Why post-merge:** If filtering happened before merge, a disabled regex entity could be absent from conflict resolution, allowing a suppressed BERT entity to survive that would otherwise have been overridden. Post-merge filtering ensures the disabled entity is correctly absent from both the output text and the mapping dictionary, while conflict resolution ran on complete data.

**Behavior:**
- Disabled entities are left intact in `masked_text`.
- Disabled entities are excluded from `mapping` and `entities` in `RedactionResult`.
- `X-Obscura-Skip-Redaction` header values are merged with `disabled_entities` from `obscura.toml` (union, not override) by the Rust proxy before passing to the engine.
- Passing `disabled_entities=None` or `[]` disables no types — all detected entities are redacted.

## 12. Long-Text Chunking Strategy
BERT's WordPiece tokenizer has a hard 512-token ceiling. Long texts must be split before inference.

**Trigger:** Input texts exceeding `chunk_size = 1500` characters (approximately 300 tokens, providing safe margin below 512).

**Split boundaries:** Chunks are split at sentence boundaries — characters `.`, `;`, and `\n`. The algorithm collects all boundary positions, then greedily extends each chunk to include as many complete sentences as fit within `chunk_size`.

**Stride overlap:** The last sentence of each chunk is repeated as the first sentence of the next chunk. This preserves context for entities that span chunk boundaries (e.g., a name split across a sentence).

**Deduplication:** Entities captured twice due to stride overlap are deduplicated using the same `merge_entities()` span-merge algorithm used for BERT/regex conflict resolution. The chunk's local character offsets are mapped back to global positions before merge.

**Fast path:** Texts at or below 1500 characters bypass chunking entirely and run in a single pipeline call.

## 13. Context-Aware Scoring Reference
Three entity types use context scoring to reduce false positives. Each scorer examines `context_window = 10` words on each side of the match, strips punctuation from context tokens, and applies additive/subtractive scoring clamped to `[0.0, 1.0]`.

### SSN Dashless (9-digit numbers)
```
base score:    0.40
+ 0.35        if any trigger word in context
              (ssn, social, security, taxpayer, tin, tax, identification,
               ss#, w-2, w-9, w2, w9, 1099, itin, identity, verification, background)
+ 0.20        if trigger phrase bigram in context
              (social security, tax id, taxpayer identification, taxpayer id,
               background check, identity verification)
- 0.35        if any negative word in context
              (phone, call, fax, tel, telephone, mobile, cell, dial, ext,
               order, confirmation, tracking, serial, account, routing,
               invoice, zip, postal, code, ref, reference, case#, ticket, po)
threshold:    0.70
```

### DOB (dates in MM/DD/YYYY or YYYY-MM-DD)
```
base score:    0.50
+ 0.40        if any trigger word in context
              (dob, birth, born, birthday, birthdate, age, newborn, neonatal)
+ 0.10        if trigger phrase bigram in context
              (date of)
- 0.35        if any negative word in context
              (meeting, appointment, scheduled, deadline, due, expires,
               expiration, created, updated, filed, issued, effective, invoice, report)
threshold:    0.70
```

### Credit Card Undashed (16 consecutive digits)
```
base score:    0.50
+ 0.45        if any trigger word in context
              (visa, mastercard, amex, discover, card, credit, debit, cc, pan, cvv)
- 0.35        if any negative word in context
              (id, identifier, tracking, order, receipt, routing, transaction)
threshold:    0.70
```

Dashed/spaced credit card formats (`NNNN-NNNN-NNNN-NNNN`) are unambiguous and receive a fixed score of `0.99` without context analysis.

## 14. ProviderAdapter Trait & OpenAIAdapter
The `ProviderAdapter` trait in `src/providers/adapter.rs` abstracts provider-specific JSON schemas:

```rust
pub trait ProviderAdapter {
    fn extract_request_text(&self, body: &str) -> Option<String>;
    fn extract_response_text(&self, body: &str) -> Option<String>;
    fn extract_response_delta_text(&self, _body: &str) -> Option<String> {
        None  // default: streaming not supported
    }
}
```

**OpenAIAdapter** (`src/providers/openai.rs`) handles the OpenAI Chat Completions schema:

| Method | Input JSON path | Content handling |
|---|---|---|
| `extract_request_text` | `messages[*].content` (role=user only) | String or `[{"type":"text","text":"..."}]` parts array; multiple user messages joined with `\n` |
| `extract_response_text` | `choices[0].message.content` | String or parts array |
| `extract_response_delta_text` | Two formats (see below) | String only |

**Streaming delta formats handled by `extract_response_delta_text`:**
1. OpenAI Responses API: `{"type": "response.output_text.delta", "delta": "..."}`
2. Chat Completions SSE chunk: `{"choices": [{"delta": {"content": "..."}}]}`

## 15. HTTP API Reference
The Rust proxy exposes the following endpoints on `0.0.0.0:8080`:

| Method | Path | Response | Description |
|---|---|---|---|
| `GET` | `/health` | `200 {"status": "ok"}` | Liveness check. Responds before config/model load. |
| `GET` | `/metrics` | `200 <prometheus text>` | Prometheus-format counter for `http_requests_total`. Served by `src/main.rs`. |
| `*` | `/*` | Upstream response | Pass-through to `upstream_url + path`. Forwards all headers except `Host`. Returns `502 Bad Gateway` on upstream failure. |

**`X-Obscura-Skip-Redaction` header:**
- Format: comma-separated entity type strings, e.g. `"PERSON,EMAIL,SSN"`
- These are merged (union) with `disabled_entities` from `obscura.toml` for the duration of the request.
- Type strings must exactly match canonical entity type names (case-sensitive).

## 16. obscura.toml Schema Reference
Full schema for the local configuration file. In production, secrets come from environment variables.

```toml
[app]
env = "dev"          # Environment: dev | staging | prod
host = "0.0.0.0"     # Bind address
port = 8080          # Bind port
debug = true         # Enable verbose logging

[llm]
provider = "mock"           # LLM provider: openai | gemini | mock
model = "gpt-4.1-mini"      # Model name (provider-specific)
max_output_tokens = 512     # Max response tokens
temperature = 0.3           # Sampling temperature (0.0 = deterministic)
stream = true               # Use server-sent events / chunked streaming

[pii]
token_prefix = "<"          # Token bracket prefix (default "<", actual code uses "[")
token_suffix = ">"          # Token bracket suffix (default ">", actual code uses "]")
mapping_store = "memory"    # Token map backend: memory (dev) | redis (future)
enable_leak_checks = false  # Reject upstream responses that appear to leak raw PII
```

**Note:** The `config` crate's `Config` struct currently parses only `upstream_url` (flat key) and `disabled_entities` (flat key). The `[app]`/`[llm]`/`[pii]` section parsing is defined in `obscura.toml` for future use but not yet wired into the Rust `Config` struct.

## 17. File-Level Module Map
One-line description per source file. Use this to find the right file before editing.

### Python (`ml/`)
| File | Owns |
|---|---|
| `ml/pii_engine.py` | `PIIEngine` — hybrid detect/redact/restore pipeline, chunking, `merge_entities()`, `REGEX_AUTHORITATIVE_TYPES` |
| `ml/regex_detector.py` | `RegexDetector` — all 9 regex entity types, pre-compiled patterns, context scoring |
| `ml/schemas.py` | `DetectedEntity` (PII-safe repr/to_dict), `RedactionResult` |
| `ml/export_onnx.py` | CLI: exports fine-tuned BERT to ONNX (opset 14, dynamic sequence axes), packages `tokenizer.json` |
| `ml/fine_tune.py` | BERT fine-tuning script on IOB2 NER datasets |
| `ml/evaluate.py` | Evaluation harness — BIO tag alignment, macro F1 via seqeval, hybrid entity resolution |
| `ml/GAMEPLAN.md` | Pipeline architecture spec — conflict resolution rules, thresholds, design rationale |

### Rust
| File | Owns |
|---|---|
| `proxy/src/main.rs` | Hyper v1.0 HTTP server, `/health` endpoint, config load, header parsing, pass-through forwarding |
| `inference/src/lib.rs` | `ModelEnvironment` — reads `NER_MODEL_PATH`, `NER_TOKENIZER_PATH` from env; validates paths at startup |
| `inference/src/mapping.rs` | `MappingDictionary` — `HashMap<String, String>` for token-to-PII restoration |
| `config/src/lib.rs` | `Config` struct — `upstream_url`, `disabled_entities`; `load_from_file("obscura.toml")` |
| `src/main.rs` | Legacy metrics server — `GET /metrics` Prometheus endpoint, `http_requests_total` counter |
| `src/providers/adapter.rs` | `ProviderAdapter` trait definition |
| `src/providers/openai.rs` | `OpenAIAdapter` — request/response/delta text extraction for OpenAI Chat Completions schema |

## 18. Current Implementation Status
**This section documents what is actually implemented vs. what the pipeline design targets.**

### Fully Implemented
- Python hybrid NER pipeline: BERT + regex, conflict resolution, chunking, context scoring, disabled_entities
- All 9 regex entity types with full test coverage
- `OpenAIAdapter` — request/response/streaming delta extraction, multimodal content arrays
- Config loading (`obscura.toml`), `X-Obscura-Skip-Redaction` header parsing
- `ModelEnvironment` — env var validation at startup (fail-closed)
- `MappingDictionary` — Rust data structure for token-to-PII map
- `/health` endpoint (Rust proxy)
- `/metrics` endpoint (legacy src/ server, Prometheus format)
- Pass-through HTTP forwarding to `upstream_url`

### Not Yet Implemented (Wiring Gap)
- **ONNX inference is not wired into the request handler.** `ModelEnvironment` is loaded at startup but the proxy's `handle_request` function does not call any inference code. Requests are forwarded as-is without redaction.
- The `ort` / HuggingFace `tokenizers` crates are not yet used in the inference crate — `ModelEnvironment` only validates path existence.
- Response unmasking (token → original PII replacement) is not implemented in the Rust proxy path; it is only available via `PIIEngine.restore()` in Python.
- `disabled_entities` from `obscura.toml` are parsed and stored in `skipped_entities` but not passed to any inference function.

An agent reading only the pipeline design in §3 would over-implement the Rust side. Build new Rust inference work on top of `inference/src/lib.rs` and `inference/src/mapping.rs`.
