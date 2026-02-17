# Obscura — AI/ML Engine

Cloud-native zero-trust security proxy with BERT-based NER for real-time PII/PHI redaction.

## Project Structure (Monorepo)

- `/src` — Rust proxy core (Cargo workspaces: proxy logic, inference crate, networking)
- `/ml` — Python: BERT model training, evaluation, ONNX conversion
- `/infra` — Kubernetes manifests, OCI config
- `/models` — Serialized `.onnx` model files (NOT committed — pulled from OCI Object Storage)

## Environment Setup

**IMPORTANT: Always activate the Python virtual environment before running any Python command.**

```bash
# Python (run from /ml)
source .venv/bin/activate  # ALWAYS do this first
poetry install             # or: uv sync

# Rust (run from /src)
cargo build
cargo clippy -- -D warnings
cargo test
```

If the `.venv` directory doesn't exist, create it:
```bash
cd ml && python3.11 -m venv .venv && source .venv/bin/activate && poetry install
```

**NEVER install Python packages globally. Always install inside the venv.**

## My Ownership Scope

I own the Python ML pipeline and the ONNX/inference integration layer:
- `/ml/**` — model training, evaluation, ONNX export, preprocessing
- `/models/**` — exported model artifacts
- `/src/inference/` — Rust inference crate (co-owned with Rainier)

**ASK before modifying code outside these directories**, especially `/src/proxy/`, `/src/net/`, or `/infra/`.

## Code Standards

### Python
- Python 3.11+, type hints on all functions
- Ruff for linting, Black for formatting (config in `pyproject.toml`)
- Docstrings on every public function — must explain mathematical logic for masking/redaction
- Run before committing: `ruff check ml/ && black --check ml/`

### Rust
- Stable toolchain 1.75+
- `cargo clippy` with zero warnings is mandatory
- Doc comments on all public items in the inference crate
- Async via `tokio`, HTTP via `hyper`, inference via `ort` or `tch-rs`

### Commits
- Format: `type(scope): OBS-N description`
- Example: `feat(ML): OBS-7 integrate BERT-NER model via ONNX`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## How I Want You to Work

- **Write code from scratch** when asked — don't just outline
- **Always explain what the code does** so I can follow along and learn
- **Debug proactively** — read error messages carefully, check venv activation, verify imports
- **Refactor when needed** — clean up directory structure, dead code, messy scripts
- **Balance maintainability and performance** — don't over-engineer simple scripts
- **Ask before making architectural decisions** (model architecture changes, new dependencies, API contract changes with Rust side)
- **Ask before working outside my ownership scope**

## Security & Compliance — CRITICAL

- **NEVER hardcode API keys or secrets.** Use env vars injected via OCI Vault / K8s Secrets.
- **NEVER commit raw datasets**, model weights (`.bin`, `.pt`), or HuggingFace cache. Use Git LFS or OCI Object Storage.
- **NEVER log raw PII/PHI.** The proxy is zero-knowledge — log only that a redaction occurred (timestamp + entity type).
- **Only use synthetic/anonymized data for local testing.** Zero risk of real PHI leakage.
- **Verify `.onnx` model checksums** before loading into the Rust backend.
- HIPAA/GDPR compliance: no content logging, only security event metadata to Grafana Cloud.

## Technical Constraints

- **Inference latency budget: ≤30-40ms** (out of 60ms total proxy overhead)
- **Target hardware: OCI Always Free ARM (Ampere A1)** — models must be quantized to fit limited RAM
- **Docker images: multi-stage builds**, distroless/Alpine final layer, no dev dependencies in production
- **Model accuracy target: 90%+ F1** on NER benchmarks (composite scoring, BERTScore validation)

## Key Libraries & Frameworks

### Python
- `transformers` (HuggingFace) — BERT-based NER models (`dslim/bert-base-NER`, clinical variants)
- `optimum` — ONNX conversion and optimization
- `torch` — PyTorch backend
- `scispaCy` — medical entity detection
- Presidio-style hybrid: transformer predictions + deterministic regex for SSNs, phone numbers

### Rust
- `tokio` — async runtime
- `hyper` — HTTP proxy layer
- `ort` — ONNX Runtime bindings (primary inference path)
- `tch-rs` — PyTorch bindings (fallback)

## Integration Contract (Python ↔ Rust)

The Python side exports:
1. A serialized `.onnx` (or `.pt`) model file
2. A JSON schema defining input/output format
3. A "Mapping Dictionary" spec: original PII tokens ↔ masked replacements

The Rust side consumes these via `ort` or `tch-rs`. Contract specs live in Jira/Confluence. When modifying the export format, flag it — Rainier needs to update the Rust consumer.

## Active Jira Tickets

- **OBS-3**: Benchmarking — establish baseline accuracy, select optimal model architecture
- **OBS-7**: Integration — bridge Python/AI layer with Rust backend, mapping dictionary
- **OBS-12**: Hardening — validate differential privacy noise doesn't degrade utility

## Testing

```bash
# Python tests
source ml/.venv/bin/activate && pytest ml/tests/ -v

# Rust tests
cargo test --workspace

# Lint everything
ruff check ml/ && black --check ml/ && cargo clippy -- -D warnings
```

## When Compacting

Always preserve: list of modified files, active Jira ticket context, and any pending integration contract changes with the Rust side.

## Project Context
For full project scope, architecture, and sprint roadmap, read the `.context/` directory. For ML-specific or inference-crate-specific context, read the `CLAUDE.md` in those subdirectories. Only read these when you need broader context — don't load them by default.