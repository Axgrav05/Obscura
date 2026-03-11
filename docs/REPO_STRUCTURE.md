# Obscura Repo Structure

## Top Level

- `src/` Rust backend source
- `ml/` Machine learning utilities (future use)
- `models/` Model artifacts
- `deploy/` Deployment-related files
- `context/` Project context and references
- `docs/` Documentation
- `Weekly Progress/` Sprint reports

---

## Rust Backend Layout (`src/`)

- `main.rs` – Application entry point
- `api/` – HTTP routes and request handlers
- `masking/` – Reversible anonymization logic
- `llm/` – LLM provider integration
- `config/` – Configuration handling (`obscura.toml`)

---

## Standards

- Use `cargo fmt` before committing
- No secrets committed to Git
- All sprint updates go in `Weekly Progress/weekX`
- Configuration defaults live in `obscura.toml`
