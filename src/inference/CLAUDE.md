# Inference Crate

Rust crate that loads ONNX models and executes NER inference within the proxy pipeline.

## Ownership
Co-owned by Arjun (AI/ML Lead) and Rainier (Backend Engineer).
Changes here should be coordinated between both owners.

## Key Dependencies
- `ort` — ONNX Runtime bindings (primary)
- `tch-rs` — PyTorch bindings (fallback)
- `tokio` — async runtime

## Contract
Consumes `.onnx` files exported by the Python ML pipeline.
Input/output schema defined in the integration contract (Jira/Confluence).

## Performance
This crate must contribute ≤30-40ms to total proxy overhead.
Profile with `cargo bench` before submitting PRs.

## Testing
```bash
cargo test -p inference
```
