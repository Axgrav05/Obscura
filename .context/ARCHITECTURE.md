# Obscura — System Architecture

## Deployment Pattern
Kubernetes Sidecar. Obscura runs alongside the application in the same Pod, intercepting traffic on localhost.

## Infrastructure
Managed Kubernetes (OKE) on OCI Always Free ARM Ampere A1 instances.

## Request Lifecycle
1. **Ingress**: Proxy intercepts raw JSON requests containing PII
2. **Redact**: BERT-NER identifies entities → Presidio-style hybrid logic replaces with tokens
3. **Egress**: Sanitized text is forwarded to the LLM (OpenAI/Gemini)
4. **Response**: Proxy restores tokens to original values using local mapping dictionary

## Component Boundaries
- Rust proxy (`/src`): Networking, async request handling, inference orchestration
- Python ML (`/ml`): Model training, evaluation, ONNX export
- Inference crate (`/src/inference/`): Loads `.onnx` models via `ort`, executes NER predictions
- Infrastructure (`/infra`): K8s manifests, OCI config, Grafana dashboards

## Integration Contract (Python → Rust)
Python exports: `.onnx` model file + JSON input/output schema + mapping dictionary spec
Rust consumes via `ort` crate. Contract details tracked in Jira/Confluence.
