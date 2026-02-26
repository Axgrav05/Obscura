# SECURITY_TRACEABILITY.md
## Obscura — Threat → Mitigation → Jira Ticket Mapping (Standalone Proxy)

This matrix maps each threat from the standalone threat model to the Jira tickets (OBS-*) that implement or validate mitigations.  
**Note:** Some threats are mitigated indirectly by “review/documentation/testing” tickets; where a *direct* implementation ticket is missing, it’s called out as a **GAP**.

---

## Legend
- **Threat ID**: from SECURITY.md threat register (T1–T18)
- **Mitigation summary**: the control(s) that reduce the threat
- **Jira tickets**: OBS-* issues that implement/verify the mitigation

---

## Threat Traceability Matrix

| Threat ID | Threat (plain meaning) | Mitigation summary | Jira ticket(s) |
|---:|---|---|---|
| T1 | Fail-open redaction on timeout/load (under stress, prompts forward unmasked) | Enforce fail-closed behavior; load/stress test; performance optimization/capacity planning | **OBS-32**, OBS-17, OBS-29, OBS-38, OBS-41 |
| T2 | Secrets leaked via logs/traces/errors (keys/tokens appear in observability) | Security review + harden logging/runbook/docs to prevent sensitive logging | **OBS-18**, OBS-54, OBS-53 *(GAP: add explicit “central log redaction + prohibit auth header/prompt logging” ticket)* |
| T3 | PII detection evasion (NER/rules miss obfuscated PII) | Model selection + ONNX inference + regex rules + integration tests + pen testing | **OBS-8**, OBS-14, OBS-25, OBS-34, OBS-42 *(plus pipeline support: OBS-13)* |
| T4 | DoS via prompt bombs/expensive inference (CPU/RAM exhaustion) | Rate limiting + stress/load testing + performance optimization + buffering controls | **OBS-27**, OBS-17, OBS-29, OBS-38, OBS-41, OBS-20 |
| T5 | Auth/header passthrough mistake (internal tokens forwarded to vendor) | Auth middleware + API key auth + correct forwarding behavior + secure integration guidance + security review | **OBS-12**, **OBS-22**, OBS-7, OBS-52, OBS-18 |
| T6 | Unrestricted egress / misrouting (data sent to unintended host) | Constrain forwarding behavior; deployment hardening; security review; pen test; document allowed endpoints | OBS-7, **OBS-15**, OBS-18, OBS-36, OBS-42 *(GAP: add explicit “egress allowlist to approved LLM domains only” ticket)* |
| T7 | Reversible masking key compromise (stolen key enables re-identification) | Implement reversible mapping; protect via auth controls; document + review; backup/recovery procedures | **OBS-26**, OBS-12, OBS-18, OBS-54, OBS-39 |
| T8 | Supply-chain compromise (malicious deps/model/image/CI) | CI/CD pipeline + pinned artifacts + image optimization + security docs and deployment guidance | **OBS-9**, OBS-13, OBS-23, OBS-36, OBS-54 *(GAP: add explicit “SBOM + dependency scan + image scan/signing” ticket)* |
| T9 | Config tampering/misconfig (redaction disabled/weak model/wrong policy) | Secure auth/config patterns; strong deployment/integration docs; security review | OBS-12, OBS-36, OBS-51, OBS-52, **OBS-18**, OBS-54 |
| T10 | Telemetry leaks sensitive info (PII in metrics/labels/alerts/dashboard) | Safe metrics design; redaction event metrics; compliance dashboard; security documentation | **OBS-11**, **OBS-24**, **OBS-35**, OBS-54 |
| T11 | Public exposure of proxy/health/metrics ports (internet reachable unintentionally) | Deployment verification + security review + pen testing + smoke tests + docs | **OBS-15**, **OBS-18**, **OBS-42**, **OBS-57**, OBS-36 |
| T12 | Regex ReDoS (crafted input stalls service via regex) | Regex engine design + stress testing + performance optimization | **OBS-25**, OBS-29, OBS-38 |
| T13 | Over-permissioned IAM / EC2 role abuse (broad cloud access if compromised) | Secure deployment practices + security review + pen testing + infra docs | **OBS-15**, **OBS-18**, OBS-42, OBS-36, OBS-51 |
| T14 | Restore-path authorization bypass (unauthorized unmask/restore) | Reversible mapping + strong auth middleware + API key auth + security review | **OBS-26**, **OBS-12**, **OBS-22**, **OBS-18** |
| T15 | Data retention via buffering/retries/caching (violates “in-memory only”) | Correct SSE streaming + buffering/reassembly + integration tests + ops/security docs | **OBS-19**, **OBS-20**, **OBS-21**, **OBS-34**, OBS-53, OBS-54 |
| T16 | Insider misuse (authorized users view prompts/PII in ops tooling) | SRE runbook + security docs + security review processes | **OBS-53**, **OBS-54**, **OBS-18** |
| T17 | Provider response parsing bugs (malformed responses crash/overflow) | Forwarding implementation + E2E tests + integration tests + multi-provider tests | **OBS-7**, **OBS-16**, **OBS-34**, **OBS-40** |
| T18 | Tenant/policy mix-up (wrong key/policy applied; cross-tenant risk) | Multi-provider test suite + integration guide + security documentation | **OBS-40**, **OBS-52**, **OBS-54** |

---

## Stretch Goal Traceability (Membership Inference / Differential Privacy)

| Item | Risk/Goal | Mitigation summary | Jira ticket(s) |
|---|---|---|---|
| MI-1 | Membership inference / privacy leakage from aggregates | Differential privacy noise injection + validate utility impact + compliance view | **OBS-31**, **OBS-37**, OBS-35 |

---

## Recommended “Gap” Tickets to Add (Optional but strengthens traceability)
- **OBS-SEC-LOG-1:** Central log redaction + ban prompt/auth logging by default (covers T2/T10/T16)
- **OBS-SEC-NET-1:** Egress allowlist to approved LLM domains only (covers T6)
- **OBS-SEC-SUP-1:** SBOM + dependency scanning + container image scanning/signing (covers T8)
