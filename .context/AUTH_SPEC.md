# Obscura Authentication Flow Specification

**Status:** (v1)  
**Last Updated:** 2026-03-04  
**Owner:** Obscura Proxy Team  
**Applies To:** Rust proxy (`/src`) deployed as a Kubernetes sidecar

---

## 1. Purpose

This document defines the authentication (AuthN) and authorization (AuthZ) flow for Obscura, a privacy-preserving middleware proxy deployed as a Kubernetes sidecar. The design prioritizes:

- **Fail-closed security:** unauthenticated or ambiguous requests are blocked and never forwarded upstream.
- **Zero-knowledge logging/telemetry:** no raw request content or secrets are logged or emitted.
- **Memory safety:** no unsafe Rust in auth paths; secrets handled carefully.
- **Sub-millisecond overhead:** auth checks must add negligible latency to the request path.

---

## 2. Security Posture and Assumptions

### 2.1 Zero-Trust
Obscura assumes:
- The adjacent application container may be compromised.
- Requests to the proxy may be crafted or malicious.
- Any upstream LLM provider is untrusted with respect to sensitive data.

### 2.2 Trust Boundary
Authentication is enforced **between the application and the sidecar**. Even though the proxy is bound to localhost within a pod network namespace, “localhost” is not sufficient authentication.

### 2.3 Threats (Auth-relevant)
- **T1:** Unauthorized access attempts to pass raw PII/PHI through the proxy.
- **T2:** Brute force / guessing attempts on static tokens.
- **T3:** Timing attacks against token comparisons.
- **T4:** Key leakage via logs, metrics, panic messages, error responses, or debug prints.
- **T5:** Replay of authenticated requests if an attacker can observe traffic.
- **T6:** Bypass attempts via alternate endpoints, methods, or header ambiguity.

Mitigations appear throughout the requirements and flow below.

---

## 3. Requirements

### 3.1 Functional Requirements
- Obscura **MUST** authenticate each inbound request **before** payload parsing and **before** any upstream call.
- Obscura **MUST** reject requests without a valid API key for authenticated endpoints.
- Obscura **MUST** enforce strict header parsing and reject ambiguous forms.
- Obscura **MUST** support key rotation without downtime (multiple valid keys).
- Obscura **MUST** return deterministic errors that do not leak secrets or payloads.

### 3.2 Non-Functional Requirements
- Auth check latency target: **< 1ms** per request under normal load.
- **No raw secrets** in logs or telemetry.
- No `.unwrap()` in production auth paths.
- **Constant-time equality** for token comparison.

---

## 4. Credentials and Secret Management

### 4.1 Credential Type
Obscura uses a static bearer token for intra-pod authorization:

- **Header:** `Authorization: Bearer <token>`
- **Secret Source:** environment variable `OBSCURA_API_KEY`

### 4.2 Key Injection
- `OBSCURA_API_KEY` is injected via Kubernetes Secrets or an equivalent secure mechanism (e.g., AWS KMS → Secret).
- The proxy **MUST NOT** hardcode any secrets.
- The proxy **MUST NOT** print environment variables, headers, or token values.

### 4.3 Key Rotation (Multiple Keys)
The proxy supports rotation by allowing multiple valid tokens:

- `OBSCURA_API_KEY` value may be:
  - a single token, or
  - a comma-separated set: `keyA,keyB,keyC`

Parsing rules:
- Split on literal commas `,`.
- Trim ASCII whitespace around each token.
- Reject empty tokens.
- Enforce bounds:
  - max keys: **8**
  - per-token length: **16..256** bytes (ASCII)

Rotation procedure:
1. Deploy secret containing both old and new tokens.
2. Update caller to use new token.
3. Deploy secret removing old token.

---

## 5. Allowed Endpoints and AuthZ Policy

### 5.1 Endpoint Policy (v1)
Obscura exposes these endpoints:

- `GET /healthz` — **unauthenticated** (liveness)
- `GET /readyz` — **unauthenticated** (readiness)
- `GET /metrics` — **authenticated**
- `POST /v1/forward` — **authenticated** (primary forwarding endpoint)

All other paths:
- **MUST** return `404 Not Found` with a minimal response body and no details.

### 5.2 Authorization Model (v1)
Authorization is currently binary, with an explicit future extension point:

- **Authenticated endpoints** require a valid bearer token.
- **403 Forbidden** is reserved for **authenticated-but-not-authorized** cases (e.g., future per-endpoint scopes, tenant allowlists/denylists, or revoked-but-recognized keys).
- **v1 default:** there are no scopes/roles; therefore most access failures are handled as **401** (see §8), but the **403 response format is defined** in this spec for forward compatibility and to satisfy interface stability.

---

## 6. Authentication Flow (Request Path)

Authentication occurs at the earliest possible stage:

1. Receive HTTP request.
2. If the request targets an unauthenticated endpoint (`/healthz`, `/readyz`):
   - return success/failure per health logic (no auth required).
3. For all other endpoints:
   - validate method/path is allowed for that endpoint (e.g., `POST /v1/forward` only).
   - extract and validate `Authorization` header strictly.
   - constant-time compare token to configured valid key(s).
4. If valid:
   - proceed to request handling/redaction pipeline.
5. If invalid/missing/malformed/ambiguous:
   - fail-closed: return 401 immediately and do not parse body, do not forward upstream.
6. If valid but not authorized by a future policy (scopes/tenant rules/denylist):
   - fail-closed: return 403 and do not forward upstream.

---

## 7. Header Validation Rules

### 7.1 Authorization Header Requirements
- Exactly **one** `Authorization` header is allowed for authenticated endpoints.
- Header name is case-insensitive per HTTP rules, but multiple instances are **ambiguous**.

Accepted format:
- `Authorization: Bearer <token>`
- Exactly one ASCII space after `Bearer`.
- `<token>` must be ASCII, length **16..256** bytes, with no leading/trailing whitespace.

Allowed token character set (recommended, strict):
- Base64url-safe: `[A-Za-z0-9_-]` plus optional `.` and `~`
- If a token contains any other byte, reject as malformed.

Reject conditions:
- Missing `Authorization`
- Multiple `Authorization` headers
- Wrong scheme (anything other than `Bearer`)
- Token outside length bounds
- Token contains disallowed bytes
- Token has leading/trailing whitespace
- Any parsing ambiguity

### 7.2 Constant-Time Comparison
- Token comparisons **MUST** be constant-time.
- Implementation must avoid early exits that create measurable timing differences.

Minimum requirement:
- Use a vetted constant-time equality function for byte slices.
- Compare against all configured keys in a loop; authenticate if any match.

---

## 8. Error Response Format (JSON Schema)

### 8.1 Standard Response Headers (All Errors)
All error responses **MUST** include:
- `Content-Type: application/json`
- `X-Request-Id: <generated-id>`

`WWW-Authenticate: Bearer`:
- **MUST** be included on `401 Unauthorized`.
- **MUST NOT** be included on `403 Forbidden` or `429 Too Many Requests`.

### 8.2 Request ID
- Generate a per-request identifier early (UUIDv4 or equivalent).
- Request ID **must not** be derived from request content.
- Request ID may be used for correlation (no payload included).
## 8.3 JSON Error Body Rules

- `error` is a machine-friendly identifier (see §8.4).
- `message` is a short, non-sensitive human-readable string.
- Do not include request content, headers, tokens, stack traces, or internal diagnostics.

## 8.4 Error Codes and Status Mapping

### 401 Unauthorized

Used when the client fails to authenticate:

- Missing `Authorization`
- Malformed `Authorization`
- Wrong scheme
- Invalid token
- Multiple `Authorization` headers

Response:

- **Status:** `401`
- **Headers:** include `WWW-Authenticate: Bearer`
- **Body example:**
  ```json
  { "error": "unauthorized", "request_id": "<id>", "message": "Authentication required." }

403 Forbidden
Used when authentication succeeds but the request is not permitted by policy:

valid token but endpoint is not allowed for that token (future scope model)
valid token is recognized but explicitly blocked (denylist / revoked key policy)
Response: - Status: 403 - Headers: no WWW-Authenticate

Body example:

{ "error": "forbidden", "request_id": "<id>", "message": "Not authorized to access this resource." }
v1 default behavior: unless an explicit AuthZ policy is implemented, Obscura will typically not emit 403. The schema and semantics are defined here for forward compatibility and stable client integration.

404 Not Found
Used when path is not in the allowlist.

Response: - Status: 404

Body example:

{ "error": "not_found", "request_id": "<id>", "message": "Not found." }
405 Method Not Allowed
Used when the path is allowed but the method is not allowed.

Response: - Status: 405

Body example:

{ "error": "method_not_allowed", "request_id": "<id>", "message": "Method not allowed." }
415 Unsupported Media Type
Used when content-type is unsupported for /v1/forward.

Response: - Status: 415

Body example:

{ "error": "unsupported_media_type", "request_id": "<id>", "message": "Unsupported C>}
429 Too Many Requests
Used when an authenticated client exceeds rate limits.

Response: - Status: 429

Body example:

{ "error": "rate_limited", "request_id": "<id>", "message": "Rate limit exceeded." }
500 Internal Server Error
Used for unexpected internal errors; must not leak diagnostics.

Response: - Status: 500

Body example:

{ "error": "internal_error", "request_id": "<id>", "message": "Internal error." }
9. Telemetry Requirements (Zero-Knowledge)
Auth telemetry may emit: - auth_requests_total - auth_ok_total - auth_denied_total{reason="missing|malformed|invalid|ambiguous"} - authz_denied_total{reas} (future policy) - auth_latency_ms histogram

Rate limiting telemetry may emit: - rate_limited_total - rate_limiter_errors_total

Telemetry must never emit: - tokens, headers, or payloads - any PII/PHI - secrets from environment variables

Logging policy: - Prefer metrics over logs. - If logs exist, they must be structured and contain only: - request_id - endpoint (coarse) - status code - reason code (coarse)

10. Rate Limiting (Integration Plan)
Rate limiting is applied after successful auth and before payload parsing/inference.

Default behavior: - Per-key token-bucket limiter in memory (bounded).

On limit exceeded: - return 429 Too Many Requests with schema in §8 - do not forward upstream

Fail-closed: - If the limiter subsystem fails, return 503 Service Unavailable (or 500 if preferred by platform policy) and do not forward.

Key partitioning: - Do not emit API keys in logs/metrics. - If per-key metrics are required, derive a non-reversible identifier using HMAC with a proxy-local salt and truncate. - Do not log raw identifiers unless explicitly needed.

11. Replay Protection (v1 Decision)
11.1 v1 Decision: Not Implemented
v1 does not implement request nonces, timestamps, or signatures. The bearer token model is considered sufficient for the intended intra-pod use case.

11.2 Risk Statement
If an attacker can observe localhost or pod traffic (node compromise, privileged workload, misconfiguration), they may replay authenticated requests.

11.3 Compensating Controls (Required)
Proxy binds only to 127.0.0.1:8080 within the pod network namespace.
Strict endpoint allowlist (see §5.1).
Tight upstream timeouts and request size limits (DoS reduction).
Metrics/logs do not contain payloads or secrets.
11.4 Upgrade Path (v2+)
If replay resistance is required, add one of: - mTLS between app and sidecar via SPIFFE/SPIRE or a service mesh. - Signed requests: X-Obscura-Timestamp + X-Obscura-Nonce + X-Obscura-Signature (HMAC over method/path/body hash). - Short-lived tokens with rotation via external identity provider.

12. Error Handling and Panic Safety
Auth code must avoid panics.
Any unexpected internal auth error returns 500 with schema in §8 and must not forward upstream.
Secret handling guidance: - Avoid formatting or cloning secrets. - Prefer byte slices and minimize allocations. - Do not convert secrets into String for logging.

13. Validation Checklist (Implementation Acceptance)
Auth check happens before any body parsing and before any upstream call.
Only allowlisted endpoints exist; all others return 404.
/healthz and /readyz are unauthenticated; /metrics and /v1/forward require auth.
Multiple Authorization headers are rejected.
Strict Bearer <token> parsing with bounds and allowed charset.
Constant-time token comparison used.
401/403/429 response formats match §8 and contain no sensitive data.
Fail-closed behavior verified in tests (missing/malformed/invalid/ambiguous).
No secrets, headers, or payloads in logs/metrics.
Key rotation works (multiple keys).
Rate limiting enforces 429 and fails closed on limiter errors.
cargo clippy clean; no unwraps in production auth path.
Tests use synthetic data only.
