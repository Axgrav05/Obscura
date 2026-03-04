# Backend Execution Plan & Modifications Guide

This document contains strict instructions and context for executing backend modifications to support the new AI engine capabilities (Bring-Your-Own-Model, Configurable Redaction, and Categorization Expansion).

**Execution requires strictly following the phases outlined below.** 
*Crucially: You must complete Phase 0 and wait for explicit confirmation before proceeding to any code execution in Phase 1.*

---

## Phase 0: Preliminary Subtask Generation (STOP AND WAIT)

**Objective:** Before executing any code, analyze the planned backend modifications stemming from the AI engine and generate additional necessary subtasks for both Ticket OBS-1 and Ticket OBS-7.

### Generated Subtasks Additions:

**For Ticket OBS-1 (Scaffold Rust proxy crate):**
* **OBS-1e [Configurable Redaction Parsing]:** Extend config loading to parse a `disabled_entities` array from `obscura.toml`. *(Context: Needed to support global configurable redaction overrides without modifying code).*
* **OBS-1f [Dynamic Resource Loader]:** Implement dynamic model and tokenizer startup loading using the `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` environment variables. Ensure the server strictly implements a **Fail-Closed** policy (crash on startup) if these critical assets are missing. *(Context: Needed to support Bring-Your-Own-Model features).*

**For Ticket OBS-7 (Implement request interception):**
* **OBS-7e [Dynamic Override Header Parsing]:** Parse the `X-Obscura-Skip-Redaction` HTTP header to support per-request, dynamic masking overrides. *(Context: Needed for clients who need to conditionally ignore redaction for specific payloads).*
* **OBS-7f [Dynamic Key Deserialization]:** Ensure the "Mapping Dictionary" structs that handle entity tracking serialization/deserialization between Python/Rust use dynamic string keys (e.g., `["[DOB_1]"]`) rather than strictly typed Rust enums. *(Context: Needed to support transparent categorization expansion without crashing on unknown PII types).*
* **OBS-7g [Redaction Configuration Merger]:** Implement logic to merge the global `disabled_entities` config with the dynamic `X-Obscura-Skip-Redaction` header overrides before invoking the ML engine inference pipeline. *(Context: Required to pass the unified skip list to the engine).*

> 🛑 **WAIT:** Pause execution here. User must review the above new subtasks, add them to the Jira board, and explicitly confirm before proceeding to Phase 1.

---

## Phase 1: Scaffold Rust Proxy Crate (Ticket OBS-1)

**Branch Focus:** `SCRUM-160-OBS-1-Scaffold-Rust-proxy-crate-with-tokio-`

**Objective:** Initialize the Rust workspace with a basic async HTTP server using `tokio` and `hyper`. Bind to `0.0.0.0:8080`, accept incoming requests, and return `200 OK` with static JSON to serve as the application skeleton. Code must pass `cargo clippy` with zero warnings natively.

**Subtasks to Execute:**
* **OBS-1a [Workspace Initialization]:** Initialize Cargo workspace with `proxy`, `inference`, and `config` crates.
* **OBS-1b [Health Server]:** Implement basic `hyper` server with a `/health` endpoint.
* **OBS-1c [Config Loader]:** Implement `obscura.toml` configuration loading using the `toml` crate.
* **OBS-1d [Smoke Tests]:** Write initial smoke tests.
* **OBS-1e [Configurable Redaction Parsing]:** Extend config loading to parse `disabled_entities` array from `obscura.toml`.
* **OBS-1f [Dynamic Resource Loader]:** Implement dynamic model/tokenizer startup loading using `NER_MODEL_PATH` and `NER_TOKENIZER_PATH` with Fail-Closed policy on asset missing.

---

## Phase 2: Checkpoint 1 Verification

**Objective:** Verify the functionality and working order of Phase 1.
* Do not proceed until tests pass (`cargo test`).
* Verify the baseline server is operational and can cleanly load mocked configs/env vars without `unwrap()` panics.

---

## Phase 3: Implement Request Interception (Ticket OBS-7)

**Branch Focus:** `SCRUM-106-OBS-7-Implement-request-interception-and-forwarding`

**Objective:** Extend the proxy to intercept HTTP requests, extract the JSON body, and forward them to a configurable upstream LLM URL. Return the upstream response unchanged (pure pass-through, no redaction).

**Subtasks to Execute:**
* **OBS-7a [Hyper HTTP Client]:** Implement HTTP client using `hyper` for upstream requests.
* **OBS-7b [Header & Body Forwarding]:** Parse and forward request headers and body.
* **OBS-7c [Response Proxying]:** Return the upstream response to the original client.
* **OBS-7d [Upstream URL Configuration]:** Add a configurable upstream URL sourced from `obscura.toml`.
* **OBS-7e [Dynamic Override Header Parsing]:** Parse `X-Obscura-Skip-Redaction` HTTP header for overrides.
* **OBS-7f [Dynamic Key Deserialization]:** Implement the "Mapping Dictionary" structs with dynamic string keys.
* **OBS-7g [Redaction Configuration Merger]:** Merge global override config with header overrides.

---

## Phase 4: Checkpoint 2 Verification

**Objective:** Verify the functionality and working order of Phase 3. 
* Write unit/integration tests mimicking a mock LLM upstream.
* Do not proceed until the pure pass-through logic and header parsing are confirmed robust.

---

## Phase 5: AI Engine Modifications & Final Verification

**Objective:** Execute backend integration code edits based on the required AI engine features.
* Link the parsed pass-through logic to the `ort` or `tch-rs` inference bindings.
* Conduct final end-to-end verification of all integrated backend + ML components using synthetic mock payloads containing PII.
