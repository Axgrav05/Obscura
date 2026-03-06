
# Weekly Progress Report

  

**Leader:** Rainier Pederson

**Project Name:** Obscura

**Week #:** 3 | **Date Range:** 2/27 - 3/5

  

---

  

## Individual Status Updates

  

| Team Member | Tasks Completed | Planned Tasks (Next Week) | Issues / Challenges | Time Spent |
|:------------|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------| :--- |
| **Arjun** | - Deprioritized real-time feedback loop integration due to high operational overhead and privacy constraints. Focus shifted to categorization expansion and BYOM.<br>- Fine-tuned BERT NER model on synthetic PII dataset (macro F1: 95.76%)<br>- Expanded PII/PHI regex coverage: added DOB, Credit Card, IPv4, US Passport detection patterns<br>- Implemented configurable redaction (`disabled_entities` parameter) for selective entity masking<br>- Built BYOM ONNX Exporter CLI for PyTorch-to-ONNX model conversion<br>- Wrote 36 new unit tests covering regex validation and redaction configuration<br>- Reviewed and validated architectural implementations for OBS-1, OBS-7, and OBS-12 proxy backend features | - Integrate ONNX-exported model with Rust proxy backend<br>- Validate latency on EC2 (currently 40ms macOS, target <=30ms)<br>- Add context-aware DOB scoring to reduce false positives | Deprioritized real-time feedback loop integration due to high operational overhead and privacy constraints. Focus shifted to categorization expansion and BYOM. | 21 hrs |
| **Bigan** | - Implemented OpenAI provider adapter (OBS-10) for the proxy system<br>- Defined provider adapter trait and implemented request/response text extraction from OpenAI chat completion payloads<br>- Added support for parsing streaming delta responses<br>- Wrote unit tests with sample payloads to validate extraction logic | - Begin OBS-16: Midterm demo integration (end-to-end test)<br>- Develop Python demo script to send requests with PII through the deployed proxy<br>- Verify correct PII redaction before upstream request and restoration after response<br>- Prepare script as the midterm demo artifact | Initial learning curve understanding proxy architecture and integration points between provider adapters and request pipeline | 8 hrs |
| **Kyle** | - Adjusted Build Process and Infrastructure as Project evolved<br>- Reviewed PRs and helped resolve merge conflicts | - Continue modifying build process / infrastructure as needed<br>- Working on the Demo + Getting it publicly accessible | | 5-7 hrs |
| **Rainer** | - Scaffolded foundational Rust proxy crate with `tokio`, Hyper health server, and `obscura.toml` configuration loader (OBS-1)<br>- Implemented HTTP request interception, dynamic header parsing (`X-Obscura-Skip-Redaction`), and upstream LLM forwarding (OBS-7)<br>- Designed authentication middleware architecture using Tower, establishing fail-closed specs and strict error response schemas (OBS-12) | - Implement Tower authentication middleware into main proxy routing pipeline<br>- Finalize dynamic key injection security handling | Navigating hyper v1.0 migration and Tower compatibility for global middleware interception layers | 10-12 hrs |
| **Eduardo** | - Implemented Prometheus `/metrics` endpoint and `http_requests_total` counter (OBS-11) | - Install k6, run baseline load tests on EC2 proxy, measure req/s and latency, document results with graphs (OBS-17) | - Initial git push (HTTP 500) and authorization issues | 7 hrs |
| **Billy** | Defined Tower auth middleware plan for Obscura: validate `Authorization: Bearer <token>` against `OBSCURA_API_KEY` (env var), return `401` when missing/malformed, return `403` when invalid, and use constant-time comparison for token validation. | Implement Tower middleware for auth (**OBS-22a**).<br>Add constant-time key comparison (**OBS-22b**).<br>Write tests for 401, 403, and 200 paths (**OBS-22c**). | 5 hrs|

  

**Total Team Time Spent:**

  

---

  

## Weekly Summary
