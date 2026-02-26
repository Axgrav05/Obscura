# Weekly Progress Report

**Leader:** Arjun Agravat
**Project Name:** Obscura  
**Week #:** 2 | **Date Range:** 2/20 - 2/27

---

## Individual Status Updates

| Team Member | Tasks Completed | Planned Tasks (Next Week) | Issues / Challenges | Time Spent |
| :--- | :--- | :--- | :--- | :--- |
| **Arjun** | - Implemented synthetic PII dataset generator<br>- Built NER model evaluation harness<br>- Evaluated candidate models (distilbert, bert-base, stanford)<br>- Wrote and tested PIIEngine for redaction<br>- Completed Round 3 verification for SSN pipeline (serialization fixes, boundary tests) | - Integrate engine with Rust backend | HuggingFace file locks caused background download scripts to hang, requiring offline-mode execution. | 8 hrs |
| **Bigan** | - Completed OBS-4: Define repo structure and configuration standards <br>- Finalized monorepo directory layout (src/api, masking, llm, config, ml, models, deploy, context)<br>- Added documented obscura.toml with baseline configuration fields <br>- Standardized .gitignore and .dockerignore <br>- Configured pre-commit hooks (rustfmt, clippy, hygiene checks) <br>- Verified pre-commit passes on all files <br>- Opened PR and moved task to In Review | - Begin implementation of reversible masking module (OBS-5) - Begin implementation of reversible masking module (OBS-5) <br>- Define masking/unmasking interface within Rust backend <br>- Align configuration handling with backend entrypoint | - Initial setup required configuring Rust toolchain (cargo, rustfmt, clippy) to enable pre-commit hooks. Resolved and verified locally. | 4-6 hrs |
| **Kyle** | - Created a LLM redaction server/client demo to validate AWS Tech stack and ensure no issues with streaming LLM generation. <br><br> - Created a Github Action that builds the Docker image and uploads to Amazon ECR before notifying EC2 Instances to deploy.  | -Adjust CD/CI processs as program evolves. <br><br> -Adjust Infrastructure as program evolves <br><br> - Help other group members. | N/A - No Major issues Encountered. | 6-8 hrs |
| **Rainer** | - Initialize the Rust workspace with a basic async HTTP server using tokio and hyper. <br> - Set up Cargo workspace with proxy/inference/config<br> - Wrote smoke tests | - Work with Arjun on integrating the backend with everything | Smoke test return is successful, which is good. Under review, still needs to be integrated. | 4 hrs |
|**Eduardo** | - Set up Grafana Cloud workspace<br>- Configured Prometheus datasource<br>- Created Compliance, Redaction Detail, and Performance dashboards | - Implement /metrics endpoint in Rust proxy<br>- Add Prometheus counters and latency histogram<br>- Deploy Prometheus via Docker<br>- Configure scrape targets<br>- Verify metrics appear in Performance dashboard | - Grafana Cloud Free plan limited to one stack, auto-named after account rather than project, and restricted to a maximum of 3 team members. | 4 hrs |
| **Billy** | - Task A | - Task B | [Detail] | X hrs |

**Total Team Time Spent:** [Sum] Hours

---

## Weekly Summary
[Provide a high-level overview of the team's progress and the current state of the project.]

The ML team successfully finalized the hybrid BERT + regex pipeline for PII/PHI redaction, completing 3 rounds of verification. The engine correctly masks semantic entities via standard NER while enforcing structure/context disambiguation on SSNs, passing all 30 robust boundary tests with an F1 score of 1.00.
