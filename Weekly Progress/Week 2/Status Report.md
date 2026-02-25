# Weekly Progress Report

**Leader:** Arjun Agravat
**Project Name:** Obscura  
**Week #:** 2 | **Date Range:** 2/20 - 2/27

---

## Individual Status Updates

| Team Member | Tasks Completed | Planned Tasks (Next Week) | Issues / Challenges | Time Spent |
| :--- | :--- | :--- | :--- | :--- |
| **Arjun** | - Implemented synthetic PII dataset generator<br>- Built NER model evaluation harness<br>- Evaluated candidate models (distilbert, bert-base, stanford)<br>- Wrote and tested PIIEngine for redaction | - Integrate engine with Rust backend | HuggingFace file locks caused background download scripts to hang, requiring offline-mode execution. | 8 hrs |
| **Bigan** | - Completed OBS-4: Define repo structure and configuration standards <br>- Finalized monorepo directory layout (src/api, masking, llm, config, ml, models, deploy, context)<br>- Added documented obscura.toml with baseline configuration fields <br>- Standardized .gitignore and .dockerignore <br>- Configured pre-commit hooks (rustfmt, clippy, hygiene checks) <br>- Verified pre-commit passes on all files <br>- Opened PR and moved task to In Review | - Begin implementation of reversible masking module (OBS-5) - Begin implementation of reversible masking module (OBS-5) <br>- Define masking/unmasking interface within Rust backend <br>- Align configuration handling with backend entrypoint | - Initial setup required configuring Rust toolchain (cargo, rustfmt, clippy) to enable pre-commit hooks. Resolved and verified locally. | 4-6 hrs |
| **Kyle** | - Created a LLM redaction server/client demo to validate AWS Tech stack and ensure no issues with streaming LLM generation. <br><br> - Created a Github Action that builds the Docker image and uploads to Amazon ECR before notifying EC2 Instances to deploy.  | -Adjust CD/CI processs as program evolves. <br><br> -Adjust Infrastructure as program evolves <br><br> - Help other group members. | N/A - No Major issues Encountered. | 6-8 hrs |
| **Rainer** | - Task A | - Task B | [Detail] | X hrs |
|**Eduardo** | - Set up Grafana Cloud workspace<br>- Configured Prometheus datasource<br>- Created Compliance, Redaction Detail, and Performance dashboards | - Implement /metrics endpoint in Rust proxy<br>- Add Prometheus counters and latency histogram<br>- Deploy Prometheus via Docker<br>- Configure scrape targets<br>- Verify metrics appear in Performance dashboard | - Grafana Cloud Free plan limited to one stack, auto-named after account rather than project, and restricted to a maximum of 3 team members. | 4 hrs |
| **Billy** | - Task A | - Task B | [Detail] | X hrs |

**Total Team Time Spent:** [Sum] Hours

---

## Weekly Summary
[Provide a high-level overview of the team's progress and the current state of the project.]
