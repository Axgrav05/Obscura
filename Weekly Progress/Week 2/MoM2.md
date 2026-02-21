# Meeting Minutes - Week 2
## Action Items
- [ ] Lead: Refactor the project specification document to reflect the middleware pivot.

- [ ] Kyle: Finalize the transition to AWS EC2 for the core hosting environment.

- [ ] Team: Align internal AI workflows (ChatGPT, Gemini, Claude) for development tasks.

- [ ] Lead: Finalize the architecture diagram for team reference.

## Key Discussion Points
- Infrastructure Pivot: Formally transitioned development from OCI to AWS EC2. This change ensures Obscura can support a wider target audience by not being restricted to OCI-specific environments.

- AI Engine Optimization: The team is currently benchmarking 4 distinct BERT models.

- Performance Goals: Establishing an F1 score baseline and testing small-scale models first to minimize RAM usage and latency.

## Risks and Blockers
- Hardware Constraints: Monitoring RAM and latency impact during the initial BERT model tests to ensure compatibility with standard cloud instances.
