# Meeting Minutes - Week 5

## Action Items
- [ ] Lead (Rainier Pederson): Enhance the `SPEC.md` file with granular details, including a comprehensive list of all PII types and their specific redaction rules.
- [ ] Lead (Rainier Pederson): Port recent development work to the appropriate feature branch to ensure repository consistency.
- [ ] Team: Review and close out all pending Pull Requests to maintain a clean main branch.
- [ ] Arjun: Initiate the development of the Proprietary Code Redaction ML pipeline (CodeBERT).
- [ ] Arjun: Guide the integration of CodeBERT ONNX inference into the Rust Proxy.
- [ ] Bigan: Implement Server-Sent Events (SSE) re-emission (OBS-21) to support streaming responses, ensuring compatibility with the OpenAI Python SDK.
- [ ] Kyle: Finalize and deploy the Mid Term demo to the AWS infrastructure and continue experimenting with model accuracy.
- [ ] Rainier Pederson: Author a comprehensive report on HIPAA considerations across four key states (TX, CA, FL, NY) and evaluate EU/UK health data privacy restrictions.
- [ ] Billy: Implement Tower middleware for strict `Authorization: Bearer` header validation, including constant-time comparison for security.

## Key Discussion Points
- **Tokenization & Chunking:** Analyzed the character-to-token ratio and reviewed context-aware chunking strategies. Addressed "Context Chunking Leaks" where overlapping chunks caused entity overrides; implemented descending length sorting to prevent PII loss.
- **Architectural Bottlenecks:** Discussed the impact of the `ort` v2 ONNX runtime forcing a blocking Mutex on the inference engine, creating a bottleneck for Tokio's async throughput.
- **Namespace Collisions:** Resolved issues where multi-message prompts caused duplicate tokens (e.g., `[PERSON_1]`) to overwrite each other by implementing global tracking middleware.
- **Security & Fail-Closed Logic:** Reinforced the "fail-closed" mandate to block unredacted leaks from unrecognized schemas (e.g., legacy `{"prompt"}` formats).
- **Feature Expansion:** Explored functional extensions for the redaction engine and identified key roadmap items for the coming weeks.

## Risks and Blockers
- **Infrastructure Access:** Billy noted that EC2/security group details are pending for the audit report, requiring either direct AWS access or screenshots from teammates.
- **Model Initialization:** Addressed initial proxy startup failures caused by missing model artifacts and environment variables; documented the required setup for local development.
- **Branch Management:** Emphasized the importance of porting Rainier's work to the correct branch to maintain structural integrity.
