# Weekly Progress Report

**Leader:**  

**Project Name:** Obscura

**Week #:** 4 | **Date Range:** 

---

## Individual Status Updates

| Team Member | Tasks Completed | Planned Tasks (Next Week) | Issues / Challenges | Time Spent |
|:------------|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------| :--- |
| **Arjun** | - Conducted formal independent cross-reviews of ML Python implementations and OBS-13 ONNX export pipeline.<br>- Engineered detailed implementation blueprints for a new Proprietary Code Redaction model.<br>- Researched and defined explicit redaction criteria based on historical enterprise leaks (Apple, Samsung) and constrained strategies to Obscura's strict latency/RAM architecture.<br>- Extended PII detection engine to cover IPv6 addresses and international phone numbers (ITU-T E.164) with edge-case false-positive protections.<br>- Implemented Context-Aware Text Chunking into the NLP engine, resolving inference failures on long payloads by splitting at natural sentence boundaries with overlapping stride and global offset mapping.<br>- Reviewed and merged k6 baseline load testing framework (OBS-17). | - Begin development of the Proprietary Code Redaction ML pipeline (CodeBERT).<br>- Guide integration of CodeBERT ONNX inference into Rust Proxy. | Balancing strict <30ms latency budgets and <500MB RAM constraints while massively expanding ML pipeline capabilities. | 10-12 hrs |
| **Bigan** | | | | |
| **Kyle** | | | | |
| **Rainer** | - Integrated the ort crate <br> - Loaded the ONNX model <br> - re-wired the request pipeline <br> - Researched HIPPA guidelines (applicable in US) <br> - Began research on Texas guidelines as it pertains to health information and the sharing thereof (TLDR: we absolutely would be liable if information went through, which would be expensive) | - Write a report with considerations of HIPPA in four test states: Texas, California, Florida, and New York. See what all data has to be protected and what the pentalties for failing to protect this data are. <br> - These four states were chosen because they are the largest by population. They also have a nice split in that 2 of them are red and 2 of them are blue, and (liekly more relevant for the sharing of information) 2 of them are one-party consent states, while two of them are all-party consent states. I promise this is a normal legal sentence. <br> - Additionally consider EU/UK health data privacy restrictions. | - API Volatility vs. Concurrency: The ort v2 ONNX RC forced us to wrap the inference engine in a blocking Mutex, creating an unavoidable bottleneck on Tokio's async routing throughput. <br> -Context Chunking Leaks: Overlapping 512-token chunks caused partial entities to override full entities; had to redesign the array sorting logic to explicitly sort by length descending to prevent PII loss. <br> - JSON Array Namespace Collisions: Multi-message prompts caused duplicate [PERSON_1] tokens to overwrite each other in the HashMap; built a global tracking middleware to namespace tokens across the entire request body. <br> - Schema Security Risk: The proxy silently failed-open on unrecognized, non-chat payloads (like legacy {"prompt"} schemas); had to implement a strict fail-closed fallback to block unredacted leaks. <br> - No lawyer in the history of ever has written things normally. | 10 hrs|
| **Eduardo** | | | | |
| **Billy** | | | | |

**Total Team Time Spent:** 10-12 hrs

---

## Weekly Summary
