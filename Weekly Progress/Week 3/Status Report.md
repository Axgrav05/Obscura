
# Weekly Progress Report

  

**Leader:** Rainier Pederson

**Project Name:** Obscura

**Week #:** 3 | **Date Range:** 2/27 - 3/5

  

---

  

## Individual Status Updates

  

| Team Member | Tasks Completed | Planned Tasks (Next Week) | Issues / Challenges | Time Spent |

|:------------|:---------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------| :--- |

| **Arjun** | - Deprioritized real-time feedback loop integration due to high operational overhead and privacy constraints. Focus shifted to categorization expansion and BYOM.<br>- Fine-tuned BERT NER model on synthetic PII dataset (macro F1: 95.76%)<br>- Expanded PII/PHI regex coverage: added DOB, Credit Card, IPv4, US Passport detection patterns<br>- Implemented configurable redaction (`disabled_entities` parameter) for selective entity masking<br>- Built BYOM ONNX Exporter CLI for PyTorch-to-ONNX model conversion<br>- Wrote 36 new unit tests covering regex validation and redaction configuration | - Integrate ONNX-exported model with Rust proxy backend<br>- Validate latency on EC2 (currently 40ms macOS, target <=30ms)<br>- Add context-aware DOB scoring to reduce false positives | Deprioritized real-time feedback loop integration due to high operational overhead and privacy constraints. Focus shifted to categorization expansion and BYOM. | 10 hrs |

| **Bigan** | | | | |

| **Kyle** | <br>-Adjusted Build Process and Infrastructure as Project evolved, <br> - Reviewed PR's and helped resolve merge conflcits | Continue modifying build process / infrastructure as needed. <br> Working on the Demo + Getting it publicly accessible | 5-7 hours |

| **Rainer** | | | | |

| **Eduardo** | -Implemented Prometheus `/metrics` endpoint and `http_requests_total` counter (OBS-11) | -Install k6, run baseline load tests on EC2 proxy, measure req/s and latency, document results with graphs (OBS-17) | -Initial git push (HTTP 500) and authorization issues | 7 hrs |

| **Billy** | | | | |

  

**Total Team Time Spent:**

  

---

  

## Weekly Summary