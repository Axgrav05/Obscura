# Obscura — Vision & Scope

## Mission
Cloud-native, zero-trust security layer that redacts PII/PHI in real-time for enterprise AI orchestration.

## Problem
Data breaches average $4.45M in cost. Enterprises face "Membership Inference Attacks" where adversaries extract sensitive context from LLM outputs.

## Solution
A high-performance middleware proxy that masks sensitive entities using BERT-based NER before data leaves the secure perimeter.

## Scope
- Real-time redaction/masking of Names, SSNs, Medical IDs
- Reversible restoration for authorized end-users via local mapping dictionary
- Cloud-native deployment on Oracle Cloud Infrastructure (OCI)

## Team (6 members)
- Arjun — AI/ML Lead (Python ML pipeline, ONNX inference integration)
- Rainier — Backend Engineer (Rust proxy core, inference crate consumer)
- Bignan — Integration Engineer (Python↔Rust bridge)
- Eduardo — Infrastructure (OCI, Kubernetes, CI/CD)
- [Others as needed]
