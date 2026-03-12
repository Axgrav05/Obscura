# Baseline Load Test Results (Local Networking)

This document records the baseline performance of the Obscura proxy in a local environment. These metrics reflect the proxy's networking overhead (request forwarding and response handling) without the active ML redaction engine.

## Test Environment
- **Host**: Apple M1 (Local)
- **Upstream**: Local Python Mock Server (50ms base simulated latency)
- **Payload**: Synthetic JSON (Realistic PII)
- **Duration**: 30s per test

## Summary of Results

| Concurrency (VUs) | Avg Req Rate | Avg Latency | p95 Latency | Error Rate |
|-------------------|--------------|-------------|-------------|------------|
| 10                | 9.84 req/s   | 10.65ms     | 22.14ms     | 0.00%      |
| 50                | 48.73 req/s  | 18.51ms     | 64.48ms     | 0.00%      |
| 100               | 97.01 req/s  | 20.90ms     | 79.26ms     | 0.00%      |

## Analysis
- **Scalability**: The proxy scales linearly from 10 to 100 VUs with minimal increase in average latency.
- **Latency**: The average latency remains low (~20ms), well within the project target of <60ms.
- **Error Rate**: 0% error rate across all tested levels, confirming stability under concurrent networking load.

> [!NOTE]
> These results represent the **networking baseline**. The Next Phase of testing will involve integrating the ML redaction engine to measure the inference bottleneck.
