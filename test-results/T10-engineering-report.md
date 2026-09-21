# Litmus RAG Evaluation Harness ? Engineering Test Report

## Test Series Summary

| Test | Area | Result | Status |
|---|---|---:|---|
| T01 | Automated backend tests | 117/117 passed | PASS |
| T02 | V1 golden retrieval | Hit@4 98%, MRR 95%, Context Precision 56.5% | PASS |
| T03 | V3 Multi-Query | Hit@4 98%, MRR 95%, Context Precision 58.0% | PASS |
| T04 | V4 Decomposition | Hit@4 98%, MRR 95%, Context Precision 56.5% | PASS |
| T05 | Retrieval comparison | 0 recoveries, 0 regressions | PASS |
| T06 | Production API reliability | Health/API/Eval/CORS passed | PASS |
| T07 | Failure recovery | Recovery passed; invalid mode returns 500 | PASS* |
| T08 | Production latency | p50 597.5 ms, p95 1257.7 ms, p99 1861.14 ms | PASS |
| T09 | Production golden regression | 50/50 passed | PASS |

* T07 contains a documented validation defect: invalid mode produces HTTP 500 instead of a controlled client error.

## T01 ? Full Automated Backend Test Suite

- Tests collected: 117
- Tests passed: 117
- Tests failed: 0
- Pass rate: 100%
- Runtime: 51.72 seconds
- Warnings: 3

The suite covered retrieval, evaluation, API, chunking, embeddings, ingestion, LLM client, multi-query retrieval, decomposition, routing, SQL source, and V1?V6 smoke tests.

## T02 ? V1 Golden Retrieval Benchmark

- Dataset: 50 questions
- Retrieval K: 4
- Hit Rate@4: 98.00%
- MRR: 95.00%
- Context Precision: 56.50%

Q050 is an intentionally out-of-scope question with no reference document.

## T03 ? V3 Multi-Query Benchmark

- Dataset: 50 questions
- Retrieval K: 4
- Hit Rate@4: 98.00%
- MRR: 95.00%
- Context Precision: 58.00%

Only Q050 failed the source-based benchmark.

Compared with V1:
- Hit Rate@4: unchanged
- MRR: unchanged
- Context Precision: +1.5 percentage points

## T04 ? V4 Decomposition Benchmark

- Dataset: 50 questions
- Retrieval K: 4
- Maximum sub-questions: 4
- Hit Rate@4: 98.00%
- MRR: 95.00%
- Context Precision: 56.50%

Only Q050 failed the source-based benchmark.

## T05 ? Retrieval Improvement Analysis

V1, V3, and V4 were compared using the same 50-question benchmark.

- V1 failures: Q050
- V3 recovered over V1: none
- V4 recovered over V1: none
- V3 regressions versus V1: none
- V4 regressions versus V1: none
- V3 Context Precision improvement: +1.5 percentage points
- V4 aggregate metrics: unchanged from V1

These results are specific to this 50-question benchmark.

## T06 ? Production API Reliability

Production Cloud Run endpoint:

https://litmus-rag-api-516781219845.asia-south1.run.app

Verified:

- /health: HTTP 200
- /ask: successful production request
- /eval/latest: successful
- Vercel CORS: successful
- Production V1 evaluation metrics matched local evaluation metrics

A known V1 answer-grounding observation was identified: one job-search response stated that setting "Open to Work" was the first step, while the retrieved evidence did not directly establish that claim. This was preserved as an observation and not changed during the benchmark series.

## T07 ? Failure Recovery

### Invalid K

k=0 returned HTTP 422 with controlled FastAPI validation.

After correcting to k=4, the request succeeded.

### Invalid Mode

mode=invalid returned HTTP 500.

After correcting to mode=v1, the request succeeded.

### Known defect

Invalid mode should ideally be rejected during request validation with a controlled 4xx response instead of HTTP 500.

## T08 ? Production Latency

Endpoint: POST /ask

- Mode: v1
- K: 4
- Samples: 20
- Successful requests: 20/20

| Metric | Result |
|---|---:|
| Minimum | 298 ms |
| Average | 689.9 ms |
| p50 | 597.5 ms |
| p95 | 1257.7 ms |
| p99 | 1861.14 ms |
| Maximum | 2012 ms |

This was a 20-request single-question sample, not a large-scale load test or universal production SLA.

## T09 ? Production Golden Regression

- Dataset: 50 questions
- Production endpoint: Cloud Run /ask
- Mode: v1
- K: 4
- Passed: 50
- Failed: 0
- Pass rate: 100%

The regression verifies whether the expected reference document appears in the production top-4 retrieved sources.

Q050 has no reference document and is explicitly treated as out-of-scope by the regression runner.

Saved artifact:

	est-results/t09-production-regression.json

## Overall Engineering Findings

1. The automated backend suite passed all 117 tests.
2. V1, V3, and V4 achieved the same 98% Hit Rate@4 on the 50-question benchmark.
3. V3 improved Context Precision from 56.5% to 58.0% on this benchmark.
4. V4 did not improve the aggregate benchmark metrics.
5. Production retrieval matched the expected reference document for all 50 golden regression questions.
6. Production /ask completed all 20 latency samples successfully.
7. The API recovered successfully after invalid requests were corrected.
8. Invalid mode remains a validation defect because it produces HTTP 500.
9. One V1 answer-grounding observation was identified and intentionally left unchanged during the test series.
10. The latency results are a small sample and should not be presented as a production SLA.

## Test-Series Status

T01 through T09 completed.

Next engineering activity is documentation and project presentation, including README updates and carefully scoped CV-ready metrics.

## Evidence

Primary saved T09 artifact:

	est-results/t09-production-regression.json

T01?T08 results are based on the recorded test executions completed during this engineering test series.
