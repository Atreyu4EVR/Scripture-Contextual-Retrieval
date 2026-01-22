# RAG Retrieval Comparison Report

**Run ID:** run_20260110_165131
**Generated:** 2026-01-10 17:00:55
**Total Queries:** 30

## Summary

This report compares three retrieval approaches:
1. **Legacy RAG** - Raw verse embeddings without chapter context
2. **Contextual RAG** - Verse embeddings with chapter summary context
3. **Contextual + Rerank** - Contextual retrieval with Cohere reranking

## Metrics Overview

| Metric | Legacy RAG | Contextual RAG | Contextual + Rerank |
| --- | --- | --- | --- |
| MRR | 0.942 | 0.967 | 0.938 |
| Precision@1 | 0.900 | 0.933 | 0.900 |
| Precision@3 | 0.878 | 0.933 | 0.844 |
| Precision@5 | 0.893 | 0.913 | 0.813 |
| Precision@10 | 0.837 | 0.893 | 0.790 |
| NDCG@5 | 0.834 | 0.845 | 0.757 |
| NDCG@10 | 0.832 | 0.865 | 0.794 |
| Avg Relevance | 2.195 | 2.292 | 2.123 |
| Avg Latency (ms) | 341.4 | 346.7 | 565.0 |


## Improvement Analysis

### Legacy RAG → Contextual RAG

- MRR: 0.942 → 0.967 (+2.7%)
- Avg Relevance: 2.195 → 2.292 (+4.4%)
- Precision@5: 0.893 → 0.913 (+2.2%)
- NDCG@10: 0.832 → 0.865 (+3.9%)
- Latency: 341.4ms → 346.7ms (+5.3ms)

### Contextual RAG → Contextual + Rerank

- MRR: 0.967 → 0.938 (-3.0%)
- Avg Relevance: 2.292 → 2.123 (-7.3%)
- Precision@5: 0.913 → 0.813 (-10.9%)
- NDCG@10: 0.865 → 0.794 (-8.2%)
- Latency: 346.7ms → 565.0ms (+218.4ms)

## Performance by Query Type

### LEGACY

- **cross_reference** (10 queries): Avg Relevance=2.495, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.450, MRR=0.875
- **thematic** (10 queries): Avg Relevance=2.640, MRR=0.950

### CONTEXTUAL

- **cross_reference** (10 queries): Avg Relevance=2.515, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.665, MRR=0.900
- **thematic** (10 queries): Avg Relevance=2.695, MRR=1.000

### CONTEXTUAL_RERANK

- **cross_reference** (10 queries): Avg Relevance=2.435, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.425, MRR=0.814
- **thematic** (10 queries): Avg Relevance=2.510, MRR=1.000

## Challenging Queries

### LEGACY - Lowest Performing Queries

- **F009**: "What is the Word of Wisdom?..." (Avg Rel: 0.90)
- **F006**: "Where did Lehi's family travel after leaving Jerus..." (Avg Rel: 0.95)
- **F004**: "How many days was Jesus in the tomb?..." (Avg Rel: 1.35)

### CONTEXTUAL - Lowest Performing Queries

- **F004**: "How many days was Jesus in the tomb?..." (Avg Rel: 1.25)
- **F005**: "Who was the first prophet of the Restoration?..." (Avg Rel: 1.35)
- **F007**: "What are the names of the twelve apostles?..." (Avg Rel: 1.50)

### CONTEXTUAL_RERANK - Lowest Performing Queries

- **F008**: "Who built the ark?..." (Avg Rel: 0.95)
- **F006**: "Where did Lehi's family travel after leaving Jerus..." (Avg Rel: 1.00)
- **F009**: "What is the Word of Wisdom?..." (Avg Rel: 1.10)

## Methodology

- **Evaluation Method:** LLM-as-Judge using GPT-5.2
- **Relevance Scale:** 0-3 (0=Not Relevant, 3=Highly Relevant)
- **Relevance Threshold:** 2 (scores >= 2 considered relevant)
- **Statistical Test:** Paired t-test (p < 0.05 for significance)

## Files

- Raw results: `evaluations/results/raw/run_20260110_165131/`
- Judgments: `evaluations/results/evaluated/run_20260110_165131/judgments.json`
- Metrics: `evaluations/results/evaluated/run_20260110_165131/metrics.json`