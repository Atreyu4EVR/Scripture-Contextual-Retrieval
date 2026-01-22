# RAG Retrieval Comparison Report

**Run ID:** run_20260110_153805
**Generated:** 2026-01-10 15:45:45
**Total Queries:** 30

## Summary

This report compares three retrieval approaches:
1. **Legacy RAG** - Raw verse embeddings without chapter context
2. **Contextual RAG** - Verse embeddings with chapter summary context
3. **Contextual + Rerank** - Contextual retrieval with Cohere reranking

## Metrics Overview

| Metric | Legacy RAG | Contextual RAG | Contextual + Rerank |
| --- | --- | --- | --- |
| MRR | 0.949 | 0.983 | 0.936 |
| Precision@1 | 0.933 | 0.967 | 0.900 |
| Precision@3 | 0.911 | 0.944 | 0.900 |
| Precision@5 | 0.887 | 0.920 | 0.913 |
| Precision@10 | 0.880 | 0.893 | 0.000 |
| NDCG@5 | 0.856 | 0.877 | 0.950 |
| NDCG@10 | 0.933 | 0.940 | 0.000 |
| Avg Relevance | 2.457 | 2.433 | 2.547 |
| Avg Latency (ms) | 378.4 | 356.1 | 554.5 |


## Improvement Analysis

### Legacy RAG → Contextual RAG

- MRR: 0.949 → 0.983 (+3.6%)
- Avg Relevance: 2.457 → 2.433 (-0.9%)
- Precision@5: 0.887 → 0.920 (+3.8%)
- NDCG@10: 0.933 → 0.940 (+0.8%)
- Latency: 378.4ms → 356.1ms (-22.3ms)

### Contextual RAG → Contextual + Rerank

- MRR: 0.983 → 0.936 (-4.8%)
- Avg Relevance: 2.433 → 2.547 (+4.7%)
- Precision@5: 0.920 → 0.913 (-0.7%)
- NDCG@10: 0.940 → 0.000 (-100.0%)
- Latency: 356.1ms → 554.5ms (+198.4ms)

## Performance by Query Type

### LEGACY

- **cross_reference** (10 queries): Avg Relevance=2.660, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.980, MRR=0.848
- **thematic** (10 queries): Avg Relevance=2.730, MRR=1.000

### CONTEXTUAL

- **cross_reference** (10 queries): Avg Relevance=2.610, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.900, MRR=0.950
- **thematic** (10 queries): Avg Relevance=2.790, MRR=1.000

### CONTEXTUAL_RERANK

- **cross_reference** (10 queries): Avg Relevance=2.680, MRR=1.000
- **factual** (10 queries): Avg Relevance=2.100, MRR=0.808
- **thematic** (10 queries): Avg Relevance=2.860, MRR=1.000

## Challenging Queries

### LEGACY - Lowest Performing Queries

- **F001**: "Who was Nephi?..." (Avg Rel: 1.30)
- **F006**: "Where did Lehi's family travel after leaving Jerus..." (Avg Rel: 1.40)
- **F004**: "How many days was Jesus in the tomb?..." (Avg Rel: 1.70)

### CONTEXTUAL - Lowest Performing Queries

- **F008**: "Who built the ark?..." (Avg Rel: 1.50)
- **F001**: "Who was Nephi?..." (Avg Rel: 1.70)
- **F004**: "How many days was Jesus in the tomb?..." (Avg Rel: 1.80)

### CONTEXTUAL_RERANK - Lowest Performing Queries

- **F001**: "Who was Nephi?..." (Avg Rel: 1.20)
- **F002**: "Who was Nephi's father?..." (Avg Rel: 1.60)
- **F005**: "Who was the first prophet of the Restoration?..." (Avg Rel: 1.80)

## Methodology

- **Evaluation Method:** LLM-as-Judge using GPT-5.2
- **Relevance Scale:** 0-3 (0=Not Relevant, 3=Highly Relevant)
- **Relevance Threshold:** 2 (scores >= 2 considered relevant)
- **Statistical Test:** Paired t-test (p < 0.05 for significance)

## Files

- Raw results: `evaluations/results/raw/run_20260110_153805/`
- Judgments: `evaluations/results/evaluated/run_20260110_153805/judgments.json`
- Metrics: `evaluations/results/evaluated/run_20260110_153805/metrics.json`