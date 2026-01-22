# RAG Retrieval Comparison Report

**Run ID:** run_20260110_160638
**Generated:** 2026-01-10 16:14:10
**Total Queries:** 30

## Summary

This report compares three retrieval approaches:
1. **Legacy RAG** - Raw verse embeddings without chapter context
2. **Contextual RAG** - Verse embeddings with chapter summary context
3. **Contextual + Rerank** - Contextual retrieval with Cohere reranking

## Metrics Overview

| Metric | Legacy RAG | Contextual RAG | Contextual + Rerank |
| --- | --- | --- | --- |
| MRR | 0.958 | 0.983 | 0.967 |
| Precision@1 | 0.933 | 0.967 | 0.933 |
| Precision@3 | 0.878 | 0.956 | 0.933 |
| Precision@5 | 0.900 | 0.947 | 0.933 |
| Precision@10 | 0.843 | 0.880 | 0.000 |
| NDCG@5 | 0.873 | 0.879 | 0.959 |
| NDCG@10 | 0.936 | 0.941 | 0.000 |
| Avg Relevance | 2.327 | 2.430 | 2.567 |
| Avg Latency (ms) | 338.4 | 394.6 | 672.7 |


## Improvement Analysis

### Legacy RAG → Contextual RAG

- MRR: 0.958 → 0.983 (+2.6%)
- Avg Relevance: 2.327 → 2.430 (+4.4%)
- Precision@5: 0.900 → 0.947 (+5.2%)
- NDCG@10: 0.936 → 0.941 (+0.5%)
- Latency: 338.4ms → 394.6ms (+56.1ms)

### Contextual RAG → Contextual + Rerank

- MRR: 0.983 → 0.967 (-1.7%)
- Avg Relevance: 2.430 → 2.567 (+5.6%)
- Precision@5: 0.947 → 0.933 (-1.4%)
- NDCG@10: 0.941 → 0.000 (-100.0%)
- Latency: 394.6ms → 672.7ms (+278.2ms)

## Performance by Query Type

### LEGACY

- **cross_reference** (10 queries): Avg Relevance=2.610, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.690, MRR=0.875
- **thematic** (10 queries): Avg Relevance=2.680, MRR=1.000

### CONTEXTUAL

- **cross_reference** (10 queries): Avg Relevance=2.650, MRR=1.000
- **factual** (10 queries): Avg Relevance=1.860, MRR=0.950
- **thematic** (10 queries): Avg Relevance=2.780, MRR=1.000

### CONTEXTUAL_RERANK

- **cross_reference** (10 queries): Avg Relevance=2.740, MRR=1.000
- **factual** (10 queries): Avg Relevance=2.100, MRR=0.900
- **thematic** (10 queries): Avg Relevance=2.860, MRR=1.000

## Challenging Queries

### LEGACY - Lowest Performing Queries

- **F006**: "Where did Lehi's family travel after leaving Jerus..." (Avg Rel: 1.10)
- **F009**: "What is the Word of Wisdom?..." (Avg Rel: 1.20)
- **F002**: "Who was Nephi's father?..." (Avg Rel: 1.50)

### CONTEXTUAL - Lowest Performing Queries

- **F004**: "How many days was Jesus in the tomb?..." (Avg Rel: 1.60)
- **F008**: "Who built the ark?..." (Avg Rel: 1.70)
- **F002**: "Who was Nephi's father?..." (Avg Rel: 1.80)

### CONTEXTUAL_RERANK - Lowest Performing Queries

- **F001**: "Who was Nephi?..." (Avg Rel: 1.40)
- **F002**: "Who was Nephi's father?..." (Avg Rel: 1.80)
- **F005**: "Who was the first prophet of the Restoration?..." (Avg Rel: 1.80)

## Methodology

- **Evaluation Method:** LLM-as-Judge using GPT-5.2
- **Relevance Scale:** 0-3 (0=Not Relevant, 3=Highly Relevant)
- **Relevance Threshold:** 2 (scores >= 2 considered relevant)
- **Statistical Test:** Paired t-test (p < 0.05 for significance)

## Files

- Raw results: `evaluations/results/raw/run_20260110_160638/`
- Judgments: `evaluations/results/evaluated/run_20260110_160638/judgments.json`
- Metrics: `evaluations/results/evaluated/run_20260110_160638/metrics.json`