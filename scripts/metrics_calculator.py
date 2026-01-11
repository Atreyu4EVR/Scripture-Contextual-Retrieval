"""
Academic metrics calculator for RAG evaluation.

Computes standard Information Retrieval metrics:
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Average Relevance Score
- Latency statistics
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from scipy import stats


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query: str
    method: str
    precision_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    avg_relevance: float = 0.0
    latency_ms: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries for a method."""
    method: str
    num_queries: int = 0
    mean_precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    mean_ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mean_avg_relevance: float = 0.0
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    total_relevant: int = 0
    total_retrieved: int = 0


@dataclass
class ComparisonResult:
    """Statistical comparison between two methods."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    improvement_pct: float
    p_value: float
    significant: bool


class MetricsCalculator:
    """Calculator for IR evaluation metrics."""

    def __init__(self, relevance_threshold: int = 2):
        """
        Initialize calculator.

        Args:
            relevance_threshold: Minimum score to consider a document relevant (0-3 scale)
        """
        self.relevance_threshold = relevance_threshold

    def _is_relevant(self, score: int) -> bool:
        """Check if a document is considered relevant."""
        return score >= self.relevance_threshold

    def precision_at_k(self, relevance_scores: list[int], k: int) -> float:
        """
        Calculate Precision@k.

        Args:
            relevance_scores: List of relevance scores in rank order
            k: Number of top results to consider

        Returns:
            Precision@k value (0.0 to 1.0)
        """
        if not relevance_scores or k <= 0:
            return 0.0

        top_k = relevance_scores[:k]
        relevant_count = sum(1 for s in top_k if self._is_relevant(s))
        return relevant_count / k

    def reciprocal_rank(self, relevance_scores: list[int]) -> float:
        """
        Calculate Reciprocal Rank.

        Args:
            relevance_scores: List of relevance scores in rank order

        Returns:
            Reciprocal rank (1/rank of first relevant doc, or 0 if none)
        """
        for rank, score in enumerate(relevance_scores, start=1):
            if self._is_relevant(score):
                return 1.0 / rank
        return 0.0

    def dcg_at_k(self, relevance_scores: list[int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at k.

        Uses the formula: sum((2^rel - 1) / log2(rank + 1))
        """
        if not relevance_scores or k <= 0:
            return 0.0

        top_k = relevance_scores[:k]
        dcg = 0.0

        for rank, score in enumerate(top_k, start=1):
            gain = (2 ** score) - 1
            discount = math.log2(rank + 1)
            dcg += gain / discount

        return dcg

    def ndcg_at_k(self, relevance_scores: list[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.

        Args:
            relevance_scores: List of relevance scores in rank order
            k: Number of top results to consider

        Returns:
            NDCG@k value (0.0 to 1.0)
        """
        if not relevance_scores or k <= 0:
            return 0.0

        dcg = self.dcg_at_k(relevance_scores, k)

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = self.dcg_at_k(ideal_scores, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_query_metrics(
        self,
        query_id: str,
        query: str,
        method: str,
        relevance_scores: list[int],
        latency_ms: float,
        k_values: list[int] = [1, 3, 5, 10],
    ) -> QueryMetrics:
        """
        Calculate all metrics for a single query.

        Args:
            query_id: Unique query identifier
            query: Query text
            method: Retrieval method name
            relevance_scores: List of relevance scores in rank order
            latency_ms: Retrieval latency in milliseconds
            k_values: List of k values for Precision@k and NDCG@k

        Returns:
            QueryMetrics object with all computed metrics
        """
        metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            method=method,
            latency_ms=latency_ms,
            num_retrieved=len(relevance_scores),
            num_relevant=sum(1 for s in relevance_scores if self._is_relevant(s)),
        )

        for k in k_values:
            if k <= len(relevance_scores):
                metrics.precision_at_k[k] = self.precision_at_k(relevance_scores, k)
                metrics.ndcg_at_k[k] = self.ndcg_at_k(relevance_scores, k)

        metrics.reciprocal_rank = self.reciprocal_rank(relevance_scores)

        if relevance_scores:
            metrics.avg_relevance = sum(relevance_scores) / len(relevance_scores)

        return metrics

    def aggregate_metrics(
        self,
        query_metrics_list: list[QueryMetrics],
    ) -> AggregateMetrics:
        """
        Aggregate metrics across multiple queries.

        Args:
            query_metrics_list: List of QueryMetrics objects for the same method

        Returns:
            AggregateMetrics with means and totals
        """
        if not query_metrics_list:
            return AggregateMetrics(method="unknown")

        method = query_metrics_list[0].method
        num_queries = len(query_metrics_list)

        k_values = set()
        for qm in query_metrics_list:
            k_values.update(qm.precision_at_k.keys())

        mean_precision_at_k = {}
        mean_ndcg_at_k = {}

        for k in sorted(k_values):
            precisions = [qm.precision_at_k.get(k, 0) for qm in query_metrics_list]
            ndcgs = [qm.ndcg_at_k.get(k, 0) for qm in query_metrics_list]

            mean_precision_at_k[k] = sum(precisions) / num_queries
            mean_ndcg_at_k[k] = sum(ndcgs) / num_queries

        latencies = [qm.latency_ms for qm in query_metrics_list]
        reciprocal_ranks = [qm.reciprocal_rank for qm in query_metrics_list]
        avg_relevances = [qm.avg_relevance for qm in query_metrics_list]

        return AggregateMetrics(
            method=method,
            num_queries=num_queries,
            mean_precision_at_k=mean_precision_at_k,
            mrr=sum(reciprocal_ranks) / num_queries,
            mean_ndcg_at_k=mean_ndcg_at_k,
            mean_avg_relevance=sum(avg_relevances) / num_queries,
            mean_latency_ms=sum(latencies) / num_queries,
            std_latency_ms=(sum((x - sum(latencies)/num_queries)**2 for x in latencies) / num_queries) ** 0.5,
            total_relevant=sum(qm.num_relevant for qm in query_metrics_list),
            total_retrieved=sum(qm.num_retrieved for qm in query_metrics_list),
        )

    def compare_methods(
        self,
        metrics_a: list[QueryMetrics],
        metrics_b: list[QueryMetrics],
        metric_name: str = "avg_relevance",
        alpha: float = 0.05,
    ) -> ComparisonResult:
        """
        Perform statistical comparison between two methods.

        Uses paired t-test for significance testing.

        Args:
            metrics_a: List of QueryMetrics for method A
            metrics_b: List of QueryMetrics for method B
            metric_name: Which metric to compare
            alpha: Significance level

        Returns:
            ComparisonResult with statistical analysis
        """
        values_a = []
        values_b = []

        for ma, mb in zip(metrics_a, metrics_b):
            if metric_name == "avg_relevance":
                values_a.append(ma.avg_relevance)
                values_b.append(mb.avg_relevance)
            elif metric_name == "reciprocal_rank":
                values_a.append(ma.reciprocal_rank)
                values_b.append(mb.reciprocal_rank)
            elif metric_name.startswith("precision_at_"):
                k = int(metric_name.split("_")[-1])
                values_a.append(ma.precision_at_k.get(k, 0))
                values_b.append(mb.precision_at_k.get(k, 0))
            elif metric_name.startswith("ndcg_at_"):
                k = int(metric_name.split("_")[-1])
                values_a.append(ma.ndcg_at_k.get(k, 0))
                values_b.append(mb.ndcg_at_k.get(k, 0))

        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)

        if mean_a > 0:
            improvement_pct = ((mean_b - mean_a) / mean_a) * 100
        else:
            improvement_pct = 0.0

        _, p_value = stats.ttest_rel(values_a, values_b)

        return ComparisonResult(
            method_a=metrics_a[0].method if metrics_a else "unknown",
            method_b=metrics_b[0].method if metrics_b else "unknown",
            metric=metric_name,
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            improvement_pct=float(improvement_pct),
            p_value=float(p_value),
            significant=bool(p_value < alpha),
        )


def main():
    """Test the metrics calculator."""
    calc = MetricsCalculator(relevance_threshold=2)

    test_scores = [3, 2, 1, 3, 0, 2, 1, 0, 0, 1]

    print("Testing Metrics Calculator")
    print("=" * 50)
    print(f"Relevance scores: {test_scores}")
    print(f"Relevance threshold: {calc.relevance_threshold}\n")

    for k in [1, 3, 5, 10]:
        p_at_k = calc.precision_at_k(test_scores, k)
        ndcg = calc.ndcg_at_k(test_scores, k)
        print(f"Precision@{k}: {p_at_k:.3f}")
        print(f"NDCG@{k}: {ndcg:.3f}")
        print()

    rr = calc.reciprocal_rank(test_scores)
    print(f"Reciprocal Rank: {rr:.3f}")


if __name__ == "__main__":
    main()
