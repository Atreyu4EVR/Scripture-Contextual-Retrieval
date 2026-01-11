"""
Main evaluation runner for RAG comparison testing.

Orchestrates the full evaluation pipeline:
1. Load test queries
2. Run all three retrieval methods
3. Evaluate with LLM-as-judge
4. Calculate metrics
5. Generate comparison report
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.retrieval_pipeline import RetrievalPipeline, RetrievalResult
from scripts.llm_judge import LLMJudge, RelevanceJudgment
from scripts.metrics_calculator import MetricsCalculator, QueryMetrics, AggregateMetrics


load_dotenv()

EVALUATIONS_DIR = Path(__file__).parent.parent / "evaluations"
CONFIG_PATH = EVALUATIONS_DIR / "config" / "evaluation_config.json"
QUERIES_PATH = EVALUATIONS_DIR / "queries" / "test_queries.json"
RESULTS_DIR = EVALUATIONS_DIR / "results"
REPORTS_DIR = EVALUATIONS_DIR / "reports"

MAX_WORKERS = 20


def load_config() -> dict:
    """Load evaluation configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries() -> list[dict]:
    """Load test queries."""
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def create_run_directory() -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    raw_dir = RESULTS_DIR / "raw" / run_id
    evaluated_dir = RESULTS_DIR / "evaluated" / run_id

    raw_dir.mkdir(parents=True, exist_ok=True)
    evaluated_dir.mkdir(parents=True, exist_ok=True)

    return run_id, raw_dir, evaluated_dir


def run_retrieval(
    pipeline: RetrievalPipeline,
    queries: list[dict],
    config: dict,
) -> dict[str, list[RetrievalResult]]:
    """Run all retrieval methods on all queries."""
    top_k = config.get("top_k", 20)
    rerank_initial_k = config.get("rerank_initial_k", 150)
    rerank_top_n = config.get("rerank_top_n", 20)

    results = {
        "legacy": [],
        "contextual": [],
        "contextual_rerank": [],
        "contextual_custom_rerank": [],  # Custom scripture reranker
        "hybrid_rrf": [],  # Hybrid sparse+dense with RRF
        "hybrid_rerank": [],  # Hybrid + Pinecone reranking
    }

    print(f"\nRunning retrieval on {len(queries)} queries...")
    print(f"  Config: top_k={top_k}, rerank_initial_k={rerank_initial_k}, rerank_top_n={rerank_top_n}")
    print(f"  Custom reranker: {'enabled' if pipeline.custom_reranker else 'disabled'}")
    print(f"  Hybrid methods: enabled")

    for i, query_data in enumerate(queries):
        query = query_data["query"]
        query_id = query_data["id"]

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processing query {i + 1}/{len(queries)}: {query_id}")

        all_results = pipeline.retrieve_all(query, top_k, rerank_initial_k, rerank_top_n)

        for method, result in all_results.items():
            results[method].append({
                "query_id": query_id,
                "query": query,
                "query_type": query_data.get("type", "unknown"),
                "result": result,
            })

    return results


def evaluate_results(
    judge: LLMJudge,
    retrieval_results: dict[str, list],
) -> dict[str, list]:
    """Run LLM-as-judge evaluation on all results."""
    evaluations = {}

    for method, results in retrieval_results.items():
        print(f"\nEvaluating {method} results...")
        method_evaluations = []

        for i, item in enumerate(results):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Evaluating query {i + 1}/{len(results)}")

            query = item["query"]
            result = item["result"]
            documents = result.documents

            judgments = judge.judge_relevance_batch(query, documents)

            method_evaluations.append({
                "query_id": item["query_id"],
                "query": query,
                "query_type": item["query_type"],
                "method": method,
                "latency_ms": result.latency_ms,
                "judgments": [
                    {
                        "document_id": j.document_id,
                        "reference": j.reference,
                        "score": j.score,
                        "reasoning": j.reasoning,
                    }
                    for j in judgments
                ],
            })

        evaluations[method] = method_evaluations

    return evaluations


def calculate_all_metrics(
    evaluations: dict[str, list],
    config: dict,
) -> dict[str, dict]:
    """Calculate metrics for all methods."""
    relevance_threshold = config.get("relevance_threshold", 2)
    precision_k_values = config.get("metrics", {}).get("precision_k_values", [1, 3, 5, 10])
    ndcg_k_values = config.get("metrics", {}).get("ndcg_k_values", [5, 10])

    k_values = list(set(precision_k_values + ndcg_k_values))

    calculator = MetricsCalculator(relevance_threshold=relevance_threshold)

    all_metrics = {}

    for method, method_evals in evaluations.items():
        query_metrics_list = []

        for eval_item in method_evals:
            relevance_scores = [j["score"] for j in eval_item["judgments"]]

            qm = calculator.calculate_query_metrics(
                query_id=eval_item["query_id"],
                query=eval_item["query"],
                method=method,
                relevance_scores=relevance_scores,
                latency_ms=eval_item["latency_ms"],
                k_values=k_values,
            )
            query_metrics_list.append(qm)

        aggregate = calculator.aggregate_metrics(query_metrics_list)

        all_metrics[method] = {
            "query_metrics": [asdict(qm) for qm in query_metrics_list],
            "aggregate": asdict(aggregate),
        }

    return all_metrics


def run_statistical_comparisons(
    evaluations: dict[str, list],
    config: dict,
) -> list[dict]:
    """Run statistical comparisons between methods."""
    relevance_threshold = config.get("relevance_threshold", 2)
    calculator = MetricsCalculator(relevance_threshold=relevance_threshold)

    # Collect metrics by method
    method_metrics = {}

    for method, method_evals in evaluations.items():
        method_metrics[method] = []
        for eval_item in method_evals:
            relevance_scores = [j["score"] for j in eval_item["judgments"]]

            qm = calculator.calculate_query_metrics(
                query_id=eval_item["query_id"],
                query=eval_item["query"],
                method=method,
                relevance_scores=relevance_scores,
                latency_ms=eval_item["latency_ms"],
                k_values=[1, 3, 5, 10],
            )
            method_metrics[method].append(qm)

    comparisons = []
    metrics_to_compare = ["avg_relevance", "reciprocal_rank", "precision_at_5", "ndcg_at_10"]

    # Define comparison pairs (method_a vs method_b)
    comparison_pairs = [
        ("legacy", "contextual"),
        ("contextual", "contextual_rerank"),
        ("contextual_rerank", "hybrid_rrf"),
        ("contextual_rerank", "hybrid_rerank"),
        ("hybrid_rrf", "hybrid_rerank"),
    ]

    for metric in metrics_to_compare:
        for method_a, method_b in comparison_pairs:
            if method_a in method_metrics and method_b in method_metrics:
                if method_metrics[method_a] and method_metrics[method_b]:
                    comp = calculator.compare_methods(
                        method_metrics[method_a],
                        method_metrics[method_b],
                        metric,
                    )
                    comparisons.append(asdict(comp))

    return comparisons


def save_results(
    run_id: str,
    raw_dir: Path,
    evaluated_dir: Path,
    retrieval_results: dict,
    evaluations: dict,
    metrics: dict,
    comparisons: list,
):
    """Save all results to files."""
    for method, results in retrieval_results.items():
        serializable = []
        for item in results:
            serializable.append({
                "query_id": item["query_id"],
                "query": item["query"],
                "query_type": item["query_type"],
                "latency_ms": item["result"].latency_ms,
                "documents": item["result"].documents,
            })

        filepath = raw_dir / f"{method}_results.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    judgments_path = evaluated_dir / "judgments.json"
    with open(judgments_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)

    metrics_path = evaluated_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_queries": len(next(iter(evaluations.values()))),
        },
        "results": {
            method: data["aggregate"]
            for method, data in metrics.items()
        },
        "statistical_comparisons": comparisons,
    }

    report_path = REPORTS_DIR / f"comparison_report_{run_id.replace('run_', '')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Raw results: {raw_dir}")
    print(f"  Evaluations: {evaluated_dir}")
    print(f"  Report: {report_path}")


def print_summary(metrics: dict, comparisons: list):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for method, data in metrics.items():
        agg = data["aggregate"]
        print(f"\n{method.upper()}")
        print(f"  MRR: {agg['mrr']:.3f}")
        print(f"  Precision@5: {agg['mean_precision_at_k'].get('5', 0):.3f}")
        print(f"  NDCG@10: {agg['mean_ndcg_at_k'].get('10', 0):.3f}")
        print(f"  Avg Relevance: {agg['mean_avg_relevance']:.3f}")
        print(f"  Avg Latency: {agg['mean_latency_ms']:.1f}ms")

    print("\n" + "-" * 60)
    print("STATISTICAL COMPARISONS")
    print("-" * 60)

    for comp in comparisons:
        if comp["metric"] == "avg_relevance":
            sig = "*" if comp["significant"] else ""
            print(f"\n{comp['method_a']} vs {comp['method_b']} ({comp['metric']})")
            print(f"  {comp['mean_a']:.3f} → {comp['mean_b']:.3f} ({comp['improvement_pct']:+.1f}%)")
            print(f"  p-value: {comp['p_value']:.4f} {sig}")


def main():
    """Run the full evaluation pipeline."""
    print("RAG Retrieval Comparison Evaluation")
    print("=" * 60)

    config = load_config()
    queries = load_queries()

    print(f"Loaded {len(queries)} test queries")
    print(f"Config: top_k={config['top_k']}, rerank_top_n={config['rerank_top_n']}")

    run_id, raw_dir, evaluated_dir = create_run_directory()
    print(f"Run ID: {run_id}")

    pipeline = RetrievalPipeline()
    judge = LLMJudge()

    retrieval_results = run_retrieval(pipeline, queries, config)

    evaluations = evaluate_results(judge, retrieval_results)

    print("\nCalculating metrics...")
    metrics = calculate_all_metrics(evaluations, config)

    print("Running statistical comparisons...")
    comparisons = run_statistical_comparisons(evaluations, config)

    save_results(
        run_id,
        raw_dir,
        evaluated_dir,
        retrieval_results,
        evaluations,
        metrics,
        comparisons,
    )

    print_summary(metrics, comparisons)

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
