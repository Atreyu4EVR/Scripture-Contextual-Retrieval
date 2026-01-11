"""
Quick comparison of all retrieval methods.

Runs a set of test queries through all retrieval approaches and
compares results side-by-side without full LLM-as-judge evaluation.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.retrieval_pipeline import RetrievalPipeline


load_dotenv()

# Test queries covering different types
TEST_QUERIES = [
    # Exact phrase queries (sparse should excel)
    {"id": "exact_1", "query": "faith without works is dead", "type": "exact_phrase"},
    {"id": "exact_2", "query": "baptism by immersion", "type": "exact_phrase"},
    {"id": "exact_3", "query": "plan of salvation", "type": "exact_phrase"},

    # Semantic/conceptual queries (dense should excel)
    {"id": "semantic_1", "query": "How can I overcome temptation?", "type": "semantic"},
    {"id": "semantic_2", "query": "What does it mean to have a broken heart?", "type": "semantic"},
    {"id": "semantic_3", "query": "How should I treat my enemies?", "type": "semantic"},

    # Doctrinal queries (hybrid should excel)
    {"id": "doctrinal_1", "query": "What is the nature of God?", "type": "doctrinal"},
    {"id": "doctrinal_2", "query": "What happens after we die?", "type": "doctrinal"},
    {"id": "doctrinal_3", "query": "How do I receive the Holy Ghost?", "type": "doctrinal"},

    # Character queries
    {"id": "character_1", "query": "Who was Nephi and what did he do?", "type": "character"},
    {"id": "character_2", "query": "What did Jesus teach about prayer?", "type": "character"},
]


def calculate_overlap(results_a: list[dict], results_b: list[dict], k: int = 10) -> float:
    """Calculate overlap between two result sets at k."""
    refs_a = {doc["reference"] for doc in results_a[:k]}
    refs_b = {doc["reference"] for doc in results_b[:k]}

    if not refs_a or not refs_b:
        return 0.0

    intersection = refs_a & refs_b
    return len(intersection) / k


def format_top_results(documents: list[dict], n: int = 5) -> list[str]:
    """Format top n results for display."""
    formatted = []
    for doc in documents[:n]:
        ref = doc["reference"]
        score = doc["score"]
        in_both = " [BOTH]" if doc.get("in_both") else ""
        formatted.append(f"{ref} ({score:.4f}){in_both}")
    return formatted


def run_comparison():
    """Run comparison across all methods."""
    print("Loading retrieval pipeline...")
    pipeline = RetrievalPipeline(load_custom_reranker=False)

    print(f"\nRunning {len(TEST_QUERIES)} test queries across all methods...\n")

    all_results = []
    method_latencies = {}
    method_overlap_with_contextual_rerank = {}

    for i, query_data in enumerate(TEST_QUERIES):
        query = query_data["query"]
        query_id = query_data["id"]
        query_type = query_data["type"]

        print(f"[{i+1}/{len(TEST_QUERIES)}] {query_id}: {query[:50]}...")

        results = pipeline.retrieve_all(
            query,
            top_k=20,
            rerank_initial_k=150,
            rerank_top_n=20,
            include_custom_rerank=False,
            include_hybrid=True,
        )

        # Track latencies
        for method, result in results.items():
            if method not in method_latencies:
                method_latencies[method] = []
            method_latencies[method].append(result.latency_ms)

        # Calculate overlap with contextual_rerank (current best)
        baseline = results["contextual_rerank"].documents
        for method, result in results.items():
            if method != "contextual_rerank":
                if method not in method_overlap_with_contextual_rerank:
                    method_overlap_with_contextual_rerank[method] = []
                overlap = calculate_overlap(result.documents, baseline, k=10)
                method_overlap_with_contextual_rerank[method].append(overlap)

        all_results.append({
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "results": {
                method: {
                    "latency_ms": result.latency_ms,
                    "top_5": format_top_results(result.documents, 5),
                }
                for method, result in results.items()
            },
        })

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print("\nAverage Latency (ms):")
    print("-" * 40)
    for method in sorted(method_latencies.keys()):
        avg = sum(method_latencies[method]) / len(method_latencies[method])
        print(f"  {method:30s} {avg:8.1f} ms")

    print("\nAverage Overlap with contextual_rerank @ 10:")
    print("-" * 40)
    for method in sorted(method_overlap_with_contextual_rerank.keys()):
        overlaps = method_overlap_with_contextual_rerank[method]
        avg = sum(overlaps) / len(overlaps)
        print(f"  {method:30s} {avg*100:5.1f}%")

    # Print detailed results for a few queries
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON (Sample Queries)")
    print("=" * 80)

    sample_queries = ["exact_1", "semantic_1", "doctrinal_1"]
    for result in all_results:
        if result["query_id"] in sample_queries:
            print(f"\n--- {result['query_id']}: \"{result['query']}\" ---")
            for method, data in result["results"].items():
                print(f"\n  {method} ({data['latency_ms']:.0f}ms):")
                for ref in data["top_5"][:3]:
                    print(f"    - {ref}")

    # Save full results
    output_path = Path("evaluations/results/quick_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "queries": all_results,
            "summary": {
                "avg_latency_ms": {
                    method: sum(lats) / len(lats)
                    for method, lats in method_latencies.items()
                },
                "avg_overlap_with_contextual_rerank": {
                    method: sum(overlaps) / len(overlaps)
                    for method, overlaps in method_overlap_with_contextual_rerank.items()
                },
            },
        }, f, indent=2)

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_comparison()
