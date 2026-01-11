"""
Report generator for RAG comparison evaluation results.

Generates human-readable reports and analysis from evaluation runs.
Can be run standalone to analyze existing results without re-running evaluations.
"""

import json
from datetime import datetime
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).parent.parent / "evaluations"
RESULTS_DIR = EVALUATIONS_DIR / "results"
REPORTS_DIR = EVALUATIONS_DIR / "reports"


def get_latest_run() -> str | None:
    """Find the most recent evaluation run."""
    evaluated_dir = RESULTS_DIR / "evaluated"
    if not evaluated_dir.exists():
        return None

    runs = sorted(evaluated_dir.iterdir(), reverse=True)
    if runs:
        return runs[0].name
    return None


def load_run_data(run_id: str) -> dict:
    """Load all data for a specific run."""
    evaluated_dir = RESULTS_DIR / "evaluated" / run_id
    raw_dir = RESULTS_DIR / "raw" / run_id

    data = {"run_id": run_id}

    judgments_path = evaluated_dir / "judgments.json"
    if judgments_path.exists():
        with open(judgments_path, "r", encoding="utf-8") as f:
            data["judgments"] = json.load(f)

    metrics_path = evaluated_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            data["metrics"] = json.load(f)

    for method in ["legacy", "contextual", "contextual_rerank"]:
        raw_path = raw_dir / f"{method}_results.json"
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                data[f"{method}_raw"] = json.load(f)

    return data


def generate_summary_table(metrics: dict) -> str:
    """Generate a markdown summary table of metrics."""
    headers = ["Metric", "Legacy RAG", "Contextual RAG", "Contextual + Rerank"]
    rows = []

    methods = ["legacy", "contextual", "contextual_rerank"]

    mrr_row = ["MRR"]
    for method in methods:
        if method in metrics:
            mrr_row.append(f"{metrics[method]['aggregate']['mrr']:.3f}")
        else:
            mrr_row.append("-")
    rows.append(mrr_row)

    for k in [1, 3, 5, 10]:
        p_row = [f"Precision@{k}"]
        for method in methods:
            if method in metrics:
                val = metrics[method]["aggregate"]["mean_precision_at_k"].get(str(k), 0)
                p_row.append(f"{val:.3f}")
            else:
                p_row.append("-")
        rows.append(p_row)

    for k in [5, 10]:
        ndcg_row = [f"NDCG@{k}"]
        for method in methods:
            if method in metrics:
                val = metrics[method]["aggregate"]["mean_ndcg_at_k"].get(str(k), 0)
                ndcg_row.append(f"{val:.3f}")
            else:
                ndcg_row.append("-")
        rows.append(ndcg_row)

    relevance_row = ["Avg Relevance"]
    for method in methods:
        if method in metrics:
            relevance_row.append(f"{metrics[method]['aggregate']['mean_avg_relevance']:.3f}")
        else:
            relevance_row.append("-")
    rows.append(relevance_row)

    latency_row = ["Avg Latency (ms)"]
    for method in methods:
        if method in metrics:
            latency_row.append(f"{metrics[method]['aggregate']['mean_latency_ms']:.1f}")
        else:
            latency_row.append("-")
    rows.append(latency_row)

    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    return table


def generate_query_type_breakdown(metrics: dict) -> str:
    """Generate breakdown by query type."""
    output = []

    for method, data in metrics.items():
        query_metrics = data.get("query_metrics", [])
        if not query_metrics:
            continue

        by_type = {}
        for qm in query_metrics:
            qtype = qm.get("query_id", "")[:1]
            if qtype == "F":
                qtype = "factual"
            elif qtype == "T":
                qtype = "thematic"
            elif qtype == "X":
                qtype = "cross_reference"
            else:
                qtype = "unknown"

            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(qm)

        output.append(f"\n### {method.upper()}\n")

        for qtype, qms in sorted(by_type.items()):
            avg_rel = sum(qm["avg_relevance"] for qm in qms) / len(qms)
            avg_mrr = sum(qm["reciprocal_rank"] for qm in qms) / len(qms)
            output.append(f"- **{qtype}** ({len(qms)} queries): Avg Relevance={avg_rel:.3f}, MRR={avg_mrr:.3f}")

    return "\n".join(output)


def generate_improvement_analysis(metrics: dict) -> str:
    """Generate improvement percentages between methods."""
    output = []

    methods = ["legacy", "contextual", "contextual_rerank"]
    method_names = ["Legacy RAG", "Contextual RAG", "Contextual + Rerank"]

    for i in range(len(methods) - 1):
        method_a = methods[i]
        method_b = methods[i + 1]

        if method_a not in metrics or method_b not in metrics:
            continue

        agg_a = metrics[method_a]["aggregate"]
        agg_b = metrics[method_b]["aggregate"]

        output.append(f"\n### {method_names[i]} → {method_names[i+1]}\n")

        mrr_a = agg_a["mrr"]
        mrr_b = agg_b["mrr"]
        if mrr_a > 0:
            mrr_imp = ((mrr_b - mrr_a) / mrr_a) * 100
            output.append(f"- MRR: {mrr_a:.3f} → {mrr_b:.3f} ({mrr_imp:+.1f}%)")

        rel_a = agg_a["mean_avg_relevance"]
        rel_b = agg_b["mean_avg_relevance"]
        if rel_a > 0:
            rel_imp = ((rel_b - rel_a) / rel_a) * 100
            output.append(f"- Avg Relevance: {rel_a:.3f} → {rel_b:.3f} ({rel_imp:+.1f}%)")

        p5_a = agg_a["mean_precision_at_k"].get("5", 0)
        p5_b = agg_b["mean_precision_at_k"].get("5", 0)
        if p5_a > 0:
            p5_imp = ((p5_b - p5_a) / p5_a) * 100
            output.append(f"- Precision@5: {p5_a:.3f} → {p5_b:.3f} ({p5_imp:+.1f}%)")

        ndcg_a = agg_a["mean_ndcg_at_k"].get("10", 0)
        ndcg_b = agg_b["mean_ndcg_at_k"].get("10", 0)
        if ndcg_a > 0:
            ndcg_imp = ((ndcg_b - ndcg_a) / ndcg_a) * 100
            output.append(f"- NDCG@10: {ndcg_a:.3f} → {ndcg_b:.3f} ({ndcg_imp:+.1f}%)")

        lat_a = agg_a["mean_latency_ms"]
        lat_b = agg_b["mean_latency_ms"]
        lat_diff = lat_b - lat_a
        output.append(f"- Latency: {lat_a:.1f}ms → {lat_b:.1f}ms ({lat_diff:+.1f}ms)")

    return "\n".join(output)


def generate_worst_queries(metrics: dict) -> str:
    """Identify queries where each method performed poorly."""
    output = []

    for method, data in metrics.items():
        query_metrics = data.get("query_metrics", [])
        if not query_metrics:
            continue

        sorted_qms = sorted(query_metrics, key=lambda x: x["avg_relevance"])
        worst_3 = sorted_qms[:3]

        output.append(f"\n### {method.upper()} - Lowest Performing Queries\n")
        for qm in worst_3:
            output.append(f"- **{qm['query_id']}**: \"{qm['query'][:50]}...\" (Avg Rel: {qm['avg_relevance']:.2f})")

    return "\n".join(output)


def generate_markdown_report(run_id: str, data: dict) -> str:
    """Generate a comprehensive markdown report."""
    metrics = data.get("metrics", {})

    report = []
    report.append(f"# RAG Retrieval Comparison Report")
    report.append(f"\n**Run ID:** {run_id}")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if metrics:
        num_queries = metrics.get("legacy", {}).get("aggregate", {}).get("num_queries", 0)
        report.append(f"**Total Queries:** {num_queries}")

    report.append("\n## Summary\n")
    report.append("This report compares three retrieval approaches:")
    report.append("1. **Legacy RAG** - Raw verse embeddings without chapter context")
    report.append("2. **Contextual RAG** - Verse embeddings with chapter summary context")
    report.append("3. **Contextual + Rerank** - Contextual retrieval with Cohere reranking")

    report.append("\n## Metrics Overview\n")
    if metrics:
        report.append(generate_summary_table(metrics))

    report.append("\n## Improvement Analysis")
    if metrics:
        report.append(generate_improvement_analysis(metrics))

    report.append("\n## Performance by Query Type")
    if metrics:
        report.append(generate_query_type_breakdown(metrics))

    report.append("\n## Challenging Queries")
    if metrics:
        report.append(generate_worst_queries(metrics))

    report.append("\n## Methodology\n")
    report.append("- **Evaluation Method:** LLM-as-Judge using GPT-5.2")
    report.append("- **Relevance Scale:** 0-3 (0=Not Relevant, 3=Highly Relevant)")
    report.append("- **Relevance Threshold:** 2 (scores >= 2 considered relevant)")
    report.append("- **Statistical Test:** Paired t-test (p < 0.05 for significance)")

    report.append("\n## Files\n")
    report.append(f"- Raw results: `evaluations/results/raw/{run_id}/`")
    report.append(f"- Judgments: `evaluations/results/evaluated/{run_id}/judgments.json`")
    report.append(f"- Metrics: `evaluations/results/evaluated/{run_id}/metrics.json`")

    return "\n".join(report)


def save_report(run_id: str, content: str, format: str = "md") -> Path:
    """Save report to file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = run_id.replace("run_", "")
    filename = f"comparison_report_{timestamp}.{format}"
    filepath = REPORTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def main():
    """Generate report for the latest or specified run."""
    import sys

    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = get_latest_run()

    if not run_id:
        print("No evaluation runs found.")
        print("Run 'python scripts/run_evaluation.py' first.")
        return

    print(f"Generating report for: {run_id}")

    data = load_run_data(run_id)

    if "metrics" not in data:
        print(f"Error: No metrics found for run {run_id}")
        return

    md_report = generate_markdown_report(run_id, data)
    md_path = save_report(run_id, md_report, "md")

    print(f"\nReport saved to: {md_path}")

    print("\n" + "=" * 60)
    print(md_report)


if __name__ == "__main__":
    main()
