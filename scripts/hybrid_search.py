"""
Hybrid Search: Combining Sparse + Dense embeddings.

This script demonstrates hybrid search by querying both indexes
and using Reciprocal Rank Fusion (RRF) to combine results.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()


def search_sparse(pc: Pinecone, query: str, top_k: int = 20) -> list[dict]:
    """Search using sparse embeddings."""
    index = pc.Index("standard-works-v2")

    results = index.search_records(
        namespace="scriptures",
        query={"inputs": {"text": query}, "top_k": top_k},
    )

    return [
        {
            "id": hit["_id"],
            "reference": hit["fields"]["reference"],
            "score": hit["_score"],
            "text": hit["fields"]["original_text"],
            "source": "sparse",
        }
        for hit in results["result"]["hits"]
    ]


def search_dense(pc: Pinecone, openai_client: OpenAI, query: str, top_k: int = 20) -> list[dict]:
    """Search using dense embeddings."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
    )
    query_vector = response.data[0].embedding

    index = pc.Index("standard-works")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "id": match["id"],
            "reference": match["metadata"]["reference"],
            "score": match["score"],
            "text": match["metadata"]["text"],
            "source": "dense",
        }
        for match in results["matches"]
    ]


def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF Score = sum(weight / (k + rank)) for each result list

    Args:
        results_list: List of result lists from different retrievers
        k: Constant to prevent high scores for top results (default 60)
        weights: Optional weights for each result list

    Returns:
        Combined and re-ranked results
    """
    if weights is None:
        weights = [1.0] * len(results_list)

    # Build fusion scores
    fusion_scores: dict[str, dict] = {}

    for weight, results in zip(weights, results_list):
        for rank, result in enumerate(results, 1):
            ref = result["reference"]
            rrf_score = weight / (k + rank)

            if ref not in fusion_scores:
                fusion_scores[ref] = {
                    "reference": ref,
                    "text": result["text"],
                    "rrf_score": 0,
                    "sources": [],
                    "ranks": {},
                }

            fusion_scores[ref]["rrf_score"] += rrf_score
            fusion_scores[ref]["sources"].append(result["source"])
            fusion_scores[ref]["ranks"][result["source"]] = rank

    # Sort by RRF score
    combined = sorted(fusion_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    return combined


def rerank_with_pinecone(pc: Pinecone, query: str, results: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank results using Pinecone's reranking model.
    """
    documents = [r["text"] for r in results]

    reranked = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=True,
    )

    return [
        {
            "reference": results[r["index"]]["reference"],
            "text": results[r["index"]]["text"],
            "rerank_score": r["score"],
            "original_sources": results[r["index"]].get("sources", ["unknown"]),
        }
        for r in reranked.data
    ]


def print_results(title: str, results: list[dict], limit: int = 5):
    """Print formatted results."""
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print("=" * 70)

    for i, r in enumerate(results[:limit], 1):
        text_preview = r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"]

        if "rrf_score" in r:
            sources = ", ".join(set(r["sources"]))
            ranks = " | ".join(f"{k}: #{v}" for k, v in r["ranks"].items())
            print(f"\n{i}. {r['reference']} (RRF: {r['rrf_score']:.4f})")
            print(f"   Sources: [{sources}] | Ranks: [{ranks}]")
        elif "rerank_score" in r:
            sources = ", ".join(set(r["original_sources"]))
            print(f"\n{i}. {r['reference']} (rerank: {r['rerank_score']:.4f})")
            print(f"   Sources: [{sources}]")
        else:
            print(f"\n{i}. {r['reference']} (score: {r['score']:.4f})")

        print(f"   {text_preview}")


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    queries = [
        "baptism by immersion",
        "overcoming temptation and sin",
        "faith hope and charity",
    ]

    for query in queries:
        print("\n" + "#" * 70)
        print(f" QUERY: \"{query}\"")
        print("#" * 70)

        # Step 1: Get results from both retrievers
        sparse_results = search_sparse(pc, query, top_k=20)
        dense_results = search_dense(pc, openai_client, query, top_k=20)

        # Step 2: Combine with Reciprocal Rank Fusion
        # Weight dense slightly higher for semantic queries
        hybrid_results = reciprocal_rank_fusion(
            [sparse_results, dense_results],
            weights=[1.0, 1.2],  # Slight preference for dense/semantic
        )

        print_results("HYBRID (RRF Fusion: Sparse + Dense)", hybrid_results, limit=5)

        # Step 3: Rerank top results with neural reranker
        reranked = rerank_with_pinecone(pc, query, hybrid_results[:15], top_n=5)

        print_results("RERANKED (Pinecone Rerank v0)", reranked, limit=5)


if __name__ == "__main__":
    main()
