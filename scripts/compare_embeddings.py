"""
Compare sparse vs dense embedding search results.

This script queries both the standard-works-v2 (sparse) and standard-works (dense)
indexes to demonstrate the difference in retrieval behavior.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()


def search_sparse(pc: Pinecone, query: str, top_k: int = 5) -> list[dict]:
    """Search using sparse embeddings (pinecone-sparse-english-v0)."""
    index = pc.Index("standard-works-v2")

    results = index.search_records(
        namespace="scriptures",
        query={"inputs": {"text": query}, "top_k": top_k},
    )

    return [
        {
            "reference": hit["fields"]["reference"],
            "score": round(hit["_score"], 2),
            "text": hit["fields"]["original_text"][:100] + "..."
            if len(hit["fields"]["original_text"]) > 100
            else hit["fields"]["original_text"],
        }
        for hit in results["result"]["hits"]
    ]


def search_dense(pc: Pinecone, openai_client: OpenAI, query: str, top_k: int = 5) -> list[dict]:
    """Search using dense embeddings (text-embedding-3-large)."""
    # Generate embedding with OpenAI
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
    )
    query_vector = response.data[0].embedding

    # Query Pinecone
    index = pc.Index("standard-works")
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "reference": match["metadata"]["reference"],
            "score": round(match["score"], 4),
            "text": match["metadata"]["text"][:100] + "..."
            if len(match["metadata"]["text"]) > 100
            else match["metadata"]["text"],
        }
        for match in results["matches"]
    ]


def print_results(title: str, results: list[dict]):
    """Print formatted results."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['reference']} (score: {r['score']})")
        print(f"   {r['text']}")


def main():
    # Initialize clients
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Test queries that highlight the difference
    queries = [
        # Exact phrase - sparse should excel
        "baptism by immersion",
        # Conceptual/semantic - dense should excel
        "overcoming temptation and sin",
        # Mixed - both should find relevant results
        "plan of salvation",
    ]

    for query in queries:
        print("\n" + "#" * 60)
        print(f" QUERY: \"{query}\"")
        print("#" * 60)

        # Search both indexes
        sparse_results = search_sparse(pc, query)
        dense_results = search_dense(pc, openai_client, query)

        print_results("SPARSE (keyword-focused)", sparse_results)
        print_results("DENSE (semantic)", dense_results)

        print("\n")


if __name__ == "__main__":
    main()
