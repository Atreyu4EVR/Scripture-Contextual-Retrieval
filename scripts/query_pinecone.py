"""
Query Pinecone index for scripture search.

Searches the 'standard-works' index and returns documents with proper field mapping.
Maps metadata.pageContent to document pageContent field.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


load_dotenv()

INDEX_NAME = "standard-works"
RESULTS_FILE = Path(__file__).parent / "results.json"


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Generate embedding for query text."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
    )
    return response.data[0].embedding


def query_index(index, query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Query Pinecone and return properly mapped documents."""
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    documents = []
    for match in results["matches"]:
        metadata = match.get("metadata", {})
        documents.append({
            "pageContent": metadata.get("text", ""),
            "metadata": {
                "context": metadata.get("context", ""),
                "reference": metadata.get("reference", ""),
                "volume": metadata.get("volume", ""),
            },
            "id": match["id"],
            "score": match.get("score", 0),
        })

    return documents


def search(query: str) -> dict:
    """Search scriptures and return results with proper mapping."""
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    print(f"Query: {query}")
    print("Generating embedding...")

    query_embedding = get_embedding(openai_client, query)

    print("Searching index...")
    documents = query_index(index, query_embedding)

    return {
        "query": query,
        "documents": documents,
    }


def main():
    queries = [
        "Who was Nephi",
    ]

    results = []
    for query in queries:
        result = search(query)
        results.append(result)
        print(f"Found {len(result['documents'])} results")
        print()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
