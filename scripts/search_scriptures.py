"""
Search the LDS Standard Works using Pinecone
Uses OpenAI text-embedding-3-large for query embeddings
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to the standard-works index
index = pc.Index("standard-works")


def embed_query(text: str) -> list[float]:
    """Generate embedding for a query using OpenAI text-embedding-3-large"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


def search_scriptures(query: str, top_k: int = 10) -> list[dict]:
    """Search the standard works for relevant scriptures"""
    # Generate query embedding
    query_embedding = embed_query(query)

    # Search Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="__default__"
    )

    return results.matches


def display_results(results: list[dict]) -> None:
    """Display search results"""
    for i, match in enumerate(results, 1):
        score = round(match.score, 4)
        metadata = match.metadata
        reference = metadata.get("reference", "Unknown")
        text = metadata.get("text", "No text available")

        print(f"\n{i}. {reference} (score: {score})")
        print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")


def interactive_search():
    """Interactive search mode"""
    print("\n" + "="*60)
    print("  LDS Standard Works Search")
    print("  Type a query to search, or 'quit' to exit")
    print("="*60)

    while True:
        try:
            query = input("\nSearch: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print(f"\nSearching for: '{query}'...")
            results = search_scriptures(query, top_k=5)

            if results:
                display_results(results)
            else:
                print("No results found.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    interactive_search()
