"""
Upsert embeddings to Pinecone index.

Uploads all scripture embeddings with metadata to the 'standard-works' index.
Processes one volume at a time to manage memory usage.
"""

import gc
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
INDEX_NAME = "standard-works"
BATCH_SIZE = 100

EMBEDDING_FILES = [
    "book-of-mormon-embeddings.json",
    "doctrine-and-covenants-embeddings.json",
    "new-testament-embeddings.json",
    "old-testament-embeddings.json",
    "pearl-of-great-price-embeddings.json",
]


def extract_context_only(full_context: str) -> str:
    """Extract just the context portion without markers or verse text."""
    if '**Context**: "' in full_context and '"\n**Verse**:' in full_context:
        start = full_context.index('**Context**: "') + len('**Context**: "')
        end = full_context.index('"\n**Verse**:')
        return full_context[start:end]
    return full_context


def create_record(embedding_data: dict) -> tuple:
    """Create a Pinecone record tuple (id, values, metadata)."""
    context_only = extract_context_only(embedding_data["context"])

    return (
        str(uuid.uuid4()),
        embedding_data["embedding"],
        {
            "volume": embedding_data.get("volume", "Unknown"),
            "reference": embedding_data.get("reference", "Unknown"),
            "context": context_only or "No context",
            "text": embedding_data.get("text", ""),
        },
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def upsert_batch(index, records: list):
    """Upsert a batch of records with retry logic."""
    index.upsert(vectors=records)


def process_volume(index, filename: str) -> int:
    """Process and upsert all embeddings from a single volume."""
    filepath = EMBEDDINGS_DIR / filename
    volume_name = filename.replace("-embeddings.json", "").replace("-", " ").title()

    print(f"Processing {volume_name}...")
    print(f"  Loading {filename}...")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = data.get("embeddings", [])
    total = len(embeddings)
    print(f"  Found {total} embeddings")

    # Process in batches
    upserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        records = [create_record(emb) for emb in batch_embeddings]

        upsert_batch(index, records)
        upserted += len(records)

        if upserted % 500 == 0 or upserted == total:
            print(f"  Upserted {upserted}/{total}")

    # Free memory
    del data
    del embeddings
    gc.collect()

    print(f"  Complete: {upserted} vectors")
    return upserted


def main():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    print("Pinecone Upsert")
    print("=" * 50)
    print(f"Index: {INDEX_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    total_vectors = 0

    for filename in EMBEDDING_FILES:
        count = process_volume(index, filename)
        total_vectors += count
        print()

    print("=" * 50)
    print(f"Total vectors upserted: {total_vectors:,}")
    print("Done!")


if __name__ == "__main__":
    main()
