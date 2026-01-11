"""
Generate legacy RAG embeddings (no context) and upsert to Pinecone.

Embeds raw verse text without chapter summaries for comparison with
contextual retrieval. Uses parallel processing to maximize throughput
while respecting OpenAI Tier 5 rate limits.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

SCRIPTURES_DIR = Path(__file__).parent.parent / "scriptures" / "source"
INDEX_NAME = "standard-work-nocontext"

SCRIPTURE_FILES = {
    "Book of Mormon": "book-of-mormon.json",
    "Doctrine and Covenants": "doctrine-and-covenants.json",
    "New Testament": "new-testament.json",
    "Old Testament": "old-testament.json",
    "Pearl of Great Price": "pearl-of-great-price.json",
}

EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
MAX_WORKERS = 50


def load_scripture(volume_name: str) -> list:
    """Load and flatten scripture verses from hierarchical JSON."""
    filepath = SCRIPTURES_DIR / SCRIPTURE_FILES[volume_name]
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    verses = []
    for book in data.get("books", []):
        book_name = book["book"]
        for chapter in book.get("chapters", []):
            chapter_num = chapter["chapter"]
            for verse in chapter.get("verses", []):
                verses.append({
                    "volume": volume_name,
                    "book": book_name,
                    "chapter": chapter_num,
                    "verse": verse["verse"],
                    "reference": verse["reference"],
                    "text": verse["text"],
                })
    return verses


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_embeddings_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts with retry logic."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def upsert_batch(index, records: list):
    """Upsert a batch of records to Pinecone with retry logic."""
    index.upsert(vectors=records)


def process_batch(args: tuple) -> int:
    """Process a batch: generate embeddings and upsert to Pinecone."""
    client, index, batch_verses, batch_idx = args

    texts = [v["text"] for v in batch_verses]
    embeddings = generate_embeddings_batch(client, texts)

    records = []
    for verse, embedding in zip(batch_verses, embeddings):
        # Create deterministic ID from reference (e.g., "1-nephi-1-1")
        verse_id = re.sub(r"[^a-zA-Z0-9]+", "-", verse["reference"].lower()).strip("-")
        records.append((
            verse_id,
            embedding,
            {
                "volume": verse["volume"],
                "reference": verse["reference"],
                "text": verse["text"],
            },
        ))

    upsert_batch(index, records)
    return len(records)


def process_volume(client: OpenAI, index, volume_name: str) -> int:
    """Process all verses in a volume: embed and upsert."""
    verses = load_scripture(volume_name)

    batches = []
    for i in range(0, len(verses), BATCH_SIZE):
        batch = verses[i : i + BATCH_SIZE]
        batches.append((client, index, batch, i // BATCH_SIZE))

    print(f"  Processing {len(verses)} verses in {len(batches)} batches...")

    total_upserted = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            count = future.result()
            total_upserted += count

            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"  Completed {completed}/{len(batches)} batches ({total_upserted} vectors)")

    return total_upserted


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    print("Legacy RAG Embedding Generator")
    print("=" * 50)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Index: {INDEX_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {MAX_WORKERS}")
    print()

    total_vectors = 0

    for volume_name in SCRIPTURE_FILES:
        print(f"Processing {volume_name}...")
        count = process_volume(client, index, volume_name)
        total_vectors += count
        print()

    print("=" * 50)
    print(f"Total vectors upserted: {total_vectors:,}")
    print("Done!")


if __name__ == "__main__":
    main()
