"""
Generate embeddings for contextualized verses using text-embedding-3-large.

Uses batch processing and parallel execution to maximize throughput
while respecting OpenAI rate limits.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


load_dotenv()

SCRIPTURES_DIR = Path(__file__).parent / "scriptures"
CONTEXTUALIZED_DIR = SCRIPTURES_DIR / "contextualized"
OUTPUT_DIR = Path(__file__).parent / "embeddings"

SCRIPTURE_FILES = {
    "Book of Mormon": "book-of-mormon-contextualized.json",
    "Doctrine and Covenants": "doctrine-and-covenants-contextualized.json",
    "New Testament": "new-testament-contextualized.json",
    "Old Testament": "old-testament-contextualized.json",
    "Pearl of Great Price": "pearl-of-great-price-contextualized.json",
}

EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 100
MAX_WORKERS = 10


def load_contextualized(volume_name: str) -> list:
    """Load contextualized verses for a volume."""
    filepath = CONTEXTUALIZED_DIR / SCRIPTURE_FILES[volume_name]
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("verses", [])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_embeddings_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts with retry logic."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def process_batch(args: tuple) -> list[dict]:
    """Process a batch of verses and generate embeddings."""
    client, batch_verses, batch_idx = args

    texts = [v["context"] for v in batch_verses]
    embeddings = generate_embeddings_batch(client, texts)

    results = []
    for verse, embedding in zip(batch_verses, embeddings):
        results.append({
            "reference": verse["reference"],
            "volume": verse["volume"],
            "book": verse["book"],
            "chapter": verse["chapter"],
            "verse": verse["verse"],
            "text": verse["text"],
            "context": verse["context"],
            "embedding": embedding,
        })

    return results


def process_volume(client: OpenAI, volume_name: str) -> list:
    """Process all verses in a volume and generate embeddings."""
    verses = load_contextualized(volume_name)

    batches = []
    for i in range(0, len(verses), BATCH_SIZE):
        batch = verses[i : i + BATCH_SIZE]
        batches.append((client, batch, i // BATCH_SIZE))

    print(f"  Processing {len(verses)} verses in {len(batches)} batches...")

    all_results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)

            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"  Completed {completed}/{len(batches)} batches")

    all_results.sort(key=lambda x: (x["chapter"], x["verse"]))

    return all_results


def save_embeddings(volume_name: str, embeddings: list):
    """Save embeddings to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = SCRIPTURE_FILES[volume_name].replace("-contextualized.json", "-embeddings.json")
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"embeddings": embeddings}, f, ensure_ascii=False)

    print(f"  Saved {len(embeddings)} embeddings to {output_path}")


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Embedding Generator")
    print("=" * 50)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Workers: {MAX_WORKERS}\n")

    total_embeddings = 0

    for volume_name in SCRIPTURE_FILES:
        print(f"Processing {volume_name}...")
        embeddings = process_volume(client, volume_name)
        save_embeddings(volume_name, embeddings)
        total_embeddings += len(embeddings)
        print()

    print("=" * 50)
    print(f"Total embeddings generated: {total_embeddings:,}")
    print("Done!")


if __name__ == "__main__":
    main()
