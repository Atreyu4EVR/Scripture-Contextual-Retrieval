"""
Upsert contextualized Standard Works to Pinecone index.

This script reads the contextualized scripture JSON files and upserts them
to the standard-works-v2 Pinecone index using the integrated sparse embedding model.
"""

import json
import os
from pathlib import Path
from pinecone import Pinecone

# Configuration
INDEX_NAME = "standard-works-v2"
NAMESPACE = "scriptures"
BATCH_SIZE = 96
SCRIPTURES_DIR = Path("scriptures/contextualized")

# Scripture files to process
SCRIPTURE_FILES = [
    "book-of-mormon-contextualized.json",
    "doctrine-and-covenants-contextualized.json",
    "new-testament-contextualized.json",
    "old-testament-contextualized.json",
    "pearl-of-great-price-contextualized.json",
]


def load_scriptures(file_path: Path) -> list[dict]:
    """Load scripture verses from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("verses", [])


def sanitize_id(text: str) -> str:
    """Sanitize text to create ASCII-only IDs."""
    # Replace common non-ASCII characters
    replacements = {
        "—": "-",  # em dash
        "–": "-",  # en dash
        "'": "",   # curly apostrophe
        "'": "",   # curly apostrophe
        """: "",   # curly quote
        """: "",   # curly quote
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-ASCII characters
    return "".join(c for c in text if ord(c) < 128)


def format_record(verse: dict) -> dict:
    """Format a verse into a Pinecone record."""
    # Create a unique ID from volume, book, chapter, verse
    volume_slug = sanitize_id(verse["volume"].lower().replace(" ", "-"))
    book_slug = sanitize_id(verse["book"].lower().replace(" ", "-"))
    record_id = f"{volume_slug}_{book_slug}_{verse['chapter']}_{verse['verse']}"

    return {
        "_id": record_id,
        "text": verse["context"],  # Use contextualized text for embedding
        "reference": verse["reference"],
        "original_text": verse["text"],
        "volume": verse["volume"],
        "book": verse["book"],
        "chapter": verse["chapter"],
        "verse": verse["verse"],
    }


def upsert_records(index, records: list[dict], namespace: str):
    """Upsert records to Pinecone in batches."""
    total = len(records)
    upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert_records(namespace=namespace, records=batch)
        upserted += len(batch)
        print(f"  Upserted {upserted}/{total} records...")

    return upserted


def main():
    # Initialize Pinecone
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    print(f"Connected to Pinecone index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print("-" * 50)

    total_upserted = 0

    for filename in SCRIPTURE_FILES:
        file_path = SCRIPTURES_DIR / filename

        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"\nProcessing {filename}...")
        verses = load_scriptures(file_path)
        print(f"  Loaded {len(verses)} verses")

        # Format records for Pinecone
        records = [format_record(verse) for verse in verses]

        # Upsert to Pinecone
        upserted = upsert_records(index, records, NAMESPACE)
        total_upserted += upserted
        print(f"  Completed: {upserted} records upserted")

    print("-" * 50)
    print(f"Total records upserted: {total_upserted}")
    print("Done!")


if __name__ == "__main__":
    main()
