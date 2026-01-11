"""
Generate contextualized verses by combining chapter summaries with verse text.

Creates embedding-ready text that includes volume, reference, chapter context,
and the verse content following the contextual retrieval method.
"""

import json
from pathlib import Path


SCRIPTURES_DIR = Path(__file__).parent / "scriptures"
SUMMARIES_DIR = SCRIPTURES_DIR / "summaries"
OUTPUT_DIR = SCRIPTURES_DIR / "contextualized"

SCRIPTURE_FILES = {
    "Book of Mormon": "book-of-mormon.json",
    "Doctrine and Covenants": "doctrine-and-covenants.json",
    "New Testament": "new-testament.json",
    "Old Testament": "old-testament.json",
    "Pearl of Great Price": "pearl-of-great-price.json",
}


def load_scripture(volume_name: str) -> dict:
    """Load scripture JSON file."""
    filepath = SCRIPTURES_DIR / SCRIPTURE_FILES[volume_name]
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summaries(volume_name: str) -> dict:
    """Load chapter summaries for a volume."""
    filename = SCRIPTURE_FILES[volume_name].replace(".json", "-summaries.json")
    filepath = SUMMARIES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_lookup(summaries: dict) -> dict:
    """Build a lookup dictionary for chapter summaries by reference."""
    lookup = {}
    for book in summaries.get("books", []):
        for chapter in book.get("chapters", []):
            lookup[chapter["reference"]] = chapter["summary"]
    return lookup


def build_context(volume: str, reference: str, chapter_summary: str, text: str) -> str:
    """Build the contextualized string for a verse with clear boundaries."""
    return (
        f'**Context**: "The following passage is from **{reference}** from the **{volume}**. '
        f'{chapter_summary}"\n'
        f'**Verse**: "{text}"'
    )


def process_volume(volume_name: str) -> list:
    """Process all verses in a volume and add context."""
    scripture_data = load_scripture(volume_name)
    summaries_data = load_summaries(volume_name)
    summary_lookup = build_summary_lookup(summaries_data)

    contextualized_verses = []

    for book in scripture_data.get("books", []):
        book_name = book["book"]

        for chapter in book.get("chapters", []):
            chapter_ref = chapter["reference"]
            chapter_summary = summary_lookup.get(chapter_ref, "")

            for verse in chapter.get("verses", []):
                verse_ref = verse["reference"]
                text = verse["text"]

                context = build_context(volume_name, verse_ref, chapter_summary, text)

                contextualized_verses.append({
                    "volume": volume_name,
                    "book": book_name,
                    "chapter": chapter["chapter"],
                    "verse": verse["verse"],
                    "reference": verse_ref,
                    "text": text,
                    "context": context,
                })

    return contextualized_verses


def save_contextualized(volume_name: str, verses: list):
    """Save contextualized verses to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = SCRIPTURE_FILES[volume_name].replace(".json", "-contextualized.json")
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"verses": verses}, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(verses)} verses to {output_path}")


def main():
    print("Contextualized Verses Generator")
    print("=" * 50 + "\n")

    total_verses = 0

    for volume_name in SCRIPTURE_FILES:
        print(f"Processing {volume_name}...")
        verses = process_volume(volume_name)
        save_contextualized(volume_name, verses)
        total_verses += len(verses)
        print()

    print("=" * 50)
    print(f"Total contextualized verses: {total_verses:,}")
    print("Done!")


if __name__ == "__main__":
    main()
