"""
LDS Standard Works Token Counter

Counts tokens for each scripture volume using OpenAI's tiktoken library
and provides a grand total across all five books.
"""

import json
from pathlib import Path

import tiktoken


SCRIPTURES_DIR = Path(__file__).parent.parent / "scriptures" / "contextualized"

SCRIPTURE_FILES = [
    "book-of-mormon-contextualized.json",
    "doctrine-and-covenants-contextualized.json",
    "new-testament-contextualized.json",
    "old-testament-contextualized.json",
    "pearl-of-great-price-contextualized.json",
]


def load_scripture(filepath: Path) -> list[dict]:
    """Load scripture JSON file and return list of verses."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("verses", [])


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens in a string using the provided encoder."""
    return len(encoder.encode(text))


def get_volume_name(filename: str) -> str:
    """Convert filename to display name."""
    name = filename.replace("-contextualized.json", "").replace("-", " ").title()
    return name


def count_scripture_tokens(filepath: Path, encoder: tiktoken.Encoding) -> dict:
    """Count tokens for a scripture volume."""
    verses = load_scripture(filepath)

    total_tokens = 0
    for verse in verses:
        text = verse.get("text", "")
        total_tokens += count_tokens(text, encoder)

    return {
        "volume": get_volume_name(filepath.name),
        "verses": len(verses),
        "tokens": total_tokens,
    }


def main():
    """Main entry point."""
    model = "gpt-5"
    encoder = tiktoken.encoding_for_model(model)

    results = []
    grand_total_tokens = 0
    grand_total_verses = 0

    print("LDS Standard Works Token Counter")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Encoding: {encoder.name}\n")

    for filename in SCRIPTURE_FILES:
        filepath = SCRIPTURES_DIR / filename

        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        result = count_scripture_tokens(filepath, encoder)
        results.append(result)

        grand_total_tokens += result["tokens"]
        grand_total_verses += result["verses"]

        print(f"{result['volume']}")
        print(f"  Verses: {result['verses']:,}")
        print(f"  Tokens: {result['tokens']:,}")
        print()

    print("=" * 50)
    print("GRAND TOTAL")
    print(f"  Total Verses: {grand_total_verses:,}")
    print(f"  Total Tokens: {grand_total_tokens:,}")


if __name__ == "__main__":
    main()
