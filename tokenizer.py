"""Token counter for LDS Standard Works using OpenAI's tiktoken library."""

import json
import tiktoken


def count_tokens_for_volume(file_path: str, encoding) -> dict:
    """Count tokens for a single scripture volume."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_tokens = 0
    verse_count = 0

    for book in data.get("books", []):
        for chapter in book.get("chapters", []):
            for verse in chapter.get("verses", []):
                text = verse.get("text", "")
                tokens = encoding.encode(text)
                total_tokens += len(tokens)
                verse_count += 1

    return {
        "tokens": total_tokens,
        "verses": verse_count
    }


def main():
    # Use o200k_base encoding (used by GPT-5 models)
    encoding = tiktoken.get_encoding("o200k_base")

    bom_path = "scriptures/source/book-of-mormon.json"

    print("Counting tokens for the Book of Mormon...")
    print(f"Using encoding: {encoding.name}")
    print("-" * 50)

    result = count_tokens_for_volume(bom_path, encoding)

    print(f"Book of Mormon:")
    print(f"  Verses: {result['verses']:,}")
    print(f"  Tokens: {result['tokens']:,}")


if __name__ == "__main__":
    main()
