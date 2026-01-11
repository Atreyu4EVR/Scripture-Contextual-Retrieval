"""
Normalize scripture JSON files to a consistent structure.

Converts D&C and Pearl of Great Price to match the standard format:
books → chapters → verses
"""

import json
from pathlib import Path


SCRIPTURES_DIR = Path(__file__).parent / "scriptures"


def normalize_doctrine_and_covenants():
    """Convert D&C from sections → verses to books → chapters → verses."""
    input_path = SCRIPTURES_DIR / "doctrine-and-covenants.json"
    output_path = SCRIPTURES_DIR / "doctrine-and-covenants.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("sections", [])

    normalized = {
        "books": [
            {
                "book": "Doctrine and Covenants",
                "chapters": [
                    {
                        "chapter": section["section"],
                        "reference": section["reference"],
                        "verses": section["verses"],
                    }
                    for section in sections
                ],
            }
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    chapter_count = len(sections)
    verse_count = sum(len(s["verses"]) for s in sections)
    print(f"Doctrine and Covenants: {chapter_count} sections, {verse_count} verses")


def normalize_pearl_of_great_price():
    """Convert Pearl of Great Price from nested dict to books → chapters → verses."""
    input_path = SCRIPTURES_DIR / "reference" / "pearl-of-great-price-reference.json"
    output_path = SCRIPTURES_DIR / "pearl-of-great-price.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    skip_keys = {"last_modified", "version"}
    book_names = [k for k in data.keys() if k not in skip_keys]

    books = []
    total_chapters = 0
    total_verses = 0

    for book_name in book_names:
        book_data = data[book_name]
        chapters = []

        for chapter_num, verses_dict in book_data.items():
            chapter_num_int = int(chapter_num)
            verses = []

            for verse_num, text in verses_dict.items():
                verse_num_int = int(verse_num)
                reference = f"{book_name} {chapter_num}:{verse_num}"
                verses.append({
                    "reference": reference,
                    "text": text,
                    "verse": verse_num_int,
                })

            verses.sort(key=lambda v: v["verse"])

            chapters.append({
                "chapter": chapter_num_int,
                "reference": f"{book_name} {chapter_num}",
                "verses": verses,
            })
            total_verses += len(verses)

        chapters.sort(key=lambda c: c["chapter"])
        total_chapters += len(chapters)

        books.append({
            "book": book_name,
            "chapters": chapters,
        })

    normalized = {"books": books}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"Pearl of Great Price: {len(books)} books, {total_chapters} chapters, {total_verses} verses")


def main():
    print("Normalizing scripture files...\n")
    normalize_doctrine_and_covenants()
    normalize_pearl_of_great_price()
    print("\nDone!")


if __name__ == "__main__":
    main()
