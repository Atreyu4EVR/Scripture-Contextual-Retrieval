"""
Generate chapter summaries for all scriptures using GPT-5.2.

Uses the contextual retrieval method to create objective, factual
summaries of each chapter for embedding enrichment.
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
OUTPUT_DIR = SCRIPTURES_DIR / "summaries"

SCRIPTURE_FILES = {
    "Book of Mormon": "book-of-mormon.json",
    "Doctrine and Covenants": "doctrine-and-covenants.json",
    "New Testament": "new-testament.json",
    "Old Testament": "old-testament.json",
    "Pearl of Great Price": "pearl-of-great-price.json",
}

SYSTEM_PROMPT = """You are a scripture indexer. Your task is to generate brief, factual summaries of scripture chapters for use in a retrieval system.

Rules:
- Be strictly objective and factual
- Do not insert opinions, interpretations, or theological commentary
- Do not use phrases like "beautifully describes" or "powerfully teaches"
- Focus on: events, people, places, commandments, and topics covered
- Start with the verse count
- Use present tense
- Keep summaries between 50-75 tokens

Format: "This chapter contains [X] verses and [describes/records/covers] [factual content]."
"""

MAX_WORKERS = 50


def load_scripture(volume_name: str) -> dict:
    """Load scripture JSON file."""
    filepath = SCRIPTURES_DIR / SCRIPTURE_FILES[volume_name]
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chapter_text(chapter: dict) -> str:
    """Combine all verses in a chapter into a single text."""
    verses = chapter.get("verses", [])
    return "\n".join(f"{v['verse']}. {v['text']}" for v in verses)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_summary_with_retry(client: OpenAI, instructions: str, input_text: str) -> str:
    """Generate a summary with exponential backoff retry."""
    response = client.responses.create(
        model="gpt-5.2",
        reasoning={"effort": "low"},
        instructions=instructions,
        input=input_text,
    )
    return response.output_text


def generate_summary(client: OpenAI, volume: str, book: str, chapter: dict) -> dict:
    """Generate a summary for a single chapter using GPT-5.2."""
    chapter_num = chapter["chapter"]
    reference = chapter["reference"]
    verse_count = len(chapter["verses"])
    chapter_text = build_chapter_text(chapter)

    unit_type = "section" if volume == "Doctrine and Covenants" else "chapter"

    input_text = f"""Summarize this {unit_type} for a retrieval index.

Volume: {volume}
Book: {book}
Chapter: {chapter_num}
Reference: {reference}
Verse Count: {verse_count}

{unit_type.title()} Text:
{chapter_text}
"""

    summary = generate_summary_with_retry(client, SYSTEM_PROMPT, input_text)

    return {
        "chapter": chapter_num,
        "reference": reference,
        "verse_count": verse_count,
        "summary": summary,
    }


def process_chapter_task(args: tuple) -> dict:
    """Process a single chapter (for parallel execution)."""
    client, volume, book_name, chapter = args
    result = generate_summary(client, volume, book_name, chapter)
    return {
        "volume": volume,
        "book": book_name,
        "result": result,
    }


def process_volume_parallel(client: OpenAI, volume_name: str) -> dict:
    """Process all chapters in a volume using parallel execution."""
    data = load_scripture(volume_name)
    books = data.get("books", [])

    tasks = []
    for book in books:
        book_name = book["book"]
        for chapter in book.get("chapters", []):
            tasks.append((client, volume_name, book_name, chapter))

    volume_summaries = {"volume": volume_name, "books": []}
    book_results = {}

    print(f"  Processing {len(tasks)} chapters with {MAX_WORKERS} workers...")

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chapter_task, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            book_name = result["book"]

            if book_name not in book_results:
                book_results[book_name] = []
            book_results[book_name].append(result["result"])

            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"  Completed {completed}/{len(tasks)} chapters")

    for book in books:
        book_name = book["book"]
        chapters = sorted(book_results.get(book_name, []), key=lambda x: x["chapter"])
        volume_summaries["books"].append({
            "book": book_name,
            "chapters": chapters,
        })

    return volume_summaries


def save_summaries(volume_name: str, summaries: dict):
    """Save summaries to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = SCRIPTURE_FILES[volume_name].replace(".json", "-summaries.json")
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_path}")


def count_chapters(volume_name: str) -> int:
    """Count total chapters in a volume."""
    data = load_scripture(volume_name)
    return sum(len(book["chapters"]) for book in data.get("books", []))


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Chapter Summary Generator")
    print("=" * 50)
    print(f"Model: gpt-5.2 (reasoning: low)")
    print(f"Workers: {MAX_WORKERS}\n")

    total_chapters = 0
    for volume_name in SCRIPTURE_FILES:
        chapter_count = count_chapters(volume_name)
        total_chapters += chapter_count
        print(f"{volume_name}: {chapter_count} chapters")

    print(f"\nTotal chapters to process: {total_chapters}")
    print("=" * 50 + "\n")

    for volume_name in SCRIPTURE_FILES:
        print(f"Processing {volume_name}...")
        summaries = process_volume_parallel(client, volume_name)
        save_summaries(volume_name, summaries)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
