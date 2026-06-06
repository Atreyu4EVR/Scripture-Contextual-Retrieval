"""
Shared utilities for the Book of Mormon authorship/production study (phase one).

Provides: repo-rooted paths, config/map loading, the canonical scripture loader
and verse iterator (mirrors the repo's books -> chapters -> verses pattern), the
Segment dataclass, reference parsing, and JSONL/JSON IO helpers.

Phase one uses only in-repo texts (Book of Mormon + KJV control); no API keys needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

# Repo root is two levels up from authorship/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORSHIP_DIR = REPO_ROOT / "authorship"
CONFIG_DIR = AUTHORSHIP_DIR / "config"
DATA_DIR = AUTHORSHIP_DIR / "data"
FEATURES_DIR = AUTHORSHIP_DIR / "features"
RESULTS_DIR = AUTHORSHIP_DIR / "results"
SOURCE_DIR = REPO_ROOT / "scriptures" / "source"

SCRIPTURE_FILES = {
    "Book of Mormon": "book-of-mormon.json",
    "Old Testament": "old-testament.json",
    "New Testament": "new-testament.json",
    "Doctrine and Covenants": "doctrine-and-covenants.json",
    "Pearl of Great Price": "pearl-of-great-price.json",
}


# --------------------------------------------------------------------------- #
# Config / map loading
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict:
    return load_json(CONFIG_DIR / "authorship_config.json")


def load_narrator_map() -> dict:
    return load_json(CONFIG_DIR / "narrator_map.json")


def load_embedded_speakers() -> dict:
    return load_json(CONFIG_DIR / "embedded_speakers.json")


def load_genre_map() -> dict:
    return load_json(CONFIG_DIR / "genre_map.json")


# --------------------------------------------------------------------------- #
# Scripture loading / iteration (canonical repo pattern)
# --------------------------------------------------------------------------- #
def load_scripture(volume: str) -> dict:
    """Load a volume's source JSON by display name."""
    filename = SCRIPTURE_FILES[volume]
    return load_json(SOURCE_DIR / filename)


@dataclass
class Verse:
    volume: str
    book: str
    chapter: int
    verse: int
    reference: str
    text: str


def iter_verses(volume: str) -> Iterator[Verse]:
    """Yield every verse in a volume (books -> chapters -> verses)."""
    data = load_scripture(volume)
    for book in data.get("books", []):
        book_name = book["book"]
        for chapter in book.get("chapters", []):
            chap_num = chapter["chapter"]
            for verse in chapter.get("verses", []):
                yield Verse(
                    volume=volume,
                    book=book_name,
                    chapter=chap_num,
                    verse=verse["verse"],
                    reference=verse["reference"],
                    text=verse["text"],
                )


def iter_chapters(volume: str) -> Iterator[dict]:
    """Yield {book, chapter, reference, verses:[Verse,...]} per chapter."""
    data = load_scripture(volume)
    for book in data.get("books", []):
        book_name = book["book"]
        for chapter in book.get("chapters", []):
            verses = [
                Verse(
                    volume=volume,
                    book=book_name,
                    chapter=chapter["chapter"],
                    verse=v["verse"],
                    reference=v["reference"],
                    text=v["text"],
                )
                for v in chapter.get("verses", [])
            ]
            yield {
                "book": book_name,
                "chapter": chapter["chapter"],
                "reference": chapter["reference"],
                "verses": verses,
            }


# --------------------------------------------------------------------------- #
# Reference parsing helpers
# --------------------------------------------------------------------------- #
def parse_chapter_verse(ref: str) -> tuple[int, int]:
    """Parse the trailing 'chapter:verse' of a reference like 'Mosiah 2:1'."""
    m = re.search(r"(\d+):(\d+)\s*$", ref)
    if not m:
        raise ValueError(f"Unparseable reference: {ref!r}")
    return int(m.group(1)), int(m.group(2))


def cv_key(chapter: int, verse: int) -> tuple[int, int]:
    return (chapter, verse)


def in_span(chapter: int, verse: int, start: str, end: str) -> bool:
    """Is (chapter, verse) within an inclusive [start, end] 'ch:v' span?"""
    sc, sv = (int(x) for x in start.split(":"))
    ec, ev = (int(x) for x in end.split(":"))
    return (sc, sv) <= (chapter, verse) <= (ec, ev)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Segment model
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """A unit of analysis. Fields mirror the annotation schema in the plan."""
    segment_id: str
    segmentation: str  # which set: chapter | rolling | narrator | speaker | genre | quote_removed
    volume: str
    book: Optional[str]
    chapter: Optional[int]
    verse_range: Optional[list]  # [start_verse, end_verse] or None
    claimed_narrator: str
    speaker: str
    editorial_layer: str
    genre: str
    reliability: str = "reliable"
    word_count: int = 0
    contains_biblical_quote: bool = False
    quote_source: list = field(default_factory=list)
    quote_fraction: float = 0.0
    mixed_speaker: bool = False
    narrator_purity: float = 1.0
    text: str = ""
    text_quote_removed: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def latest_run_dir(parent: Path) -> Optional[Path]:
    if not parent.exists():
        return None
    runs = sorted([p for p in parent.iterdir() if p.is_dir()], reverse=True)
    return runs[0] if runs else None
