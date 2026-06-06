"""
Step 1: Build & validate the rule-based annotation layer for the Book of Mormon.

Resolves, for every verse, the claimed narrator, editorial layer, embedded speaker,
and genre from the config maps; validates that the maps cover the corpus and that
every embedded-speaker span resolves to real verses; computes per-narrator word
totals and a reliability flag; and writes a segment index summary.

Exposes resolver functions (annotate_verse, compute_reliability) reused by
build_segments.py.

Run: python authorship/scripts/build_annotations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

VOLUME = "Book of Mormon"


def _book_narrator(book: str, chapter: int, narrator_map: dict) -> tuple[str, str]:
    """Resolve (claimed_narrator, editorial_layer) for a book/chapter, honoring overrides."""
    overrides = narrator_map.get("book_overrides", {}).get(book, {})
    if str(chapter) in overrides:
        o = overrides[str(chapter)]
        return o["claimed_narrator"], o["editorial_layer"]
    base = narrator_map["books"][book]
    return base["claimed_narrator"], base["editorial_layer"]


def _genre_for_chapter(book: str, chapter: int, genre_map: dict) -> str:
    """Resolve the (pre-quote, pre-embedded) genre for a chapter."""
    for ov in genre_map.get("chapter_overrides", []):
        if ov["book"] == book and chapter in ov["chapters"]:
            return ov["genre"]
    return genre_map["book_default"][book]


def _embedded_for_verse(book: str, chapter: int, verse: int, embedded: dict, rely_only: bool = True):
    """Return the embedded-speaker span dict covering this verse, or None."""
    for span in embedded["spans"]:
        if rely_only and not span.get("rely", False):
            continue
        if span["book"] != book:
            continue
        if ac.in_span(chapter, verse, span["start"], span["end"]):
            return span
    return None


def annotate_verse(
    book: str,
    chapter: int,
    verse: int,
    narrator_map: dict,
    embedded: dict,
    genre_map: dict,
) -> dict:
    """Resolve the full rule-based annotation for a single Book of Mormon verse."""
    claimed_narrator, editorial_layer = _book_narrator(book, chapter, narrator_map)
    genre = _genre_for_chapter(book, chapter, genre_map)
    speaker = claimed_narrator

    span = _embedded_for_verse(book, chapter, verse, embedded, rely_only=True)
    if span is not None:
        speaker = span["speaker"]
        genre = span.get("genre", genre)
        # Embedded epistles carry their own editorial layer; otherwise keep host layer.
        editorial_layer = span.get("editorial_layer", editorial_layer)

    return {
        "claimed_narrator": claimed_narrator,
        "speaker": speaker,
        "editorial_layer": editorial_layer,
        "genre": genre,
    }


def compute_reliability(narrator_word_totals: dict, config: dict, narrator_map: dict) -> dict:
    """Assign a reliability flag per claimed narrator from word totals."""
    rel_cfg = config["reliability"]
    composite = set(narrator_map.get("composite_unreliable_narrators", []))
    out = {}
    for narrator, words in narrator_word_totals.items():
        if narrator in composite:
            flag = "composite_unreliable"
        elif words >= rel_cfg["min_word_count_reliable"]:
            flag = "reliable"
        elif words >= rel_cfg["min_word_count_borderline"]:
            flag = "borderline"
        else:
            flag = "report_only"
        out[narrator] = {"word_count": words, "reliability": flag}
    return out


def validate_maps(narrator_map: dict, embedded: dict, genre_map: dict) -> list[str]:
    """Return a list of validation errors (empty == ok)."""
    errors = []
    data = ac.load_scripture(VOLUME)
    book_names = {b["book"] for b in data["books"]}

    # Every book must be mapped for narrator and genre default.
    for b in book_names:
        if b not in narrator_map["books"]:
            errors.append(f"narrator_map missing book: {b}")
        if b not in genre_map["book_default"]:
            errors.append(f"genre_map.book_default missing book: {b}")

    # Build a chapter/verse index for span validation.
    valid_cv = {}
    for ch in ac.iter_chapters(VOLUME):
        key = (ch["book"], ch["chapter"])
        valid_cv[key] = {v.verse for v in ch["verses"]}

    for span in embedded["spans"]:
        b = span["book"]
        if b not in book_names:
            errors.append(f"embedded span references unknown book: {b}")
            continue
        for endpoint in ("start", "end"):
            c, v = (int(x) for x in span[endpoint].split(":"))
            if (b, c) not in valid_cv:
                errors.append(f"embedded span {span['speaker']} {b} {span[endpoint]}: chapter {c} not found")
            elif v not in valid_cv[(b, c)]:
                errors.append(f"embedded span {span['speaker']} {b} {span[endpoint]}: verse {v} not found")
    return errors


def main():
    config = ac.load_config()
    narrator_map = ac.load_narrator_map()
    embedded = ac.load_embedded_speakers()
    genre_map = ac.load_genre_map()

    print("=" * 60)
    print("Step 1: Build & validate annotations")
    print("=" * 60)

    errors = validate_maps(narrator_map, embedded, genre_map)
    if errors:
        print("\nMAP VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Map validation passed.")

    # Per-narrator word totals (using claimed_narrator with overrides).
    narrator_word_totals: dict[str, int] = {}
    speaker_word_totals: dict[str, int] = {}
    n_books = set()
    n_chapters = 0
    n_verses = 0

    for ch in ac.iter_chapters(VOLUME):
        n_chapters += 1
        n_books.add(ch["book"])
        for v in ch["verses"]:
            n_verses += 1
            ann = annotate_verse(ch["book"], ch["chapter"], v.verse, narrator_map, embedded, genre_map)
            wc = ac.word_count(v.text)
            narrator_word_totals[ann["claimed_narrator"]] = (
                narrator_word_totals.get(ann["claimed_narrator"], 0) + wc
            )
            speaker_word_totals[ann["speaker"]] = speaker_word_totals.get(ann["speaker"], 0) + wc

    reliability = compute_reliability(narrator_word_totals, config, narrator_map)
    speaker_reliability = compute_reliability(speaker_word_totals, config, narrator_map)

    # Sanity check against known corpus stats.
    assert n_chapters == 239, f"expected 239 chapters, got {n_chapters}"
    assert n_verses == 6604, f"expected 6604 verses, got {n_verses}"
    assert len(n_books) == 15, f"expected 15 books, got {len(n_books)}"
    print(f"Corpus sanity OK: {len(n_books)} books / {n_chapters} chapters / {n_verses} verses")

    index = {
        "volume": VOLUME,
        "n_books": len(n_books),
        "n_chapters": n_chapters,
        "n_verses": n_verses,
        "narrator_word_totals": narrator_word_totals,
        "narrator_reliability": reliability,
        "speaker_word_totals": speaker_word_totals,
        "speaker_reliability": speaker_reliability,
    }
    out_path = ac.DATA_DIR / "segments" / "segment_index.json"
    ac.write_json(out_path, index)
    print(f"\nWrote {out_path}")

    print("\nNarrator reliability:")
    for narrator, info in sorted(reliability.items(), key=lambda kv: -kv[1]["word_count"]):
        print(f"  {narrator:20s} {info['word_count']:7d} words  [{info['reliability']}]")


if __name__ == "__main__":
    main()
