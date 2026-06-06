"""
Step 2: Produce the segmentation sets for the Book of Mormon.

Generates (JSONL, one Segment per line):
  - segments_chapter.jsonl     chapter-based (239 units; primary unit for LOCO-CV)
  - segments_rolling1000.jsonl fixed rolling word windows (length-matched units)
  - segments_narrator.jsonl    claimed-narrator AND by-speaker macro-segments
  - segments_genre.jsonl       genre-aggregated segments (isolates the genre confound)

The biblical-quotation-removed set (segments_quote_removed.jsonl) and the quote
fields on the chapter set are produced by Step 3 (detect_biblical_quotes.py).

Run: python authorship/scripts/build_segments.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from build_annotations import annotate_verse

VOLUME = "Book of Mormon"
SEG_DIR = ac.DATA_DIR / "segments"


def _reliability_lookup(index: dict):
    narr = {k: v["reliability"] for k, v in index["narrator_reliability"].items()}
    spk = {k: v["reliability"] for k, v in index["speaker_reliability"].items()}
    return narr, spk


def _dominant(counter: Counter):
    """Return (key, fraction) of the most common key by accumulated weight."""
    total = sum(counter.values())
    if total == 0:
        return None, 0.0
    key, val = counter.most_common(1)[0]
    return key, val / total


def build_chapter_segments(maps, narr_rel) -> list[dict]:
    narrator_map, embedded, genre_map = maps
    segments = []
    for ch in ac.iter_chapters(VOLUME):
        book, chap = ch["book"], ch["chapter"]
        verses = ch["verses"]
        text = " ".join(v.text for v in verses)

        speaker_words = Counter()
        genre_words = Counter()
        claimed = editorial = None
        for v in verses:
            ann = annotate_verse(book, chap, v.verse, narrator_map, embedded, genre_map)
            wc = ac.word_count(v.text)
            speaker_words[ann["speaker"]] += wc
            genre_words[ann["genre"]] += wc
            claimed = ann["claimed_narrator"]
            editorial = ann["editorial_layer"]

        speaker, _ = _dominant(speaker_words)
        genre, _ = _dominant(genre_words)
        mixed_speaker = len([s for s in speaker_words if speaker_words[s] > 0]) > 1

        seg = ac.Segment(
            segment_id=f"BoM:{book}:{chap}",
            segmentation="chapter",
            volume=VOLUME,
            book=book,
            chapter=chap,
            verse_range=[verses[0].verse, verses[-1].verse],
            claimed_narrator=claimed,
            speaker=speaker,
            editorial_layer=editorial,
            genre=genre,
            reliability=narr_rel.get(claimed, "reliable"),
            word_count=ac.word_count(text),
            mixed_speaker=mixed_speaker,
            text=text,
            text_quote_removed=text,  # placeholder until Step 3
        )
        segments.append(seg.to_dict())
    return segments


def _annotated_word_stream(maps):
    """Yield (word, annotation, book) across the whole volume in canonical order."""
    narrator_map, embedded, genre_map = maps
    for ch in ac.iter_chapters(VOLUME):
        book, chap = ch["book"], ch["chapter"]
        for v in ch["verses"]:
            ann = annotate_verse(book, chap, v.verse, narrator_map, embedded, genre_map)
            for w in v.text.split():
                yield w, ann, book


def build_rolling_segments(maps, narr_rel, window: int, stride: int) -> list[dict]:
    # Windows do not cross book boundaries.
    by_book = defaultdict(list)
    for w, ann, book in _annotated_word_stream(maps):
        by_book[book].append((w, ann))

    segments = []
    widx = 0
    for book, stream in by_book.items():
        i = 0
        while i < len(stream):
            chunk = stream[i : i + window]
            if len(chunk) < max(50, window // 4):  # drop tiny trailing remainder
                break
            words = [w for w, _ in chunk]
            narr_c = Counter(a["claimed_narrator"] for _, a in chunk)
            spk_c = Counter(a["speaker"] for _, a in chunk)
            genre_c = Counter(a["genre"] for _, a in chunk)
            claimed, purity = _dominant(narr_c)
            speaker, _ = _dominant(spk_c)
            genre, _ = _dominant(genre_c)
            editorial = Counter(a["editorial_layer"] for _, a in chunk).most_common(1)[0][0]

            seg = ac.Segment(
                segment_id=f"BoM:win:{widx:05d}",
                segmentation="rolling",
                volume=VOLUME,
                book=book,
                chapter=None,
                verse_range=None,
                claimed_narrator=claimed,
                speaker=speaker,
                editorial_layer=editorial,
                genre=genre,
                reliability=narr_rel.get(claimed, "reliable"),
                word_count=len(words),
                narrator_purity=round(purity, 4),
                mixed_speaker=len([s for s in spk_c if spk_c[s] > 0]) > 1,
                text=" ".join(words),
                text_quote_removed=" ".join(words),
            )
            segments.append(seg.to_dict())
            widx += 1
            i += stride
    return segments


def build_narrator_segments(maps, narr_rel, spk_rel) -> list[dict]:
    narrator_map, embedded, genre_map = maps
    by_narr = defaultdict(list)
    by_spk = defaultdict(list)
    narr_layer = {}
    for ch in ac.iter_chapters(VOLUME):
        for v in ch["verses"]:
            ann = annotate_verse(ch["book"], ch["chapter"], v.verse, narrator_map, embedded, genre_map)
            by_narr[ann["claimed_narrator"]].append(v.text)
            by_spk[ann["speaker"]].append(v.text)
            narr_layer.setdefault(ann["claimed_narrator"], ann["editorial_layer"])

    segments = []
    for narrator, texts in by_narr.items():
        text = " ".join(texts)
        seg = ac.Segment(
            segment_id=f"BoM:narrator:{narrator}",
            segmentation="narrator",
            volume=VOLUME, book=None, chapter=None, verse_range=None,
            claimed_narrator=narrator, speaker=narrator,
            editorial_layer=narr_layer.get(narrator, "unknown"),
            genre="mixed",
            reliability=narr_rel.get(narrator, "reliable"),
            word_count=ac.word_count(text),
            text=text, text_quote_removed=text,
        )
        segments.append(seg.to_dict())

    for speaker, texts in by_spk.items():
        text = " ".join(texts)
        seg = ac.Segment(
            segment_id=f"BoM:speaker:{speaker}",
            segmentation="speaker",
            volume=VOLUME, book=None, chapter=None, verse_range=None,
            claimed_narrator=speaker, speaker=speaker,
            editorial_layer="mixed", genre="mixed",
            reliability=spk_rel.get(speaker, "reliable"),
            word_count=ac.word_count(text),
            text=text, text_quote_removed=text,
        )
        segments.append(seg.to_dict())
    return segments


def build_genre_segments(maps) -> list[dict]:
    narrator_map, embedded, genre_map = maps
    by_genre = defaultdict(list)
    for ch in ac.iter_chapters(VOLUME):
        for v in ch["verses"]:
            ann = annotate_verse(ch["book"], ch["chapter"], v.verse, narrator_map, embedded, genre_map)
            by_genre[ann["genre"]].append(v.text)
    segments = []
    for genre, texts in by_genre.items():
        text = " ".join(texts)
        seg = ac.Segment(
            segment_id=f"BoM:genre:{genre}",
            segmentation="genre",
            volume=VOLUME, book=None, chapter=None, verse_range=None,
            claimed_narrator="mixed", speaker="mixed",
            editorial_layer="mixed", genre=genre,
            reliability="aggregate",
            word_count=ac.word_count(text),
            text=text, text_quote_removed=text,
        )
        segments.append(seg.to_dict())
    return segments


def main():
    config = ac.load_config()
    maps = (ac.load_narrator_map(), ac.load_embedded_speakers(), ac.load_genre_map())

    index_path = SEG_DIR / "segment_index.json"
    if not index_path.exists():
        print("segment_index.json not found; run build_annotations.py first.")
        sys.exit(1)
    index = ac.load_json(index_path)
    narr_rel, spk_rel = _reliability_lookup(index)

    print("=" * 60)
    print("Step 2: Build segmentation sets")
    print("=" * 60)

    chap = build_chapter_segments(maps, narr_rel)
    ac.write_jsonl(SEG_DIR / "segments_chapter.jsonl", chap)
    print(f"  chapter segments:  {len(chap)} -> segments_chapter.jsonl")

    win = config["segmentation"]["rolling_window_size"]
    stride = config["segmentation"]["rolling_window_stride"]
    rolling = build_rolling_segments(maps, narr_rel, win, stride)
    ac.write_jsonl(SEG_DIR / "segments_rolling1000.jsonl", rolling)
    print(f"  rolling segments:  {len(rolling)} (window={win}, stride={stride}) -> segments_rolling1000.jsonl")

    narr = build_narrator_segments(maps, narr_rel, spk_rel)
    ac.write_jsonl(SEG_DIR / "segments_narrator.jsonl", narr)
    print(f"  narrator+speaker:  {len(narr)} -> segments_narrator.jsonl")

    genre = build_genre_segments(maps)
    ac.write_jsonl(SEG_DIR / "segments_genre.jsonl", genre)
    print(f"  genre segments:    {len(genre)} -> segments_genre.jsonl")

    # Update the segment index with set counts for the sanity check.
    total_words_chapter = sum(s["word_count"] for s in chap)
    index["segment_counts"] = {
        "chapter": len(chap),
        "rolling": len(rolling),
        "narrator_and_speaker": len(narr),
        "genre": len(genre),
    }
    index["chapter_total_words"] = total_words_chapter
    ac.write_json(index_path, index)

    assert len(chap) == 239, f"expected 239 chapter segments, got {len(chap)}"
    print(f"\nChapter-count sanity OK (239). Total chapter words: {total_words_chapter}")


if __name__ == "__main__":
    main()
