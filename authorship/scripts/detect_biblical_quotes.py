"""
Step 3: Detect KJV biblical quotation in the Book of Mormon (n-gram shingling).

Builds an n-gram (default 7-gram) index over the in-repo KJV Old + New Testament,
then scans each Book of Mormon chapter to flag KJV-quoted words, compute a per-verse
and per-segment quote fraction, identify quote spans and their KJV source, and emit
a quote-removed text variant.

Outputs:
  data/quotes/kjv_shingles.pkl     cached KJV n-gram index (gitignored)
  data/quotes/quote_spans.jsonl    merged quote spans with source + coverage
  data/quotes/quote_fractions.json per-chapter quote fraction summary
  data/segments/segments_chapter.jsonl       (updated in place with quote fields)
  data/segments/segments_quote_removed.jsonl (chapter granularity, quotes stripped)

Run: python authorship/scripts/detect_biblical_quotes.py [--skip-quotes]
Pure stdlib + the source JSON; no API keys.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

VOLUME = "Book of Mormon"
KJV_VOLUMES = ["Old Testament", "New Testament"]
SEG_DIR = ac.DATA_DIR / "segments"
QUOTE_DIR = ac.DATA_DIR / "quotes"

_norm_re = re.compile(r"[^a-z0-9]")


def normalize_word(w: str) -> str:
    return _norm_re.sub("", w.lower())


def gram_hash(gram: tuple[str, ...]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(gram).encode("utf-8"), digest_size=8).digest(), "big")


def build_kjv_index(n: int) -> dict[int, str]:
    """Map each KJV n-gram hash -> 'Book Chapter' source label (cross-verse grams included)."""
    index: dict[int, str] = {}
    for volume in KJV_VOLUMES:
        for ch in ac.iter_chapters(volume):
            label = f"{ch['book']} {ch['chapter']}"
            norm = [normalize_word(w) for v in ch["verses"] for w in v.text.split()]
            for i in range(len(norm) - n + 1):
                gram = tuple(norm[i : i + n])
                if "" in gram:
                    continue
                h = gram_hash(gram)
                index.setdefault(h, label)
    return index


def scan_chapter(ch: dict, kjv: dict[int, str], n: int, dilation: int):
    """Return per-verse coverage info + chapter quote-removed text."""
    tokens = []  # (orig_word, norm_word, verse_num)
    for v in ch["verses"]:
        for w in v.text.split():
            tokens.append((w, normalize_word(w), v.verse))
    norm = [t[1] for t in tokens]
    covered = [False] * len(tokens)
    source_counter: Counter = Counter()

    for i in range(len(norm) - n + 1):
        gram = tuple(norm[i : i + n])
        if "" in gram:
            continue
        h = gram_hash(gram)
        src = kjv.get(h)
        if src is not None:
            source_counter[src] += 1
            for j in range(i, i + n):
                covered[j] = True

    # Dilate covered runs to remove partial-fragment edges.
    if dilation > 0:
        dilated = covered[:]
        for i, c in enumerate(covered):
            if c:
                for j in range(max(0, i - dilation), min(len(covered), i + dilation + 1)):
                    dilated[j] = True
        covered = dilated

    # Per-verse aggregation.
    verse_tokens: dict[int, list[int]] = defaultdict(list)
    for idx, (_, _, vn) in enumerate(tokens):
        verse_tokens[vn].append(idx)
    per_verse = {}
    for vn, idxs in verse_tokens.items():
        cov = sum(1 for k in idxs if covered[k])
        per_verse[vn] = {"covered": cov, "total": len(idxs),
                         "fraction": cov / len(idxs) if idxs else 0.0}

    quote_removed = " ".join(tokens[k][0] for k in range(len(tokens)) if not covered[k])
    total = len(tokens)
    covered_total = sum(covered)
    return {
        "per_verse": per_verse,
        "quote_fraction": covered_total / total if total else 0.0,
        "covered_total": covered_total,
        "total": total,
        "sources": [s for s, _ in source_counter.most_common(5)],
        "quote_removed": quote_removed,
    }


def merge_spans(book: str, per_verse: dict, threshold: float) -> list[dict]:
    """Merge contiguous high-coverage verses into spans."""
    spans = []
    cur = None
    for vn in sorted(per_verse):
        hot = per_verse[vn]["fraction"] >= threshold
        if hot and cur is None:
            cur = [vn, vn]
        elif hot:
            cur[1] = vn
        elif cur is not None:
            spans.append({"book": book, "start_verse": cur[0], "end_verse": cur[1]})
            cur = None
    if cur is not None:
        spans.append({"book": book, "start_verse": cur[0], "end_verse": cur[1]})
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-quotes", action="store_true",
                        help="Reuse cached KJV index if present")
    args = parser.parse_args()

    config = ac.load_config()
    qc = config["biblical_quotes"]
    n = qc["ngram_n"]
    vthr = qc["verse_quote_threshold"]
    mostly = qc["segment_mostly_quote_threshold"]
    dilation = qc["span_dilation_words"]

    print("=" * 60)
    print("Step 3: Detect biblical quotation (KJV %d-gram shingling)" % n)
    print("=" * 60)

    cache = QUOTE_DIR / "kjv_shingles.pkl"
    if args.skip_quotes and cache.exists():
        with open(cache, "rb") as f:
            kjv = pickle.load(f)
        print(f"Loaded cached KJV index ({len(kjv):,} grams)")
    else:
        print("Building KJV index ...")
        kjv = build_kjv_index(n)
        QUOTE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(kjv, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Built KJV index ({len(kjv):,} grams) -> {cache}")

    chapter_segments = ac.read_jsonl(SEG_DIR / "segments_chapter.jsonl")
    seg_by_id = {s["segment_id"]: s for s in chapter_segments}

    all_spans = []
    quote_fractions = {}
    quote_removed_segments = []

    for ch in ac.iter_chapters(VOLUME):
        seg_id = f"BoM:{ch['book']}:{ch['chapter']}"
        res = scan_chapter(ch, kjv, n, dilation)

        seg = seg_by_id[seg_id]
        seg["quote_fraction"] = round(res["quote_fraction"], 4)
        seg["contains_biblical_quote"] = res["quote_fraction"] > 0.05
        seg["quote_source"] = res["sources"]
        seg["text_quote_removed"] = res["quote_removed"]

        quote_fractions[seg_id] = {
            "book": ch["book"], "chapter": ch["chapter"],
            "quote_fraction": round(res["quote_fraction"], 4),
            "covered": res["covered_total"], "total": res["total"],
            "sources": res["sources"],
        }
        for span in merge_spans(ch["book"], res["per_verse"], vthr):
            span["chapter"] = ch["chapter"]
            span["segment_id"] = seg_id
            all_spans.append(span)

        # Quote-removed variant of the chapter segment.
        qr = dict(seg)
        qr["segmentation"] = "quote_removed"
        qr["text"] = res["quote_removed"]
        qr["word_count"] = ac.word_count(res["quote_removed"])
        if res["quote_fraction"] > mostly:
            qr["genre"] = "biblical_quotation"
            qr["reliability"] = "mostly_quotation"
        quote_removed_segments.append(qr)

    ac.write_jsonl(SEG_DIR / "segments_chapter.jsonl", list(seg_by_id.values()))
    ac.write_jsonl(SEG_DIR / "segments_quote_removed.jsonl", quote_removed_segments)
    ac.write_jsonl(QUOTE_DIR / "quote_spans.jsonl", all_spans)
    ac.write_json(QUOTE_DIR / "quote_fractions.json", quote_fractions)

    n_quote_chapters = sum(1 for s in quote_fractions.values() if s["quote_fraction"] > mostly)
    top = sorted(quote_fractions.values(), key=lambda x: -x["quote_fraction"])[:8]
    print(f"\nChapters mostly quotation (>{mostly}): {n_quote_chapters}")
    print("Highest-quotation chapters:")
    for s in top:
        print(f"  {s['book']} {s['chapter']:>3}  frac={s['quote_fraction']:.2f}  src={s['sources'][:2]}")
    print(f"\nWrote quote_spans.jsonl ({len(all_spans)} spans), quote_fractions.json, "
          f"segments_quote_removed.jsonl")


if __name__ == "__main__":
    main()
