"""
Phase 2 — Step B: Clean the fetched external corpus.

For each raw text:
  - strip Project Gutenberg START/END boilerplate (and license trailer)
  - strip Google/archive scan boilerplate
  - de-hyphenate line-break splits ("giv-\\nen" -> "given")
  - drop OCR-garbage lines (low alphabetic ratio, mostly short/non-word tokens)
  - normalize whitespace

Writes authorship/data/external/clean/<key>.txt and clean_manifest.json with
retained word counts and an OCR-noise estimate (dropped-line fraction).

Run: python authorship/scripts/clean_external_corpus.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

RAW_DIR = ac.DATA_DIR / "external" / "raw"
CLEAN_DIR = ac.DATA_DIR / "external" / "clean"

GUT_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUT_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*", re.I | re.S)
GOOGLE_BOILER = re.compile(r"this is a digital copy of a book.*?(google)", re.I | re.S)
_word_re = re.compile(r"[A-Za-z]+")


def strip_boilerplate(text: str) -> str:
    m = GUT_START.search(text)
    if m:
        text = text[m.end():]
    m = GUT_END.search(text)
    if m:
        text = text[: m.start()]
    text = GOOGLE_BOILER.sub("", text)
    return text


def dehyphenate(text: str) -> str:
    # join "word-\n word" -> "wordword"
    return re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", text)


def is_good_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 20:
        return False
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return False
    alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
    if alpha_ratio < 0.72:
        return False
    words = _word_re.findall(s)
    if len(words) < 4:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.5 or avg_len > 11:
        return False
    # fraction of real-ish words (length >= 3)
    longish = sum(1 for w in words if len(w) >= 3) / len(words)
    return longish >= 0.5


def clean_text(raw: str) -> tuple[str, float]:
    raw = strip_boilerplate(raw)
    raw = dehyphenate(raw)
    lines = raw.split("\n")
    kept, dropped = [], 0
    for ln in lines:
        if ln.strip() == "":
            continue
        if is_good_line(ln):
            kept.append(ln.strip())
        else:
            dropped += 1
    total = len([l for l in lines if l.strip()])
    drop_frac = dropped / total if total else 0.0
    blob = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return blob, drop_frac


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Phase 2 Step B: Clean external corpus")
    print("=" * 60)

    manifest = {}
    for raw_path in sorted(RAW_DIR.glob("*.txt")):
        key = raw_path.stem
        raw = raw_path.read_text(encoding="utf-8", errors="replace")
        cleaned, drop_frac = clean_text(raw)
        out = CLEAN_DIR / f"{key}.txt"
        out.write_text(cleaned, encoding="utf-8")
        wc = ac.word_count(cleaned)
        manifest[key] = {"words": wc, "dropped_line_fraction": round(drop_frac, 3),
                         "raw_bytes": raw_path.stat().st_size}
        print(f"  {key:20s} {wc:7d} words  (dropped {drop_frac*100:4.1f}% of lines as noise)")

    ac.write_json(ac.DATA_DIR / "external" / "clean_manifest.json", manifest)
    print(f"\nCleaned {len(manifest)} sources -> {CLEAN_DIR}")


if __name__ == "__main__":
    main()
