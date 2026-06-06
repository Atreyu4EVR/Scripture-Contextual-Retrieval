"""
Phase 2 — Step C: Build fixed-window segments for the external comparison corpus.

Produces equal-length (default 1000-word) windows per author, balanced by capping the
number of windows per author, from:
  - cleaned external texts (Ethan Smith, Spalding, Edwards, Bunyan, Irving)
  - in-repo volumes (Joseph Smith = Doctrine and Covenants; KJV control = New Testament)

Windows are the unit for open-set attribution. Book of Mormon windows come from the
phase-one rolling set and are tagged separately at the feature step.

Writes authorship/data/external/segments_external.jsonl and external_segment_index.json.
Run: python authorship/scripts/build_external_segments.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

CLEAN_DIR = ac.DATA_DIR / "external" / "clean"
WINDOW = 1000
MAX_WINDOWS_PER_AUTHOR = 45  # balance classes; Spalding (smallest candidate) ~41 windows
MIN_WINDOWS = 8


def windows_from_tokens(tokens, window, cap, rng):
    chunks = [tokens[i:i + window] for i in range(0, len(tokens), window)]
    chunks = [c for c in chunks if len(c) >= window // 2]
    if len(chunks) > cap:
        chunks = rng.sample(chunks, cap)
    return chunks


def tokens_for_author(a, rng):
    src = a["source"]
    if src["type"] == "in_repo_volume":
        text = " ".join(v.text for v in ac.iter_verses(src["volume"]))
        return text.split()
    # external cleaned file
    path = CLEAN_DIR / f"{a['key']}.txt"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").split()


def main():
    cfg = ac.load_json(ac.CONFIG_DIR / "external_corpus.json")
    config = ac.load_config()
    rng = random.Random(config["random_seed"])

    print("=" * 60)
    print("Phase 2 Step C: Build external segments")
    print("=" * 60)

    segments = []
    index = {}
    for a in cfg["authors"]:
        toks = tokens_for_author(a, rng)
        chunks = windows_from_tokens(toks, WINDOW, MAX_WINDOWS_PER_AUTHOR, rng)
        if len(chunks) < MIN_WINDOWS:
            print(f"  {a['display']:28s} only {len(chunks)} windows (<{MIN_WINDOWS}); "
                  f"included but flagged low_support")
        for i, ch in enumerate(chunks):
            text = " ".join(ch)
            segments.append({
                "segment_id": f"EXT:{a['key']}:{i:03d}",
                "author": a["key"], "display": a["display"], "role": a["role"],
                "register": a.get("register", "unknown"), "is_bom": False,
                "word_count": len(ch), "text": text,
            })
        index[a["key"]] = {"display": a["display"], "role": a["role"],
                           "register": a.get("register", "unknown"),
                           "total_words": len(toks), "windows": len(chunks),
                           "low_support": len(chunks) < MIN_WINDOWS}
        print(f"  {a['display']:28s} {len(toks):7d} words -> {len(chunks):3d} windows  "
              f"[{a['role']}]")

    ac.write_jsonl(ac.DATA_DIR / "external" / "segments_external.jsonl", segments)
    ac.write_json(ac.DATA_DIR / "external" / "external_segment_index.json",
                  {"window_size": WINDOW, "max_per_author": MAX_WINDOWS_PER_AUTHOR,
                   "authors": index, "total_segments": len(segments)})
    print(f"\nTotal external segments: {len(segments)} -> segments_external.jsonl")


if __name__ == "__main__":
    main()
