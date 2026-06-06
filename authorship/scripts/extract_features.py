"""
Step 4: Extract stylometric features for each segmentation set.

Produces, per set, a feature table (one row per segment) with raw relative-frequency
features plus the annotation metadata columns, in two variants:
  - full    (includes punct_* rates and sent_* sentence-length stats)
  - nopunct (drops punct_*/sent_*; char n-grams recomputed on punctuation-stripped text)

Also writes a Burrows's Delta matrix (z-scored top-N word frequencies) per set and a
feature manifest recording the fitted vocab and parameters.

Feature tables hold RAW relative frequencies (not globally z-scored) so the supervised
classifier can scale inside CV folds without leakage. The Delta matrix is a separate
descriptive artifact.

NOTE: 1830-edition Book of Mormon punctuation is the typesetter's, not authorial; the
nopunct variant is the mandatory ablation, not optional.

Run: python authorship/scripts/extract_features.py [--quick]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import spacy

import authorship_common as ac

SEG_DIR = ac.DATA_DIR / "segments"
SETS = {
    "chapter": "segments_chapter.jsonl",
    "rolling": "segments_rolling1000.jsonl",
    "narrator": "segments_narrator.jsonl",
    "genre": "segments_genre.jsonl",
    "quote_removed": "segments_quote_removed.jsonl",
}

META_COLS = [
    "segment_id", "segmentation", "volume", "book", "chapter", "claimed_narrator",
    "speaker", "editorial_layer", "genre", "reliability", "word_count",
    "contains_biblical_quote", "quote_fraction", "mixed_speaker", "narrator_purity",
]

# ~150 function words incl. KJV-archaic forms (the genre-robust core signal).
FUNCTION_WORDS = [
    "a", "an", "the", "this", "that", "these", "those", "such", "same", "other",
    "and", "but", "or", "nor", "for", "so", "yet", "if", "then", "than", "as",
    "because", "although", "though", "while", "whereas", "wherefore", "therefore",
    "thus", "hence", "notwithstanding", "nevertheless", "save", "lest",
    "of", "in", "on", "at", "by", "to", "from", "with", "without", "within",
    "into", "unto", "upon", "over", "under", "above", "beneath", "before",
    "after", "behind", "between", "among", "amongst", "against", "about",
    "through", "throughout", "toward", "towards", "concerning", "according",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "thou", "thee", "thy", "thine", "thyself", "ye", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "who", "whom", "whose", "which", "what", "whatsoever",
    "whosoever", "whoso",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doth", "dost", "hath", "shall",
    "should", "will", "would", "may", "might", "must", "can", "could", "ought",
    "not", "no", "yea", "nay", "all", "any", "some", "every", "each", "more",
    "most", "much", "many", "few", "now", "here", "there", "when", "where",
    "how", "why", "again", "even", "also", "only", "very", "behold", "verily",
]


_alnum_re = re.compile(r"[a-z0-9']+")
_punct_chars = ",;:.!?\"'()-"


def tokenize_words(text: str) -> list[str]:
    return _alnum_re.findall(text.lower())


def strip_punct(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def char_ngram_counter(text: str, sizes, stripped: bool) -> Counter:
    base = strip_punct(text) if stripped else text.lower()
    base = re.sub(r"\s+", " ", base)
    c = Counter()
    for n in sizes:
        for i in range(len(base) - n + 1):
            c[base[i : i + n]] += 1
    return c


def word_ngram_counter(tokens, sizes) -> Counter:
    c = Counter()
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            c["_".join(tokens[i : i + n])] += 1
    return c


def pos_ngram_counter(pos_tags, sizes) -> Counter:
    c = Counter()
    for n in sizes:
        for i in range(len(pos_tags) - n + 1):
            c["_".join(pos_tags[i : i + n])] += 1
    return c


def top_k_keys(global_counter: Counter, k: int) -> list[str]:
    return [key for key, _ in global_counter.most_common(k)]


def build_feature_table(segments, nlp, config, manifest_out=None):
    """Return (df_full, df_nopunct, delta_df, manifest) for a list of segment dicts."""
    fcfg = config["features"]
    char_sizes = fcfg["char_ngram_sizes"]
    word_sizes = fcfg["word_ngram_sizes"]
    pos_sizes = fcfg["pos_ngram_sizes"]

    # --- Pass 1: per-segment primitives + global counters for vocab selection ---
    prim = []
    g_char_full, g_char_np = Counter(), Counter()
    g_word, g_pos = Counter(), Counter()
    g_wordfreq = Counter()

    texts = [s["text"] for s in segments]
    docs = nlp.pipe(texts, batch_size=16)
    for seg, doc in zip(segments, docs):
        text = seg["text"]
        tokens = tokenize_words(text)
        pos_tags = [t.pos_ for t in doc if not t.is_space and not t.is_punct]
        try:
            sent_lens = [len([t for t in s if not t.is_space and not t.is_punct]) for s in doc.sents]
        except ValueError:
            sent_lens = []

        cf = char_ngram_counter(text, char_sizes, stripped=False)
        cnp = char_ngram_counter(text, char_sizes, stripped=True)
        wn = word_ngram_counter(tokens, word_sizes) if fcfg["enable_word_ngrams"] else Counter()
        pn = pos_ngram_counter(pos_tags, pos_sizes)

        g_char_full.update({k: 1 for k in cf})
        g_char_np.update({k: 1 for k in cnp})
        g_word.update({k: 1 for k in wn})
        g_pos.update({k: 1 for k in pn})
        g_wordfreq.update(tokens)

        prim.append({
            "seg": seg, "tokens": tokens, "pos": pos_tags, "sent_lens": sent_lens,
            "char_full": cf, "char_np": cnp, "word_ng": wn, "pos_ng": pn, "raw_text": text,
        })

    char_full_vocab = top_k_keys(g_char_full, fcfg["char_ngram_top_k"])
    char_np_vocab = top_k_keys(g_char_np, fcfg["char_ngram_top_k"])
    word_vocab = top_k_keys(g_word, fcfg["word_ngram_top_k"]) if fcfg["enable_word_ngrams"] else []
    pos_vocab = top_k_keys(g_pos, fcfg["pos_ngram_top_k"])
    delta_words = [w for w, _ in g_wordfreq.most_common(fcfg["delta_top_n_words"])]

    # --- Pass 2: build rows ---
    rows_full, rows_np, delta_rows = [], [], []
    for p in prim:
        seg = p["seg"]
        tokens = p["tokens"]
        n_tok = max(len(tokens), 1)
        per = fcfg["function_word_per"]
        tok_counter = Counter(tokens)

        row_meta = {c: seg.get(c) for c in META_COLS}

        # Function words (per-1000).
        fw = {f"fw_{w}": tok_counter.get(w, 0) / n_tok * per for w in FUNCTION_WORDS}

        # Lexical richness.
        types = len(set(tokens))
        ttr = types / n_tok
        hapax = sum(1 for _, c in tok_counter.items() if c == 1) / n_tok
        dis = sum(1 for _, c in tok_counter.items() if c == 2) / n_tok
        win = fcfg["mattr_window"]
        if len(tokens) >= win:
            ratios = []
            for i in range(0, len(tokens) - win + 1, max(1, win // 2)):
                w = tokens[i : i + win]
                ratios.append(len(set(w)) / win)
            mattr = float(np.mean(ratios)) if ratios else ttr
        else:
            mattr = ttr
        lex = {"ttr": ttr, "mattr": mattr, "hapax_ratio": hapax, "dis_ratio": dis}

        # POS n-grams (relative).
        pn_total = max(sum(p["pos_ng"].values()), 1)
        pos = {f"pos_{k}": p["pos_ng"].get(k, 0) / pn_total for k in pos_vocab}

        # Word n-grams (relative).
        wn_total = max(sum(p["word_ng"].values()), 1)
        word_ng = {f"wng_{k}": p["word_ng"].get(k, 0) / wn_total for k in word_vocab}

        # Char n-grams full + nopunct (relative).
        cf_total = max(sum(p["char_full"].values()), 1)
        cnp_total = max(sum(p["char_np"].values()), 1)
        char_full = {f"cng_{k}": p["char_full"].get(k, 0) / cf_total for k in char_full_vocab}
        char_np = {f"cng_{k}": p["char_np"].get(k, 0) / cnp_total for k in char_np_vocab}

        # Punctuation rates (per-1000), full only.
        raw = p["raw_text"]
        n_chars_words = max(len(raw.split()), 1)
        punct = {f"punct_{name}": raw.count(ch) / n_chars_words * per
                 for name, ch in [("comma", ","), ("semicolon", ";"), ("colon", ":"),
                                  ("period", "."), ("question", "?"), ("exclaim", "!"),
                                  ("dash", "-"), ("quote", '"')]}

        # Sentence-length stats, full only.
        sl = p["sent_lens"]
        if sl:
            sent = {"sent_mean": float(np.mean(sl)), "sent_std": float(np.std(sl)),
                    "sent_median": float(np.median(sl)),
                    "sent_iqr": float(np.subtract(*np.percentile(sl, [75, 25])))}
        else:
            sent = {"sent_mean": 0.0, "sent_std": 0.0, "sent_median": 0.0, "sent_iqr": 0.0}

        rows_full.append({**row_meta, **fw, **lex, **pos, **word_ng, **char_full, **punct, **sent})
        rows_np.append({**row_meta, **fw, **lex, **pos, **word_ng, **char_np})

        # Delta relative freqs (z-scored later).
        delta_rows.append({"segment_id": seg["segment_id"],
                           **{f"d_{w}": tok_counter.get(w, 0) / n_tok for w in delta_words}})

    df_full = pd.DataFrame(rows_full)
    df_np = pd.DataFrame(rows_np)

    # Burrows's Delta: z-score each word column across segments.
    delta_df = pd.DataFrame(delta_rows).set_index("segment_id")
    means = delta_df.mean(axis=0)
    stds = delta_df.std(axis=0).replace(0, 1.0)
    delta_df = (delta_df - means) / stds

    manifest = {
        "n_segments": len(segments),
        "function_words": FUNCTION_WORDS,
        "char_full_vocab_size": len(char_full_vocab),
        "char_np_vocab_size": len(char_np_vocab),
        "word_ngram_vocab_size": len(word_vocab),
        "pos_ngram_vocab_size": len(pos_vocab),
        "delta_top_n_words": len(delta_words),
        "params": fcfg,
    }
    return df_full, df_np, delta_df, manifest


def load_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 5_000_000
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Process only the rolling set with reduced vocab")
    parser.add_argument("--set", choices=list(SETS.keys()), default=None,
                        help="Process a single set")
    args = parser.parse_args()

    config = ac.load_config()
    if args.quick:
        config["features"]["char_ngram_top_k"] = 100
        config["features"]["word_ngram_top_k"] = 50
        config["features"]["pos_ngram_top_k"] = 50

    nlp = load_nlp()
    # --quick reduces the feature vocab (above) but still builds every set, since the
    # downstream calibration/analysis steps need the chapter and quote-removed tables.
    sets = [args.set] if args.set else list(SETS.keys())

    print("=" * 60)
    print("Step 4: Extract stylometric features")
    print("=" * 60)

    ac.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name in sets:
        segments = ac.read_jsonl(SEG_DIR / SETS[name])
        df_full, df_np, delta_df, manifest = build_feature_table(segments, nlp, config)
        df_full.to_parquet(ac.FEATURES_DIR / f"features_{name}.parquet", index=False)
        df_np.to_parquet(ac.FEATURES_DIR / f"features_{name}_nopunct.parquet", index=False)
        delta_df.to_parquet(ac.FEATURES_DIR / f"delta_matrix_{name}.parquet")
        manifests[name] = manifest
        print(f"  {name:14s} {len(segments):4d} segments, "
              f"{df_full.shape[1]} full cols / {df_np.shape[1]} nopunct cols")

    ac.write_json(ac.FEATURES_DIR / "feature_manifest.json", manifests)
    print(f"\nWrote feature tables + delta matrices + feature_manifest.json")


if __name__ == "__main__":
    main()
