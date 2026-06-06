"""
Phase 2 — Step D: Extract attribution features in a shared space.

Builds one feature table over the UNION of external-author windows and Book of Mormon
rolling windows, so authors and the Book of Mormon live in the same feature space (same
vocab, same Delta basis) — required for open-set attribution.

Outputs:
  authorship/features/features_attribution.parquet      (relative-frequency features)
  authorship/features/delta_matrix_attribution.parquet  (z-scored top-N word freqs)
Both carry author / display / role / register / is_bom columns.

Run: python authorship/scripts/extract_attribution_features.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from extract_features import build_feature_table, load_nlp

EXT_DIR = ac.DATA_DIR / "external"
SEG_DIR = ac.DATA_DIR / "segments"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = ac.load_config()
    if args.quick:
        config["features"]["char_ngram_top_k"] = 100
        config["features"]["word_ngram_top_k"] = 50
        config["features"]["pos_ngram_top_k"] = 50

    print("=" * 60)
    print("Phase 2 Step D: Extract attribution features (shared space)")
    print("=" * 60)

    external = ac.read_jsonl(EXT_DIR / "segments_external.jsonl")
    bom = ac.read_jsonl(SEG_DIR / "segments_rolling1000.jsonl")
    for b in bom:
        b["author"] = "book_of_mormon"
        b["display"] = "Book of Mormon"
        b["role"] = "target"
        b["register"] = "dictated_translation"
        b["is_bom"] = True

    union = external + bom
    meta = {s["segment_id"]: {"author": s["author"], "display": s["display"],
                              "role": s["role"], "register": s.get("register", "unknown"),
                              "is_bom": s["is_bom"]} for s in union}

    nlp = load_nlp()
    print(f"Extracting features for {len(union)} segments "
          f"({len(external)} author windows + {len(bom)} BoM windows) ...")
    df_full, _, delta_df, manifest = build_feature_table(union, nlp, config)

    # Attach attribution metadata.
    extra = pd.DataFrame([{"segment_id": sid, **meta[sid]} for sid in df_full["segment_id"]])
    df_full = df_full.merge(extra, on="segment_id", how="left")

    df_full.to_parquet(ac.FEATURES_DIR / "features_attribution.parquet", index=False)
    delta_df.to_parquet(ac.FEATURES_DIR / "delta_matrix_attribution.parquet")
    ac.write_json(ac.FEATURES_DIR / "attribution_feature_manifest.json", manifest)

    print(f"  wrote features_attribution.parquet {df_full.shape}")
    print(f"  wrote delta_matrix_attribution.parquet {delta_df.shape}")
    print("  authors:", df_full[~df_full["is_bom"]]["author"].value_counts().to_dict())


if __name__ == "__main__":
    main()
