"""
Step 5: Method-validation / calibration harness (known-authorship controls).

Runs the same feature + attribution pipeline on KJV text, where authorship contrast
is real and known, to calibrate true-positive accuracy and false-positive behavior
BEFORE any Book of Mormon result is trusted.

Experiments (in-repo data only):
  Positive A : Old Testament vs New Testament (register/translation-era contrast)
  Positive B : Pauline epistles vs Synoptic Gospels vs Johannine writings
               (author-like contrast within one translation -- mirrors the BoM challenge)
  Negative   : arbitrary halves of a single book (Isaiah) -- should NOT separate
  Negative   : label-permutation on BoM narrators -- LOCO accuracy must collapse to baseline

Writes data/validation/segments_kjv_validation.jsonl, the feature table, and
authorship/results/raw/run_<ts>/validation_calibration.json (the gate read by Step 7).

Run: python authorship/scripts/run_validation_harness.py [--run-id run_...]
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from extract_features import build_feature_table, load_nlp
from analyze_supervised import feature_columns, make_classifier

from sklearn.model_selection import StratifiedKFold, cross_val_predict, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

PAULINE = {"Romans", "1 Corinthians", "2 Corinthians", "Galatians"}
SYNOPTIC = {"Matthew", "Mark", "Luke"}
JOHANNINE = {"John", "1 John", "2 John", "3 John"}


def kjv_chapter_segments(books_filter=None, volumes=("Old Testament", "New Testament"),
                         sample_per_volume=None, seed=42):
    """Build chapter-level segments from KJV volumes, optionally filtered/sampled."""
    rng = random.Random(seed)
    out = []
    for volume in volumes:
        vol_chapters = []
        for ch in ac.iter_chapters(volume):
            if books_filter is not None and ch["book"] not in books_filter:
                continue
            text = " ".join(v.text for v in ch["verses"])
            vol_chapters.append({
                "segment_id": f"{volume[:2]}:{ch['book']}:{ch['chapter']}",
                "segmentation": "kjv_validation",
                "volume": volume, "book": ch["book"], "chapter": ch["chapter"],
                "claimed_narrator": ch["book"], "speaker": ch["book"],
                "editorial_layer": "kjv", "genre": "scripture",
                "reliability": "reliable", "word_count": ac.word_count(text),
                "contains_biblical_quote": False, "quote_fraction": 0.0,
                "mixed_speaker": False, "narrator_purity": 1.0,
                "text": text, "text_quote_removed": text,
            })
        if sample_per_volume and len(vol_chapters) > sample_per_volume:
            vol_chapters = rng.sample(vol_chapters, sample_per_volume)
        out.extend(vol_chapters)
    return out


def classify(df, label_col, classifier, seed, folds=5):
    cols = feature_columns(df)
    X = df[cols].fillna(0).to_numpy()
    y = df[label_col].to_numpy()
    labels = sorted(set(y))
    min_count = min((y == l).sum() for l in labels)
    cv = StratifiedKFold(n_splits=min(folds, min_count), shuffle=True, random_state=seed)
    pipe = make_classifier(classifier, seed)
    pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, y_pred=pred, labels=labels, average="macro", zero_division=0))
    base = DummyClassifier(strategy="most_frequent").fit(X, y)
    base_acc = float((base.predict(X) == y).mean())
    return {"accuracy": acc, "macro_f1": mf1, "baseline_accuracy": base_acc,
            "labels": labels, "n": len(y)}


def label_permutation_bom(seed, trials):
    """Negative control: shuffle BoM narrator labels; accuracy should fall to baseline."""
    df = pd.read_parquet(ac.FEATURES_DIR / "features_chapter.parquet")
    df = df[df["reliability"].isin({"reliable", "borderline"})].copy()
    counts = df["claimed_narrator"].value_counts()
    df = df[df["claimed_narrator"].isin(counts[counts >= 4].index)].reset_index(drop=True)
    cols = feature_columns(df)
    X = df[cols].fillna(0).to_numpy()
    y_true = df["claimed_narrator"].to_numpy()
    rng = np.random.default_rng(seed)
    accs = []
    pipe = make_classifier("logistic", seed)
    for _ in range(trials):
        y = y_true.copy()
        rng.shuffle(y)
        pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=-1)
        accs.append(float(accuracy_score(y, pred)))
    base = float(max((y_true == l).mean() for l in set(y_true)))
    return {"permuted_mean_acc": float(np.mean(accs)), "permuted_std_acc": float(np.std(accs)),
            "majority_baseline": base, "trials": trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = ac.load_config()
    seed = config["random_seed"]
    vcfg = config["validation"]
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_dir = ac.RESULTS_DIR / "raw" / run_id

    print("=" * 60)
    print(f"Step 5: Validation harness (calibration)  (run {run_id})")
    print("=" * 60)

    nlp = load_nlp()

    # Union of chapters needed across KJV experiments (one feature extraction pass).
    ot_nt = kjv_chapter_segments(sample_per_volume=150, seed=seed)
    authors = kjv_chapter_segments(books_filter=PAULINE | SYNOPTIC | JOHANNINE,
                                   volumes=("New Testament",))
    isaiah = kjv_chapter_segments(books_filter={"Isaiah"}, volumes=("Old Testament",))

    union = {s["segment_id"]: s for s in (ot_nt + authors + isaiah)}
    segments = list(union.values())
    print(f"Extracting features for {len(segments)} KJV chapters ...")
    df_full, _, _, _ = build_feature_table(segments, nlp, config)
    ac.write_jsonl(ac.DATA_DIR / "validation" / "segments_kjv_validation.jsonl", segments)

    # Attach experiment labels.
    df_full["author_group"] = df_full["book"].map(
        lambda b: "Pauline" if b in PAULINE else "Synoptic" if b in SYNOPTIC
        else "Johannine" if b in JOHANNINE else None)

    # Positive A: OT vs NT.
    ot_nt_ids = {s["segment_id"] for s in ot_nt}
    dfA = df_full[df_full["segment_id"].isin(ot_nt_ids)]
    posA = classify(dfA, "volume", "logistic", seed)
    print(f"  Positive A (OT vs NT)        acc={posA['accuracy']:.3f} "
          f"macroF1={posA['macro_f1']:.3f} (n={posA['n']})")

    # Positive B: Pauline vs Synoptic vs Johannine.
    dfB = df_full[df_full["author_group"].notna()]
    posB = classify(dfB, "author_group", "logistic", seed)
    print(f"  Positive B (3 NT author grps) acc={posB['accuracy']:.3f} "
          f"macroF1={posB['macro_f1']:.3f} (n={posB['n']})")

    # Negative: Isaiah arbitrary halves.
    isa_ids = [s["segment_id"] for s in isaiah]
    dfN = df_full[df_full["segment_id"].isin(isa_ids)].copy()
    rng = random.Random(seed)
    half = dfN["segment_id"].tolist()
    rng.shuffle(half)
    cut = len(half) // 2
    assign = {sid: ("half_a" if i < cut else "half_b") for i, sid in enumerate(half)}
    dfN["arbitrary_half"] = dfN["segment_id"].map(assign)
    negI = classify(dfN, "arbitrary_half", "logistic", seed)
    print(f"  Negative (Isaiah halves)      acc={negI['accuracy']:.3f} "
          f"(baseline={negI['baseline_accuracy']:.3f})")

    # Negative: BoM label permutation.
    negP = label_permutation_bom(seed, vcfg["label_permutation_trials"])
    print(f"  Negative (BoM label perm)     permuted_acc={negP['permuted_mean_acc']:.3f}"
          f"±{negP['permuted_std_acc']:.3f} (majority={negP['majority_baseline']:.3f})")

    # Pass/fail gates.
    passA = posA["macro_f1"] >= vcfg["positive_control_ot_vs_nt_min_macro_f1"]
    passB = posB["macro_f1"] >= vcfg["positive_control_authors_min_macro_f1"]
    neg_ok = negI["accuracy"] <= negI["baseline_accuracy"] + 0.20  # within tolerance of chance
    passed = bool(passA and passB)

    summary = (f"PositiveA(OT/NT) macroF1={posA['macro_f1']:.3f} [{'PASS' if passA else 'FAIL'}]; "
               f"PositiveB(authors) macroF1={posB['macro_f1']:.3f} [{'PASS' if passB else 'FAIL'}]; "
               f"NegativeIsaiah acc={negI['accuracy']:.3f} [{'OK' if neg_ok else 'SUSPECT'}]")

    calibration = {
        "run_id": run_id, "passed": passed, "neg_control_ok": neg_ok, "summary": summary,
        "positive_a_ot_vs_nt": posA, "positive_b_authors": posB,
        "negative_isaiah_halves": negI, "negative_bom_permutation": negP,
        "gates": {
            "ot_vs_nt_min_macro_f1": vcfg["positive_control_ot_vs_nt_min_macro_f1"],
            "authors_min_macro_f1": vcfg["positive_control_authors_min_macro_f1"],
        },
    }
    ac.write_json(raw_dir / "validation_calibration.json", calibration)

    print(f"\n{'CALIBRATION PASSED' if passed else 'CALIBRATION FAILED'}: {summary}")
    print(f"Results -> {raw_dir / 'validation_calibration.json'}")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
