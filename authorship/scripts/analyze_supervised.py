"""
Step 7: Supervised internal authorship attribution (leave-one-chapter-out CV).

Classifies Book of Mormon chapter segments by claimed_narrator (reliable + borderline
classes only) using leave-one-chapter-out cross-validation, so a held-out chapter never
leaks into training.

Core 2x2x2 grid:
  quote   : include_quotes (features_chapter) vs exclude_quotes (features_quote_removed)
  genre   : raw (all chapters)               vs narrative_only
  features: full                              vs nopunct (1830 punctuation is editorial)

Each cell is run with LogisticRegression and LinearSVC, compared against a stratified
baseline and a word-count-only confound probe.

Honors the calibration gate: if validation_calibration.json reports a failed positive
control and config.validation.require_calibration is true, results are marked UNCALIBRATED.

Writes authorship/results/raw/run_<ts>/attribution.json.
Run: python authorship/scripts/analyze_supervised.py [--run-id run_...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from stylometry_metrics import AttributionMetrics, macro_f1_from_confusion

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

META_PREFIXES = ("fw_", "cng_", "wng_", "pos_", "punct_", "sent_")
SCALAR_FEATS = ("ttr", "mattr", "hapax_ratio", "dis_ratio")
ELIGIBLE_RELIABILITY = {"reliable", "borderline"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(META_PREFIXES)]
    cols += [c for c in SCALAR_FEATS if c in df.columns]
    return cols


def make_classifier(name: str, seed: int):
    if name == "logistic":
        clf = LogisticRegression(max_iter=3000, C=1.0)
    elif name == "svc_linear":
        clf = LinearSVC(C=1.0, dual="auto")
    else:
        raise ValueError(name)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def eligible_frame(df: pd.DataFrame, min_per_class: int, narrative_only: bool) -> pd.DataFrame:
    sub = df[df["reliability"].isin(ELIGIBLE_RELIABILITY)].copy()
    if narrative_only:
        sub = sub[sub["genre"] == "narrative"]
    counts = sub["claimed_narrator"].value_counts()
    keep = counts[counts >= min_per_class].index
    return sub[sub["claimed_narrator"].isin(keep)].reset_index(drop=True)


def run_cell(df, condition, classifier, seed):
    cols = feature_columns(df)
    X = df[cols].fillna(0).to_numpy()
    y = df["claimed_narrator"].to_numpy()
    labels = sorted(set(y))
    if len(labels) < 2 or len(y) < 4:
        return None

    cv = LeaveOneOut()
    pipe = make_classifier(classifier, seed)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    acc = float(accuracy_score(y, y_pred))
    mf1 = float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y, y_pred, labels=labels)
    per_class = dict(zip(labels, f1_score(y, y_pred, labels=labels, average=None, zero_division=0)))

    base = DummyClassifier(strategy="stratified", random_state=seed)
    base_pred = cross_val_predict(base, X, y, cv=cv)
    base_acc = float(accuracy_score(y, base_pred))

    return AttributionMetrics(
        condition=condition, classifier=classifier, n_samples=len(y), n_classes=len(labels),
        accuracy=acc, macro_f1=mf1, baseline_accuracy=base_acc,
        per_class_f1={k: float(v) for k, v in per_class.items()},
        confusion=cm.tolist(), labels=labels,
    )


def wordcount_probe(df, seed):
    """Confound probe: can narrator be predicted from word_count alone?"""
    sub = df[df["reliability"].isin(ELIGIBLE_RELIABILITY)].copy()
    counts = sub["claimed_narrator"].value_counts()
    keep = counts[counts >= 4].index
    sub = sub[sub["claimed_narrator"].isin(keep)]
    if sub["claimed_narrator"].nunique() < 2:
        return None
    X = sub[["word_count"]].to_numpy()
    y = sub["claimed_narrator"].to_numpy()
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)
    return float(accuracy_score(y, pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = ac.load_config()
    seed = config["random_seed"]
    min_per_class = config["reliability"]["min_segments_for_classification"]
    classifiers = config["analysis"]["classifiers"]
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_dir = ac.RESULTS_DIR / "raw" / run_id

    print("=" * 60)
    print(f"Step 7: Supervised attribution (LOCO-CV)  (run {run_id})")
    print("=" * 60)

    # Calibration gate.
    calib_path = raw_dir / "validation_calibration.json"
    calibrated, calib_note = True, "calibration file not found (run validation harness)"
    if calib_path.exists():
        calib = ac.load_json(calib_path)
        calibrated = bool(calib.get("passed", False))
        calib_note = calib.get("summary", "")
    uncalibrated_flag = config["validation"]["require_calibration"] and not calibrated

    feat_full = pd.read_parquet(ac.FEATURES_DIR / "features_chapter.parquet")
    feat_qr = pd.read_parquet(ac.FEATURES_DIR / "features_quote_removed.parquet")
    feat_full_np = pd.read_parquet(ac.FEATURES_DIR / "features_chapter_nopunct.parquet")
    feat_qr_np = pd.read_parquet(ac.FEATURES_DIR / "features_quote_removed_nopunct.parquet")

    sources = {
        ("include_quotes", "full"): feat_full,
        ("exclude_quotes", "full"): feat_qr,
        ("include_quotes", "nopunct"): feat_full_np,
        ("exclude_quotes", "nopunct"): feat_qr_np,
    }

    results = []
    for (quote_cond, feat_variant), src in sources.items():
        # quote-removed parquet keeps mostly_quotation segments flagged; drop them when excluding.
        base_df = src
        if quote_cond == "exclude_quotes":
            base_df = src[src["reliability"] != "mostly_quotation"]
        for genre_cond, narrative_only in [("raw", False), ("narrative_only", True)]:
            df = eligible_frame(base_df, min_per_class, narrative_only)
            for clf in classifiers:
                cond = f"{quote_cond}|{genre_cond}|{feat_variant}"
                res = run_cell(df, cond, clf, seed)
                if res is None:
                    continue
                results.append(res)
                print(f"  {cond:34s} {clf:11s} acc={res.accuracy:.3f} "
                      f"macroF1={res.macro_f1:.3f} base={res.baseline_accuracy:.3f} "
                      f"(n={res.n_samples}, {res.n_classes} classes)")

    probe = wordcount_probe(feat_full, seed)

    out = {
        "run_id": run_id,
        "uncalibrated": uncalibrated_flag,
        "calibration_note": calib_note,
        "wordcount_only_accuracy": probe,
        "results": [r.__dict__ for r in results],
    }
    ac.write_json(raw_dir / "attribution.json", out)

    if uncalibrated_flag:
        print("\n*** RESULTS MARKED UNCALIBRATED: positive control failed and "
              "require_calibration=true. Interpret with extreme caution. ***")
    print(f"\nWord-count-only probe accuracy: {probe}")
    print(f"Results -> {raw_dir / 'attribution.json'}")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
