"""
Phase 2 — Step E: Open-set authorship attribution of the Book of Mormon (H1/H2).

Places candidate authors (Joseph Smith via D&C, Ethan Smith, Solomon Spalding),
distractors (Edwards, Bunyan, Irving), and a translation control (KJV) in one feature
space with Book of Mormon windows, then asks which author — IF ANY — the Book of Mormon
text most resembles, with an explicit "none of the above" option.

Steps:
  1. Closed-set author CV — confirm the authors are separable from each other.
  2. Open-set calibration (leave-one-author-out) — measure whether a genuinely unseen
     author is rejected; pick a max-probability rejection threshold.
  3. Probabilistic attribution of BoM windows (logistic) + rejection at the threshold.
  4. Distance attribution (Burrows's Delta nearest-centroid) of BoM windows.

Writes authorship/results/raw/run_<ts>/open_set.json and figures.
Run: python authorship/scripts/analyze_open_set.py [--run-id run_...]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from analyze_supervised import feature_columns, make_classifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix


def logistic_proba_pipe(seed):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, C=1.0))])


def closed_set_cv(authors_df, cols, seed):
    X = authors_df[cols].fillna(0).to_numpy()
    y = authors_df["display"].to_numpy()
    labels = sorted(set(y))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(make_classifier("logistic", seed), X, y, cv=cv, n_jobs=-1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "labels": labels,
            "confusion": confusion_matrix(y, pred, labels=labels).tolist()}


def open_set_calibration(authors_df, cols, seed):
    """Leave-one-author-out: max-prob of held-out (unknown) windows should be low."""
    authors = sorted(authors_df["display"].unique())
    unknown_scores, known_scores = [], []
    for held in authors:
        train = authors_df[authors_df["display"] != held]
        test = authors_df[authors_df["display"] == held]
        if train["display"].nunique() < 2:
            continue
        pipe = logistic_proba_pipe(seed)
        pipe.fit(train[cols].fillna(0).to_numpy(), train["display"].to_numpy())
        proba = pipe.predict_proba(test[cols].fillna(0).to_numpy())
        unknown_scores.extend(proba.max(axis=1).tolist())
    # Known: in-distribution CV max-prob.
    X = authors_df[cols].fillna(0).to_numpy()
    y = authors_df["display"].to_numpy()
    proba_known = cross_val_predict(logistic_proba_pipe(seed), X, y,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                    method="predict_proba", n_jobs=-1)
    known_scores = proba_known.max(axis=1).tolist()
    # Threshold: reject ~90% of unknown windows.
    threshold = float(np.quantile(unknown_scores, 0.90)) if unknown_scores else 0.5
    unknown_reject_rate = float(np.mean([s < threshold for s in unknown_scores])) if unknown_scores else 0.0
    known_retain_rate = float(np.mean([s >= threshold for s in known_scores])) if known_scores else 0.0
    return {"threshold": threshold,
            "unknown_reject_rate": unknown_reject_rate,
            "known_retain_rate": known_retain_rate,
            "unknown_median_maxprob": float(np.median(unknown_scores)) if unknown_scores else None,
            "known_median_maxprob": float(np.median(known_scores)) if known_scores else None}


def probabilistic_attribution(authors_df, bom_df, cols, threshold, seed):
    pipe = logistic_proba_pipe(seed)
    pipe.fit(authors_df[cols].fillna(0).to_numpy(), authors_df["display"].to_numpy())
    classes = list(pipe.named_steps["clf"].classes_)
    proba = pipe.predict_proba(bom_df[cols].fillna(0).to_numpy())
    maxp = proba.max(axis=1)
    pred = [classes[i] for i in proba.argmax(axis=1)]
    rejected = maxp < threshold
    kept = Counter(p for p, r in zip(pred, rejected) if not r)
    return {
        "n_bom_windows": len(bom_df),
        "rejected_none_of_the_above": int(rejected.sum()),
        "reject_fraction": float(rejected.mean()),
        "predicted_when_not_rejected": dict(kept),
        "mean_class_probability": {c: float(proba[:, j].mean()) for j, c in enumerate(classes)},
        "median_max_probability": float(np.median(maxp)),
    }


def delta_attribution(delta_df, authors_df, bom_df):
    author_ids = {d: authors_df[authors_df["display"] == d]["segment_id"].tolist()
                  for d in sorted(authors_df["display"].unique())}
    centroids = {d: delta_df.loc[delta_df.index.intersection(ids)].mean(axis=0)
                 for d, ids in author_ids.items()}
    bom_ids = bom_df["segment_id"].tolist()
    bom_vecs = delta_df.loc[delta_df.index.intersection(bom_ids)]

    nearest = Counter()
    for sid, row in bom_vecs.iterrows():
        dists = {d: float(np.abs(row - c).mean()) for d, c in centroids.items()}
        nearest[min(dists, key=dists.get)] += 1
    bom_centroid = bom_vecs.mean(axis=0)
    mean_dist = {d: float(np.abs(bom_centroid - c).mean()) for d, c in centroids.items()}
    return {"nearest_author_distribution": dict(nearest),
            "bom_centroid_distance_to_author": dict(sorted(mean_dist.items(), key=lambda kv: kv[1]))}


def plot_bar(d, title, ylabel, path, sort_asc=False):
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=not sort_asc)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([k for k, _ in items], [v for _, v in items], color="#4477aa")
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.set_xticklabels([k for k, _ in items], rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = ac.load_config()
    seed = config["random_seed"]
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw_dir = ac.RESULTS_DIR / "raw" / run_id
    fig_dir = ac.RESULTS_DIR / "figures" / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 2 Step E: Open-set attribution  (run {run_id})")
    print("=" * 60)

    df = pd.read_parquet(ac.FEATURES_DIR / "features_attribution.parquet")
    delta_df = pd.read_parquet(ac.FEATURES_DIR / "delta_matrix_attribution.parquet")
    cols = feature_columns(df)

    authors_df = df[~df["is_bom"]].reset_index(drop=True)
    bom_df = df[df["is_bom"]].reset_index(drop=True)

    closed = closed_set_cv(authors_df, cols, seed)
    print(f"  Closed-set author accuracy: {closed['accuracy']:.3f} "
          f"({len(closed['labels'])} authors)")

    calib = open_set_calibration(authors_df, cols, seed)
    print(f"  Open-set threshold={calib['threshold']:.3f}  "
          f"unknown-reject={calib['unknown_reject_rate']:.2f}  "
          f"known-retain={calib['known_retain_rate']:.2f}")

    prob = probabilistic_attribution(authors_df, bom_df, cols, calib["threshold"], seed)
    print(f"  BoM probabilistic: reject(none-of-the-above)={prob['reject_fraction']:.2f}; "
          f"kept-predictions={prob['predicted_when_not_rejected']}")

    dist = delta_attribution(delta_df, authors_df, bom_df)
    print(f"  BoM Delta nearest-author distribution: {dist['nearest_author_distribution']}")
    nearest_overall = next(iter(dist["bom_centroid_distance_to_author"]))
    print(f"  BoM centroid nearest author (Delta): {nearest_overall}")

    plot_bar(prob["mean_class_probability"], "Mean logistic probability of each author for BoM windows",
             "mean P(author | BoM window)", fig_dir / "openset_mean_probability.png")
    plot_bar(dist["bom_centroid_distance_to_author"],
             "Burrows's Delta distance: BoM centroid to each author (lower = closer)",
             "Delta distance", fig_dir / "openset_delta_distance.png", sort_asc=True)

    out = {
        "run_id": run_id,
        "closed_set": closed,
        "open_set_calibration": calib,
        "bom_probabilistic_attribution": prob,
        "bom_delta_attribution": dist,
        "authors": authors_df["display"].value_counts().to_dict(),
    }
    ac.write_json(raw_dir / "open_set.json", out)
    print(f"\nResults -> {raw_dir / 'open_set.json'}")
    print(f"Figures -> {fig_dir}")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
