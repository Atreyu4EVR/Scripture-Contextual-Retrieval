"""
Step 6: Unsupervised analysis of Book of Mormon segments.

- Standardize -> PCA -> UMAP 2D projection of chapter segments.
- KMeans (over a k grid) + Agglomerative clustering; silhouette + ARI/AMI of cluster
  assignments against four label sets: claimed_narrator, genre, editorial_layer,
  quote_status.
- Burrows's Delta pairwise distance matrices between reliable narrators
  (full corpus and quote-removed).

Writes results JSON to authorship/results/raw/run_<ts>/projection.json and
delta_matrix.json, and figures (UMAP scatters, Delta heatmaps) to
authorship/results/figures/run_<ts>/.

Run: python authorship/scripts/analyze_unsupervised.py [--run-id run_YYYYMMDD_HHMMSS]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac
from stylometry_metrics import ClusterMetrics

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

META_PREFIXES = ("fw_", "cng_", "wng_", "pos_", "punct_", "sent_")
SCALAR_FEATS = ("ttr", "mattr", "hapax_ratio", "dis_ratio")
LABEL_SETS = ["claimed_narrator", "genre", "editorial_layer", "quote_status"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(META_PREFIXES)]
    cols += [c for c in SCALAR_FEATS if c in df.columns]
    return cols


def add_quote_status(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["quote_status"] = np.where(df["quote_fraction"].fillna(0) > threshold,
                                  "mostly_quote", "original")
    return df


def run_projection(df, config, seed):
    cols = feature_columns(df)
    X = df[cols].fillna(0).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(config["analysis"]["pca_components"], Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp = pca.fit_transform(Xs)

    import umap
    reducer = umap.UMAP(n_neighbors=config["analysis"]["umap_neighbors"],
                        min_dist=config["analysis"]["umap_min_dist"],
                        n_components=2, random_state=seed)
    emb = reducer.fit_transform(Xp)
    return Xp, emb


def cluster_and_score(Xp, emb, df, config, seed):
    results = []
    best = None
    for k in config["analysis"]["kmeans_k_grid"]:
        if k >= Xp.shape[0]:
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xp)
        sil = float(silhouette_score(Xp, km.labels_))
        cm = ClusterMetrics(method="kmeans", k=k, silhouette=sil)
        for ls in LABEL_SETS:
            if ls in df.columns:
                cm.ari_vs[ls] = float(adjusted_rand_score(df[ls].astype(str), km.labels_))
                cm.ami_vs[ls] = float(adjusted_mutual_info_score(df[ls].astype(str), km.labels_))
        results.append(cm)
        if best is None or sil > best[0]:
            best = (sil, km.labels_, k)

    agg_k = best[2] if best else 4
    agg = AgglomerativeClustering(n_clusters=agg_k).fit(Xp)
    agg_cm = ClusterMetrics(method="agglomerative", k=agg_k,
                            silhouette=float(silhouette_score(Xp, agg.labels_)))
    for ls in LABEL_SETS:
        if ls in df.columns:
            agg_cm.ari_vs[ls] = float(adjusted_rand_score(df[ls].astype(str), agg.labels_))
            agg_cm.ami_vs[ls] = float(adjusted_mutual_info_score(df[ls].astype(str), agg.labels_))
    results.append(agg_cm)
    return results, (best[1] if best else None)


def plot_umap(emb, df, label_set, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = df[label_set].astype(str).to_numpy()
    for lab in sorted(set(labels)):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.75, label=lab)
    ax.set_title(f"UMAP of BoM chapter segments — colored by {label_set}")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=7, markerscale=1.2, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def narrator_delta_matrix(delta_df, chapter_df, reliabilities=("reliable", "borderline")):
    groups = {}
    rel = chapter_df.set_index("segment_id")["reliability"].to_dict()
    narr = chapter_df.set_index("segment_id")["claimed_narrator"].to_dict()
    for sid in delta_df.index:
        if rel.get(sid) in reliabilities:
            groups.setdefault(narr[sid], []).append(sid)
    names = sorted(groups)
    means = {n: delta_df.loc[delta_df.index.intersection(groups[n])].mean(axis=0) for n in names}
    mat = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = float(np.abs(means[a] - means[b]).mean())
    return names, mat


def plot_heatmap(names, mat, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
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
    print(f"Step 6: Unsupervised analysis  (run {run_id})")
    print("=" * 60)

    df = pd.read_parquet(ac.FEATURES_DIR / "features_chapter.parquet")
    df = add_quote_status(df, config["biblical_quotes"]["segment_mostly_quote_threshold"])
    delta_df = pd.read_parquet(ac.FEATURES_DIR / "delta_matrix_chapter.parquet")

    Xp, emb = run_projection(df, config, seed)
    cluster_results, best_labels = cluster_and_score(Xp, emb, df, config, seed)

    for ls in LABEL_SETS:
        plot_umap(emb, df, ls, fig_dir / f"umap_{ls}.png")

    names, mat = narrator_delta_matrix(delta_df, df)
    plot_heatmap(names, mat, "Burrows's Delta between narrators (full)",
                 fig_dir / "delta_narrators_full.png")

    # Quote-removed Delta.
    delta_qr = pd.read_parquet(ac.FEATURES_DIR / "delta_matrix_quote_removed.parquet")
    names_qr, mat_qr = narrator_delta_matrix(delta_qr, df)
    plot_heatmap(names_qr, mat_qr, "Burrows's Delta between narrators (quote-removed)",
                 fig_dir / "delta_narrators_quote_removed.png")

    projection = {
        "run_id": run_id,
        "umap": [{"segment_id": sid, "x": float(emb[i, 0]), "y": float(emb[i, 1]),
                  "claimed_narrator": df.iloc[i]["claimed_narrator"],
                  "genre": df.iloc[i]["genre"],
                  "editorial_layer": df.iloc[i]["editorial_layer"],
                  "quote_status": df.iloc[i]["quote_status"]}
                 for i, sid in enumerate(df["segment_id"])],
        "clusters": [c.__dict__ for c in cluster_results],
    }
    ac.write_json(raw_dir / "projection.json", projection)
    ac.write_json(raw_dir / "delta_matrix.json", {
        "full": {"names": names, "matrix": mat.tolist()},
        "quote_removed": {"names": names_qr, "matrix": mat_qr.tolist()},
    })

    print("\nCluster agreement (ARI) vs label sets (best KMeans by silhouette):")
    best = max((c for c in cluster_results if c.method == "kmeans"),
               key=lambda c: c.silhouette, default=None)
    if best:
        print(f"  k={best.k}, silhouette={best.silhouette:.3f}")
        for ls, ari in best.ari_vs.items():
            print(f"    ARI vs {ls:18s}: {ari:.3f}")
    print(f"\nFigures -> {fig_dir}")
    print(f"Results -> {raw_dir}")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
