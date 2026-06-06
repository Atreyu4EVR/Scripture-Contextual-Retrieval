"""
Stylometry metrics for the authorship study.

Extends the repo's metrics_calculator.py dataclass + scipy idiom with structures
for authorship attribution, Burrows's Delta distance, and clustering agreement.
These are deliberately separate from the IR metrics (P@K/NDCG) in
scripts/metrics_calculator.py, which do not apply here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class AttributionMetrics:
    """Result of a supervised internal-attribution experiment."""
    condition: str               # e.g. "include_quotes|raw"
    classifier: str
    n_samples: int
    n_classes: int
    accuracy: float
    macro_f1: float
    baseline_accuracy: float     # stratified/most-frequent baseline
    per_class_f1: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)   # rows=true, cols=pred
    labels: list = field(default_factory=list)


@dataclass
class DeltaResult:
    narrator_a: str
    narrator_b: str
    delta: float


@dataclass
class ClusterMetrics:
    method: str
    k: int
    silhouette: float
    ari_vs: dict = field(default_factory=dict)   # label_set -> adjusted Rand index
    ami_vs: dict = field(default_factory=dict)   # label_set -> adjusted mutual info


def burrows_delta(delta_matrix, ids_a: list[str], ids_b: list[str]) -> float:
    """Mean absolute z-score difference between two groups' mean Delta vectors."""
    a = delta_matrix.loc[delta_matrix.index.intersection(ids_a)].mean(axis=0)
    b = delta_matrix.loc[delta_matrix.index.intersection(ids_b)].mean(axis=0)
    return float(np.abs(a - b).mean())


def significance(values_a: list[float], values_b: list[float], alpha: float = 0.05):
    """Welch's t-test between two unpaired metric samples. Returns (p, significant)."""
    if len(values_a) < 2 or len(values_b) < 2:
        return 1.0, False
    _, p = stats.ttest_ind(values_a, values_b, equal_var=False)
    return float(p), bool(p < alpha)


def macro_f1_from_confusion(confusion: np.ndarray) -> float:
    """Macro-averaged F1 directly from a confusion matrix (rows=true, cols=pred)."""
    f1s = []
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0
