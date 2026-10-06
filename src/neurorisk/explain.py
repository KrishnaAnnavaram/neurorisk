"""Attributions computed on HELD-OUT rows: grouped (domain) and single-feature permutation importance.

Permutation importance on held-out rows measures the drop in ROC AUC when the values of a
feature, or of all features of one domain together, are shuffled. Training rows are never used,
so the ranking does not reward memorised training patterns. SHAP is an optional extra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import FeatureSet


def grouped_permutation_importance(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]],
                                   n_repeats: int = 10, seed: int = 0) -> dict[str, dict]:
    """Mean and standard deviation of the AUROC drop when each group of columns is shuffled together.

    One row permutation is used for all columns of a group, so the correlation inside the
    group is kept and only the link to the label is broken.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    out: dict[str, dict] = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            shuffled = X.copy()
            shuffled[cols] = X[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        out[name] = {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "columns": list(cols)}
    return out


def domain_importance(model, X, y, fs: FeatureSet, n_repeats: int = 10, seed: int = 0) -> dict[str, dict]:
    return grouped_permutation_importance(model, X, y, {d: list(c) for d, c in fs.domains.items()}, n_repeats, seed)


def feature_importance(model, X, y, fs: FeatureSet, n_repeats: int = 10, seed: int = 0) -> dict[str, dict]:
    return grouped_permutation_importance(model, X, y, {c: [c] for c in fs.columns}, n_repeats, seed)


def average_importance(per_fold: list[dict[str, dict]]) -> list[dict]:
    """Average fold-level importances. Return rows sorted by mean drop, largest first."""
    if not per_fold:
        return []
    names = per_fold[0].keys()
    rows = []
    for name in names:
        means = [fold[name]["mean"] for fold in per_fold]
        rows.append({"name": name, "mean_auroc_drop": float(np.mean(means)),
                     "fold_std": float(np.std(means)), "folds": len(means)})
    return sorted(rows, key=lambda r: r["mean_auroc_drop"], reverse=True)


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame, max_rows: int = 200):  # pragma: no cover
    """Model-agnostic SHAP values for the positive-class probability (optional extra)."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP needs the optional extra: pip install 'neurorisk[shap]'") from exc
    explainer = shap.Explainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1],
                               background.iloc[:max_rows])
    return explainer(X.iloc[:max_rows])
