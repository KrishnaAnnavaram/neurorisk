"""Nested cross-validation: the outer folds measure, the inner folds tune and calibrate.

For each outer fold:
1. Select the feature-set columns (whitelist) for the outer-train and outer-test rows.
2. Tune hyperparameters with inner CV on the outer-train rows only.
3. Calibrate the tuned pipeline with inner CV on the outer-train rows only.
4. Predict calibrated probabilities for the outer-test rows (out-of-fold predictions).
5. Optionally compute permutation importance on the outer-test rows.

Splits are stratified by label and grouped by patient ID, so one patient never sits on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from . import explain
from .config import Settings
from .data import Dataset
from .features import FeatureSet, select
from .models import build_pipeline, get_model_spec
from .tuning import TuningResult, tune


@dataclass
class FoldRecord:
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    params: dict
    inner_auc: float | None
    domain_importance: dict | None = None
    feature_importance: dict | None = None


@dataclass
class NestedCVResult:
    feature_set: str
    model: str
    y: np.ndarray
    oof_probability: np.ndarray
    folds: list[FoldRecord] = field(default_factory=list)
    seed: int = 0


def fit_calibrated(pipeline, params: dict, X, y, inner_folds: int, seed: int, method: str, model_name: str):
    """Return the tuned pipeline wrapped in inner-CV probability calibration (fit on X, y only)."""
    tuned = clone(pipeline).set_params(**params)
    if model_name == "prior":
        return tuned.fit(X, y)
    calibrated = CalibratedClassifierCV(tuned, method=method,
                                        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed))
    return calibrated.fit(X, y)


def outer_splits(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def run_nested_cv(dataset: Dataset, fs: FeatureSet, model: str, settings: Settings, explain_folds: bool = False,
                  tuner: Callable[..., TuningResult] = tune) -> NestedCVResult:
    spec = get_model_spec(model)
    pipeline = build_pipeline(fs, model, settings.seed)
    X_all = select(dataset.frame, fs)
    y = dataset.y
    oof = np.full(len(y), np.nan)
    result = NestedCVResult(fs.name, model, y, oof, seed=settings.seed)

    for k, (train_idx, test_idx) in enumerate(outer_splits(y, dataset.groups, settings.outer_folds, settings.seed)):
        X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        tuning = tuner(pipeline, spec.grid, X_tr, y_tr, settings.inner_folds, settings.seed,
                       backend=settings.tuning_backend, n_trials=settings.optuna_trials)
        fitted = fit_calibrated(pipeline, tuning.params, X_tr, y_tr, settings.inner_folds, settings.seed,
                                settings.calibration, model)
        oof[test_idx] = fitted.predict_proba(X_te)[:, 1]
        record = FoldRecord(k, train_idx, test_idx, tuning.params, tuning.inner_auc)
        if explain_folds:
            record.domain_importance = explain.domain_importance(
                fitted, X_te, y_te, fs, settings.permutation_repeats, settings.seed + k)
            record.feature_importance = explain.feature_importance(
                fitted, X_te, y_te, fs, settings.permutation_repeats, settings.seed + k)
        result.folds.append(record)

    if np.isnan(oof).any():
        raise RuntimeError("some rows received no out-of-fold prediction")
    return result
