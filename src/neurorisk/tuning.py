"""Hyperparameter tuning. It sees only the rows that the caller gives it (an outer training fold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


@dataclass
class TuningResult:
    params: dict
    inner_auc: float | None


def tune(pipeline, grid: dict, X: pd.DataFrame, y: np.ndarray, inner_folds: int, seed: int,
         backend: str = "grid", n_trials: int = 20) -> TuningResult:
    """Choose hyperparameters by inner cross-validation (ROC AUC) on X, y only."""
    if not grid:
        return TuningResult({}, None)
    cv = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    if backend == "grid":
        search = GridSearchCV(clone(pipeline), grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=False)
        search.fit(X, y)
        return TuningResult(dict(search.best_params_), float(search.best_score_))
    if backend == "optuna":
        return _tune_optuna(pipeline, grid, X, y, cv, seed, n_trials)
    raise ValueError(f"unknown tuning backend {backend!r}")


def _tune_optuna(pipeline, grid, X, y, cv, seed, n_trials) -> TuningResult:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("tuning backend 'optuna' needs: pip install 'neurorisk[optuna]'") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in grid.items()}
        model = clone(pipeline).set_params(**params)
        return float(np.mean(cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=1)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return TuningResult(dict(study.best_params), float(study.best_value))
