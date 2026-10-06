"""Fit one final calibrated model on all rows, save it with metadata, and score new rows."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .config import Settings
from .data import Dataset
from .features import FeatureSet, select
from .models import build_pipeline, get_model_spec
from .nested_cv import fit_calibrated
from .risk import RiskBands
from .schema import ID_COLUMN, SCHEMA_VERSION, SchemaError, validate
from .tuning import tune


@dataclass
class TrainedModel:
    estimator: object
    feature_set: FeatureSet
    model: str
    params: dict
    risk_bands: tuple[float, ...]
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        report = validate(frame, require_label=False)
        if not report.ok:
            raise SchemaError(report.errors)
        X = select(frame, self.feature_set).apply(pd.to_numeric, errors="coerce")
        p = self.estimator.predict_proba(X)[:, 1]
        bands = RiskBands(self.risk_bands).assign(p)
        out = pd.DataFrame({"risk_probability": np.round(p, 6), "risk_band": bands})
        if ID_COLUMN in frame.columns:
            out.insert(0, ID_COLUMN, frame[ID_COLUMN].to_numpy())
        return out


def train_final(dataset: Dataset, fs: FeatureSet, model: str, settings: Settings) -> TrainedModel:
    """Tune with CV on all rows, then fit inner-CV calibration on all rows.

    The final model has no held-out estimate of its own. Use `evaluate` or `compare`
    (nested CV) for the performance numbers.
    """
    spec = get_model_spec(model)
    pipeline = build_pipeline(fs, model, settings.seed)
    X = select(dataset.frame, fs)
    y = dataset.y
    tuning = tune(pipeline, spec.grid, X, y, settings.inner_folds, settings.seed,
                  backend=settings.tuning_backend, n_trials=settings.optuna_trials)
    fitted = fit_calibrated(pipeline, tuning.params, X, y, settings.inner_folds, settings.seed,
                            settings.calibration, model)
    metadata = {
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(y)),
        "training_prevalence": float(y.mean()),
        "data_sha256": dataset.sha256,
        "data_source": dataset.source,
        "seed": settings.seed,
        "calibration": settings.calibration,
        "inner_cv_auroc": tuning.inner_auc,
    }
    return TrainedModel(fitted, fs, model, tuning.params, tuple(settings.risk_bands), metadata)


def save(model: TrainedModel, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> TrainedModel:
    """Load a saved model. joblib files run code on load: open only files that you made."""
    obj = joblib.load(path)
    if not isinstance(obj, TrainedModel):
        raise TypeError(f"{path} does not contain a neurorisk TrainedModel")
    return obj
