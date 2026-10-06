"""Preprocessing and model pipelines. All preprocessing is fit inside the pipeline, on training rows only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureSet
from .schema import BY_NAME


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Impute and scale numeric columns, one-hot encode nominal columns.

    Scaling puts cholesterol (150-300) and binary flags (0/1) on one scale, so no feature
    dominates a linear model because of its unit.
    """
    nominal = [c for c in columns if BY_NAME[c].kind == "nominal"]
    numeric = [c for c in columns if c not in nominal]
    transformers = []
    if numeric:
        transformers.append(
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric)
        )
    if nominal:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                nominal,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def _logreg(seed: int):
    return LogisticRegression(max_iter=5000, random_state=seed)


def _rf(seed: int):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1)


def _hgb(seed: int):
    return HistGradientBoostingClassifier(random_state=seed, max_iter=200, early_stopping=False)


def _xgb(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("model 'xgb' needs the optional extra: pip install 'neurorisk[xgboost]'") from exc
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                         eval_metric="logloss", random_state=seed, n_jobs=1)


def _stack(seed: int):
    return StackingClassifier(
        estimators=[("logreg", _logreg(seed)), ("rf", _rf(seed)), ("hgb", _hgb(seed))],
        final_estimator=LogisticRegression(max_iter=5000),
        cv=StratifiedKFold(3, shuffle=True, random_state=seed),
        stack_method="predict_proba",
        n_jobs=1,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    factory: Callable[[int], object]
    grid: dict[str, list] = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    "prior": ModelSpec("prior", "Baseline: predicts the training prevalence for every row",
                       lambda seed: DummyClassifier(strategy="prior")),
    "logreg": ModelSpec("logreg", "L2 logistic regression (interpretable baseline)", _logreg,
                        {"model__C": [0.01, 0.1, 1.0, 10.0]}),
    "rf": ModelSpec("rf", "Random forest", _rf,
                    {"model__max_depth": [None, 6], "model__max_features": ["sqrt", 0.5]}),
    "hgb": ModelSpec("hgb", "Histogram gradient boosting", _hgb,
                     {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, None]}),
    "xgb": ModelSpec("xgb", "XGBoost (optional extra)", _xgb,
                     {"model__max_depth": [2, 4], "model__learning_rate": [0.03, 0.1]}),
    "stack": ModelSpec("stack", "Stacking of logreg, rf and hgb with a logistic meta-model", _stack),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]


def build_pipeline(feature_set: FeatureSet, model: str, seed: int) -> Pipeline:
    spec = get_model_spec(model)
    return Pipeline([("prep", build_preprocessor(feature_set.columns)), ("model", spec.factory(seed))])
