import numpy as np
import pytest

from neurorisk.features import FeatureSetError, parse_feature_sets, select
from neurorisk.models import MODELS, build_pipeline, get_model_spec

SYMPTOMS = {"MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems", "Confusion",
            "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"}


def test_feature_sets_are_nested(feature_sets):
    a, b, c = (set(feature_sets[k].columns) for k in "ABC")
    assert a < b < c


def test_main_set_has_no_symptom_or_cognitive_columns(feature_sets):
    # Problem 1: the main claim is about cardiometabolic factors, so set A must not contain symptoms.
    assert not SYMPTOMS & set(feature_sets["A"].columns)
    assert SYMPTOMS <= set(feature_sets["C"].columns)
    assert feature_sets["A"].role == "main" and feature_sets["C"].role == "ceiling"


def test_derived_columns_never_reach_the_model(dataset, feature_sets):
    # Problem 6: extra columns (old risk scores, categories, the label) are ignored by the whitelist.
    df = dataset.frame.copy()
    df["AlzheimersRiskScore"] = df["Diagnosis"] * 10.0
    df["CardiovascularRiskScore"] = np.random.default_rng(0).random(len(df))
    X = select(df, feature_sets["C"])
    assert list(X.columns) == feature_sets["C"].columns
    assert "AlzheimersRiskScore" not in X and "Diagnosis" not in X and "PatientID" not in X


def test_select_returns_a_copy(dataset, feature_sets):
    X = select(dataset.frame, feature_sets["A"])
    X.iloc[0, 0] = -999
    assert dataset.frame.loc[0, X.columns[0]] != -999


@pytest.mark.parametrize("raw, message", [
    ({"X": {"domains": {"d": ["Diagnosis"]}}}, "label"),
    ({"X": {"domains": {"d": ["AlzheimersRiskScore"]}}}, "not a feature"),
    ({"X": {"domains": {"d": ["BMI", "BMI"]}}}, "twice"),
    ({"X": {"extends": "Y", "domains": {}}, "Y": {"extends": "X", "domains": {}}}, "circular"),
    ({"X": {"domains": {}}}, "no columns"),
])
def test_invalid_feature_sets_are_rejected(raw, message):
    with pytest.raises(FeatureSetError, match=message):
        parse_feature_sets(raw)


def test_scaling_makes_units_irrelevant(dataset, feature_sets):
    # Problem 4: the prototype multiplied raw, unscaled values by weights, so cholesterol dominated.
    fs = feature_sets["A"]
    X = select(dataset.frame, fs)
    y = dataset.y
    p1 = build_pipeline(fs, "logreg", 0).fit(X, y).predict_proba(X)[:, 1]
    X2 = X.copy()
    X2["CholesterolTotal"] = X2["CholesterolTotal"] * 1000.0
    p2 = build_pipeline(fs, "logreg", 0).fit(X2, y).predict_proba(X2)[:, 1]
    assert np.allclose(p1, p2, atol=1e-6)


def test_preprocessing_handles_missing_values(feature_sets):
    from neurorisk import synthetic
    df = synthetic.generate(200, seed=8, missing_rate=0.2)
    fs = feature_sets["C"]
    model = build_pipeline(fs, "logreg", 0).fit(select(df, fs), df["Diagnosis"])
    p = model.predict_proba(select(df, fs))[:, 1]
    assert np.isfinite(p).all()


@pytest.mark.parametrize("name", ["prior", "logreg", "rf", "hgb", "stack"])
def test_core_models_fit_and_predict(name, dataset, feature_sets):
    fs = feature_sets["B"]
    X, y = select(dataset.frame, fs), dataset.y
    p = build_pipeline(fs, name, 0).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and ((p >= 0) & (p <= 1)).all()


def test_xgboost_model_with_optional_extra(dataset, feature_sets):
    pytest.importorskip("xgboost")
    fs = feature_sets["A"]
    p = build_pipeline(fs, "xgb", 0).fit(select(dataset.frame, fs), dataset.y).predict_proba(
        select(dataset.frame, fs))[:, 1]
    assert p.shape == (len(dataset.y),)


def test_unknown_model_name():
    with pytest.raises(ValueError):
        get_model_spec("magic")
    assert "logreg" in MODELS and "prior" in MODELS
