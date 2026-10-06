import numpy as np
import pandas as pd
import pytest

from neurorisk import explain
from neurorisk.models import build_pipeline
from neurorisk.features import select
from neurorisk.nested_cv import run_nested_cv
from neurorisk.stats import CircularTestError, compare_groups


class SpyModel:
    """Records the row index of every frame that it scores."""

    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(set(X.index))
        return self.inner.predict_proba(X)


def test_importance_uses_only_the_rows_it_receives(dataset, feature_sets):
    # Problem 7: importance must come from held-out rows, never from training rows.
    fs = feature_sets["C"]
    X = select(dataset.frame, fs)
    train, test = X.iloc[:300], X.iloc[300:]
    model = SpyModel(build_pipeline(fs, "logreg", 0).fit(train, dataset.y[:300]))
    explain.domain_importance(model, test, dataset.y[300:], fs, n_repeats=2)
    assert all(rows == set(test.index) for rows in model.seen)


def test_domain_importance_finds_the_informative_domain(dataset, feature_sets):
    fs = feature_sets["C"]
    X = select(dataset.frame, fs)
    model = build_pipeline(fs, "logreg", 0).fit(X.iloc[:250], dataset.y[:250])
    imp = explain.domain_importance(model, X.iloc[250:], dataset.y[250:], fs, n_repeats=5, seed=0)
    assert imp["cognitive_functional"]["mean"] > imp["lifestyle"]["mean"]
    assert set(imp) == set(fs.domains)


def test_nested_cv_importance_is_per_held_out_fold(dataset, feature_sets, fast_settings):
    res = run_nested_cv(dataset, feature_sets["C"], "logreg", fast_settings, explain_folds=True)
    assert all(f.domain_importance is not None for f in res.folds)
    rows = explain.average_importance([f.domain_importance for f in res.folds])
    assert rows[0]["mean_auroc_drop"] >= rows[-1]["mean_auroc_drop"]
    assert rows[0]["folds"] == fast_settings.outer_folds


def test_stats_refuse_model_derived_scores(dataset):
    # Problem 5: testing a diagnosis-weighted score for a diagnosis difference is circular.
    df = dataset.frame.copy()
    df["AlzheimersRiskScore"] = df["Diagnosis"] * 2.0
    with pytest.raises(CircularTestError):
        compare_groups(df, ["AlzheimersRiskScore"])


def test_stats_adjust_for_multiple_tests(dataset):
    table = compare_groups(dataset.frame)
    assert len(table) > 20
    assert (table["p_holm"] >= table["p_value"] - 1e-12).all()
    assert (table["p_bh"] >= table["p_value"] - 1e-12).all()
    assert table["rank_biserial"].between(-1, 1).all()


def test_stats_find_the_generated_effect(dataset):
    table = compare_groups(dataset.frame, ["MMSE", "DietQuality"]).set_index("feature")
    assert table.loc["MMSE", "rank_biserial"] < 0  # cases have lower MMSE in the generator
    assert table.loc["MMSE", "p_value"] < table.loc["DietQuality", "p_value"]


def test_stats_skip_empty_groups():
    df = pd.DataFrame({"BMI": [np.nan, np.nan, 20.0, 21.0], "Diagnosis": [1, 1, 0, 0]})
    assert compare_groups(df, ["BMI"]).empty
