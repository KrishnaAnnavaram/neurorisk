import numpy as np
import pandas as pd

from neurorisk.data import load, SyntheticSource
from neurorisk.nested_cv import outer_splits, run_nested_cv
from neurorisk.tuning import TuningResult, tune


def test_tuning_never_sees_outer_test_rows(dataset, feature_sets, fast_settings):
    # Problem 3: the prototype tuned on the same 20% split that it reported as the test set.
    seen = []

    def spy(pipeline, grid, X, y, inner_folds, seed, **kw):
        seen.append(set(X.index))
        return tune(pipeline, grid, X, y, inner_folds, seed, **kw)

    res = run_nested_cv(dataset, feature_sets["A"], "logreg", fast_settings, tuner=spy)
    assert len(seen) == fast_settings.outer_folds
    for rows, fold in zip(seen, res.folds):
        assert rows.isdisjoint(set(fold.test_index))
        assert rows == set(fold.train_index)


def test_every_row_gets_exactly_one_out_of_fold_prediction(dataset, feature_sets, fast_settings):
    res = run_nested_cv(dataset, feature_sets["A"], "logreg", fast_settings)
    all_test = np.concatenate([f.test_index for f in res.folds])
    assert sorted(all_test.tolist()) == list(range(len(dataset.y)))
    assert np.isfinite(res.oof_probability).all()
    assert ((res.oof_probability > 0) & (res.oof_probability < 1)).all()


def test_runs_are_reproducible_with_a_seed(dataset, feature_sets, fast_settings):
    a = run_nested_cv(dataset, feature_sets["B"], "logreg", fast_settings)
    b = run_nested_cv(dataset, feature_sets["B"], "logreg", fast_settings)
    assert np.allclose(a.oof_probability, b.oof_probability)


def test_one_patient_never_sits_on_both_sides():
    y = np.array([0, 1] * 30)
    groups = np.repeat(np.arange(20), 3)
    for train, test in outer_splits(y, groups, 4, seed=0):
        assert set(groups[train]).isdisjoint(set(groups[test]))


def test_repeated_patient_rows_stay_in_one_fold(feature_sets, fast_settings):
    ds = load(SyntheticSource(n_rows=200, seed=11))
    frame = pd.concat([ds.frame, ds.frame.iloc[:50]], ignore_index=True)  # 50 patients appear twice
    ds2 = type(ds)(frame=frame, report=ds.report, source="dup", sha256="x")
    res = run_nested_cv(ds2, feature_sets["A"], "prior", fast_settings)
    ids = frame["PatientID"].to_numpy()
    for f in res.folds:
        assert set(ids[f.train_index]).isdisjoint(set(ids[f.test_index]))


def test_ceiling_set_scores_higher_than_risk_factor_set(dataset, feature_sets, fast_settings):
    from sklearn.metrics import roc_auc_score
    a = run_nested_cv(dataset, feature_sets["A"], "logreg", fast_settings)
    c = run_nested_cv(dataset, feature_sets["C"], "logreg", fast_settings)
    assert roc_auc_score(c.y, c.oof_probability) > roc_auc_score(a.y, a.oof_probability) + 0.1


def test_prior_baseline_is_near_chance(dataset, feature_sets, fast_settings):
    from sklearn.metrics import roc_auc_score
    res = run_nested_cv(dataset, feature_sets["A"], "prior", fast_settings)
    assert abs(roc_auc_score(res.y, res.oof_probability) - 0.5) < 0.1


def test_tune_with_empty_grid_returns_no_params(dataset, feature_sets):
    from neurorisk.models import build_pipeline
    result = tune(build_pipeline(feature_sets["A"], "prior", 0), {}, None, None, 2, 0)
    assert result == TuningResult({}, None)
