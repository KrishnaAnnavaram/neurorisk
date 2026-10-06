import numpy as np
import pytest

from neurorisk import metrics
from neurorisk.risk import RiskBands


@pytest.fixture()
def calibrated_sample():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 4000)
    y = (rng.random(4000) < p).astype(int)
    return y, p


def test_bootstrap_ci_contains_point(calibrated_sample):
    y, p = calibrated_sample
    est = metrics.bootstrap_ci(y, p, "auroc", n_boot=200, seed=1)
    assert est.low <= est.value <= est.high
    assert est.high - est.low < 0.05


def test_perfect_and_random_auroc():
    y = np.array([0, 0, 1, 1] * 25)
    assert metrics.METRICS["auroc"](y, y.astype(float)) == 1.0
    rng = np.random.default_rng(1)
    assert abs(metrics.METRICS["auroc"](np.tile(y, 20), rng.random(2000)) - 0.5) < 0.05


def test_ece_and_slope_for_calibrated_predictions(calibrated_sample):
    y, p = calibrated_sample
    assert metrics.ece(y, p) < 0.03
    slope, intercept = metrics.calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.15 and abs(intercept) < 0.15


def test_overconfident_predictions_have_slope_below_one(calibrated_sample):
    y, p = calibrated_sample
    logit = np.log(p / (1 - p)) * 3
    slope, _ = metrics.calibration_slope_intercept(y, 1 / (1 + np.exp(-logit)))
    assert slope < 0.5


def test_calibration_table_counts_add_up(calibrated_sample):
    y, p = calibrated_sample
    rows = metrics.calibration_table(y, p)
    assert sum(r["count"] for r in rows) == len(y)


def test_paired_bootstrap_detects_a_real_difference(calibrated_sample):
    y, p = calibrated_sample
    noisy = np.clip(p + np.random.default_rng(2).normal(0, 0.4, p.size), 0, 1)
    d = metrics.paired_bootstrap(y, noisy, p, "auroc", n_boot=200, seed=0)
    assert d.difference > 0 and d.low > 0 and d.p_value < 0.05


def test_paired_bootstrap_of_identical_predictions(calibrated_sample):
    y, p = calibrated_sample
    d = metrics.paired_bootstrap(y, p, p, "auroc", n_boot=100, seed=0)
    assert d.difference == 0 and d.p_value == 1.0


def test_holm_and_bh_adjustments():
    p = [0.01, 0.04, 0.03, 0.5]
    assert np.allclose(metrics.holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.allclose(metrics.benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.5])
    assert all(a >= r for a, r in zip(metrics.holm(p), p))


def test_decision_curve_reference_lines():
    y = np.array([1, 0, 0, 0] * 50)
    rows = metrics.decision_curve(y, y.astype(float), thresholds=(0.1, 0.5))
    assert rows[0]["treat_none"] == 0.0
    assert rows[0]["model"] == pytest.approx(0.25)  # a perfect model gains the prevalence
    assert rows[1]["treat_all"] == pytest.approx(0.25 - 0.75)


def test_summarize_has_all_metrics(calibrated_sample):
    y, p = calibrated_sample
    s = metrics.summarize(y, p, n_boot=50)
    assert {"auroc", "auprc", "brier", "log_loss", "ece", "calibration_slope"} <= set(s)


def test_risk_bands_assign_monotone_labels():
    bands = RiskBands((0.1, 0.3, 0.6))
    labels = bands.assign([0.0, 0.05, 0.1, 0.29, 0.3, 0.7, 1.0])
    assert labels.tolist() == ["low", "low", "moderate", "moderate", "elevated", "high", "high"]


def test_risk_bands_reject_invalid_input():
    with pytest.raises(ValueError):
        RiskBands((0.5, 0.2))
    with pytest.raises(ValueError):
        RiskBands((0.1, 0.3)).assign([1.2])
    with pytest.raises(ValueError):
        RiskBands((0.1,), labels=("a", "b", "c"))


def test_risk_band_table_reports_observed_rates(calibrated_sample):
    # Problem 4: a band label is backed by the observed event rate of the rows in that band.
    y, p = calibrated_sample
    table = RiskBands((0.1, 0.3, 0.6)).table(y, p)
    assert sum(r["count"] for r in table) == len(y)
    rates = [r["observed_rate"] for r in table]
    assert rates == sorted(rates)
    for r in table:
        assert abs(r["observed_rate"] - r["mean_predicted"]) < 0.05


def test_generic_band_names():
    assert RiskBands((0.5,)).names == ("band_0", "band_1")
