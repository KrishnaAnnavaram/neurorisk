"""Discrimination, calibration and clinical-utility metrics with bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit logit(P(y=1)) = a + b * logit(p). A calibrated model has a=0 and b=1."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if np.ptp(logit) < 1e-9:
        return float("nan"), float("nan")
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Rows of (bin range, count, mean predicted risk, observed event rate) for non-empty bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({"low": float(edges[b]), "high": float(edges[b + 1]), "count": int(mask.sum()),
                         "mean_predicted": float(p[mask].mean()), "observed_rate": float(y[mask].mean())})
    return rows


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": lambda y, p: float(roc_auc_score(y, p)),
    "auprc": lambda y, p: float(average_precision_score(y, p)),
    "brier": lambda y, p: float(brier_score_loss(y, p)),
    "log_loss": lambda y, p: float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
    "ece": ece,
}


@dataclass
class Estimate:
    value: float
    low: float
    high: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.3f}, {self.high:.3f}]"


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)])


def bootstrap_ci(y, p, metric: str, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Estimate:
    """Point estimate and percentile CI. Resampling is stratified so both classes stay present."""
    y, p = np.asarray(y), np.asarray(p)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        stats[i] = fn(y[idx], p[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return Estimate(fn(y, p), float(lo), float(hi))


@dataclass
class PairedDifference:
    metric: str
    difference: float
    low: float
    high: float
    p_value: float
    p_adjusted: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(y, p_a, p_b, metric: str = "auroc", n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> PairedDifference:
    """Difference metric(p_b) - metric(p_a) on the same rows, with a two-sided bootstrap p-value."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        diffs[i] = fn(y[idx], p_b[idx]) - fn(y[idx], p_a[idx])
    point = fn(y, p_b) - fn(y, p_a)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    tail = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = float(min(1.0, 2 * max(tail, 1.0 / n_boot)))
    return PairedDifference(metric, float(point), float(lo), float(hi), p_value)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)[::-1]
    adjusted = np.empty(m)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        adjusted[i] = running
    return adjusted.tolist()


def decision_curve(y, p, thresholds=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)) -> list[dict]:
    """Net benefit of the model, of "treat all" and of "treat none" at each threshold probability."""
    y, p = np.asarray(y), np.asarray(p)
    n = y.size
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        flagged = p >= t
        tp = np.sum(flagged & (y == 1)) / n
        fp = np.sum(flagged & (y == 0)) / n
        w = t / (1 - t)
        rows.append({"threshold": float(t), "model": float(tp - fp * w),
                     "treat_all": float(prevalence - (1 - prevalence) * w), "treat_none": 0.0})
    return rows


def summarize(y, p, n_boot: int = 1000, seed: int = 0) -> dict:
    """All headline metrics with CIs, plus calibration slope and intercept."""
    out = {name: bootstrap_ci(y, p, name, n_boot, seed).to_dict() for name in METRICS}
    slope, intercept = calibration_slope_intercept(y, p)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    out["prevalence"] = float(np.mean(y))
    out["rows"] = int(np.size(y))
    return out
