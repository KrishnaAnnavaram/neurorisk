"""Descriptive group comparisons on RAW schema features only, with multiple-testing correction.

The prototype tested model-derived scores (weighted by the diagnosis) for a diagnosis
difference, which is circular. This module refuses any column that is not a raw feature
of the schema, and it reports effect sizes and adjusted p-values for many tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .metrics import benjamini_hochberg, holm
from .schema import FEATURE_COLUMNS, LABEL_COLUMN


class CircularTestError(ValueError):
    """A requested column is not a raw feature (for example a model-derived score)."""


def compare_groups(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = list(FEATURE_COLUMNS) if columns is None else list(columns)
    bad = [c for c in columns if c not in FEATURE_COLUMNS]
    if bad:
        raise CircularTestError(f"only raw schema features can be tested, not {bad}")
    y = df[LABEL_COLUMN].to_numpy()
    rows = []
    for col in columns:
        a = df.loc[y == 1, col].dropna().to_numpy(dtype=float)
        b = df.loc[y == 0, col].dropna().to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial correlation: +1 means every case value is above every control value
        effect = 2 * u / (a.size * b.size) - 1
        rows.append({"feature": col, "median_case": float(np.median(a)), "median_control": float(np.median(b)),
                     "mean_case": float(a.mean()), "mean_control": float(b.mean()),
                     "rank_biserial": float(effect), "p_value": float(p)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"].tolist())
    out["p_bh"] = benjamini_hochberg(out["p_value"].tolist())
    return out.sort_values("p_value").reset_index(drop=True)
