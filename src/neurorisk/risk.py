"""Risk = a calibrated predicted probability. Bands are labels on that probability, with their observed rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LABELS = ("low", "moderate", "elevated", "high")


@dataclass(frozen=True)
class RiskBands:
    cutoffs: tuple[float, ...] = (0.10, 0.30, 0.60)
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if any(b <= a for a, b in zip(self.cutoffs, self.cutoffs[1:])):
            raise ValueError("band cut-offs must increase strictly")
        if any(not 0 < c < 1 for c in self.cutoffs):
            raise ValueError("band cut-offs must be inside (0, 1)")
        if self.labels is not None and len(self.labels) != len(self.cutoffs) + 1:
            raise ValueError("need exactly one more label than cut-offs")

    @property
    def names(self) -> tuple[str, ...]:
        if self.labels is not None:
            return self.labels
        if len(self.cutoffs) + 1 == len(DEFAULT_LABELS):
            return DEFAULT_LABELS
        return tuple(f"band_{i}" for i in range(len(self.cutoffs) + 1))

    def ranges(self) -> list[tuple[str, float, float]]:
        edges = (0.0, *self.cutoffs, 1.0)
        return [(name, edges[i], edges[i + 1]) for i, name in enumerate(self.names)]

    def assign(self, probabilities) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
            raise ValueError("risk probabilities must be inside [0, 1]")
        idx = np.searchsorted(np.asarray(self.cutoffs), p, side="right")
        return np.asarray(self.names, dtype=object)[idx]

    def table(self, y, probabilities) -> list[dict]:
        """For each band: count, mean predicted risk and observed event rate on held-out rows."""
        y = np.asarray(y)
        p = np.asarray(probabilities, dtype=float)
        bands = self.assign(p)
        rows = []
        for name, lo, hi in self.ranges():
            mask = bands == name
            rows.append({
                "band": name, "low": lo, "high": hi, "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else None,
                "observed_rate": float(y[mask].mean()) if mask.any() else None,
            })
        return rows
