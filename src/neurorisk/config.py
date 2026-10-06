"""Settings read from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

CALIBRATION_METHODS = ("sigmoid", "isotonic")
TUNING_BACKENDS = ("grid", "optuna")


class ConfigError(ValueError):
    """A setting has an invalid value."""


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[key.strip()] = value
    return values


def _int(env: Mapping[str, str], key: str, default: int, minimum: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def parse_bands(raw: str) -> tuple[float, ...]:
    """Parse risk band cut-offs such as "0.1,0.3,0.6" (strictly increasing, inside (0, 1))."""
    try:
        cuts = tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ConfigError(f"risk bands must be numbers, got {raw!r}") from exc
    if not cuts:
        raise ConfigError("risk bands need at least one cut-off")
    if any(not 0.0 < c < 1.0 for c in cuts):
        raise ConfigError(f"risk band cut-offs must be inside (0, 1), got {cuts}")
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ConfigError(f"risk band cut-offs must increase strictly, got {cuts}")
    return cuts


@dataclass(frozen=True)
class Settings:
    data_path: Path | None = None
    output_dir: Path = Path("outputs")
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    n_bootstrap: int = 1000
    permutation_repeats: int = 10
    calibration: str = "sigmoid"
    tuning_backend: str = "grid"
    optuna_trials: int = 20
    risk_bands: tuple[float, ...] = field(default=(0.10, 0.30, 0.60))

    def __post_init__(self) -> None:
        if self.calibration not in CALIBRATION_METHODS:
            raise ConfigError(f"calibration must be one of {CALIBRATION_METHODS}, got {self.calibration!r}")
        if self.tuning_backend not in TUNING_BACKENDS:
            raise ConfigError(f"tuning backend must be one of {TUNING_BACKENDS}, got {self.tuning_backend!r}")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ConfigError("outer and inner folds must be >= 2")

    def with_overrides(self, **changes) -> "Settings":
        clean = {k: v for k, v in changes.items() if v is not None}
        return replace(self, **clean)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: Path | None = Path(".env")) -> "Settings":
        merged: dict[str, str] = {}
        if env is None:
            if dotenv is not None:
                merged.update(_read_dotenv(dotenv))
            merged.update(os.environ)
        else:
            merged.update(env)
        data = merged.get("NEURORISK_DATA") or None
        bands_raw = merged.get("NEURORISK_RISK_BANDS")
        return cls(
            data_path=Path(data) if data else None,
            output_dir=Path(merged.get("NEURORISK_OUTPUT_DIR") or "outputs"),
            seed=_int(merged, "NEURORISK_SEED", 42, 0),
            outer_folds=_int(merged, "NEURORISK_OUTER_FOLDS", 5, 2),
            inner_folds=_int(merged, "NEURORISK_INNER_FOLDS", 3, 2),
            n_bootstrap=_int(merged, "NEURORISK_BOOTSTRAP", 1000, 50),
            permutation_repeats=_int(merged, "NEURORISK_PERMUTATION_REPEATS", 10, 1),
            calibration=merged.get("NEURORISK_CALIBRATION") or "sigmoid",
            tuning_backend=merged.get("NEURORISK_TUNING") or "grid",
            optuna_trials=_int(merged, "NEURORISK_OPTUNA_TRIALS", 20, 1),
            risk_bands=parse_bands(bands_raw) if bands_raw else (0.10, 0.30, 0.60),
        )
