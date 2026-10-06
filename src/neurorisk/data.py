"""Data sources behind one small interface, plus loading with schema validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from . import synthetic
from .schema import ID_COLUMN, LABEL_COLUMN, SchemaError, ValidationReport, coerce, validate


class DataSource(Protocol):
    """Anything that returns the raw table. A remote source can implement the same method."""

    name: str

    def read(self) -> pd.DataFrame: ...


@dataclass
class CSVSource:
    path: Path
    name: str = "csv"

    def read(self) -> pd.DataFrame:
        if not Path(self.path).is_file():
            raise FileNotFoundError(
                f"{self.path} not found. See data/README.md for the download steps, "
                "or run `neurorisk synth` for an offline table."
            )
        return pd.read_csv(self.path)


@dataclass
class SyntheticSource:
    n_rows: int = 2000
    seed: int = 42
    missing_rate: float = 0.0
    name: str = "synthetic"

    def read(self) -> pd.DataFrame:
        return synthetic.generate(self.n_rows, self.seed, self.missing_rate)


@dataclass
class Dataset:
    frame: pd.DataFrame
    report: ValidationReport
    source: str
    sha256: str

    @property
    def y(self) -> np.ndarray:
        return self.frame[LABEL_COLUMN].to_numpy(dtype=int)

    @property
    def groups(self) -> np.ndarray:
        return self.frame[ID_COLUMN].to_numpy()


def frame_sha256(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def load(source: DataSource, require_label: bool = True) -> Dataset:
    """Read, validate and type-coerce a table. Raise SchemaError if the contract is broken."""
    raw = source.read()
    report = validate(raw, require_label=require_label)
    if not report.ok:
        raise SchemaError(report.errors)
    frame = coerce(raw).reset_index(drop=True)
    return Dataset(frame=frame, report=report, source=source.name, sha256=frame_sha256(frame))


def source_for(path: str | Path | None, synthetic_rows: int = 2000, seed: int = 42) -> DataSource:
    if path is None or str(path) == "synthetic":
        return SyntheticSource(n_rows=synthetic_rows, seed=seed)
    return CSVSource(Path(path))
