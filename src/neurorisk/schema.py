"""Column contract for the tabular Alzheimer's dataset and a validator for it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1"

ID_COLUMN = "PatientID"
LABEL_COLUMN = "Diagnosis"
IGNORED_COLUMNS = ("DoctorInCharge",)


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # "id", "label", "binary", "nominal", "ordinal", "continuous", "text"
    low: float | None = None
    high: float | None = None
    allowed: tuple[int, ...] | None = None

    @property
    def is_feature(self) -> bool:
        return self.kind in {"binary", "nominal", "ordinal", "continuous"}


def _b(name: str) -> ColumnSpec:
    return ColumnSpec(name, "binary", allowed=(0, 1))


def _c(name: str, low: float, high: float) -> ColumnSpec:
    return ColumnSpec(name, "continuous", low, high)


COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(ID_COLUMN, "id"),
    _c("Age", 18, 120),
    _b("Gender"),
    ColumnSpec("Ethnicity", "nominal", allowed=(0, 1, 2, 3)),
    ColumnSpec("EducationLevel", "ordinal", allowed=(0, 1, 2, 3)),
    _c("BMI", 10, 60),
    _b("Smoking"),
    _c("AlcoholConsumption", 0, 20),
    _c("PhysicalActivity", 0, 10),
    _c("DietQuality", 0, 10),
    _c("SleepQuality", 0, 10),
    _b("FamilyHistoryAlzheimers"),
    _b("CardiovascularDisease"),
    _b("Diabetes"),
    _b("Depression"),
    _b("HeadInjury"),
    _b("Hypertension"),
    _c("SystolicBP", 60, 250),
    _c("DiastolicBP", 30, 150),
    _c("CholesterolTotal", 50, 500),
    _c("CholesterolLDL", 10, 400),
    _c("CholesterolHDL", 5, 200),
    _c("CholesterolTriglycerides", 10, 1000),
    _c("MMSE", 0, 30),
    _c("FunctionalAssessment", 0, 10),
    _b("MemoryComplaints"),
    _b("BehavioralProblems"),
    _c("ADL", 0, 10),
    _b("Confusion"),
    _b("Disorientation"),
    _b("PersonalityChanges"),
    _b("DifficultyCompletingTasks"),
    _b("Forgetfulness"),
    ColumnSpec(LABEL_COLUMN, "label", allowed=(0, 1)),
    ColumnSpec("DoctorInCharge", "text"),
)

BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in COLUMNS}
FEATURE_COLUMNS: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.is_feature)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(c.name for c in COLUMNS if c.kind != "text")


class SchemaError(ValueError):
    """The data does not match the column contract."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:10]) + (" ..." if len(errors) > 10 else ""))
        self.errors = errors


@dataclass
class ValidationReport:
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(df: pd.DataFrame, require_label: bool = True) -> ValidationReport:
    """Check columns, types, value ranges, categories, missing labels and duplicate IDs.

    Missing feature values are allowed (the pipelines impute them on the training fold only)
    but they are counted as warnings. Unknown columns are ignored and reported as warnings,
    because model inputs come only from an explicit feature list.
    """
    report = ValidationReport(rows=len(df))
    required = [c for c in REQUIRED_COLUMNS if require_label or c != LABEL_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"missing columns: {missing}")
    unknown = [c for c in df.columns if c not in BY_NAME]
    if unknown:
        report.warnings.append(f"unknown columns are ignored: {unknown}")
    if len(df) == 0:
        report.errors.append("the table has no rows")

    for name in df.columns:
        spec = BY_NAME.get(name)
        if spec is None or spec.kind == "text":
            continue
        if spec.kind == "label" and not require_label:
            continue
        col = pd.to_numeric(df[name], errors="coerce")
        bad_type = int((col.isna() & df[name].notna()).sum())
        if bad_type:
            report.errors.append(f"{name}: {bad_type} non-numeric values")
        n_missing = int(df[name].isna().sum())
        if n_missing:
            if spec.kind in {"id", "label"}:
                report.errors.append(f"{name}: {n_missing} missing values")
            else:
                report.warnings.append(f"{name}: {n_missing} missing values (imputed inside the pipeline)")
        values = col.dropna()
        if spec.allowed is not None:
            outside = sorted(set(values.unique()) - set(spec.allowed))
            if outside:
                report.errors.append(f"{name}: values {outside[:5]} not in {list(spec.allowed)}")
        if spec.low is not None and (values < spec.low).any():
            report.errors.append(f"{name}: {int((values < spec.low).sum())} values below {spec.low}")
        if spec.high is not None and (values > spec.high).any():
            report.errors.append(f"{name}: {int((values > spec.high).sum())} values above {spec.high}")

    if ID_COLUMN in df.columns:
        dupes = int(df[ID_COLUMN].duplicated().sum())
        if dupes:
            report.warnings.append(
                f"{ID_COLUMN}: {dupes} repeated IDs (splits keep all rows of one patient in one fold)"
            )
    if require_label and LABEL_COLUMN in df.columns and report.ok:
        classes = set(pd.to_numeric(df[LABEL_COLUMN]).unique())
        if len(classes) < 2:
            report.errors.append(f"{LABEL_COLUMN}: needs both classes, found {sorted(classes)}")
    return report


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric dtypes for every numeric column of the contract."""
    out = df.copy()
    for name in out.columns:
        spec = BY_NAME.get(name)
        if spec is not None and spec.kind != "text":
            out[name] = pd.to_numeric(out[name], errors="coerce")
    if LABEL_COLUMN in out.columns and out[LABEL_COLUMN].notna().all():
        out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(np.int64)
    return out
