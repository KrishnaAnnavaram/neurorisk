import numpy as np
import pytest

from neurorisk import synthetic
from neurorisk.config import ConfigError, Settings, parse_bands
from neurorisk.data import CSVSource, load
from neurorisk.schema import FEATURE_COLUMNS, SchemaError, validate


def test_settings_from_env_reads_values():
    s = Settings.from_env({"NEURORISK_SEED": "9", "NEURORISK_OUTER_FOLDS": "4", "NEURORISK_RISK_BANDS": "0.2,0.5",
                           "NEURORISK_CALIBRATION": "isotonic", "NEURORISK_DATA": "x.csv"})
    assert s.seed == 9 and s.outer_folds == 4 and s.risk_bands == (0.2, 0.5)
    assert s.calibration == "isotonic" and str(s.data_path) == "x.csv"


def test_settings_defaults_without_env():
    s = Settings.from_env({})
    assert s.data_path is None and s.outer_folds == 5 and s.inner_folds == 3


@pytest.mark.parametrize("env", [{"NEURORISK_SEED": "abc"}, {"NEURORISK_OUTER_FOLDS": "1"},
                                 {"NEURORISK_CALIBRATION": "magic"}, {"NEURORISK_RISK_BANDS": "0.5,0.2"},
                                 {"NEURORISK_RISK_BANDS": "1.5"}])
def test_settings_reject_bad_values(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_parse_bands():
    assert parse_bands("0.1, 0.3,0.6") == (0.1, 0.3, 0.6)


def test_synthetic_data_passes_schema_and_is_deterministic():
    a = synthetic.generate(300, seed=1)
    b = synthetic.generate(300, seed=1)
    assert a.equals(b)
    assert validate(a).ok
    assert 0.2 < a["Diagnosis"].mean() < 0.55


def test_synthetic_missing_values_are_warnings_not_errors():
    df = synthetic.generate(300, seed=2, missing_rate=0.1)
    report = validate(df)
    assert report.ok
    assert any("missing values" in w for w in report.warnings)


def test_schema_rejects_out_of_range_and_bad_categories():
    df = synthetic.generate(100, seed=3)
    df.loc[0, "MMSE"] = 45
    df.loc[1, "Ethnicity"] = 7
    df.loc[2, "Diagnosis"] = 2
    report = validate(df)
    assert not report.ok
    text = " ".join(report.errors)
    assert "MMSE" in text and "Ethnicity" in text and "Diagnosis" in text


def test_schema_reports_missing_columns_and_missing_labels():
    df = synthetic.generate(100, seed=4).drop(columns=["BMI"])
    df.loc[0, "Diagnosis"] = np.nan
    report = validate(df)
    assert any("missing columns" in e for e in report.errors)
    assert any("Diagnosis" in e and "missing" in e for e in report.errors)


def test_schema_needs_both_classes():
    df = synthetic.generate(100, seed=5)
    df["Diagnosis"] = 0
    assert not validate(df).ok


def test_load_raises_schema_error(tmp_path):
    df = synthetic.generate(50, seed=6)
    df["Age"] = df["Age"].astype(object)
    df.loc[0, "Age"] = "old"
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SchemaError):
        load(CSVSource(path))


def test_missing_csv_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load(CSVSource(tmp_path / "nope.csv"))


def test_feature_columns_exclude_id_label_and_doctor():
    assert "PatientID" not in FEATURE_COLUMNS
    assert "Diagnosis" not in FEATURE_COLUMNS
    assert "DoctorInCharge" not in FEATURE_COLUMNS
