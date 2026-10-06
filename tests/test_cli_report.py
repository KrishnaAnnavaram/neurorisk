import json

import pandas as pd
import pytest

from neurorisk import synthetic
from neurorisk.cli import main
from neurorisk.training import load_model

FAST = ["--outer-folds", "3", "--inner-folds", "2", "--bootstrap", "60", "--synthetic-rows", "300"]


def test_synth_and_validate(tmp_path, capsys):
    out = tmp_path / "s.csv"
    assert main(["synth", "--rows", "120", "--out", str(out)]) == 0
    assert main(["validate", "--data", str(out)]) == 0
    assert "schema: OK" in capsys.readouterr().out


def test_validate_fails_on_bad_file(tmp_path, capsys):
    df = synthetic.generate(60, seed=1)
    df.loc[0, "MMSE"] = 99
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    assert main(["validate", "--data", str(path)]) == 1
    assert "schema: FAILED" in capsys.readouterr().out


def test_compare_writes_report_with_synthetic_notice(tmp_path, capsys):
    # Problem 2: every report states that the data is synthetic.
    out = tmp_path / "rep"
    assert main(["compare", "--data", "synthetic", "--feature-sets", "A,C", "--out", str(out), *FAST]) == 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Synthetic data" in report and "ceiling" in report
    record = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert record["comparisons"][0]["comparison"] == "C - A"
    assert "p_adjusted" in record["comparisons"][0]
    card = (out / "model_card_A_logreg.md").read_text(encoding="utf-8")
    assert "not a diagnostic tool" in card


def test_evaluate_with_explain(tmp_path, capsys):
    out = tmp_path / "ev"
    assert main(["evaluate", "--feature-set", "B", "--model", "logreg", "--explain", "--out", str(out), *FAST]) == 0
    record = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "domain_importance" in record["results"][0]


def test_train_then_predict_round_trip(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    assert main(["train", "--feature-set", "A", "--out", str(model_path), *FAST]) == 0
    new = synthetic.generate(40, seed=99).drop(columns=["Diagnosis"])
    data_path = tmp_path / "new.csv"
    new.to_csv(data_path, index=False)
    preds_path = tmp_path / "p.csv"
    assert main(["predict", "--model", str(model_path), "--data", str(data_path), "--out", str(preds_path)]) == 0
    preds = pd.read_csv(preds_path)
    assert list(preds.columns) == ["PatientID", "risk_probability", "risk_band"]
    assert preds["risk_probability"].between(0, 1).all()
    trained = load_model(model_path)
    assert trained.metadata["training_rows"] == 300 and trained.feature_set.name == "A"


def test_predict_rejects_invalid_rows(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    main(["train", "--feature-set", "A", "--out", str(model_path), *FAST])
    bad = synthetic.generate(30, seed=5)
    bad.loc[0, "BMI"] = 500
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    assert main(["predict", "--model", str(model_path), "--data", str(path)]) == 2


def test_stats_command(capsys):
    assert main(["stats", "--synthetic-rows", "200"]) == 0
    assert "Holm" in capsys.readouterr().out


def test_unknown_feature_set_is_a_clean_error(capsys):
    assert main(["evaluate", "--feature-set", "Z", *FAST]) == 2


def test_load_model_rejects_other_objects(tmp_path):
    import joblib
    path = tmp_path / "x.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(TypeError):
        load_model(path)
