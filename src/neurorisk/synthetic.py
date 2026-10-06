"""Synthetic generator with the same column contract as the public dataset.

The generator makes the label depend weakly on cardiometabolic factors and age and strongly on
cognitive, functional and symptom columns. This copies the structure that matters for the study:
a "ceiling" feature set with disease symptoms is much easier than a risk-factor-only set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COLUMNS, LABEL_COLUMN


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = 2000, seed: int = 42, missing_rate: float = 0.0, prevalence: float = 0.35) -> pd.DataFrame:
    """Return `n_rows` schema-valid synthetic records. The same seed gives the same table."""
    if n_rows < 20:
        raise ValueError("n_rows must be >= 20")
    if not 0.0 <= missing_rate < 0.5:
        raise ValueError("missing_rate must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    n = n_rows

    def binary(p: float) -> np.ndarray:
        return (rng.random(n) < p).astype(int)

    age = rng.integers(60, 91, n).astype(float)
    bmi = np.clip(rng.normal(27.5, 5.0, n), 15, 40)
    diabetes = binary(0.15)
    hypertension = binary(0.15)
    cvd = binary(0.14)
    smoking = binary(0.29)
    sbp = np.clip(rng.normal(132, 18, n) + 10 * hypertension, 90, 179).round()
    dbp = np.clip(rng.normal(88, 12, n) + 5 * hypertension, 60, 119).round()
    chol_ldl = np.clip(rng.normal(125, 35, n), 50, 200)
    chol_hdl = np.clip(rng.normal(60, 18, n), 20, 100)
    trig = np.clip(rng.normal(225, 80, n) + 30 * diabetes, 50, 400)
    chol_total = np.clip(chol_ldl + chol_hdl + trig / 5 + rng.normal(0, 10, n), 150, 300)

    # Latent disease status, then symptoms that follow from it.
    risk_logit = (
        0.05 * (age - 75)
        + 0.25 * hypertension
        + 0.25 * diabetes
        + 0.20 * cvd
        + 0.004 * (sbp - 132)
        + 0.003 * (chol_ldl - 125)
        - 0.004 * (chol_hdl - 60)
    )
    offset = np.log(prevalence / (1 - prevalence))
    disease = (rng.random(n) < _sigmoid(offset + risk_logit)).astype(int)

    mmse = np.clip(rng.normal(17, 7, n) - 4 * disease, 0, 30)
    functional = np.clip(rng.normal(5.8, 2.6, n) - 2.0 * disease, 0, 10)
    adl = np.clip(rng.normal(5.6, 2.7, n) - 1.8 * disease, 0, 10)

    def symptom(base: float, lift: float) -> np.ndarray:
        return (rng.random(n) < base + lift * disease).astype(int)

    data = {
        "PatientID": np.arange(100000, 100000 + n),
        "Age": age,
        "Gender": binary(0.5),
        "Ethnicity": rng.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1]),
        "EducationLevel": rng.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        "BMI": bmi,
        "Smoking": smoking,
        "AlcoholConsumption": rng.uniform(0, 20, n),
        "PhysicalActivity": rng.uniform(0, 10, n),
        "DietQuality": rng.uniform(0, 10, n),
        "SleepQuality": rng.uniform(4, 10, n),
        "FamilyHistoryAlzheimers": binary(0.25),
        "CardiovascularDisease": cvd,
        "Diabetes": diabetes,
        "Depression": binary(0.2),
        "HeadInjury": binary(0.09),
        "Hypertension": hypertension,
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "CholesterolTotal": chol_total,
        "CholesterolLDL": chol_ldl,
        "CholesterolHDL": chol_hdl,
        "CholesterolTriglycerides": trig,
        "MMSE": mmse,
        "FunctionalAssessment": functional,
        "MemoryComplaints": symptom(0.10, 0.30),
        "BehavioralProblems": symptom(0.08, 0.22),
        "ADL": adl,
        "Confusion": symptom(0.20, 0.02),
        "Disorientation": symptom(0.15, 0.02),
        "PersonalityChanges": symptom(0.15, 0.01),
        "DifficultyCompletingTasks": symptom(0.15, 0.02),
        "Forgetfulness": symptom(0.30, 0.01),
        LABEL_COLUMN: disease,
        "DoctorInCharge": "SYNTHETIC",
    }
    df = pd.DataFrame(data)[[c.name for c in COLUMNS]]

    if missing_rate > 0:
        for col in ("BMI", "CholesterolLDL", "SleepQuality", "MMSE"):
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan
    return df
