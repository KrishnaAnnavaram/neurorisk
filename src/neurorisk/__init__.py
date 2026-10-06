"""neurorisk: leakage-safe, calibrated Alzheimer's risk models on tabular data.

The core package needs only numpy, pandas, scipy and scikit-learn.
XGBoost, Optuna and SHAP are optional extras that are imported lazily.
"""

__version__ = "0.1.0"

SYNTHETIC_DATA_NOTICE = (
    "The public Alzheimer's dataset and the bundled generator both produce SYNTHETIC records. "
    "Results describe the data generator, not real patients. This is not a medical device and "
    "not a diagnostic tool."
)
