from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "pima_diabetes_model.joblib"
METRICS_PATH = MODEL_PATH.with_suffix(".metrics.json")
DATASET_NAME = "Pima Indians Diabetes Database"
DATASET_REFERENCE = (
    "Smith et al. (1988), Using the ADAP learning algorithm to forecast "
    "the onset of diabetes mellitus."
)
MODEL_VERSION = "Pima-RandomForest-v1"
FEATURE_COLUMNS = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
FEATURE_NAMES = (
    "Number of pregnancies",
    "2-hour plasma glucose after OGTT (mg/dL)",
    "Diastolic blood pressure (mmHg)",
    "Triceps skinfold thickness (mm)",
    "2-hour serum insulin (uU/mL)",
    "BMI (kg/m^2)",
    "Diabetes pedigree function",
    "Age (years)",
)
ZERO_IS_MISSING_FEATURES = (
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)

def load_model_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)

def _prepare_values(values):
    features = np.asarray(values, dtype=float).reshape(1, -1)
    if features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature values.")
    if not np.isfinite(features).all():
        raise ValueError("Prediction inputs must be finite numeric values.")
    for column in ZERO_IS_MISSING_FEATURES:
        index = FEATURE_COLUMNS.index(column)
        if features[0, index] == 0:
            features[0, index] = np.nan
    return pd.DataFrame(features, columns=FEATURE_COLUMNS)

def predict_diabetes_risk(model, values):
    features = _prepare_values(values)
    score = float(model.predict_proba(features)[0][1])
    label = "Higher model score" if score >= 0.5 else "Lower model score"
    return {"label": label, "score": score}

def explain_diabetes_prediction(model, values):
    import shap

    original_values = np.asarray(values, dtype=float).reshape(1, -1)
    features = _prepare_values(values)
    transformed = model.named_steps["imputer"].transform(features)
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed)

    if isinstance(raw_values, list):
        positive_class_values = np.asarray(raw_values[1])[0]
    else:
        raw_values = np.asarray(raw_values)
        positive_class_values = raw_values[0, :, 1] if raw_values.ndim == 3 else raw_values[0]

    base_value = np.asarray(explainer.expected_value)
    if base_value.ndim > 0:
        base_value = base_value[-1]
    contributions = [
        {
            "feature": name,
            "value": float(value),
            "contribution": float(contribution),
        }
        for name, value, contribution in zip(
            FEATURE_NAMES, original_values[0], positive_class_values
        )
    ]
    return {"base_value": float(base_value), "contributions": contributions}
