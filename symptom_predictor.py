from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
TRAINING_PATH = ROOT / "Training.csv"
TESTING_PATH = ROOT / "Testing.csv"
MODEL_PATH = ROOT / "models" / "symptom_disease_model.joblib"
METRICS_PATH = MODEL_PATH.with_suffix(".metrics.json")
TARGET_COLUMN = "prognosis"
DATASET_NAME = "User-provided symptom/prognosis dataset"
MODEL_VERSION = "Symptom-RandomForest-v1"


def _load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{path.name} must contain a '{TARGET_COLUMN}' target column.")
    if frame.empty:
        raise ValueError(f"{path.name} contains no rows.")
    if frame.isna().any().any():
        raise ValueError(f"{path.name} contains missing values.")

    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError(f"{path.name} contains no symptom columns.")

    features = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    if not features.isin([0, 1]).all().all():
        raise ValueError(f"{path.name} symptom columns must contain only 0 or 1.")

    labels = frame[TARGET_COLUMN].astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError(f"{path.name} contains an empty '{TARGET_COLUMN}' label.")
    return features.astype("uint8"), labels


def load_training_data():
    features, labels = _load_dataset(TRAINING_PATH)
    label_counts_by_pattern = pd.concat(
        [features, labels.rename(TARGET_COLUMN)], axis=1
    ).groupby(list(features.columns), dropna=False)[TARGET_COLUMN].nunique()
    if (label_counts_by_pattern > 1).any():
        raise ValueError("A symptom pattern maps to multiple prognosis labels.")
    return features, labels


def load_test_data():
    features, labels = _load_dataset(TESTING_PATH)
    training_features, _ = load_training_data()
    if list(features.columns) != list(training_features.columns):
        raise ValueError("Testing.csv symptom columns do not match Training.csv.")
    return features, labels


def load_symptom_features():
    features, _ = load_training_data()
    return list(features.columns)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_symptom_model.py first."
        )
    return joblib.load(MODEL_PATH)


def load_model_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics not found: {METRICS_PATH}. Run train_symptom_model.py first."
        )
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    class_counts = metrics.get("class_counts_unique", {})
    if not class_counts:
        raise ValueError(
            "Model metrics lack unique-pattern class counts. "
            "Run train_symptom_model.py to regenerate them."
        )
    counts = [int(count) for count in class_counts.values()]
    metrics.setdefault("min_unique_patterns_per_class", min(counts))
    metrics.setdefault("max_unique_patterns_per_class", max(counts))
    metrics.setdefault(
        "holdout_method",
        "Stratified 20% holdout of unique training patterns",
    )
    metrics.setdefault("holdout_seed", 42)
    metrics.setdefault("external_validation", False)
    metrics.setdefault(
        "evaluation_note",
        "This small internal split is not external, prospective, or clinical validation.",
    )
    return metrics


def format_symptom_name(name):
    if name.endswith(".1"):
        name = f"{name[:-2]} (second indicator)"
    return " ".join(name.replace("_", " ").split()).title()


def predict_disease_candidates(model, selected_symptoms, top_k=3):
    feature_columns = list(model.feature_names_in_)
    selected = set(selected_symptoms)
    unknown = selected.difference(feature_columns)
    if unknown:
        raise ValueError(f"Unknown symptom columns: {', '.join(sorted(unknown))}")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    features = pd.DataFrame(
        [[int(column in selected) for column in feature_columns]],
        columns=feature_columns,
    )
    scores = model.predict_proba(features)[0]
    indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    return [
        {"label": str(model.classes_[index]), "score": float(scores[index])}
        for index in indices
    ]
