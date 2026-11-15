"""Train the experimental diabetes model on the documented Pima dataset."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predictor import FEATURE_COLUMNS, MODEL_PATH, ZERO_IS_MISSING_FEATURES

DATA_PATH = Path(__file__).parent / "data" / "pima_indians_diabetes.csv"
DATASET_COLUMNS = list(FEATURE_COLUMNS) + ["Outcome"]
SEED = 42


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH, header=None, names=DATASET_COLUMNS)
    if len(frame) != 768:
        raise ValueError(f"Expected 768 dataset rows, found {len(frame)}.")
    if frame.isna().any().any():
        raise ValueError("The bundled dataset contains unrecognized missing values.")
    if not set(frame["Outcome"].unique()).issubset({0, 1}):
        raise ValueError("Outcome must contain only binary values 0 and 1.")

    features = frame[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise").astype(float)
    labels = frame["Outcome"].astype(int)
    missing_columns = list(ZERO_IS_MISSING_FEATURES)
    features.loc[:, missing_columns] = features.loc[
        :, missing_columns
    ].replace(0, np.nan)
    return features, labels


def main():
    features, labels = load_dataset()
    outcome_counts = labels.value_counts().sort_index()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    metrics = {
        "holdout_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "dataset_rows": int(len(labels)),
        "outcome_counts": {
            str(int(outcome)): int(count)
            for outcome, count in outcome_counts.items()
        },
    }

    model.fit(features, labels)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    dump(model, MODEL_PATH)

    metrics_path = MODEL_PATH.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Dataset: Pima Indians Diabetes Database (768 records)")
    print("Outcome counts:", outcome_counts.to_dict())
    print("Stratified holdout metrics (20%; not clinical validation):")
    print(json.dumps(metrics, indent=2))
    print("Saved model:", MODEL_PATH)
    print("Saved holdout metrics:", metrics_path)


if __name__ == "__main__":
    main()
