"""Train an educational symptom-pattern classifier from the supplied CSV files."""
import json

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from symptom_predictor import (
    MODEL_PATH,
    MODEL_VERSION,
    TRAINING_PATH,
    load_test_data,
    load_training_data,
)


SEED = 42


def build_dataset_metadata(raw_training_rows, unique_features, unique_labels, test_features, test_labels):
    train_patterns = set(unique_features.itertuples(index=False, name=None))
    test_patterns = list(test_features.itertuples(index=False, name=None))
    class_counts = unique_labels.value_counts().sort_index()
    return {
        "training_rows": int(raw_training_rows),
        "unique_training_patterns": int(len(unique_features)),
        "duplicate_training_rows_removed": int(raw_training_rows - len(unique_features)),
        "symptom_features": int(unique_features.shape[1]),
        "disease_classes": int(unique_labels.nunique()),
        "class_counts_unique": {
            str(label): int(count) for label, count in class_counts.items()
        },
        "min_unique_patterns_per_class": int(class_counts.min()),
        "max_unique_patterns_per_class": int(class_counts.max()),
        "provided_test_rows": int(len(test_features)),
        "provided_test_classes": int(test_labels.nunique()),
        "provided_test_unique_patterns": int(len(set(test_patterns))),
        "provided_test_rows_overlapping_training": int(
            sum(pattern in train_patterns for pattern in test_patterns)
        ),
        "provided_test_rows_used_for_holdout_metrics": 0,
        "holdout_method": "Stratified 20% holdout of unique training patterns",
        "holdout_seed": SEED,
        "external_validation": False,
        "evaluation_note": (
            "Metrics use a stratified holdout of unique Training.csv symptom patterns. "
            "Testing.csv overlaps training patterns and was not used for evaluation. "
            "This small internal split is not external, prospective, or clinical validation."
        ),
    }


def main():
    raw_training = pd.read_csv(TRAINING_PATH)
    features, labels = load_training_data()
    unique_data = features.copy()
    unique_data["prognosis"] = labels
    unique_data = unique_data.drop_duplicates()
    unique_features = unique_data.drop(columns="prognosis")
    unique_labels = unique_data["prognosis"]
    duplicate_rows_removed = len(raw_training) - len(unique_data)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        unique_features,
        unique_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=unique_labels,
    )
    model = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            )
        ]
    )
    model.fit(x_train, y_train)
    holdout_predictions = model.predict(x_holdout)
    holdout_scores = model.predict_proba(x_holdout)
    top_k = min(3, len(model.classes_))

    test_features, test_labels = load_test_data()
    metadata = build_dataset_metadata(
        len(raw_training),
        unique_features,
        unique_labels,
        test_features,
        test_labels,
    )
    test_overlap_rows = metadata["provided_test_rows_overlapping_training"]

    metrics = {
        "model_version": MODEL_VERSION,
        **metadata,
        "holdout_rows": int(len(y_holdout)),
        "holdout_accuracy": float(accuracy_score(y_holdout, holdout_predictions)),
        "holdout_balanced_accuracy": float(
            balanced_accuracy_score(y_holdout, holdout_predictions)
        ),
        "holdout_macro_f1": float(
            f1_score(y_holdout, holdout_predictions, average="macro", zero_division=0)
        ),
        "holdout_top_3_accuracy": float(
            top_k_accuracy_score(
                y_holdout,
                holdout_scores,
                k=top_k,
                labels=model.classes_,
            )
        ),
    }

    model.fit(unique_features, unique_labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    MODEL_PATH.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Training rows:", len(raw_training))
    print("Unique symptom patterns:", len(unique_data))
    print("Duplicate rows removed:", duplicate_rows_removed)
    print("Disease classes:", unique_labels.nunique())
    print(
        f"Testing.csv exact symptom-pattern overlap: {test_overlap_rows}/"
        f"{len(test_features)} rows"
    )
    print("Stratified holdout metrics (unique training patterns only):")
    print(json.dumps(metrics, indent=2))
    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", MODEL_PATH.with_suffix(".metrics.json"))


if __name__ == "__main__":
    main()
