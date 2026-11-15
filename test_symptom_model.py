import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import symptom_predictor
from train_symptom_model import build_dataset_metadata


class SymptomEvaluationMetadataTests(unittest.TestCase):
    def test_metadata_reports_deduplicated_class_sizes_and_test_overlap(self):
        features = pd.DataFrame(
            [[0, 0], [1, 0], [0, 1]],
            columns=["symptom_a", "symptom_b"],
        )
        labels = pd.Series(["Class A", "Class A", "Class B"], name="prognosis")
        test_features = pd.DataFrame(
            [[0, 0], [1, 1]],
            columns=["symptom_a", "symptom_b"],
        )
        test_labels = pd.Series(["Class A", "Class B"])

        metadata = build_dataset_metadata(
            5,
            features,
            labels,
            test_features,
            test_labels,
        )

        self.assertEqual(metadata["training_rows"], 5)
        self.assertEqual(metadata["unique_training_patterns"], 3)
        self.assertEqual(metadata["duplicate_training_rows_removed"], 2)
        self.assertEqual(metadata["class_counts_unique"], {"Class A": 2, "Class B": 1})
        self.assertEqual(metadata["min_unique_patterns_per_class"], 1)
        self.assertEqual(metadata["max_unique_patterns_per_class"], 2)
        self.assertEqual(metadata["provided_test_rows_overlapping_training"], 1)
        self.assertEqual(metadata["provided_test_rows_used_for_holdout_metrics"], 0)
        self.assertEqual(metadata["holdout_seed"], 42)
        self.assertFalse(metadata["external_validation"])
        self.assertIn("not external", metadata["evaluation_note"])

    def test_existing_metrics_get_non_validation_metadata_at_load_time(self):
        with TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "legacy.metrics.json"
            metrics_path.write_text(
                json.dumps({"class_counts_unique": {"A": 5, "B": 8}}),
                encoding="utf-8",
            )
            with patch.object(symptom_predictor, "METRICS_PATH", metrics_path):
                metrics = symptom_predictor.load_model_metrics()

        self.assertEqual(metrics["min_unique_patterns_per_class"], 5)
        self.assertEqual(metrics["max_unique_patterns_per_class"], 8)
        self.assertEqual(metrics["holdout_seed"], 42)
        self.assertFalse(metrics["external_validation"])
        self.assertIn("not external", metrics["evaluation_note"])


if __name__ == "__main__":
    unittest.main()
