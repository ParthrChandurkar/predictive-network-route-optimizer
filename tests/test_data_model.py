import tempfile
import unittest
from pathlib import Path

from data_generator import FEATURES, generate_network_dataset, save_dataset
from model import load_model, predict_link, train_model


class SyntheticDatasetTests(unittest.TestCase):
    def test_generation_is_reproducible_and_has_expected_schema(self):
        first = generate_network_dataset(100, random_state=9)
        second = generate_network_dataset(100, random_state=9)

        self.assertTrue(first.equals(second))
        self.assertEqual(list(first.columns), FEATURES + ["failure"])
        self.assertTrue(first["bandwidth"].between(10, 1000).all())
        self.assertTrue(first["latency"].between(1, 300).all())
        self.assertTrue(first["packet_loss"].between(0, 20).all())
        self.assertTrue(first["traffic_load"].between(0, 100).all())
        self.assertEqual(set(first["failure"].unique()), {0, 1})

    def test_generation_rejects_non_positive_sample_count(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            generate_network_dataset(0)

    def test_save_dataset_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "dataset.csv"
            frame = save_dataset(output, n_samples=25, random_state=5)

            self.assertTrue(output.exists())
            self.assertEqual(len(frame), 25)


class ModelTests(unittest.TestCase):
    def test_training_artifacts_can_be_loaded_and_used(self):
        dataset = generate_network_dataset(300, random_state=12)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            scaler_path = Path(temp_dir) / "scaler.pkl"
            _, _, metrics = train_model(
                dataset,
                model_path=model_path,
                scaler_path=scaler_path,
            )
            model, scaler = load_model(model_path, scaler_path)
            prediction = predict_link(model, scaler, dataset.iloc[0].to_dict())

            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            self.assertTrue(0.0 <= prediction["failure_prob"] <= 1.0)
            self.assertIn(prediction["risk_label"], {"Low", "Medium", "High"})
            self.assertIn("f1", metrics)

    def test_training_rejects_missing_required_columns(self):
        dataset = generate_network_dataset(100).drop(columns=["latency"])

        with self.assertRaisesRegex(ValueError, "latency"):
            train_model(dataset, persist=False)

    def test_load_returns_empty_pair_when_artifacts_are_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model, scaler = load_model(
                Path(temp_dir) / "missing-model.pkl",
                Path(temp_dir) / "missing-scaler.pkl",
            )

            self.assertIsNone(model)
            self.assertIsNone(scaler)


if __name__ == "__main__":
    unittest.main()

