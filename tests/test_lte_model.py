import tempfile
import unittest
from pathlib import Path

from modeling.lte_model import (
    load_lte_risk_model,
    predict_lte_risk,
    train_lte_risk_model,
)
from telemetry import load_lte_dataset


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lte_sample.csv"


class LteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_lte_dataset(FIXTURE).frame

    def test_model_round_trip_and_prediction_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.pkl"
            metadata_path = root / "metadata.json"
            _, metadata = train_lte_risk_model(
                self.frame, model_path, metadata_path
            )
            loaded = load_lte_risk_model(model_path)
            predictions = predict_lte_risk(loaded, self.frame)

        self.assertEqual(len(predictions), len(self.frame))
        self.assertTrue(predictions["degradation_score"].between(0, 1).all())
        self.assertTrue(set(predictions["risk_label"]).issubset({"Low", "Medium", "High"}))
        self.assertEqual(metadata["training_rows"], 4)
        self.assertIn("sha256", metadata["artifact"])

    def test_corrupt_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.pkl"
            path.write_bytes(b"not a model")

            self.assertIsNone(load_lte_risk_model(path))


if __name__ == "__main__":
    unittest.main()
