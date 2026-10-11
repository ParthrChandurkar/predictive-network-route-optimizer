import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from telemetry import TelemetryValidationError
from telemetry.prepare_lte import prepare_lte_data


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lte_sample.csv"


class PrepareLteTests(unittest.TestCase):
    def test_prepare_writes_canonical_splits_and_report(self):
        source = pd.read_csv(FIXTURE)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "raw" / "train.csv"
            test = root / "raw" / "test.csv"
            train.parent.mkdir()
            source.to_csv(train, sep=";", index=False)
            source.drop(columns=["Unusual"]).to_csv(test, sep=";", index=False)

            train_output = root / "processed" / "train.csv"
            test_output = root / "processed" / "test.csv"
            report_path = root / "report.json"
            report = prepare_lte_data(
                train, test, train_output, test_output, report_path
            )

            self.assertTrue(train_output.exists())
            self.assertTrue(test_output.exists())
            self.assertEqual(report["validation"]["shared_cells"], 2)
            self.assertAlmostEqual(report["validation"]["positive_rate"], 0.5)
            self.assertEqual(json.loads(report_path.read_text())["schema_version"], 1)

    def test_prepare_rejects_labeled_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train = root / "train.csv"
            test = root / "test.csv"
            train.write_bytes(FIXTURE.read_bytes())
            test.write_bytes(FIXTURE.read_bytes())

            with self.assertRaisesRegex(TelemetryValidationError, "must not contain labels"):
                prepare_lte_data(
                    train,
                    test,
                    root / "train-output.csv",
                    root / "test-output.csv",
                    root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
