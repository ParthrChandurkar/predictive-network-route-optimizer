import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd

from telemetry import TelemetryValidationError, load_lte_dataset
from telemetry.schemas import LTE_FEATURE_COLUMNS, LTE_LABEL_COLUMN


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lte_sample.csv"


class LteTelemetryTests(unittest.TestCase):
    def test_labeled_dataset_is_normalized_and_described(self):
        dataset = load_lte_dataset(FIXTURE)

        self.assertEqual(
            list(dataset.frame.columns),
            list(LTE_FEATURE_COLUMNS) + [LTE_LABEL_COLUMN],
        )
        self.assertEqual(dataset.report.rows, 4)
        self.assertEqual(dataset.report.cells, 2)
        self.assertTrue(dataset.report.labeled)
        self.assertEqual(dataset.report.class_counts, {"0": 2, "1": 2})
        self.assertEqual(dataset.frame["minute_of_day"].tolist(), [0, 15, 0, 15])

    def test_documented_column_aliases_are_supported(self):
        frame = pd.read_csv(FIXTURE).rename(
            columns={"CellName": "CellName1", "Unusual": "Label"}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aliased.csv"
            frame.to_csv(path, index=False)
            dataset = load_lte_dataset(path)

        self.assertEqual(dataset.report.cells, 2)
        self.assertTrue(dataset.report.labeled)

    def test_unlabeled_competition_file_is_allowed_explicitly(self):
        frame = pd.read_csv(FIXTURE).drop(columns=["Unusual"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.csv"
            frame.to_csv(path, index=False)
            dataset = load_lte_dataset(path, require_label=False)

        self.assertFalse(dataset.report.labeled)
        self.assertNotIn(LTE_LABEL_COLUMN, dataset.frame.columns)

    def test_uploaded_file_like_source_is_rewound(self):
        upload = BytesIO(FIXTURE.read_bytes())
        upload.name = "uploaded.csv"
        upload.seek(10)

        dataset = load_lte_dataset(upload)

        self.assertEqual(dataset.report.source, "uploaded.csv")
        self.assertEqual(dataset.report.rows, 4)

    def test_kaggle_semicolon_delimiter_is_detected(self):
        frame = pd.read_csv(FIXTURE)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "kaggle.csv"
            frame.to_csv(path, index=False, sep=";")
            dataset = load_lte_dataset(path)

        self.assertEqual(dataset.report.rows, 4)
        self.assertEqual(dataset.report.cells, 2)

    def test_missing_numeric_values_are_reported_not_silently_filled(self):
        frame = pd.read_csv(FIXTURE)
        frame.loc[0, "maxUE_UL"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.csv"
            frame.to_csv(path, index=False)
            dataset = load_lte_dataset(path)

        self.assertEqual(dataset.report.missing_values["max_ue_ul"], 1)
        self.assertTrue(pd.isna(dataset.frame.loc[0, "max_ue_ul"]))

    def test_missing_required_column_is_rejected(self):
        frame = pd.read_csv(FIXTURE).drop(columns=["PRBUsageUL"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.csv"
            frame.to_csv(path, index=False)

            with self.assertRaisesRegex(TelemetryValidationError, "PRBUsageUL"):
                load_lte_dataset(path)

    def test_invalid_time_and_label_are_rejected(self):
        frame = pd.read_csv(FIXTURE)
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_time = Path(temp_dir) / "invalid-time.csv"
            frame.assign(Time="not-a-time").to_csv(invalid_time, index=False)
            with self.assertRaisesRegex(TelemetryValidationError, "HH:MM"):
                load_lte_dataset(invalid_time)

            invalid_label = Path(temp_dir) / "invalid-label.csv"
            frame.assign(Unusual=2).to_csv(invalid_label, index=False)
            with self.assertRaisesRegex(TelemetryValidationError, "binary"):
                load_lte_dataset(invalid_label)


if __name__ == "__main__":
    unittest.main()

