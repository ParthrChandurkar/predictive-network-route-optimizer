import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.lte_features import MODEL_FEATURES, build_lte_features, feature_duplicate_groups
from telemetry import TelemetryValidationError, load_lte_dataset


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lte_sample.csv"


class LteFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_lte_dataset(FIXTURE).frame

    def test_feature_schema_and_cyclical_time(self):
        features = build_lte_features(self.frame)

        self.assertEqual(list(features.columns), list(MODEL_FEATURES))
        self.assertAlmostEqual(features.loc[0, "time_sin"], 0.0)
        self.assertAlmostEqual(features.loc[0, "time_cos"], 1.0)
        self.assertTrue(np.isfinite(features.select_dtypes("number")).all().all())

    def test_features_do_not_depend_on_label(self):
        with_label = build_lte_features(self.frame)
        without_label = build_lte_features(self.frame.drop(columns=["is_unusual"]))

        pd.testing.assert_frame_equal(with_label, without_label)

    def test_duplicate_grouping_ignores_label_but_not_features(self):
        duplicated = pd.concat([self.frame.iloc[[0]], self.frame.iloc[[0]]], ignore_index=True)
        duplicated.loc[1, "is_unusual"] = 1 - duplicated.loc[0, "is_unusual"]
        groups = feature_duplicate_groups(duplicated)

        self.assertEqual(groups.iloc[0], groups.iloc[1])

    def test_missing_input_is_rejected(self):
        with self.assertRaisesRegex(TelemetryValidationError, "prb_usage_ul"):
            build_lte_features(self.frame.drop(columns=["prb_usage_ul"]))


if __name__ == "__main__":
    unittest.main()
