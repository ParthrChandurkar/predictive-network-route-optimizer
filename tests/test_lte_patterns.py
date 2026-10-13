import unittest

import pandas as pd

from analysis.lte_patterns import discover_lte_patterns, wilson_interval
from telemetry.schemas import TelemetryValidationError


def _pattern_fixture() -> pd.DataFrame:
    rows = []
    for cell_index, cell in enumerate(("A", "B", "C", "D", "E")):
        for index in range(40):
            unusual = int(index % 4 == 0)
            rows.append(
                {
                    "time_of_day": f"{index % 24:02d}:00",
                    "minute_of_day": (index % 24) * 60,
                    "cell_id": cell,
                    "prb_usage_ul": 1.0 + unusual + cell_index * 0.01,
                    "prb_usage_dl": 2.0 + unusual,
                    "mean_throughput_dl": 3.0 - unusual * 0.5,
                    "mean_throughput_ul": 0.5,
                    "max_throughput_dl": 5.0,
                    "max_throughput_ul": 1.0,
                    "mean_ue_dl": 1.2,
                    "mean_ue_ul": 0.5 if unusual else 1.011,
                    "max_ue_dl": 4,
                    "max_ue_ul": 3,
                    "max_ue_total": 7,
                    "is_unusual": unusual,
                }
            )
    return pd.DataFrame(rows)


class LTEPatternTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(25, 100)
        self.assertLess(low, 0.25)
        self.assertGreater(high, 0.25)

    def test_discovery_keeps_cells_out_of_validation_training(self):
        report = discover_lte_patterns(_pattern_fixture())
        rules = report["validated_interpretable_rules"]
        self.assertTrue(set(rules["discovery_cells"]).isdisjoint(rules["validation_cells"]))
        self.assertEqual(report["dataset"]["cells"], 5)
        self.assertEqual(len(report["hour_patterns"]), 24)

    def test_discovery_requires_labels(self):
        with self.assertRaises(TelemetryValidationError):
            discover_lte_patterns(_pattern_fixture().drop(columns=["is_unusual"]))


if __name__ == "__main__":
    unittest.main()
