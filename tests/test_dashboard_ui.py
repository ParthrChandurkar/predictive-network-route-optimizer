import unittest

import pandas as pd

from analytics import lte_degradation_timeline
from ui.dashboard import network_operating_state, prepare_link_inventory


class DashboardStateTests(unittest.TestCase):
    def test_nominal_network(self):
        state = network_operating_state(
            {"failed_links": 0, "high_risk": 0, "health_score": 0.9}, True
        )
        self.assertEqual(state, ("Nominal", "healthy"))

    def test_high_risk_network_warns(self):
        state = network_operating_state(
            {"failed_links": 0, "high_risk": 2, "health_score": 0.8}, True
        )
        self.assertEqual(state, ("At risk", "warning"))

    def test_failure_is_degraded(self):
        state = network_operating_state(
            {"failed_links": 1, "high_risk": 0, "health_score": 0.8}, True
        )
        self.assertEqual(state, ("Degraded", "critical"))

    def test_link_inventory_prioritizes_failure_state(self):
        frame = pd.DataFrame(
            {
                "risk_label": ["Low", "High", "Medium"],
                "failed": [True, False, False],
                "routing_risk": [0.1, 0.9, 0.5],
            }
        )
        inventory = prepare_link_inventory(frame)
        self.assertEqual(
            inventory["operating_state"].tolist(),
            ["FAILED", "CRITICAL", "WATCH"],
        )
        self.assertEqual(inventory["routing_risk_percent"].tolist(), [10.0, 90.0, 50.0])

    def test_lte_timeline_aggregates_time_slots(self):
        frame = pd.DataFrame(
            {
                "time_of_day": ["00:00", "00:00", "00:15"],
                "unusual_behavior": [0, 1, 1],
            }
        )
        figure = lte_degradation_timeline(frame)
        self.assertEqual(list(figure.data[0].x), ["00:00", "00:15"])
        self.assertEqual(list(figure.data[0].y), [50.0, 100.0])


if __name__ == "__main__":
    unittest.main()
