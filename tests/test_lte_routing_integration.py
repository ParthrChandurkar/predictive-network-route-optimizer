import unittest

import networkx as nx
import pandas as pd

from integration.lte_routing import (
    apply_cell_risk_to_graph,
    build_demo_edge_cell_mapping,
    cell_risk_snapshot,
)
from routing import route_ai
from simulation import network_health


class LteRoutingIntegrationTests(unittest.TestCase):
    def test_snapshot_averages_repeated_cell_observations(self):
        rows = pd.DataFrame(
            {
                "time_of_day": ["10:00", "10:00", "10:15"],
                "cell_id": ["A", "A", "A"],
                "degradation_score": [0.2, 0.6, 0.9],
            }
        )

        snapshot = cell_risk_snapshot(rows, "10:00")

        self.assertAlmostEqual(snapshot["A"], 0.4)

    def test_lte_risk_changes_route_without_claiming_failure_probability(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", distance=1, failure_prob=0.0, risk_label="Low")
        graph.add_edge("A", "C", distance=2, failure_prob=0.0, risk_label="Low")
        graph.add_edge("C", "B", distance=2, failure_prob=0.0, risk_label="Low")
        mapping = {("A", "B"): "hot", ("A", "C"): "cool", ("B", "C"): "cool"}

        summary = apply_cell_risk_to_graph(
            graph, {"hot": 0.95, "cool": 0.01}, mapping
        )
        result = route_ai(graph, "A", "B", penalty=200)

        self.assertEqual(result["path_str"], "A → C → B")
        self.assertEqual(graph["A"]["B"]["failure_prob"], 0.0)
        self.assertEqual(graph["A"]["B"]["risk_source"], "lte_degradation")
        self.assertEqual(summary["mapped_edges"], 3)
        self.assertGreater(network_health(graph)["avg_risk"], 0.0)

    def test_demo_mapping_is_deterministic_and_explicitly_labeled(self):
        graph = nx.path_graph(["B", "A", "C"])

        first = build_demo_edge_cell_mapping(graph, ["cell-2", "cell-1"])
        second = build_demo_edge_cell_mapping(graph, ["cell-1", "cell-2"])

        self.assertEqual(first, second)

    def test_unknown_mapping_is_rejected(self):
        graph = nx.path_graph(["A", "B"])
        with self.assertRaisesRegex(ValueError, "does not exist"):
            apply_cell_risk_to_graph(graph, {"cell": 0.5}, {("A", "Z"): "cell"})

    def test_invalid_score_does_not_partially_replace_existing_mapping(self):
        graph = nx.path_graph(["A", "B"])
        graph["A"]["B"].update(
            routing_risk=0.2, risk_source="lte_degradation", telemetry_cell="old"
        )

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            apply_cell_risk_to_graph(graph, {"new": 2.0}, {("A", "B"): "new"})

        self.assertEqual(graph["A"]["B"]["routing_risk"], 0.2)
        self.assertEqual(graph["A"]["B"]["telemetry_cell"], "old")


if __name__ == "__main__":
    unittest.main()
