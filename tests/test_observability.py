import os
import unittest
from unittest.mock import patch

import networkx as nx

from observability import configure_observability
from observability.telemetry import _clean_endpoint
from routing import route_ai, route_traditional


class ObservabilityTests(unittest.TestCase):
    def test_disabled_configuration_is_no_op(self):
        with patch.dict(os.environ, {"OBSERVABILITY_ENABLED": "false"}):
            state = configure_observability()
        self.assertFalse(state.enabled)

    def test_endpoint_normalization(self):
        self.assertEqual(_clean_endpoint("http://localhost:4318/"), "http://localhost:4318")

    def test_instrumented_routes_keep_the_same_contract(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", distance=1, routing_risk=0.2)
        traditional = route_traditional(graph, "A", "B")
        risk_aware = route_ai(graph, "A", "B")
        self.assertEqual(traditional["edges"], [("A", "B")])
        self.assertEqual(risk_aware["edges"], [("A", "B")])


if __name__ == "__main__":
    unittest.main()
