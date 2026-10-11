import unittest

import networkx as nx

from routing import (
    compute_ai_weights,
    route_ai,
    route_bellman_ford,
    route_bellman_ford_ai,
    route_floyd_warshall,
    route_floyd_warshall_ai,
    route_traditional,
)


def _edge(distance, risk):
    return {
        "distance": distance,
        "failure_prob": risk,
        "risk_label": "High" if risk >= 0.65 else "Low",
        "failed": False,
    }


class RoutingAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge("A", "B", **_edge(1, 0.9))
        self.graph.add_edge("A", "C", **_edge(2, 0.01))
        self.graph.add_edge("C", "B", **_edge(2, 0.01))

    def test_traditional_algorithms_agree_on_shortest_path(self):
        results = [
            route_traditional(self.graph, "A", "B"),
            route_bellman_ford(self.graph, "A", "B"),
            route_floyd_warshall(self.graph, "A", "B"),
        ]

        self.assertTrue(all(result["found"] for result in results))
        self.assertTrue(all(result["path_str"] == "A → B" for result in results))
        self.assertTrue(all(result["algo_cost"] == 1 for result in results))

    def test_risk_aware_algorithms_avoid_high_risk_shortcut(self):
        results = [
            route_ai(self.graph, "A", "B", penalty=200),
            route_bellman_ford_ai(self.graph, "A", "B", penalty=200),
            route_floyd_warshall_ai(self.graph, "A", "B", penalty=200),
        ]

        self.assertTrue(all(result["found"] for result in results))
        self.assertTrue(all(result["path_str"] == "A → C → B" for result in results))
        self.assertTrue(all(result["dist_cost"] == 4 for result in results))

    def test_unknown_nodes_return_not_found(self):
        results = [
            route_traditional(self.graph, "missing", "B"),
            route_ai(self.graph, "A", "missing"),
            route_bellman_ford(self.graph, "missing", "B"),
            route_bellman_ford_ai(self.graph, "A", "missing"),
            route_floyd_warshall(self.graph, "missing", "B"),
            route_floyd_warshall_ai(self.graph, "A", "missing"),
        ]

        self.assertTrue(all(not result["found"] for result in results))

    def test_negative_penalty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_ai_weights(self.graph, penalty=-1)


if __name__ == "__main__":
    unittest.main()

