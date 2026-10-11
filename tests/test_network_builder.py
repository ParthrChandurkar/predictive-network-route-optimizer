import unittest

from network_builder import (
    LINK_DEFAULTS,
    add_link,
    add_router,
    build_default_topology,
    graph_summary,
    remove_link,
    set_link_failed,
    update_link,
)


class NetworkBuilderTests(unittest.TestCase):
    def test_default_topology_has_complete_link_schema(self):
        graph = build_default_topology()
        summary = graph_summary(graph)

        self.assertEqual(summary["routers"], 8)
        self.assertEqual(summary["links"], 12)
        self.assertTrue(summary["connected"])
        for _, _, attributes in graph.edges(data=True):
            self.assertTrue(set(LINK_DEFAULTS).issubset(attributes))

    def test_router_and_link_lifecycle(self):
        graph = build_default_topology()

        added, _ = add_router(graph, "R9")
        linked, _ = add_link(graph, "R8", "R9", {"distance": 7})
        updated, _ = update_link(graph, "R8", "R9", {"latency": 99.0})
        failed, _ = set_link_failed(graph, "R8", "R9", True)

        self.assertTrue(added and linked and updated and failed)
        self.assertEqual(graph["R8"]["R9"]["distance"], 7)
        self.assertEqual(graph["R8"]["R9"]["ai_weight"], 7.0)
        self.assertEqual(graph["R8"]["R9"]["latency"], 99.0)
        self.assertTrue(graph["R8"]["R9"]["failed"])

        removed, _ = remove_link(graph, "R8", "R9")
        self.assertTrue(removed)
        self.assertFalse(graph.has_edge("R8", "R9"))

    def test_invalid_links_are_rejected(self):
        graph = build_default_topology()

        self.assertFalse(add_link(graph, "R1", "R1", {})[0])
        self.assertFalse(add_link(graph, "R1", "missing", {})[0])
        self.assertFalse(add_link(graph, "R1", "R2", {})[0])


if __name__ == "__main__":
    unittest.main()

