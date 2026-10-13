import json
import tempfile
import unittest
from pathlib import Path
from evaluation.phase8 import controlled_routing_evaluation, write_report


class Phase8EvaluationTests(unittest.TestCase):
    def test_controlled_scenario_validates_all_algorithm_families(self):
        result = controlled_routing_evaluation()

        self.assertEqual(len(result["routes"]), 6)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["routes"]["dijkstra"]["path"], ["R1", "R2", "R4", "R7", "R8"])
        self.assertEqual(result["routes"]["risk_aware_dijkstra"]["path"], ["R1", "R4", "R7", "R8"])

    def test_report_write_is_atomic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "phase8.json"
            report = {"phase": "phase-8", "status": "complete"}
            write_report(report, destination)

            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), report)
            self.assertFalse(destination.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
