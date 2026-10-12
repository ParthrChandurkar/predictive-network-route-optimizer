import json
import tempfile
import unittest
from pathlib import Path

from experiments.mininet_plan import build_mininet_plan, write_plan
from experiments.mininet_runner import (
    aggregate_trials,
    dry_run_summary,
    load_and_validate_plan,
    parse_iperf_output,
    parse_ping_output,
)


class MininetExperimentTests(unittest.TestCase):
    def test_plan_has_distinct_valid_routes(self):
        plan = build_mininet_plan()

        traditional = plan["routes"]["traditional"]["path"]
        risk_aware = plan["routes"]["risk_aware"]["path"]
        self.assertNotEqual(traditional, risk_aware)
        self.assertEqual(plan["scenario"]["impaired_edge"], ["R1", "R2"])
        self.assertIn("R1", traditional)
        self.assertIn("R8", risk_aware)
        self.assertEqual(plan["measurements"]["repetitions"], 3)
        self.assertTrue(all(link["bandwidth_mbps"] <= 100 for link in plan["topology"]["links"]))

    def test_plan_round_trip_and_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            write_plan(build_mininet_plan(), path)
            loaded = load_and_validate_plan(path)
            summary = dry_run_summary(loaded)

        self.assertTrue(summary["ready_for_mininet"])
        self.assertEqual(summary["routers"], 8)
        self.assertEqual(summary["links"], 12)

    def test_measurement_parsers(self):
        ping = parse_ping_output(
            "10 packets transmitted, 9 received, 10% packet loss, time 9010ms\n"
            "rtt min/avg/max/mdev = 1.000/2.500/4.000/0.500 ms\n"
        )
        iperf = parse_iperf_output(
            json.dumps(
                {
                    "end": {
                        "sum_sent": {"bits_per_second": 9_000_000, "retransmits": 3},
                        "sum_received": {"bits_per_second": 8_000_000},
                    }
                }
            )
        )

        self.assertEqual(ping["packet_loss_percent"], 10.0)
        self.assertEqual(ping["average_rtt_ms"], 2.5)
        self.assertEqual(iperf["throughput_mbps"], 8.0)
        self.assertEqual(iperf["retransmits"], 3)

    def test_trial_aggregation(self):
        trials = []
        for route, throughput, rtt in (
            ("traditional", 1.0, 100.0),
            ("traditional", 3.0, 120.0),
            ("risk_aware", 5.0, 40.0),
            ("risk_aware", 7.0, 60.0),
        ):
            trials.append(
                {
                    "route": route,
                    "ping": {"packet_loss_percent": 0.0, "average_rtt_ms": rtt},
                    "iperf": {"throughput_mbps": throughput, "retransmits": 1},
                }
            )

        aggregate = aggregate_trials(trials)

        self.assertEqual(aggregate["traditional"]["throughput_mbps"]["mean"], 2.0)
        self.assertEqual(aggregate["risk_aware"]["average_rtt_ms"]["mean"], 50.0)

    def test_invalid_route_link_is_rejected(self):
        plan = build_mininet_plan()
        plan["routes"]["traditional"]["path"] = ["R1", "R8"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing link"):
                load_and_validate_plan(path)


if __name__ == "__main__":
    unittest.main()
