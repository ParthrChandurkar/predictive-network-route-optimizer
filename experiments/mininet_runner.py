"""Execute or dry-run a generated routing experiment inside Linux/Mininet."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


PING_SUMMARY = re.compile(
    r"(?P<sent>\d+) packets transmitted, (?P<received>\d+) received, "
    r"(?P<loss>[\d.]+)% packet loss"
)
PING_RTT = re.compile(r"= [\d.]+/(?P<average>[\d.]+)/")


def load_and_validate_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "topology", "scenario", "routes", "measurements"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError("Experiment plan is missing: " + ", ".join(missing))
    if plan["schema_version"] != 1:
        raise ValueError("Unsupported experiment-plan schema")

    routers = set(plan["topology"]["routers"])
    link_edges = {
        frozenset((link["source"], link["target"]))
        for link in plan["topology"]["links"]
    }
    for name, route in plan["routes"].items():
        path_nodes = route["path"]
        if len(path_nodes) < 2 or not set(path_nodes).issubset(routers):
            raise ValueError(f"Route {name} contains invalid routers")
        for left, right in zip(path_nodes, path_nodes[1:]):
            if frozenset((left, right)) not in link_edges:
                raise ValueError(f"Route {name} uses missing link {left}<->{right}")
    return plan


def parse_ping_output(output: str) -> dict:
    summary = PING_SUMMARY.search(output)
    if not summary:
        raise ValueError("Could not parse ping output")
    rtt = PING_RTT.search(output)
    return {
        "packets_sent": int(summary.group("sent")),
        "packets_received": int(summary.group("received")),
        "packet_loss_percent": float(summary.group("loss")),
        "average_rtt_ms": float(rtt.group("average")) if rtt else None,
    }


def parse_iperf_output(output: str) -> dict:
    payload = json.loads(output)
    end = payload.get("end", {})
    received = end.get("sum_received", {})
    sent = end.get("sum_sent", {})
    bits_per_second = received.get("bits_per_second", sent.get("bits_per_second", 0.0))
    return {
        "throughput_mbps": float(bits_per_second) / 1_000_000,
        "retransmits": int(sent.get("retransmits", 0)),
    }


def dry_run_summary(plan: dict) -> dict:
    return {
        "mode": "dry-run",
        "routers": len(plan["topology"]["routers"]),
        "links": len(plan["topology"]["links"]),
        "scenario": plan["scenario"]["name"],
        "impaired_edge": plan["scenario"]["impaired_edge"],
        "routes": {
            name: " -> ".join(route["path"])
            for name, route in plan["routes"].items()
        },
        "ready_for_mininet": True,
    }


def _mininet_imports():
    try:
        from mininet.link import TCLink
        from mininet.net import Mininet
        from mininet.node import Node
    except ImportError as exc:
        raise RuntimeError(
            "Mininet is unavailable. Install mininet, openvswitch-switch, and iperf3 in Linux."
        ) from exc
    return Mininet, Node, TCLink


def _build_network(plan: dict, route_path: list[str]):
    Mininet, Node, TCLink = _mininet_imports()

    class LinuxRouter(Node):
        def config(self, **params):
            result = super().config(**params)
            self.cmd("sysctl -q -w net.ipv4.ip_forward=1")
            return result

        def terminate(self):
            self.cmd("sysctl -q -w net.ipv4.ip_forward=0")
            super().terminate()

    network = Mininet(controller=None, build=False, link=TCLink)
    routers = {
        name: network.addHost(name.lower(), cls=LinuxRouter, ip=None)
        for name in plan["topology"]["routers"]
    }
    link_addresses = {}
    for index, link in enumerate(plan["topology"]["links"], start=1):
        left, right = link["source"], link["target"]
        subnet_octet = index * 4
        left_ip = f"10.1.{subnet_octet // 256}.{subnet_octet % 256 + 1}"
        right_ip = f"10.1.{subnet_octet // 256}.{subnet_octet % 256 + 2}"
        network.addLink(
            routers[left],
            routers[right],
            cls=TCLink,
            params1={"ip": f"{left_ip}/30"},
            params2={"ip": f"{right_ip}/30"},
            bw=link["bandwidth_mbps"],
            delay=f"{link['delay_ms']}ms",
            loss=link["loss_percent"],
        )
        link_addresses[frozenset((left, right))] = {
            left: left_ip,
            right: right_ip,
        }

    hsrc = network.addHost("hsrc", ip="10.0.0.2/24", defaultRoute="via 10.0.0.1")
    hdst = network.addHost("hdst", ip="10.0.1.2/24", defaultRoute="via 10.0.1.1")
    network.addLink(hsrc, routers[plan["scenario"]["source_router"]], cls=TCLink,
                    params2={"ip": "10.0.0.1/24"}, bw=1000)
    network.addLink(hdst, routers[plan["scenario"]["destination_router"]], cls=TCLink,
                    params2={"ip": "10.0.1.1/24"}, bw=1000)
    network.build()

    impairment = plan["scenario"]["impairment"]
    impaired_nodes = [routers[name] for name in plan["scenario"]["impaired_edge"]]
    for interface in network.linksBetween(*impaired_nodes)[0].intf1, network.linksBetween(*impaired_nodes)[0].intf2:
        interface.config(
            bw=impairment["bandwidth_mbps"],
            delay=f"{impairment['delay_ms']}ms",
            loss=impairment["loss_percent"],
        )

    for left, right in zip(route_path, route_path[1:]):
        addresses = link_addresses[frozenset((left, right))]
        routers[left].cmd(f"ip route replace 10.0.1.0/24 via {addresses[right]}")
        routers[right].cmd(f"ip route replace 10.0.0.0/24 via {addresses[left]}")
    return network, hsrc, hdst


def _run_trial(plan: dict, route_name: str) -> dict:
    route = plan["routes"][route_name]
    network = hsrc = hdst = None
    server = None
    try:
        network, hsrc, hdst = _build_network(plan, route["path"])
        ping_output = hsrc.cmd(
            f"ping -c {plan['measurements']['ping_count']} -W 2 10.0.1.2"
        )
        server = hdst.popen(["iperf3", "-s", "-1"])
        time.sleep(0.5)
        iperf_output = hsrc.cmd(
            "iperf3 -c 10.0.1.2 "
            f"-t {plan['measurements']['iperf_seconds']} "
            f"-P {plan['measurements']['iperf_parallel_streams']} -J"
        )
        server.wait(timeout=10)
        return {
            "route": route_name,
            "path": route["path"],
            "ping": parse_ping_output(ping_output),
            "iperf": parse_iperf_output(iperf_output),
        }
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
        if network is not None:
            network.stop()


def _metric_summary(values: list[float]) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "standard_deviation": None, "minimum": None, "maximum": None}
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_trials(trials: list[dict]) -> dict:
    aggregates = {}
    for route_name in ("traditional", "risk_aware"):
        route_trials = [trial for trial in trials if trial["route"] == route_name]
        aggregates[route_name] = {
            "runs": len(route_trials),
            "packet_loss_percent": _metric_summary(
                [trial["ping"]["packet_loss_percent"] for trial in route_trials]
            ),
            "average_rtt_ms": _metric_summary(
                [trial["ping"]["average_rtt_ms"] for trial in route_trials]
            ),
            "throughput_mbps": _metric_summary(
                [trial["iperf"]["throughput_mbps"] for trial in route_trials]
            ),
            "retransmits": _metric_summary(
                [trial["iperf"]["retransmits"] for trial in route_trials]
            ),
        }
    return aggregates


def run_experiment(plan: dict) -> dict:
    if os.geteuid() != 0:
        raise PermissionError("Mininet experiments must run as root")
    trials = []
    repetitions = int(plan["measurements"].get("repetitions", 1))
    for repetition in range(1, repetitions + 1):
        order = ("traditional", "risk_aware")
        if repetition % 2 == 0:
            order = tuple(reversed(order))
        for route_name in order:
            trial = _run_trial(plan, route_name)
            trial["repetition"] = repetition
            trials.append(trial)
    aggregates = aggregate_trials(trials)
    traditional_throughput = aggregates["traditional"]["throughput_mbps"]["mean"]
    risk_aware_throughput = aggregates["risk_aware"]["throughput_mbps"]["mean"]
    traditional_rtt = aggregates["traditional"]["average_rtt_ms"]["mean"]
    risk_aware_rtt = aggregates["risk_aware"]["average_rtt_ms"]["mean"]
    throughput_gain = risk_aware_throughput - traditional_throughput
    rtt_reduction = traditional_rtt - risk_aware_rtt
    return {
        "phase": plan["phase"],
        "environment": {
            "kernel": platform.release(),
            "mininet": subprocess.check_output(["mn", "--version"], text=True).strip(),
            "iperf3": subprocess.check_output(["iperf3", "--version"], text=True).splitlines()[0],
        },
        "scenario": plan["scenario"],
        "trials": trials,
        "aggregates": aggregates,
        "comparison": {
            "throughput_gain_mbps": (
                throughput_gain
            ),
            "throughput_gain_percent": throughput_gain / traditional_throughput * 100,
            "rtt_reduction_ms": rtt_reduction,
            "rtt_reduction_percent": rtt_reduction / traditional_rtt * 100,
            "packet_loss_reduction_points": (
                aggregates["traditional"]["packet_loss_percent"]["mean"]
                - aggregates["risk_aware"]["packet_loss_percent"]["mean"]
            ),
            "interpretation": (
                "Three controlled repetitions; this is an emulator result, not evidence "
                "of production-network performance."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = load_and_validate_plan(args.plan)
    result = dry_run_summary(plan) if args.dry_run else run_experiment(plan)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
