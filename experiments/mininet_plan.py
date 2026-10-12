"""Build a portable Mininet experiment plan from the project topology."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from network_builder import build_default_topology
from routing import route_ai, route_traditional
from settings import DEFAULT_ROUTE_PENALTY, MININET_PLAN_PATH


def _edge_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def _path_nodes(result: dict) -> list[str]:
    if not result.get("found"):
        raise ValueError(f"{result.get('algorithm', 'Route')} did not find a path")
    edges = result["edges"]
    return [edges[0][0], *[target for _, target in edges]] if edges else []


def build_mininet_plan(
    source: str = "R1",
    destination: str = "R8",
    impairment_edge: tuple[str, str] = ("R1", "R2"),
    penalty: float = DEFAULT_ROUTE_PENALTY,
) -> dict:
    graph = build_default_topology()
    impaired = _edge_key(*impairment_edge)
    if not graph.has_edge(*impaired):
        raise ValueError(f"Impairment edge {impaired[0]}<->{impaired[1]} does not exist")

    for left, right, attributes in graph.edges(data=True):
        attributes["routing_risk"] = 0.05
        attributes["risk_label"] = "Low"
        attributes["risk_source"] = "controlled_mininet_scenario"
        if _edge_key(left, right) == impaired:
            attributes["routing_risk"] = 0.95
            attributes["risk_label"] = "High"

    traditional = route_traditional(graph, source, destination)
    risk_aware = route_ai(graph, source, destination, penalty=penalty)
    traditional_path = _path_nodes(traditional)
    risk_aware_path = _path_nodes(risk_aware)
    if traditional_path == risk_aware_path:
        raise ValueError("Scenario does not produce distinct comparison routes")

    links = []
    for left, right, attributes in sorted(
        graph.edges(data=True), key=lambda edge: _edge_key(edge[0], edge[1])
    ):
        source_node, target_node = _edge_key(left, right)
        links.append(
            {
                "source": source_node,
                "target": target_node,
                "bandwidth_mbps": min(float(attributes["bandwidth"]), 100.0),
                "source_bandwidth_mbps": float(attributes["bandwidth"]),
                "delay_ms": float(attributes["latency"]),
                "loss_percent": float(attributes["packet_loss"]),
                "distance": float(attributes["distance"]),
                "routing_risk": float(attributes["routing_risk"]),
            }
        )

    return {
        "schema_version": 1,
        "phase": "phase-4-mininet-routing-experiment",
        "topology": {
            "routers": sorted(graph.nodes()),
            "links": links,
            "endpoint_hosts": {"source": "hsrc", "destination": "hdst"},
        },
        "scenario": {
            "name": "single-link-severe-degradation",
            "source_router": source,
            "destination_router": destination,
            "impaired_edge": list(impaired),
            "impairment": {
                "bandwidth_mbps": 5.0,
                "delay_ms": 100.0,
                "loss_percent": 5.0,
            },
            "penalty": float(penalty),
        },
        "routes": {
            "traditional": {
                "algorithm": traditional["algorithm"],
                "path": traditional_path,
                "distance": traditional["dist_cost"],
                "average_risk": traditional["avg_risk"],
            },
            "risk_aware": {
                "algorithm": risk_aware["algorithm"],
                "path": risk_aware_path,
                "distance": risk_aware["dist_cost"],
                "average_risk": risk_aware["avg_risk"],
            },
        },
        "measurements": {
            "ping_count": 10,
            "iperf_seconds": 5,
            "iperf_parallel_streams": 1,
            "repetitions": 3,
        },
        "interpretation": (
            "This controlled experiment validates routing behavior under emulated link "
            "degradation; it does not recreate the Kaggle LTE deployment topology."
        ),
    }


def write_plan(plan: dict, destination: Path = MININET_PLAN_PATH) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MININET_PLAN_PATH)
    parser.add_argument("--source", default="R1")
    parser.add_argument("--destination", default="R8")
    parser.add_argument("--impair", nargs=2, metavar=("ROUTER_A", "ROUTER_B"), default=("R1", "R2"))
    parser.add_argument("--penalty", type=float, default=DEFAULT_ROUTE_PENALTY)
    args = parser.parse_args()
    plan = build_mininet_plan(
        source=args.source,
        destination=args.destination,
        impairment_edge=tuple(args.impair),
        penalty=args.penalty,
    )
    write_plan(plan, args.output)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
