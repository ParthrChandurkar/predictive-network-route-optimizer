"""Emit a small deterministic batch of route telemetry for stack verification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network_builder import build_default_topology
from observability import configure_observability, shutdown_observability
from routing import route_ai, route_bellman_ford, route_traditional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:4318")
    args = parser.parse_args()
    state = configure_observability(enabled=True, endpoint=args.endpoint)
    graph = build_default_topology()
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise RuntimeError("Default topology needs at least two routers")
    source, destination = nodes[0], nodes[-1]
    results = [
        route_traditional(graph, source, destination),
        route_ai(graph, source, destination),
        route_bellman_ford(graph, source, destination),
    ]
    shutdown_observability()
    print(
        json.dumps(
            {
                "telemetry_enabled": state.enabled,
                "endpoint": state.endpoint,
                "routes_emitted": len(results),
                "source": source,
                "destination": destination,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
