"""Generate a reproducible Phase 0 baseline without replacing saved artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_generator import generate_network_dataset
from model import predict_graph_edges, train_model
from network_builder import build_default_topology
from routing import (
    route_ai,
    route_bellman_ford,
    route_bellman_ford_ai,
    route_floyd_warshall,
    route_floyd_warshall_ai,
    route_traditional,
)
from settings import DEFAULT_ROUTE_PENALTY, RANDOM_SEED


def build_baseline_report(
    samples: int = 1700,
    source: str = "R1",
    destination: str = "R8",
    penalty: float = DEFAULT_ROUTE_PENALTY,
    random_state: int = RANDOM_SEED,
) -> dict:
    dataset = generate_network_dataset(samples, random_state=random_state)
    model, scaler, metrics = train_model(
        dataset,
        persist=False,
        random_state=random_state,
    )

    graph = build_default_topology()
    predict_graph_edges(model, scaler, graph)
    results = [
        route_traditional(graph, source, destination),
        route_ai(graph, source, destination, penalty),
        route_bellman_ford(graph, source, destination),
        route_bellman_ford_ai(graph, source, destination, penalty),
        route_floyd_warshall(graph, source, destination),
        route_floyd_warshall_ai(graph, source, destination, penalty),
    ]

    route_fields = (
        "algorithm",
        "found",
        "path_str",
        "hops",
        "dist_cost",
        "algo_cost",
        "risky_links",
        "avg_risk",
    )
    return {
        "phase": "phase-0-synthetic-baseline",
        "dataset": {
            "samples": len(dataset),
            "random_state": random_state,
            "failure_rate": round(float(dataset["failure"].mean()), 4),
        },
        "model": {
            key: metrics[key]
            for key in ("accuracy", "precision", "recall", "f1", "feature_importance")
        },
        "routing": {
            "source": source,
            "destination": destination,
            "penalty": penalty,
            "results": [
                {key: result[key] for key in route_fields}
                for result in results
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1700)
    parser.add_argument("--source", default="R1")
    parser.add_argument("--destination", default="R8")
    parser.add_argument("--penalty", type=float, default=DEFAULT_ROUTE_PENALTY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_baseline_report(
        samples=args.samples,
        source=args.source,
        destination=args.destination,
        penalty=args.penalty,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

