"""Build the consolidated, reproducible Phase 8 evaluation report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from integration.lte_routing import (
    apply_cell_risk_to_graph,
    build_demo_edge_cell_mapping,
    cell_risk_snapshot,
)
from modeling.lte_model import load_lte_risk_model, predict_lte_risk
from network_builder import build_default_topology
from routing import (
    route_ai,
    route_bellman_ford,
    route_bellman_ford_ai,
    route_floyd_warshall,
    route_floyd_warshall_ai,
    route_traditional,
)
from settings import (
    DEFAULT_ROUTE_PENALTY,
    LTE_MODEL_PATH,
    LTE_TRAIN_PROCESSED_PATH,
    PHASE1_REPORT_PATH,
    PHASE2_REPORT_PATH,
    PHASE3_REPORT_PATH,
    PHASE7_REPORT_PATH,
    PHASE8_REPORT_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)


EVIDENCE_PATHS = {
    "ingestion": PHASE1_REPORT_PATH,
    "model_evaluation": PHASE2_REPORT_PATH,
    "model_artifact": PHASE3_REPORT_PATH,
    "mininet": PROJECT_ROOT / "reports" / "phase4_mininet_results.json",
    "observability": PROJECT_ROOT / "reports" / "phase5_observability.json",
    "dashboard": PROJECT_ROOT / "reports" / "phase6_dashboard.json",
    "pattern_discovery": PHASE7_REPORT_PATH,
}

RouteFunction = Callable[..., dict]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Evidence report must contain a JSON object: {path}")
    return value


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(result: dict) -> list[str]:
    edges = result.get("edges", [])
    return [edges[0][0], *[right for _, right in edges]] if edges else []


def _compact_route(result: dict) -> dict:
    return {
        "algorithm": result["algorithm"],
        "found": bool(result["found"]),
        "path": _path(result),
        "distance": result["dist_cost"],
        "hops": result["hops"],
        "average_risk": result["avg_risk"],
        "objective_cost": result["algo_cost"],
    }


def _run_all_routes(graph, source: str, destination: str, penalty: float) -> dict:
    functions: tuple[tuple[str, RouteFunction, bool], ...] = (
        ("dijkstra", route_traditional, False),
        ("bellman_ford", route_bellman_ford, False),
        ("floyd_warshall", route_floyd_warshall, False),
        ("risk_aware_dijkstra", route_ai, True),
        ("risk_aware_bellman_ford", route_bellman_ford_ai, True),
        ("risk_aware_floyd_warshall", route_floyd_warshall_ai, True),
    )
    results = {}
    for name, function, risk_aware in functions:
        result = function(graph, source, destination, penalty) if risk_aware else function(
            graph, source, destination
        )
        results[name] = _compact_route(result)
    return results


def controlled_routing_evaluation(penalty: float = DEFAULT_ROUTE_PENALTY) -> dict:
    """Check algorithm agreement on the same controlled degraded-link scenario."""
    graph = build_default_topology()
    for left, right, attributes in graph.edges(data=True):
        attributes["routing_risk"] = 0.05
        attributes["risk_label"] = "Low"
        attributes["risk_source"] = "controlled_evaluation"
        if set((left, right)) == {"R1", "R2"}:
            attributes["routing_risk"] = 0.95
            attributes["risk_label"] = "High"

    routes = _run_all_routes(graph, "R1", "R8", penalty)
    traditional_paths = {tuple(routes[name]["path"]) for name in (
        "dijkstra", "bellman_ford", "floyd_warshall"
    )}
    risk_aware_paths = {tuple(routes[name]["path"]) for name in (
        "risk_aware_dijkstra", "risk_aware_bellman_ford", "risk_aware_floyd_warshall"
    )}
    traditional_path = next(iter(traditional_paths)) if len(traditional_paths) == 1 else ()
    risk_aware_path = next(iter(risk_aware_paths)) if len(risk_aware_paths) == 1 else ()
    return {
        "scenario": "single-link-severe-degradation",
        "source": "R1",
        "destination": "R8",
        "impaired_edge": ["R1", "R2"],
        "penalty": float(penalty),
        "routes": routes,
        "checks": {
            "all_routes_found": all(route["found"] for route in routes.values()),
            "traditional_algorithms_agree": len(traditional_paths) == 1,
            "risk_aware_algorithms_agree": len(risk_aware_paths) == 1,
            "risk_aware_route_differs": bool(
                traditional_path and risk_aware_path and traditional_path != risk_aware_path
            ),
            "risk_aware_route_avoids_impaired_edge": not any(
                set(edge) == {"R1", "R2"}
                for edge in zip(risk_aware_path, risk_aware_path[1:])
            ),
        },
    }


def lte_routing_demonstration(
    data_path: Path = LTE_TRAIN_PROCESSED_PATH,
    model_path: Path = LTE_MODEL_PATH,
    time_of_day: str = "00:00",
    penalty: float = DEFAULT_ROUTE_PENALTY,
) -> dict:
    """Run one deterministic LTE-score-to-route demonstration.

    This validates software integration only. The cell-to-edge assignment is synthetic.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Processed LTE data is missing: {data_path}")
    model = load_lte_risk_model(model_path)
    if model is None:
        raise FileNotFoundError(f"A compatible LTE model is missing: {model_path}")
    frame = pd.read_csv(data_path)
    selected = frame.loc[frame["time_of_day"].astype(str) == time_of_day].copy()
    if selected.empty:
        raise ValueError(f"No LTE rows are available at {time_of_day}")

    scored = predict_lte_risk(model, selected)
    cell_risks = cell_risk_snapshot(scored, time_of_day)
    graph = build_default_topology()
    mapping = build_demo_edge_cell_mapping(graph, cell_risks)
    mapping_summary = apply_cell_risk_to_graph(graph, cell_risks, mapping)
    routes = _run_all_routes(graph, "R1", "R8", penalty)
    traditional = routes["dijkstra"]
    risk_aware = routes["risk_aware_dijkstra"]
    return {
        "evidence_class": "integration_demo",
        "time_of_day": time_of_day,
        "observations": len(selected),
        "cells": len(cell_risks),
        "score_summary": {
            "minimum": min(cell_risks.values()),
            "mean": sum(cell_risks.values()) / len(cell_risks),
            "maximum": max(cell_risks.values()),
        },
        "mapping": mapping_summary,
        "routes": routes,
        "route_changed": traditional["path"] != risk_aware["path"],
        "average_path_risk_reduction": traditional["average_risk"] - risk_aware["average_risk"],
        "distance_overhead": risk_aware["distance"] - traditional["distance"],
        "limitation": (
            "LTE cells are assigned to graph links deterministically for demonstration; "
            "the dataset does not contain this physical topology."
        ),
    }


def build_phase8_report(
    evidence_paths: Optional[Dict[str, Path]] = None,
    data_path: Path = LTE_TRAIN_PROCESSED_PATH,
    model_path: Path = LTE_MODEL_PATH,
) -> Dict[str, Any]:
    paths = evidence_paths or EVIDENCE_PATHS
    evidence = {name: _load_json(Path(path)) for name, path in paths.items()}
    phase2 = evidence["model_evaluation"]
    phase4 = evidence["mininet"]
    phase7 = evidence["pattern_discovery"]
    held_out = phase2["protocols"]["unseen_cell"]["models"]["random_forest"]["aggregate"]
    controlled = controlled_routing_evaluation()
    lte_demo = lte_routing_demonstration(data_path=data_path, model_path=model_path)

    checks = {
        "all_phase_reports_present": len(evidence) == len(EVIDENCE_PATHS),
        "ingestion_model_row_count_matches": (
            phase2["dataset"]["rows"] == evidence["model_artifact"]["training_rows"]
        ),
        "pattern_source_row_count_matches": (
            phase2["dataset"]["rows"] == phase7["dataset"]["original_rows"]
        ),
        "model_artifact_hash_matches": (
            model_path.exists()
            and _sha256(model_path) == evidence["model_artifact"]["artifact"]["sha256"]
        ),
        "controlled_routing_checks_pass": all(controlled["checks"].values()),
        "lte_integration_routes_found": all(
            route["found"] for route in lte_demo["routes"].values()
        ),
        "mininet_has_repeated_trials": (
            phase4["aggregates"]["traditional"]["runs"] >= 3
            and phase4["aggregates"]["risk_aware"]["runs"] >= 3
        ),
    }

    return {
        "schema_version": 1,
        "phase": "phase-8-final-integration-and-evaluation",
        "status": "complete" if all(checks.values()) else "incomplete",
        "headline": {
            "held_out_cell_f1": held_out["f1"],
            "held_out_cell_roc_auc": held_out["roc_auc"],
            "mininet_throughput_gain_percent": phase4["comparison"]["throughput_gain_percent"],
            "mininet_rtt_reduction_percent": phase4["comparison"]["rtt_reduction_percent"],
            "mininet_packet_loss_reduction_points": phase4["comparison"]["packet_loss_reduction_points"],
            "validated_pattern_precision": phase7["headline_findings"][
                "dominant_validated_signature"
            ]["validation_unusual_rate"],
            "validated_pattern_coverage": phase7["headline_findings"][
                "dominant_validated_signature"
            ]["validation_unusual_coverage"],
        },
        "evidence_classes": {
            "held_out_data": (
                "LTE classification and pattern metrics use group-separated cells."
            ),
            "integration_demo": (
                "LTE scores drive routes through a deterministic, non-physical cell-to-link mapping."
            ),
            "controlled_emulation": (
                "Mininet measurements come from three repeated WSL2 experiments."
            ),
        },
        "controlled_routing": controlled,
        "lte_routing_integration": lte_demo,
        "measured_mininet": {
            "environment": phase4["environment"],
            "repetitions_per_policy": phase4["aggregates"]["traditional"]["runs"],
            "comparison": phase4["comparison"],
        },
        "observability": {
            "architecture": evidence["observability"]["architecture"],
            "verified_scope": evidence["observability"]["scope"],
        },
        "verification": {
            "checks": checks,
            "passed": sum(checks.values()),
            "total": len(checks),
        },
        "sources": {
            name: _portable_path(Path(path)) for name, path in paths.items()
        },
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "lte_data": _portable_path(data_path),
            "lte_model": _portable_path(model_path),
            "command": "python -m evaluation.phase8",
        },
        "claim_boundaries": [
            "LTE labels represent unusual behavior, not confirmed physical-link failures.",
            "The LTE dataset has no physical mapping to the demonstration topology.",
            "Mininet results are controlled emulator evidence, not production guarantees.",
            "Three Mininet repetitions are sufficient for a demo, not broad statistical generalization.",
            "The sharp validated KPI rule may reflect quantization or label construction rather than causality.",
        ],
    }


def write_report(report: dict, destination: Path = PHASE8_REPORT_PATH) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PHASE8_REPORT_PATH)
    parser.add_argument("--data", type=Path, default=LTE_TRAIN_PROCESSED_PATH)
    parser.add_argument("--model", type=Path, default=LTE_MODEL_PATH)
    args = parser.parse_args()
    report = build_phase8_report(data_path=args.data, model_path=args.model)
    write_report(report, args.output)
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
