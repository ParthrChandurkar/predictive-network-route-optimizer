"""Explicit LTE-cell to graph-edge mapping for routing experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import networkx as nx
import pandas as pd

from settings import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD


Edge = Tuple[str, str]


def _edge_key(source: str, target: str) -> Edge:
    return tuple(sorted((source, target)))


def _risk_label(score: float) -> str:
    if score < LOW_RISK_THRESHOLD:
        return "Low"
    if score < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def cell_risk_snapshot(scored_rows: pd.DataFrame, time_of_day: str) -> Dict[str, float]:
    """Average repeated observations for each cell at one time-of-day slot."""
    required = {"time_of_day", "cell_id", "degradation_score"}
    missing = sorted(required - set(scored_rows.columns))
    if missing:
        raise ValueError("Scored LTE rows are missing: " + ", ".join(missing))
    selected = scored_rows.loc[scored_rows["time_of_day"] == time_of_day]
    if selected.empty:
        raise ValueError(f"No LTE observations found for {time_of_day}")
    return {
        str(cell): float(score)
        for cell, score in selected.groupby("cell_id")["degradation_score"].mean().items()
    }


def build_demo_edge_cell_mapping(graph: nx.Graph, cell_ids: Iterable[str]) -> Dict[Edge, str]:
    """Create a deterministic synthetic mapping; no physical topology is inferred."""
    cells = sorted({str(cell) for cell in cell_ids})
    if not cells:
        raise ValueError("At least one LTE cell is required")
    edges = sorted(_edge_key(str(source), str(target)) for source, target in graph.edges())
    return {edge: cells[index % len(cells)] for index, edge in enumerate(edges)}


def apply_cell_risk_to_graph(
    graph: nx.Graph,
    cell_risks: Mapping[str, float],
    edge_cell_mapping: Mapping[Edge, str],
) -> dict:
    """Attach degradation scores to explicitly mapped edges in-place."""
    normalized_mapping = {
        _edge_key(str(source), str(target)): str(cell)
        for (source, target), cell in edge_cell_mapping.items()
    }
    for source, target in normalized_mapping:
        if not graph.has_edge(source, target):
            raise ValueError(f"Mapped edge {source}<->{target} does not exist")
    missing_cells = sorted(set(normalized_mapping.values()) - set(cell_risks))
    if missing_cells:
        raise ValueError("No degradation score for mapped cell(s): " + ", ".join(missing_cells))

    validated_scores = {}
    for cell in set(normalized_mapping.values()):
        score = float(cell_risks[cell])
        if not 0 <= score <= 1:
            raise ValueError(f"Risk score for {cell} must be between 0 and 1")
        validated_scores[cell] = score

    for _, _, attributes in graph.edges(data=True):
        if attributes.get("risk_source") == "lte_degradation":
            attributes.pop("routing_risk", None)
            attributes.pop("telemetry_cell", None)
            attributes.pop("risk_source", None)

    applied_scores = []
    for (source, target), cell in normalized_mapping.items():
        score = validated_scores[cell]
        attributes = graph[source][target]
        attributes["routing_risk"] = score
        attributes["risk_label"] = _risk_label(score)
        attributes["risk_source"] = "lte_degradation"
        attributes["telemetry_cell"] = cell
        applied_scores.append(score)

    return {
        "mapped_edges": len(applied_scores),
        "mapped_cells": len(set(normalized_mapping.values())),
        "mean_degradation_score": (
            sum(applied_scores) / len(applied_scores) if applied_scores else 0.0
        ),
        "mapping_semantics": "Deterministic demo mapping; not a physical LTE topology.",
    }
