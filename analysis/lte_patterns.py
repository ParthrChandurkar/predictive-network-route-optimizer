"""Discover reproducible patterns in labeled LTE unusual-behavior telemetry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, _tree

from modeling.lte_features import build_lte_features
from settings import LTE_TRAIN_PROCESSED_PATH, PHASE7_REPORT_PATH, RANDOM_SEED
from telemetry.schemas import LTE_LABEL_COLUMN, TelemetryValidationError


RAW_KPIS = (
    "prb_usage_ul",
    "prb_usage_dl",
    "mean_throughput_dl",
    "mean_throughput_ul",
    "max_throughput_dl",
    "max_throughput_ul",
    "mean_ue_dl",
    "mean_ue_ul",
    "max_ue_dl",
    "max_ue_ul",
    "max_ue_total",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(positives: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a binomial Wilson score interval."""
    if total <= 0:
        return 0.0, 0.0
    proportion = positives / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate_records(frame: pd.DataFrame, group_column: str) -> list[dict]:
    grouped = frame.groupby(group_column)[LTE_LABEL_COLUMN].agg(["size", "sum", "mean"])
    records = []
    for group, row in grouped.iterrows():
        low, high = wilson_interval(int(row["sum"]), int(row["size"]))
        records.append(
            {
                group_column: int(group) if isinstance(group, (int, np.integer)) else str(group),
                "observations": int(row["size"]),
                "unusual_count": int(row["sum"]),
                "unusual_rate": float(row["mean"]),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return sorted(records, key=lambda item: item["unusual_rate"], reverse=True)


def _feature_shifts(frame: pd.DataFrame) -> list[dict]:
    unusual = frame[LTE_LABEL_COLUMN] == 1
    records = []
    for feature in RAW_KPIS:
        positive = frame.loc[unusual, feature].astype(float)
        negative = frame.loc[~unusual, feature].astype(float)
        pooled_variance = (
            (len(positive) - 1) * positive.var(ddof=1)
            + (len(negative) - 1) * negative.var(ddof=1)
        ) / (len(positive) + len(negative) - 2)
        pooled_std = math.sqrt(max(0.0, pooled_variance))
        standardized_difference = (
            (positive.mean() - negative.mean()) / pooled_std if pooled_std else 0.0
        )
        records.append(
            {
                "feature": feature,
                "unusual_mean": float(positive.mean()),
                "normal_mean": float(negative.mean()),
                "unusual_median": float(positive.median()),
                "normal_median": float(negative.median()),
                "standardized_mean_difference": float(standardized_difference),
            }
        )
    return sorted(
        records,
        key=lambda item: abs(item["standardized_mean_difference"]),
        reverse=True,
    )


def _canonical_rule(bounds: dict[str, dict[str, float]]) -> str:
    conditions = []
    for feature in sorted(bounds):
        lower = bounds[feature].get("lower")
        upper = bounds[feature].get("upper")
        if lower is not None:
            conditions.append(f"{feature} > {lower:.4f}")
        if upper is not None:
            conditions.append(f"{feature} <= {upper:.4f}")
    return " AND ".join(conditions)


def _leaf_rules(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, str]:
    rules: dict[int, str] = {}
    structure = tree.tree_

    def walk(node: int, bounds: dict[str, dict[str, float]]) -> None:
        feature_index = structure.feature[node]
        if feature_index == _tree.TREE_UNDEFINED:
            rules[node] = _canonical_rule(bounds)
            return
        feature = feature_names[feature_index]
        threshold = float(structure.threshold[node])

        left = {name: values.copy() for name, values in bounds.items()}
        left.setdefault(feature, {})["upper"] = min(
            threshold, left.get(feature, {}).get("upper", float("inf"))
        )
        walk(structure.children_left[node], left)

        right = {name: values.copy() for name, values in bounds.items()}
        right.setdefault(feature, {})["lower"] = max(
            threshold, right.get(feature, {}).get("lower", -float("inf"))
        )
        walk(structure.children_right[node], right)

    walk(0, {})
    return rules


def _validated_rules(frame: pd.DataFrame) -> dict:
    features = build_lte_features(frame).drop(columns=["cell_id"])
    target = frame[LTE_LABEL_COLUMN].astype(int)
    groups = frame["cell_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    discovery_index, validation_index = next(splitter.split(features, target, groups))

    minimum_leaf = min(300, max(5, len(discovery_index) // 20))
    minimum_validation_support = min(100, max(5, len(validation_index) // 100))

    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=minimum_leaf,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    tree.fit(features.iloc[discovery_index], target.iloc[discovery_index])
    validation_probability = tree.predict_proba(features.iloc[validation_index])[:, 1]
    validation_prediction = validation_probability >= 0.5
    validation_target = target.iloc[validation_index].to_numpy()
    validation_leaves = tree.apply(features.iloc[validation_index])
    discovery_leaves = tree.apply(features.iloc[discovery_index])
    discovery_target = target.iloc[discovery_index].to_numpy()
    rules = _leaf_rules(tree, list(features.columns))
    baseline_rate = float(validation_target.mean())

    candidates = []
    for leaf_id, rule in rules.items():
        validation_mask = validation_leaves == leaf_id
        discovery_mask = discovery_leaves == leaf_id
        support = int(validation_mask.sum())
        if support < minimum_validation_support:
            continue
        unusual_count = int(validation_target[validation_mask].sum())
        rate = unusual_count / support
        low, high = wilson_interval(unusual_count, support)
        candidates.append(
            {
                "rule": rule,
                "validation_observations": support,
                "validation_unusual_count": unusual_count,
                "validation_unusual_rate": rate,
                "validation_lift": rate / baseline_rate if baseline_rate else 0.0,
                "validation_wilson_95_low": low,
                "validation_wilson_95_high": high,
                "validation_unusual_coverage": unusual_count / max(1, int(validation_target.sum())),
                "discovery_observations": int(discovery_mask.sum()),
                "discovery_unusual_rate": float(discovery_target[discovery_mask].mean()),
            }
        )
    candidates.sort(key=lambda item: (item["validation_lift"], item["validation_observations"]), reverse=True)
    return {
        "protocol": "A shallow decision tree was fit on discovery cells and evaluated on disjoint validation cells.",
        "discovery_cells": sorted(groups.iloc[discovery_index].unique().tolist()),
        "validation_cells": sorted(groups.iloc[validation_index].unique().tolist()),
        "discovery_rows": int(len(discovery_index)),
        "validation_rows": int(len(validation_index)),
        "validation_baseline_rate": baseline_rate,
        "validation_metrics": {
            "f1": float(f1_score(validation_target, validation_prediction)),
            "roc_auc": float(roc_auc_score(validation_target, validation_probability)),
            "precision": float(precision_score(validation_target, validation_prediction)),
            "recall": float(recall_score(validation_target, validation_prediction)),
        },
        "high_lift_rules": candidates[:5],
        "low_lift_rules": sorted(candidates, key=lambda item: item["validation_lift"])[:3],
    }


def discover_lte_patterns(frame: pd.DataFrame, source_path: Path | None = None) -> dict:
    required = {"time_of_day", "minute_of_day", "cell_id", LTE_LABEL_COLUMN, *RAW_KPIS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TelemetryValidationError("Pattern analysis is missing columns: " + ", ".join(missing))
    labels = set(frame[LTE_LABEL_COLUMN].dropna().astype(int).unique())
    if labels != {0, 1}:
        raise TelemetryValidationError("Pattern discovery requires both labeled classes")

    original_rows = len(frame)
    raw_cell_patterns = _rate_records(frame, "cell_id")
    clean = frame.drop_duplicates().reset_index(drop=True)
    clean["hour"] = (clean["minute_of_day"] // 60).astype(int)
    baseline_rate = float(clean[LTE_LABEL_COLUMN].mean())
    cell_patterns = _rate_records(clean, "cell_id")
    hour_patterns = _rate_records(clean, "hour")
    time_slot_patterns = _rate_records(clean, "time_of_day")
    feature_shifts = _feature_shifts(clean)
    validated_rules = _validated_rules(clean)

    top_five_unusual = sum(item["unusual_count"] for item in cell_patterns[:5])
    total_unusual = int(clean[LTE_LABEL_COLUMN].sum())
    top_rule = validated_rules["high_lift_rules"][0]
    return {
        "phase": "phase-7-lte-pattern-discovery",
        "dataset": {
            "source": str(source_path.as_posix()) if source_path else "in-memory",
            "sha256": _sha256(source_path) if source_path and source_path.exists() else None,
            "original_rows": original_rows,
            "deduplicated_rows": len(clean),
            "duplicates_removed": original_rows - len(clean),
            "cells": int(clean["cell_id"].nunique()),
            "unusual_count": total_unusual,
            "unusual_rate": baseline_rate,
        },
        "headline_findings": {
            "dominant_validated_signature": top_rule,
            "cell_rate_spread_points": (
                cell_patterns[0]["unusual_rate"] - cell_patterns[-1]["unusual_rate"]
            ) * 100,
            "raw_cell_rate_spread_points": (
                raw_cell_patterns[0]["unusual_rate"]
                - raw_cell_patterns[-1]["unusual_rate"]
            ) * 100,
            "hour_rate_spread_points": (
                hour_patterns[0]["unusual_rate"] - hour_patterns[-1]["unusual_rate"]
            ) * 100,
            "top_five_cell_share_of_unusual": top_five_unusual / total_unusual,
            "largest_marginal_feature_shift": feature_shifts[0],
        },
        "cell_patterns": cell_patterns,
        "raw_cell_patterns": raw_cell_patterns,
        "hour_patterns": hour_patterns,
        "time_slot_patterns": time_slot_patterns,
        "feature_shifts": feature_shifts,
        "validated_interpretable_rules": validated_rules,
        "interpretation": [
            "Unusual labels are broadly distributed across cells rather than isolated to a small hotspot set.",
            "Hour-of-day variation is modest; the source lacks dates, so recurrence and event duration cannot be established.",
            "Single-KPI mean shifts are small, while nonlinear value bands generalize strongly across held-out cells.",
            "The strongest rule uses a narrow mean_ue_ul boundary and may reflect measurement quantization or label construction rather than a physical degradation mechanism."
        ],
        "limitations": [
            "Labels represent unusual LTE behavior, not confirmed outages or physical-link failures.",
            "The analysis is descriptive and predictive; it does not establish causality.",
            "Only time of day is available, with no calendar date or event identifier.",
            "Exact duplicate rows are removed before pattern summaries.",
            "Cell-rate rankings are sensitive to duplicate removal and both raw and deduplicated summaries are retained.",
            "Interpretable rules are validated on disjoint cells but still require validation on an independent deployment."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LTE_TRAIN_PROCESSED_PATH)
    parser.add_argument("--output", type=Path, default=PHASE7_REPORT_PATH)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    report = discover_lte_patterns(frame, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["headline_findings"], indent=2))


if __name__ == "__main__":
    main()
