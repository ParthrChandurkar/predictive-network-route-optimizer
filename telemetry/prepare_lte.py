"""Prepare the raw Kaggle LTE splits and emit a reproducible ingestion report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from settings import (
    LTE_TEST_PROCESSED_PATH,
    LTE_TEST_RAW_PATH,
    LTE_TRAIN_PROCESSED_PATH,
    LTE_TRAIN_RAW_PATH,
    PHASE1_REPORT_PATH,
    PROJECT_ROOT,
)
from telemetry.lte_kaggle import load_lte_dataset
from telemetry.schemas import LTE_LABEL_COLUMN, TelemetryValidationError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "path": _portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_csv(frame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def _write_json(payload: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def prepare_lte_data(
    train_path: Path = LTE_TRAIN_RAW_PATH,
    test_path: Path = LTE_TEST_RAW_PATH,
    train_output: Path = LTE_TRAIN_PROCESSED_PATH,
    test_output: Path = LTE_TEST_PROCESSED_PATH,
    report_path: Path = PHASE1_REPORT_PATH,
) -> Dict[str, Any]:
    """Validate both source splits, write canonical CSVs, and return lineage data."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_output = Path(train_output)
    test_output = Path(test_output)
    report_path = Path(report_path)

    train = load_lte_dataset(train_path, require_label=True)
    test = load_lte_dataset(test_path, require_label=False)
    if test.report.labeled:
        raise TelemetryValidationError("The competition test split must not contain labels")

    train_features = [column for column in train.frame if column != LTE_LABEL_COLUMN]
    if train_features != list(test.frame.columns):
        raise TelemetryValidationError("Training and test feature schemas do not match")

    train_cells = set(train.frame["cell_id"].astype(str))
    test_cells = set(test.frame["cell_id"].astype(str))
    _write_csv(train.frame, train_output)
    _write_csv(test.frame, test_output)

    positive_count = train.report.class_counts.get("1", 0)
    train_validation = train.report.to_dict()
    train_validation["source"] = _portable_path(train_path)
    test_validation = test.report.to_dict()
    test_validation["source"] = _portable_path(test_path)
    report = {
        "schema_version": 1,
        "dataset": "Kaggle Anomaly Detection in 4G Cellular Networks",
        "source": {
            "competition": "anomaly-detection-in-4g-cellular-networks",
            "train": _artifact(train_path),
            "test": _artifact(test_path),
        },
        "validation": {
            "train": train_validation,
            "test": test_validation,
            "feature_columns": train_features,
            "label_column": LTE_LABEL_COLUMN,
            "shared_cells": len(train_cells & test_cells),
            "train_only_cells": sorted(train_cells - test_cells),
            "test_only_cells": sorted(test_cells - train_cells),
            "positive_rate": positive_count / train.report.rows,
        },
        "outputs": {
            "train": _portable_path(train_output),
            "test": _portable_path(test_output),
        },
        "limitations": [
            "Labels describe unusual cellular behavior, not confirmed physical link failures.",
            "Only time of day is available, so full chronological order cannot be reconstructed.",
            "Duplicate observations are retained and reported; leakage-safe handling belongs to modeling.",
        ],
    }
    _write_json(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=LTE_TRAIN_RAW_PATH)
    parser.add_argument("--test", type=Path, default=LTE_TEST_RAW_PATH)
    parser.add_argument("--train-output", type=Path, default=LTE_TRAIN_PROCESSED_PATH)
    parser.add_argument("--test-output", type=Path, default=LTE_TEST_PROCESSED_PATH)
    parser.add_argument("--report", type=Path, default=PHASE1_REPORT_PATH)
    args = parser.parse_args()
    report = prepare_lte_data(
        args.train,
        args.test,
        args.train_output,
        args.test_output,
        args.report,
    )
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
