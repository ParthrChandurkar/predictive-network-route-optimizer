"""Adapter for Kaggle's Anomaly Detection in 4G Cellular Networks data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import IO, Optional, Union

import pandas as pd

from telemetry.base import TelemetryDataset
from telemetry.schemas import (
    LTE_FEATURE_COLUMNS,
    LTE_LABEL_COLUMN,
    TelemetryValidationError,
    ValidationReport,
)


CsvSource = Union[str, Path, IO[bytes], IO[str]]

RAW_TO_CANONICAL = {
    "Time": "time_of_day",
    "CellName": "cell_id",
    "PRBUsageUL": "prb_usage_ul",
    "PRBUsageDL": "prb_usage_dl",
    "meanThr_DL": "mean_throughput_dl",
    "meanThr_UL": "mean_throughput_ul",
    "maxThr_DL": "max_throughput_dl",
    "maxThr_UL": "max_throughput_ul",
    "meanUE_DL": "mean_ue_dl",
    "meanUE_UL": "mean_ue_ul",
    "maxUE_DL": "max_ue_dl",
    "maxUE_UL": "max_ue_ul",
    "maxUE_UL+DL": "max_ue_total",
    "Unusual": LTE_LABEL_COLUMN,
}

COLUMN_ALIASES = {
    "CellName1": "CellName",
    "Label": "Unusual",
}

NUMERIC_COLUMNS = tuple(
    column
    for column in LTE_FEATURE_COLUMNS
    if column not in {"time_of_day", "minute_of_day", "cell_id"}
)


def _source_name(source: CsvSource) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return str(getattr(source, "name", "uploaded.csv"))


def _read_csv(source: CsvSource) -> pd.DataFrame:
    try:
        if not isinstance(source, (str, Path)) and hasattr(source, "seek"):
            source.seek(0)
        # Kaggle distributes the LTE files with a ``.csv`` name but uses a
        # semicolon delimiter. Automatic delimiter detection also keeps the
        # adapter compatible with ordinary comma-separated uploads.
        return pd.read_csv(source, sep=None, engine="python")
    except FileNotFoundError as exc:
        raise TelemetryValidationError(f"Telemetry file not found: {source}") from exc
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise TelemetryValidationError(f"Could not read telemetry CSV: {exc}") from exc


def _normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    for alias, canonical_raw_name in COLUMN_ALIASES.items():
        if alias in normalized.columns and canonical_raw_name not in normalized.columns:
            normalized = normalized.rename(columns={alias: canonical_raw_name})
    return normalized


def _validate_schema(frame: pd.DataFrame, require_label: bool) -> bool:
    required = set(RAW_TO_CANONICAL) - {"Unusual"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TelemetryValidationError(
            "LTE telemetry is missing required columns: " + ", ".join(missing)
        )
    labeled = "Unusual" in frame.columns
    if require_label and not labeled:
        raise TelemetryValidationError("LTE training data must include Unusual or Label")
    return labeled


def _normalize_time(frame: pd.DataFrame) -> pd.DataFrame:
    raw_time = frame["time_of_day"].astype("string").str.strip()
    if raw_time.isna().any() or raw_time.eq("").any():
        raise TelemetryValidationError("Time cannot be empty")
    parsed = pd.to_datetime(raw_time, format="%H:%M", errors="coerce")
    invalid = raw_time.notna() & parsed.isna()
    if invalid.any():
        examples = ", ".join(raw_time[invalid].drop_duplicates().head(3).tolist())
        raise TelemetryValidationError(
            f"Time must use HH:MM format; invalid value(s): {examples}"
        )
    result = frame.copy()
    result["time_of_day"] = parsed.dt.strftime("%H:%M")
    result.insert(1, "minute_of_day", parsed.dt.hour * 60 + parsed.dt.minute)
    return result


def _normalize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        raw = result[column]
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            examples = ", ".join(raw[invalid].astype(str).drop_duplicates().head(3).tolist())
            raise TelemetryValidationError(
                f"Column {column} contains non-numeric value(s): {examples}"
            )
        result[column] = converted
    return result


def _normalize_label(frame: pd.DataFrame, labeled: bool) -> pd.DataFrame:
    if not labeled:
        return frame
    result = frame.copy()
    raw = result[LTE_LABEL_COLUMN]
    label = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & label.isna()
    if invalid.any() or label.isna().any() or not set(label.unique()).issubset({0, 1}):
        raise TelemetryValidationError("Unusual/Label must contain only binary values 0 and 1")
    result[LTE_LABEL_COLUMN] = label.astype("int8")
    return result


def load_lte_dataset(source: CsvSource, require_label: bool = True) -> TelemetryDataset:
    """Load, normalize, and describe a Kaggle LTE telemetry CSV.

    Missing numeric observations are retained and reported. Imputation belongs
    to the later feature pipeline so ingestion never silently invents values.
    """
    raw = _normalize_headers(_read_csv(source))
    if raw.empty:
        raise TelemetryValidationError("LTE telemetry file is empty")
    labeled = _validate_schema(raw, require_label=require_label)

    selected_raw_columns = list(RAW_TO_CANONICAL)
    if not labeled:
        selected_raw_columns.remove("Unusual")
    frame = raw[selected_raw_columns].rename(columns=RAW_TO_CANONICAL)

    if frame["cell_id"].isna().any() or frame["cell_id"].astype(str).str.strip().eq("").any():
        raise TelemetryValidationError("CellName cannot be empty")
    frame["cell_id"] = frame["cell_id"].astype("string").str.strip()
    frame = _normalize_time(frame)
    frame = _normalize_numeric(frame)
    frame = _normalize_label(frame, labeled=labeled)

    missing_values = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if count
    }
    warnings = [
        "The source exposes time of day but no calendar date; full chronological ordering cannot be proven."
    ]
    for column in ("prb_usage_ul", "prb_usage_dl"):
        out_of_range = int(((frame[column] < 0) | (frame[column] > 100)).sum())
        if out_of_range:
            warnings.append(f"{column} has {out_of_range} value(s) outside 0-100%.")
    if missing_values:
        warnings.append("Missing numeric values were retained for an explicit preprocessing decision.")

    class_counts = {}
    if labeled:
        class_counts = {
            str(int(label)): int(count)
            for label, count in frame[LTE_LABEL_COLUMN].value_counts().sort_index().items()
        }
    report = ValidationReport(
        source=_source_name(source),
        rows=len(frame),
        cells=int(frame["cell_id"].nunique()),
        labeled=labeled,
        duplicate_rows=int(frame.duplicated().sum()),
        missing_values=missing_values,
        class_counts=class_counts,
        warnings=warnings,
    )
    return TelemetryDataset(frame=frame, report=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the Kaggle LTE CSV")
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Allow a competition test file without labels",
    )
    args = parser.parse_args()
    dataset = load_lte_dataset(args.csv, require_label=not args.allow_unlabeled)
    print(json.dumps(dataset.report.to_dict(), indent=2))
    print(dataset.frame.head().to_string(index=False))


if __name__ == "__main__":
    main()

