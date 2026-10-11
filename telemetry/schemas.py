"""Canonical schema definitions shared by telemetry data sources."""
from dataclasses import asdict, dataclass
from typing import Dict, List


LTE_FEATURE_COLUMNS = (
    "time_of_day",
    "minute_of_day",
    "cell_id",
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
LTE_LABEL_COLUMN = "is_unusual"


class TelemetryValidationError(ValueError):
    """Raised when source telemetry cannot satisfy its declared schema."""


@dataclass(frozen=True)
class ValidationReport:
    source: str
    rows: int
    cells: int
    labeled: bool
    duplicate_rows: int
    missing_values: Dict[str, int]
    class_counts: Dict[str, int]
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

