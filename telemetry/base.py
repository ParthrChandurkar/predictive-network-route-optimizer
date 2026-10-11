"""Shared contracts for pluggable telemetry sources."""
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from telemetry.schemas import ValidationReport


@dataclass(frozen=True)
class TelemetryDataset:
    frame: pd.DataFrame
    report: ValidationReport


class TelemetryLoader(Protocol):
    def __call__(self, source, require_label: bool = True) -> TelemetryDataset:
        """Load a source into a canonical frame plus validation report."""

