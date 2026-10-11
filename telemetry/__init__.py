"""Telemetry ingestion interfaces for the rebuild."""

from telemetry.schemas import TelemetryValidationError, ValidationReport


def load_lte_dataset(source, require_label=True):
    """Load the LTE adapter lazily so its module can also run as a CLI."""
    from telemetry.lte_kaggle import load_lte_dataset as load

    return load(source, require_label=require_label)


__all__ = [
    "TelemetryValidationError",
    "ValidationReport",
    "load_lte_dataset",
]

