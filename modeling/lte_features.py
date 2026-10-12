"""Leakage-safe feature engineering for canonical LTE telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from telemetry.schemas import LTE_LABEL_COLUMN, TelemetryValidationError


CATEGORICAL_FEATURES = ("cell_id",)
BASE_NUMERIC_FEATURES = (
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
DERIVED_NUMERIC_FEATURES = (
    "time_sin",
    "time_cos",
    "prb_usage_total",
    "prb_usage_imbalance",
    "mean_throughput_total",
    "max_throughput_total",
    "mean_ue_total",
    "throughput_headroom_dl",
    "throughput_headroom_ul",
)
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_lte_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-local LTE features without using labels or adjacent rows.

    The Kaggle data has time of day but no date and its rows are shuffled. All
    derived features therefore depend only on the current observation.
    """
    required = {"minute_of_day", *CATEGORICAL_FEATURES, *BASE_NUMERIC_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TelemetryValidationError(
            "Cannot build LTE features; missing columns: " + ", ".join(missing)
        )

    features = frame.loc[:, list(CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES)].copy()
    angle = 2 * np.pi * frame["minute_of_day"] / (24 * 60)
    features["time_sin"] = np.sin(angle)
    features["time_cos"] = np.cos(angle)
    features["prb_usage_total"] = frame["prb_usage_ul"] + frame["prb_usage_dl"]
    features["prb_usage_imbalance"] = (
        frame["prb_usage_dl"] - frame["prb_usage_ul"]
    ).abs()
    features["mean_throughput_total"] = (
        frame["mean_throughput_dl"] + frame["mean_throughput_ul"]
    )
    features["max_throughput_total"] = (
        frame["max_throughput_dl"] + frame["max_throughput_ul"]
    )
    features["mean_ue_total"] = frame["mean_ue_dl"] + frame["mean_ue_ul"]
    features["throughput_headroom_dl"] = (
        frame["max_throughput_dl"] - frame["mean_throughput_dl"]
    )
    features["throughput_headroom_ul"] = (
        frame["max_throughput_ul"] - frame["mean_throughput_ul"]
    )
    return features.loc[:, list(MODEL_FEATURES)]


def feature_duplicate_groups(frame: pd.DataFrame) -> pd.Series:
    """Return stable group IDs so identical observations stay in one fold."""
    source_columns = [column for column in frame if column != LTE_LABEL_COLUMN]
    return pd.util.hash_pandas_object(
        frame.loc[:, source_columns], index=False, categorize=True
    ).astype("uint64")
