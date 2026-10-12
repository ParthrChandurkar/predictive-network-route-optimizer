"""Train, persist, load, and run the LTE degradation-risk model."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from modeling.lte_features import MODEL_FEATURES, build_lte_features
from modeling.lte_pipeline import build_lte_risk_pipeline
from settings import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    LTE_MODEL_PATH,
    LTE_TRAIN_PROCESSED_PATH,
    PHASE2_REPORT_PATH,
    PHASE3_REPORT_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from telemetry.schemas import LTE_LABEL_COLUMN, TelemetryValidationError


def _atomic_pickle(value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as artifact:
            pickle.dump(value, artifact, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_lte_risk_model(
    frame: pd.DataFrame,
    model_path: Union[str, Path] = LTE_MODEL_PATH,
    metadata_path: Union[str, Path] = PHASE3_REPORT_PATH,
):
    """Fit the validated baseline on all labeled rows and persist it locally."""
    if LTE_LABEL_COLUMN not in frame:
        raise TelemetryValidationError(f"Training requires {LTE_LABEL_COLUMN}")
    target = frame[LTE_LABEL_COLUMN].astype(int)
    if set(target.unique()) != {0, 1}:
        raise TelemetryValidationError("Training requires both unusual-behavior classes")

    features = build_lte_features(frame)
    pipeline = build_lte_risk_pipeline().fit(features, target)
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    _atomic_pickle(pipeline, model_path)
    metadata = {
        "phase": "phase-3-deployable-lte-risk-model",
        "artifact": {
            "path": _portable_path(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        },
        "training_rows": len(frame),
        "cells": int(frame["cell_id"].nunique()),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in target.value_counts().sort_index().items()
        },
        "features": list(MODEL_FEATURES),
        "risk_semantics": "Score estimates unusual LTE behavior; it is not a physical-link failure probability.",
        "evaluation_report": _portable_path(PHASE2_REPORT_PATH),
        "reproducibility": {
            "random_seed": RANDOM_SEED,
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    _atomic_json(metadata, metadata_path)
    return pipeline, metadata


def load_lte_risk_model(model_path: Union[str, Path] = LTE_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            with path.open("rb") as artifact:
                model = pickle.load(artifact)
        if not hasattr(model, "predict_proba"):
            return None
        return model
    except (
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        InconsistentVersionWarning,
        pickle.UnpicklingError,
    ):
        return None


def predict_lte_risk(model, frame: pd.DataFrame) -> pd.DataFrame:
    if model is None:
        raise ValueError("LTE risk model is not loaded")
    probabilities = model.predict_proba(build_lte_features(frame))[:, 1]
    result = frame.loc[:, ["time_of_day", "cell_id"]].copy()
    result["degradation_score"] = probabilities
    result["risk_label"] = pd.cut(
        probabilities,
        bins=[-float("inf"), LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, float("inf")],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LTE_TRAIN_PROCESSED_PATH)
    parser.add_argument("--model", type=Path, default=LTE_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=PHASE3_REPORT_PATH)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    _, metadata = train_lte_risk_model(frame, args.model, args.metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
