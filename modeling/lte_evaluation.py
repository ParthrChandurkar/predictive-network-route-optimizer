"""Leakage-aware baseline evaluation for the real LTE telemetry dataset."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from modeling.lte_features import (
    MODEL_FEATURES,
    build_lte_features,
    feature_duplicate_groups,
)
from modeling.lte_pipeline import build_lte_pipeline, random_forest_estimator
from settings import (
    LTE_TRAIN_PROCESSED_PATH,
    PHASE1_REPORT_PATH,
    PHASE2_REPORT_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from telemetry.schemas import LTE_LABEL_COLUMN, TelemetryValidationError


def candidate_models() -> Dict[str, Pipeline]:
    models = {
        "majority_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        ),
        "random_forest": random_forest_estimator(),
    }
    return {
        name: build_lte_pipeline(model)
        for name, model in models.items()
    }


def _metrics(y_true: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
    }


def _evaluate_model(
    estimator: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> dict:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    predictions = np.zeros(len(target), dtype=np.int8)
    probabilities = np.zeros(len(target), dtype=float)
    folds = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError("Evaluation group leakage detected")
        fitted = clone(estimator).fit(features.iloc[train_index], target.iloc[train_index])
        fold_prediction = fitted.predict(features.iloc[validation_index])
        fold_probability = fitted.predict_proba(features.iloc[validation_index])[:, 1]
        predictions[validation_index] = fold_prediction
        probabilities[validation_index] = fold_probability
        fold_metrics = _metrics(
            target.iloc[validation_index].to_numpy(), fold_prediction, fold_probability
        )
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
                **fold_metrics,
            }
        )

    aggregate = _metrics(target.to_numpy(), predictions, probabilities)
    aggregate["confusion_matrix"] = confusion_matrix(target, predictions).tolist()
    return {"aggregate": aggregate, "folds": folds}


def _top_random_forest_features(
    estimator: Pipeline, features: pd.DataFrame, target: pd.Series, limit: int = 15
) -> list:
    fitted = clone(estimator).fit(features, target)
    names = fitted.named_steps["preprocess"].get_feature_names_out()
    importance = fitted.named_steps["model"].feature_importances_
    ranking = sorted(zip(names, importance), key=lambda item: item[1], reverse=True)
    return [
        {"feature": name.replace("numeric__", "").replace("cell__", ""), "importance": value}
        for name, value in ranking[:limit]
    ]


def evaluate_lte_baselines(frame: pd.DataFrame) -> Dict[str, Any]:
    if LTE_LABEL_COLUMN not in frame:
        raise TelemetryValidationError(f"Evaluation requires {LTE_LABEL_COLUMN}")
    features = build_lte_features(frame)
    target = frame[LTE_LABEL_COLUMN].astype(int).reset_index(drop=True)
    features = features.reset_index(drop=True)

    protocols: Iterable[Tuple[str, pd.Series, str]] = (
        (
            "duplicate_safe",
            feature_duplicate_groups(frame).reset_index(drop=True),
            "Identical source observations are kept in the same fold.",
        ),
        (
            "unseen_cell",
            frame["cell_id"].astype(str).reset_index(drop=True),
            "Every validation cell is absent from that fold's training data.",
        ),
    )
    models = candidate_models()
    results = {}
    for protocol_name, groups, description in protocols:
        results[protocol_name] = {
            "description": description,
            "groups": int(groups.nunique()),
            "models": {
                name: _evaluate_model(estimator, features, target, groups)
                for name, estimator in models.items()
            },
        }

    return {
        "phase": "phase-2-leakage-aware-lte-baselines",
        "dataset": {
            "rows": len(frame),
            "cells": int(frame["cell_id"].nunique()),
            "positive_rate": float(target.mean()),
        },
        "features": {
            "count_before_one_hot_encoding": len(MODEL_FEATURES),
            "columns": list(MODEL_FEATURES),
            "design": "Row-local features only; no lag or rolling features without trustworthy dates.",
        },
        "protocols": results,
        "random_forest_descriptive_importance": _top_random_forest_features(
            models["random_forest"], features, target
        ),
        "limitations": [
            "The source has no calendar date, so chronological forecasting cannot be evaluated.",
            "Feature importance is descriptive and does not establish causality.",
            "Unusual-behavior labels are degradation proxies, not physical-link failure labels.",
        ],
    }


def _write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LTE_TRAIN_PROCESSED_PATH)
    parser.add_argument("--report", type=Path, default=PHASE2_REPORT_PATH)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    report = evaluate_lte_baselines(frame)
    report["reproducibility"] = {
        "random_seed": RANDOM_SEED,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "phase1_report": PHASE1_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    _write_report(report, args.report)
    summary = {
        protocol: {
            model: values["aggregate"]
            for model, values in result["models"].items()
        }
        for protocol, result in report["protocols"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
