"""
model.py — RandomForest-based link failure prediction.
Exposes: train_model, load_model, predict_link, predict_graph_edges
"""
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from settings import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    MODEL_FEATURES,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TARGET_COLUMN,
)

FEATURES = list(MODEL_FEATURES)
MODEL_FILE = MODEL_PATH
SCALER_FILE = SCALER_PATH


def _validate_training_data(df: pd.DataFrame) -> None:
    required = FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")
    if len(df) < 10:
        raise ValueError("Training data must contain at least 10 rows")
    if df[required].isnull().any().any():
        raise ValueError("Training data contains missing values")
    labels = set(df[TARGET_COLUMN].unique())
    if labels != {0, 1}:
        raise ValueError("Training target must contain both binary classes 0 and 1")
    if (df[TARGET_COLUMN].value_counts() < 2).any():
        raise ValueError("Each training target class must contain at least two rows")


def _save_pickle(value, path: Union[str, Path]) -> None:
    """Atomically replace an artifact so interrupted writes do not corrupt it."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")
    try:
        with temporary_path.open("wb") as artifact_file:
            pickle.dump(value, artifact_file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(artifact_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_model(
    df: pd.DataFrame,
    *,
    model_path: Union[str, Path] = MODEL_FILE,
    scaler_path: Union[str, Path] = SCALER_FILE,
    persist: bool = True,
    random_state: int = RANDOM_SEED,
):
    """Train on df, return (model, scaler, metrics_dict)."""
    _validate_training_data(df)
    X = df[FEATURES].values
    y = df[TARGET_COLUMN].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5,
                                 class_weight="balanced", random_state=random_state, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
        "cm":        confusion_matrix(y_te, y_pred).tolist(),
        "feature_importance": dict(zip(FEATURES, clf.feature_importances_.round(4).tolist())),
        "report":    classification_report(y_te, y_pred),
    }
    if persist:
        _save_pickle(clf, model_path)
        _save_pickle(scaler, scaler_path)
    return clf, scaler, metrics


def load_model(
    model_path: Union[str, Path] = MODEL_FILE,
    scaler_path: Union[str, Path] = SCALER_FILE,
):
    """Return (model, scaler) if saved files exist, else (None, None)."""
    model_artifact = Path(model_path)
    scaler_artifact = Path(scaler_path)
    if model_artifact.exists() and scaler_artifact.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                with model_artifact.open("rb") as f: clf = pickle.load(f)
                with scaler_artifact.open("rb") as f: scaler = pickle.load(f)
            return clf, scaler
        except (
            AttributeError,
            EOFError,
            ImportError,
            IndexError,
            InconsistentVersionWarning,
            pickle.UnpicklingError,
        ):
            return None, None
    return None, None


def predict_link(model, scaler, attrs: dict) -> dict:
    """
    Predict failure probability for a single link.
    attrs keys: bandwidth, latency, packet_loss, traffic_load
    Returns: {failure_prob, risk_label}
    """
    if model is None:
        return {"failure_prob": 0.0, "risk_label": "Unknown"}
    row = np.array([[
        attrs.get("bandwidth",    300),
        attrs.get("latency",       20),
        attrs.get("packet_loss",    1),
        attrs.get("traffic_load",  50),
    ]])
    prob = float(model.predict_proba(scaler.transform(row))[0][1])
    if prob < LOW_RISK_THRESHOLD:   risk = "Low"
    elif prob < HIGH_RISK_THRESHOLD: risk = "Medium"
    else:             risk = "High"
    return {"failure_prob": round(prob, 4), "risk_label": risk}


def predict_graph_edges(model, scaler, G):
    """Update every edge in G with failure_prob and risk_label in-place."""
    if model is None:
        return
    for u, v, data in G.edges(data=True):
        result = predict_link(model, scaler, data)
        G[u][v].pop("routing_risk", None)
        G[u][v].pop("telemetry_cell", None)
        G[u][v]["failure_prob"] = result["failure_prob"]
        G[u][v]["risk_label"]   = result["risk_label"]
        G[u][v]["risk_source"]  = "synthetic_failure"
