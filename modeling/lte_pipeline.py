"""Shared scikit-learn pipeline construction for LTE models."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling.lte_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from settings import RANDOM_SEED


def build_lte_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
            (
                "cell",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def random_forest_estimator() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def build_lte_risk_pipeline() -> Pipeline:
    return build_lte_pipeline(random_forest_estimator())
