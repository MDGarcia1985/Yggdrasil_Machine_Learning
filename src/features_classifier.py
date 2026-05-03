"""
features_classifier.py

Purpose:
Feature preparation helpers for optional comparison classifiers.

Design:
This module keeps the older project's feature-selection pattern, but updates it
for the Yggdrasil circuit dataset.

Main features.py is the preferred RandomForest pipeline.
This file is useful when testing alternate classifiers such as Logistic
Regression, SVM, or KNN.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


MODEL_FEATURES = [
    "component_value",
    "circuit_component_type_count",
    "circuit_net_count",
    "ref_des_connection_count",
    "is_power_net",
    "is_ground_net",
    "is_output_net",
    "is_control_net",
    "is_timing_net",
    "is_input_pin",
    "is_output_pin",
    "is_power_pin",
    "is_ground_pin",
    "is_resistor",
    "is_capacitor",
    "is_diode",
    "is_switch",
    "is_timer",
    "is_schmitt",
    "is_primitive",
    "is_structure",
    "is_block",
    "is_part",
    "is_source",
]

DEFAULT_TARGET_COLUMN = "next_component_type"


def select_model_features(
    df: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Select only the columns intended for model input.

    Why this exists:
        Comparison models in this module are numeric/boolean only, so this
        helper enforces the shared input contract.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    X = df[feature_columns].copy()

    for col in X.columns:
        # sklearn estimators in comparison mode expect numeric dtypes.
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str] | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_end: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the model-ready feature matrix X and target vector y.

    This is numeric/boolean-only and is mainly for comparison classifiers.
    The main features.py module should be used for full one-hot categorical
    training.

    Workflow role:
        Used when evaluating alternate estimators that do not benefit from the
        full categorical one-hot pipeline.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if drop_end:
        df = df[df[target_column].fillna("end") != "end"].copy()

    df = df.dropna(subset=feature_columns).copy()

    X = select_model_features(df, feature_columns)
    y = df[target_column].astype(str).copy()

    return X, y


def get_feature_names(feature_columns: List[str] | None = None) -> List[str]:
    """
    Return the configured comparison-model feature list.
    """
    return list(feature_columns or MODEL_FEATURES)
