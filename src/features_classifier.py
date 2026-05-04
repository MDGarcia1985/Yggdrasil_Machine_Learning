"""
features_classifier.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

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
    Purpose:
        Slice a DataFrame down to numeric/boolean columns used by alt models.

    Design:
        Coerces bool columns to int so sklearn SVM/KNN pipelines stay dtype-safe.

    Workflow:
        Internal helper for `features_classifier.build_feature_matrix`.

    Data handoff:
        Inputs: preprocessed `df` containing `MODEL_FEATURES` subset.
        Outputs: `X` only—no labels.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    X = df[feature_columns].copy()

    for col in X.columns:
        if X[col].dtype == bool:
            # Many sklearn estimators reject raw bool dtype; int8/64 is fine.
            X[col] = X[col].astype(int)

    return X


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str] | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    drop_end: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Purpose:
        Produce `(X, y)` for experiments with sklearn models that skip one-hot text.

    Design:
        Optionally drops terminal `"end"` labels to avoid training on explicit
        stop tokens; drops rows with NaNs in selected features.

    Workflow:
        Notebook benchmarks comparing RF vs KNN vs logistic on the same counts.

    Data handoff:
        Inputs: fully preprocessed labeled `df` (engineered numerics present).
        Outputs: tuple consumed directly by custom training scripts—not by
        `train.train_model`, which uses `src.features.build_feature_matrix`.
    """
    if feature_columns is None:
        feature_columns = MODEL_FEATURES

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if drop_end:
        # "end" marks sequence tail from preprocessing; often excluded from classifiers.
        df = df[df[target_column].fillna("end") != "end"].copy()

    df = df.dropna(subset=feature_columns).copy()

    X = select_model_features(df, feature_columns)
    y = df[target_column].astype(str).copy()

    return X, y


def get_feature_names(feature_columns: List[str] | None = None) -> List[str]:
    """
    Purpose:
        Expose the default numeric/boolean feature name list for logging/UI.

    Design:
        Returns a shallow copy as a list to avoid accidental mutation of defaults.

    Workflow:
        Parameter wiring in comparison notebooks.

    Data handoff:
        Inputs: optional override list; outputs column names aligned with `X`.
    """
    return list(feature_columns or MODEL_FEATURES)
