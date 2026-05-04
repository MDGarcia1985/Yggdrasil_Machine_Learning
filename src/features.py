"""
features.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Purpose:
Create feature summaries and encoded feature matrices for the Yggdrasil
circuit ML pipeline.

Design:
Use the circuit dataset as a graph-like tabular representation where each row
means:

(component, pin) -> net

Workflow:
- summarize circuit dataset
- extract sample rows for UI/tests
- select raw + engineered model features
- one-hot encode categorical fields
- pass numeric/boolean fields through
- return target labels
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_FEATURES = [
    "circuit_name",
    "description",
    "ref_des",
    "component_kind",
    "component_type",
    "component_value_type",
    "pin_name",
    "net_name",
]

NUMERIC_FEATURES = [
    "component_value",
    "circuit_component_type_count",
    "circuit_net_count",
    "ref_des_connection_count",
]

BOOLEAN_FEATURES = [
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

TARGET_COLUMN = "next_component_type"


def get_available_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Purpose:
        Intersect the canonical feature lists with whatever columns exist in `df`.

    Design:
        Three buckets (categorical / numeric / boolean) mirror how
        `build_feature_matrix` assembles `X`.

    Workflow:
        Always called inside `build_feature_matrix` before encoding.

    Data handoff:
        Inputs: any circuit DataFrame (partial columns allowed for summaries).
        Outputs: dict of column names; consumed only by `build_feature_matrix`.
    """
    return {
        "categorical": [col for col in CATEGORICAL_FEATURES if col in df.columns],
        "numeric": [col for col in NUMERIC_FEATURES if col in df.columns],
        "boolean": [col for col in BOOLEAN_FEATURES if col in df.columns],
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Purpose:
        Produce JSON-serializable stats for UI dashboards and smoke tests.

    Design:
        Counts uniques on key identity columns; converts numpy/pandas scalars
        to plain Python ints/strings for stable serialization.

    Workflow:
        Optional diagnostics; not on the hot training path.

    Data handoff:
        Inputs: minimally the required summary columns on `df`.
        Outputs: dict for Streamlit/tests; no writes to disk.
    """
    required_columns = [
        "circuit_name",
        "component_kind",
        "component_type",
        "pin_name",
        "net_name",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required summary columns: {missing_columns}")

    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": [str(col) for col in df.columns],
        "num_circuits": int(df["circuit_name"].nunique()),
        "num_component_kinds": int(df["component_kind"].nunique()),
        "num_component_types": int(df["component_type"].nunique()),
        "num_nets": int(df["net_name"].nunique()),
        "circuit_names": sorted(df["circuit_name"].unique().tolist()),
        "component_kinds": sorted(df["component_kind"].unique().tolist()),
        "component_types": sorted(df["component_type"].unique().tolist()),
        "net_names": sorted(df["net_name"].unique().tolist()),
    }


def get_sample_rows(df: pd.DataFrame, n: int = 5) -> List[Dict[str, object]]:
    """
    Purpose:
        Surface a small, human-readable slice of the dataset for previews.

    Design:
        Inserts a 1-based `index` column for display; uses `orient="records"`
        so each row becomes one dict (common JSON shape).

    Workflow:
        UI tables / quick REPL inspection.

    Data handoff:
        Inputs: `df`, row cap `n`.
        Outputs: `List[dict]`; not fed back into training automatically.
    """
    sample = df.head(n).copy()
    sample.insert(0, "index", range(1, len(sample) + 1))

    return sample.to_dict(orient="records")


def build_feature_matrix(
    df: pd.DataFrame,
    fit_encoder: bool = True,
    encoder: OneHotEncoder | None = None,
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Purpose:
        Turn preprocessed tabular rows into a numeric-only design matrix `X`.

    Design:
        sklearn `OneHotEncoder` on categoricals; numeric coerced with
        `errors="coerce"`; booleans mapped to 0/1. Concatenation order is
        numeric, boolean, then one-hot blocks (stable for `feature_columns`).

    Workflow:
        Training: `fit_encoder=True` then persist encoder. Inference: reuse
        the fitted encoder with `fit_encoder=False`.

    Data handoff:
        Inputs: `df` from `preprocess` (or `predict.prepare_prediction_dataframe`).
        Outputs: `(X_encoded, encoder)`; `X_encoded` aligns with
        `train.split_data` / `predict.align_feature_columns`; `encoder` saved
        beside the model in `train.save_artifacts`.
    """
    available = get_available_feature_columns(df)

    categorical_features = available["categorical"]
    numeric_features = available["numeric"]
    boolean_features = available["boolean"]

    missing_required = [col for col in CATEGORICAL_FEATURES if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required categorical features: {missing_required}")

    categorical_data = df[categorical_features].fillna("none").astype(str)

    numeric_data = pd.DataFrame(index=df.index)
    if numeric_features:
        numeric_data = df[numeric_features].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)

    boolean_data = pd.DataFrame(index=df.index)
    if boolean_features:
        boolean_data = df[boolean_features].fillna(False).astype(int)

    if fit_encoder:
        # handle_unknown="ignore": unseen categories at inference become all-zero.
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_array = encoder.fit_transform(categorical_data)
    else:
        if encoder is None:
            raise ValueError("encoder is required when fit_encoder=False")
        encoded_array = encoder.transform(categorical_data)

    encoded_columns = encoder.get_feature_names_out(categorical_features)

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        index=df.index,
    )

    # reset_index(drop=True): concat aligns row-by-row after possible reindex drift.
    X_encoded = pd.concat(
        [
            numeric_data.reset_index(drop=True),
            boolean_data.reset_index(drop=True),
            encoded_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return X_encoded, encoder


def get_target_labels(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """
    Purpose:
        Extract the supervised label vector aligned with `build_feature_matrix` rows.

    Design:
        Fills missing labels with `"end"` so sklearn always sees strings.

    Workflow:
        Called by `train.train_model` immediately after building `X_encoded`.

    Data handoff:
        Inputs: labeled `df` from preprocessing (`next_component_type` present).
        Outputs: `y` Series passed to `preprocess.split_data` / `model.fit`.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column '{target_column}'. "
            "Run preprocessing label creation first."
        )

    return df[target_column].fillna("end").astype(str)


def get_circuit_component_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Aggregate how many connection rows each component type contributes per circuit.

    Design:
        Uses `groupby(...).size()` (row counts), not distinct ref_des counts.

    Workflow:
        Analytics / UI histograms; orthogonal to model `X`.

    Data handoff:
        Inputs: preprocessed `df` with `circuit_name` and `component_type`.
        Outputs: small summary DataFrame for reporting consumers.
    """
    return (
        df.groupby(["circuit_name", "component_type"])
        .size()
        .reset_index(name="connection_row_count")
        .sort_values(["circuit_name", "component_type"])
    )


def get_net_connection_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Count pin connections attached to each net within each circuit.

    Design:
        Sorted output for deterministic display in tests/UI.

    Workflow:
        Reporting only.

    Data handoff:
        Inputs: `df` with `circuit_name`, `net_name`.
        Outputs: per-(circuit, net) connection totals.
    """
    return (
        df.groupby(["circuit_name", "net_name"])
        .size()
        .reset_index(name="connection_count")
        .sort_values(["circuit_name", "net_name"])
    )


def get_component_kind_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Count rows per `component_kind` within each circuit for inventory views.

    Design:
        Mirrors `get_circuit_component_counts` but groups on `component_kind`.

    Workflow:
        Reporting / UI side panels.

    Data handoff:
        Inputs: `df` with `circuit_name`, `component_kind`.
        Outputs: sorted aggregate DataFrame.
    """
    return (
        df.groupby(["circuit_name", "component_kind"])
        .size()
        .reset_index(name="connection_row_count")
        .sort_values(["circuit_name", "component_kind"])
    )
