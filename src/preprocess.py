"""
preprocess.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Purpose:
Preprocessing utilities for the Yggdrasil circuit classification dataset.

Design:
Keep the same staged pipeline pattern from earlier projects:
- load tabular data
- select expected columns
- clean values
- remove incomplete rows
- engineer derived circuit features
- create model labels

Workflow:
CSV / SQL query export -> cleaned DataFrame -> engineered features -> labels -> train/test split
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATA_PATH = Path("Data/raw/circuits_sample.csv")
SECONDARY_DATA_PATH = Path("Data/raw/circuit_sample_two.csv")
DEFAULT_DATA_PATHS = [
    DEFAULT_DATA_PATH,
    SECONDARY_DATA_PATH,
]

REQUIRED_COLUMNS = [
    "circuit_name",
    "description",
    "ref_des",
    "component_kind",
    "component_type",
    "component_value",
    "component_value_type",
    "pin_name",
    "net_name",
]

TARGET_COLUMN = "next_component_type"


def _normalize_data_paths(file_path: str | Path | Iterable[str | Path]) -> List[Path]:
    """
    Purpose:
        Accept a single path or many paths and normalize them to `pathlib.Path`.

    Design:
        Uses a type check so a bare string is treated as one file, not an
        iterable of characters (which would be wrong for `Path`).

    Workflow:
        Called internally by `load_data` before reading CSVs.

    Data handoff:
        Inputs: user-supplied path(s) from CLI, tests, or `load_data`.
        Outputs: `List[Path]` consumed only by `load_data`'s read loop.
    """
    # str is iterable in Python; branch first so we never iterate characters.
    if isinstance(file_path, (str, Path)):
        return [Path(file_path)]

    return [Path(path) for path in file_path]


def load_data(file_path: str | Path | Iterable[str | Path] = DEFAULT_DATA_PATHS) -> pd.DataFrame:
    """
    Purpose:
        Read raw circuit tabular exports from disk into one stacked DataFrame.

    Design:
        Multiple CSVs are concatenated vertically with `ignore_index=True` so
        row indices stay unique; column union must already align across files.

    Workflow:
        First stage of the pipeline before `select_columns` / cleaning.

    Data handoff:
        Inputs: filesystem paths (defaults in `DEFAULT_DATA_PATHS`).
        Outputs: raw `pd.DataFrame` passed to `preprocess_dataframe` or
        `load_preprocessed_dataframe`.
    """
    paths = _normalize_data_paths(file_path)
    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        frames.append(pd.read_csv(path))

    if not frames:
        raise ValueError("At least one data file path is required.")

    # Stack files: same columns expected; new global row index 0..N-1.
    return pd.concat(frames, ignore_index=True)


def select_columns(df: pd.DataFrame, required_columns: List[str] | None = None) -> pd.DataFrame:
    """
    Purpose:
        Drop extra spreadsheet columns so downstream steps see a fixed schema.

    Design:
        Fails fast if the export is missing expected headers instead of
        silently producing NaN-only columns.

    Workflow:
        Typically the first transform after `load_data`.

    Data handoff:
        Inputs: raw `df` from `load_data`.
        Outputs: narrowed copy; fed into `clean_categorical_data` inside
        `preprocess_dataframe`.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # .copy() avoids SettingWithCopy warnings if callers chain in-place edits.
    return df[required_columns].copy()


def clean_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Normalize string fields so encoders and heuristics see stable tokens.

    Design:
        Missing values become the literal `"none"`; nets/pins use consistent
        casing (nets upper, circuit names lower) to match regex features later.

    Workflow:
        Runs after column selection, before numeric cleaning.

    Data handoff:
        Inputs: narrowed tabular rows from `select_columns` (or prediction prep
        in `predict.prepare_prediction_dataframe`).
        Outputs: same columns, cleaned strings; consumed by `clean_numeric_data`
        and `engineer_features`.
    """
    df = df.copy()

    text_columns = [
        "circuit_name",
        "description",
        "ref_des",
        "component_kind",
        "component_type",
        "component_value_type",
        "pin_name",
        "net_name",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("none").astype(str).str.strip()

    df["circuit_name"] = df["circuit_name"].str.lower()
    df["component_kind"] = df["component_kind"].str.lower()
    df["component_type"] = df["component_type"].str.lower()
    df["component_value_type"] = df["component_value_type"].str.lower()
    df["pin_name"] = df["pin_name"].str.upper()
    df["net_name"] = df["net_name"].str.upper()

    return df


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Make `component_value` a real number column for tree/linear models.

    Design:
        Coerce unreadable values to NaN then fill with 0; units stay in
        `component_value_type` / `component_type` so we do not lose semantics.

    Workflow:
        After categorical cleaning, before row filtering.

    Data handoff:
        Inputs: DataFrame with stringly-typed values from CSV.
        Outputs: numeric `component_value`; used by `engineer_features` and
        `features.NUMERIC_FEATURES`.
    """
    df = df.copy()

    cleaned = df["component_value"].astype(str)
    cleaned = cleaned.str.replace(",", "", regex=False).str.strip()
    # errors="coerce" turns "10k" style junk into NaN, then we impute for MVP.
    df["component_value"] = pd.to_numeric(cleaned, errors="coerce").fillna(0)

    return df


def clean_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Drop rows that cannot represent a valid (component, pin) → net fact.

    Design:
        Iterative filtering per column keeps logic obvious; rejects empty
        strings and placeholder `"none"` from `clean_categorical_data`.

    Workflow:
        After numeric + categorical cleaning, before feature engineering.

    Data handoff:
        Inputs: partially cleaned `df`.
        Outputs: dense connection rows for `engineer_features` and labels.
    """
    df = df.copy()

    critical_columns = [
        "circuit_name",
        "component_kind",
        "component_type",
        "pin_name",
        "net_name",
    ]

    for col in critical_columns:
        df = df[df[col].astype(str).str.strip() != ""]
        df = df[df[col].astype(str).str.lower() != "none"]

    # reset_index: row labels become 0..n-1 after drops (helps groupby/shift).
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Add boolean and count features so tabular models see circuit structure.

    Design:
        Regex flags on net/pin names plus `groupby().transform()` for circuit-
        level statistics; keeps each row self-contained for sklearn.

    Workflow:
        After cleaning; same function used in training and live prediction.

    Data handoff:
        Inputs: cleaned rows (no label column required).
        Outputs: original columns plus engineered booleans/counts; consumed by
        `features.build_feature_matrix` and `features_classifier` helpers.
    """
    df = df.copy()

    net = df["net_name"].astype(str).str.upper()
    pin = df["pin_name"].astype(str).str.upper()
    component_type = df["component_type"].astype(str).str.lower()
    component_kind = df["component_kind"].astype(str).str.lower()

    df["is_power_net"] = net.str.contains("VCC|VIN|VDD|POWER|SUPPLY", regex=True)
    df["is_ground_net"] = net.str.contains("GND|GROUND|0V", regex=True)
    df["is_output_net"] = net.str.contains("OUT|OUTPUT", regex=True)
    df["is_control_net"] = net.str.contains("CTRL|CONTROL|RESET|ENABLE|TRIG|THRESH|DISCH", regex=True)
    df["is_timing_net"] = net.str.contains("TIMING|THRESH|TRIG|DISCH|CONTROL", regex=True)

    df["is_input_pin"] = pin.isin(["IN", "INPUT", "A", "B", "TRIG", "THRESH", "BASE", "GATE", "POS"])
    df["is_output_pin"] = pin.isin(["OUT", "OUTPUT", "Y", "Q", "Q_BAR", "COLLECTOR", "DRAIN"])
    df["is_power_pin"] = pin.isin(["VCC", "VDD", "VIN", "POS", "+"])
    df["is_ground_pin"] = pin.isin(["GND", "NEG", "-", "VSS"])

    df["is_resistor"] = component_type.eq("resistor")
    df["is_capacitor"] = component_type.eq("capacitor")
    df["is_diode"] = component_type.eq("diode")
    df["is_switch"] = component_type.eq("switch")
    df["is_timer"] = component_type.isin(["555_timer", "timer"])
    df["is_schmitt"] = component_type.str.contains("schmitt", regex=False)

    df["is_primitive"] = component_kind.eq("primitive")
    df["is_structure"] = component_kind.eq("structure")
    df["is_block"] = component_kind.eq("block")
    df["is_part"] = component_kind.eq("part")
    df["is_source"] = component_kind.eq("source")

    # transform repeats one scalar per row in the group (broadcasts to each row).
    df["circuit_component_type_count"] = df.groupby("circuit_name")["component_type"].transform("nunique")
    df["circuit_net_count"] = df.groupby("circuit_name")["net_name"].transform("nunique")
    df["ref_des_connection_count"] = df.groupby(["circuit_name", "ref_des"])["pin_name"].transform("count")

    return df


def add_next_component_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Define supervised target `next_component_type` for sequence-style rows.

    Design:
        `shift(-1)` inside `groupby("circuit_name")` peeks at the next row's
        component type without crossing circuit boundaries.

    Workflow:
        Last preprocessing step before train/test split and encoding.

    Data handoff:
        Inputs: engineered feature table from `engineer_features`.
        Outputs: adds `TARGET_COLUMN`; `features.get_target_labels` and
        `split_features_and_labels` peel it off as `y`.
    """
    df = df.copy()

    # Negative shift: "next" row in file order within the same circuit_name group.
    df[TARGET_COLUMN] = df.groupby("circuit_name")["component_type"].shift(-1)
    df = df.dropna(subset=[TARGET_COLUMN])

    return df.reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
        Apply the full ordered cleaning + feature + label pipeline in one call.

    Design:
        Pure function chain (no I/O); order is fixed so tests pin behavior.

    Workflow:
        `load_preprocessed_dataframe` wraps `load_data` + this function.

    Data handoff:
        Inputs: raw-ish DataFrame (e.g. from `load_data` or tests).
        Outputs: ML-ready labeled table for `train.train_model` or split helpers.
    """
    df = select_columns(df)
    df = clean_categorical_data(df)
    df = clean_numeric_data(df)
    df = clean_missing_rows(df)
    df = engineer_features(df)
    df = add_next_component_label(df)

    return df


def load_preprocessed_dataframe(
    file_path: str | Path | Iterable[str | Path] = DEFAULT_DATA_PATHS,
) -> pd.DataFrame:
    """
    Purpose:
        Convenience entry: disk → cleaned, engineered, labeled DataFrame.

    Design:
        Thin composition of `load_data` and `preprocess_dataframe`.

    Workflow:
        Primary data source for `train.train_model` and exploratory analysis.

    Data handoff:
        Inputs: CSV path(s).
        Outputs: same as `preprocess_dataframe`; feeds `features` and `train`.
    """
    df = load_data(file_path)
    return preprocess_dataframe(df)


def split_features_and_labels(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Purpose:
        Separate supervised inputs from the target column before encoding.

    Design:
        Drops only the label column from `X`; all other columns stay for
        `features.build_feature_matrix` to filter/encode.

    Workflow:
        Used by `load_and_split_data`; alternative path if you encode manually.

    Data handoff:
        Inputs: labeled DataFrame from preprocessing.
        Outputs: `(X_raw, y)` where `X_raw` still includes categoricals;
        `train` pipeline typically calls `build_feature_matrix` on full `df`
        instead—this helper supports experiments and tests.
    """
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(str)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Purpose:
        Create reproducible train/test partitions for sklearn estimators.

    Design:
        Wraps `sklearn.model_selection.train_test_split` with `stratify=None`
        so rare classes are not dropped by stratification constraints on small
        synthetic samples.

    Workflow:
        Called from `train.train_model` (post-encoding) and `load_and_split_data`.

    Data handoff:
        Inputs: feature matrix `X` and aligned labels `y` (same index length).
        Outputs: `X_train, X_test, y_train, y_test` tuple for `.fit`/`.predict`.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )


def load_and_split_data(
    file_path: str | Path | Iterable[str | Path] = DEFAULT_DATA_PATHS,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Purpose:
        End-to-end dataset prep returning raw feature column names before OHE.

    Design:
        Splits on unencoded `X` so `feature_columns` lists original schema
        names (not expanded one-hot names).

    Workflow:
        Useful for notebooks/tests; production training often uses
        `train.train_model` which encodes first.

    Data handoff:
        Inputs: CSV path(s).
        Outputs: `X_train, X_test, y_train, y_test, feature_columns` where
        `feature_columns` is `list(X.columns)` prior to one-hot—callers that
        need encoded matrices should use `features.build_feature_matrix`.
    """
    df = load_preprocessed_dataframe(file_path)
    X, y = split_features_and_labels(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, list(X.columns)
