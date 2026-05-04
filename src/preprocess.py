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
    Normalize one or more CSV paths into Path objects.
    """
    if isinstance(file_path, (str, Path)):
        return [Path(file_path)]

    return [Path(path) for path in file_path]


def load_data(file_path: str | Path | Iterable[str | Path] = DEFAULT_DATA_PATHS) -> pd.DataFrame:
    """
    Load one or more circuit CSV datasets.

    Args:
        file_path:
            Path or iterable of paths to raw circuit CSV files.

    Returns:
        Raw pandas DataFrame. Multiple files are concatenated in order.
    """
    paths = _normalize_data_paths(file_path)
    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        frames.append(pd.read_csv(path))

    if not frames:
        raise ValueError("At least one data file path is required.")

    return pd.concat(frames, ignore_index=True)


def select_columns(df: pd.DataFrame, required_columns: List[str] | None = None) -> pd.DataFrame:
    """
    Select only the required columns for the MVP circuit ML pipeline.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[required_columns].copy()


def clean_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text/categorical fields used by the circuit model.
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
    Convert component_value to numeric.

    Missing or non-numeric values are set to 0 for MVP. The text meaning is
    preserved separately in component_value_type and component_type.
    """
    df = df.copy()

    cleaned = df["component_value"].astype(str)
    cleaned = cleaned.str.replace(",", "", regex=False).str.strip()
    df["component_value"] = pd.to_numeric(cleaned, errors="coerce").fillna(0)

    return df


def clean_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows missing critical connection identity.

    For the MVP, every usable row must have:
    - circuit_name
    - component_kind
    - component_type
    - pin_name
    - net_name
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

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived circuit features from cleaned connection rows.

    These features are intentionally simple. They help early tabular models
    learn power, ground, signal, passive, and control patterns before the
    project moves to graph neural networks.
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

    # Small circuit-level context features repeated on each row.
    df["circuit_component_type_count"] = df.groupby("circuit_name")["component_type"].transform("nunique")
    df["circuit_net_count"] = df.groupby("circuit_name")["net_name"].transform("nunique")
    df["ref_des_connection_count"] = df.groupby(["circuit_name", "ref_des"])["pin_name"].transform("count")

    return df


def add_next_component_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the MVP label: next_component_type.

    The label is generated within each circuit so rows do not leak from one
    circuit into the next.
    """
    df = df.copy()

    df[TARGET_COLUMN] = df.groupby("circuit_name")["component_type"].shift(-1)
    df = df.dropna(subset=[TARGET_COLUMN])

    return df.reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on an existing DataFrame.
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
    Load and fully preprocess a circuit dataset.
    """
    df = load_data(file_path)
    return preprocess_dataframe(df)


def split_features_and_labels(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split preprocessed data into X features and y labels.
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
    Split features and labels into training and testing sets.
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
    Full loading/preprocessing/splitting pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_columns
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
