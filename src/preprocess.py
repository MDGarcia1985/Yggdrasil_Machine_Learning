"""
This preprocesses the data after it has been organized
by parse_semicolon_data and before it is used by features.py.

Steps used:
- Load data
- Select Columns
- Clean numeric data
- Feature engineering
- Transform data
- Simplify labels
- Cleanup rows with missing critical values

The output should be clean numeric values with usable headers
for use in features, training, and classification.

Written by: Michael Garcia
Date: 03/23/2026
For: CSC373
License: MIT
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load cleaned CSV exported from Excel.

    This assumes parse_semicolon_data.ts has been run and:
    - rows have been reconstructed
    - semicolon-delimited values have been split into columns
    Args:
        file_path (str): Path to cleaned CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")


def select_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Select only the required columns for preprocessing.

    Why:
    Ensures consistency and prevents unexpected columns from affecting the pipeline.

    Args:
        df: Input dataframe.
        required_columns: List of column names to keep

    Returns:
        DataFrame with only the selected columns
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df[required_columns].copy()

REQUIRED_COLUMNS = [
    "Name",
    "PrimaryRole",
    "Number",
    "Crew",
    "Length",
    "Wingspan",
    "Height",
    "WingArea",
    "MaxSpeed",
]


def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric columns by converting to float and handling malformed values.

    Why:
    - Ensures all numeric data is in a consistent float format
    - Handles commas, whitespace, and simple text artifacts
    - Converts invalid values to NaN for later handling

    Note:
    Floats are used because some values (e.g., Length, Wingspan)
    are not whole numbers.

    Args:
        df: Input dataframe

    Returns:
        DataFrame with cleaned numeric columns
    """
    df = df.copy()

    numeric_columns = [
        "Crew",
        "Number",
        "Length",
        "Wingspan",
        "Height",
        "WingArea",
        "MaxSpeed",
    ]

    for col in numeric_columns:
        if col in df.columns:
            # Convert to string for safe processing
            cleaned = df[col].astype(str)

            # Remove commas and whitespace
            cleaned = cleaned.str.replace(",", "", regex=False).str.strip()

            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    # ------------------------
    # Create NumberBuilt
    # ------------------------
    if "Number" in df.columns:
        df["NumberBuilt"] = df["Number"]

    return df


def clean_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows missing critical values required for feature engineering.

    Why:
    Derived features such as AspectRatio and SizeIndex depend on valid
    geometric and performance data. Rows missing these values are not reliable
    for downstream modeling.

    Args:
        df: Input dataframe with numeric values already cleaned.

    Returns:
        pd.DataFrame: DataFrame with incomplete critical rows removed.
    """
    df = df.copy()

    # Define columns that must not be null for reliable feature engineering
    required_for_features = [
        "Length",
        "Wingspan",
        "Height",
        "WingArea",
        "MaxSpeed"
    ]

    # Drop rows where any critical column is null
    df = df.dropna(subset=required_for_features) # dropna is how the rows are dropped.
    df = df.reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from cleaned aircraft data.

    Why:
    Derived features can capture more meaningful relationships than
    raw dimensions alone. These engineered features help the model 
    better distinguish between aircraft classes.

    Features created:
    - AspectRatio: Wingspan squared divided by WingArea
    - SizeIndex: Product of Length, Wingspan, and Height
    - NumberBuilt_log: log-transformed production count

    Args:
        df: Input dataframe with cleaned numeric values.

    Returns:
        pd.DataFrame: DataFrame with engineered feature columns added.
    """
    df=df.copy()


    df["AspectRatio"] = (df["Wingspan"]**2) / df["WingArea"]
    df["SizeIndex"] = df["Length"] * df["Wingspan"]
    df["NumberBuilt_log"] = np.log1p(df["NumberBuilt"])

    return df


def simplify_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplify aircraft role labels into broader categories for classification.

    Why:
    The raw dataset contains many specific role names such as
    'Heavy Bomber', 'Dive Bomber', and 'Night Fighter'.
    These are useful historically, but for classification we want
    a smaller and more consistent label set.

    Mapping:
    - roles containing 'fighter' -> Fighter
    - roles containing 'bomber'  -> Bomber
    - everything else            -> Other

    Args:
        df: Input dataframe containing PrimaryRole.

    Returns:
        pd.DataFrame: Dataframe with a new RoleClass column.
    """
    df = df.copy()

    """
    Note: .fillna("") handles cases where PrimaryRole might be NaN,
          preventing errors during string operations.

    Originally, I used "roles = df["PrimaryRole"].str.lower().str.strip()"
    But this was not ideal as it created the need to use isna() from Pandas to
    reject missing values such as Not a Number (NaN), None, and a second "if"
    statement to check for empty strings.
    """
    roles = df["PrimaryRole"].fillna("").str.lower().str.strip()

    # Define role mappings for simplification
    def map_role(role: str) -> str:

        if role == "":
            return "Other"

        if "fighter" in role:
            return "Fighter"
        if "bomber" in role:
            return "Bomber"

        return "Other"

    df["RoleClass"] = roles.apply(map_role)

    return df