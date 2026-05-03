"""
This module prepares preprocessed aricraft data for machine learning.

It assumes preprocess.py has already:
- selected relevant columns
- cleaned numeric values
- removed rows with missing critical values
- engineered derived features
- simplified labels into RoleClass

Steps used:
- Select model input features
- Build features matrix x
- Build Target vector y
- Filter classification classes

Written by: Michael Garcia
Date: 03/02/2026
For: CSC373
License: MIT
"""

from __future__ import annotations

import pandas as pd
from typing import List, Tuple


MODEL_FEATURES = [
    "Crew",
    "Length",
    "Wingspan",
    "Height",
    "WingArea",
    "MaxSpeed",
    "AspectRatio",
    "SizeIndex",
    "NumberBuilt_log",
]


def select_model_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Select only the columns intended for model input.

    Why:
    This enforces a clear boundary between available 
    data and the final set used by the model.

    Args:
        df: Input dataframe from preprocess.py
        feature_columns: List of column names to keep

    Returns:
        pd.DataFrame: DataFrame containing only the selected feature columns
    """
    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    return df[feature_columns].copy()


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_columns: str = "RoleClass",
    drop_other: bool = True,
) -> Tuple[pd.DataFrame, pd.series]:
    """
    Build the model-ready feature matrix x and target vector y.

    Why:
    Machine Learning models expect a clear separation between:
    - X: input features
    - Y: target labels

    Args:
        df: Input dataframe from preprocess.py
        feature_columns: List of columns to use for x
        target_columns: Name of the target lebel column
        drop_other: If True, remove rows labeled as 'Other'

    Returns:
        Tuple[pd.Dataframe, pd.Series]:
            X = selected feature datafrome
            Y = target label series
    """
    df =df.copy()

    if target_columns not in df.columns:
        raise ValueError(f"Target Columns not found: {target_columns}")
    
    if drop_other:
        df =df[df[target_columns]!= "Other"].copy()

    # Drop rows with missing values in model features
    df = df.dropna(subset=feature_columns).copy()

    X = select_model_features(df, feature_columns)
    Y = df[target_columns].copy()
    
    return (X, Y)