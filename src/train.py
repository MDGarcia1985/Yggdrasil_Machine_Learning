"""
This module trains and evaluates a machine learning model
using the feature matrix and target labels prepared upstream.

It assumes:
- preprocess.py has cleaned and enriched the dataset
- features.py has produced X (features) and y (labels)

Steps used:
- Split data into training and test sets
- Scale feature values
- Train classification model
- Evaluate model performance

Written by: Michael Garcia
Date: 03/23/2026
For: CSC373
License: MIT
"""

from __future__ import annotations

import pandas as pd
import joblib
import os
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2, # 20% of data reserved for testinge:
    random_state: int = 42, # for reproducibility
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and labels into training and test sets.

    Why:
    The model must be evaluated on unseen data to estimate
    how well it generalizes.

    Args:
        X: Feature matrix
        y: Target labels
        test_size: Fraction of data reserved for testing
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale training and test features using StandardScaler.

    Why:
    Standardization places features on a comparable scale,
    which helps many models train more effectively.

    Important:
    The scaler is fit only on training data, then applied
    to both training and test data to avoid leakage.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train) -> LogisticRegression:
    """
    Train a logistic regression classifier.

    Why:
    Logistic regression is simple, interpretable, and
    appropriate for a first-pass structured classification model.

    Args:
        X_train: Scaled training feature matrix
        y_train: Training target labels

    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained classifier
        X_test: Scaled test feature matrix
        y_test: Test target labels

    Returns:
        Dictionary containing evaluation results
    """
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return results

def save_model(model, scaler, feature_columns, path: str = "models/model.pkl") -> None:
    """
    Save trained model, scaler, and feature metadata to disk.

    Why:
    The model alone is not sufficient for predictions. We must also
    preserve the scaler and the exact feature ordering used during training.

    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_columns: List of feature column names
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }

    joblib.dump(payload, path)


def load_model(path: str = "models/model.pkl"):
    """
    Load trained model artifacts from disk.

    Args:
        path: Path to saved model artifact

    Returns:
        Dictionary containing:
        - model
        - scaler
        - feature_columns
    """
    return joblib.load(path)