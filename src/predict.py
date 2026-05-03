"""
predict.py

Purpose:
Load a saved trained model and use it to classify a single aircraft
feature vector as Fighter or Bomber.

This module assumes the input data is already preprocessed and
feature-engineered to match the model's expected feature columns.

Workflow:
- Load saved model artifact
- Build a one-row dataframe from input features
- Reorder columns to match training
- Scale the input using the saved scaler
- Predict class and class probabilities

Written by: Michael Garcia
Date: 03/23/2026
For: CSC373
License: MIT
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any

from src.train import load_model


def predict_aircraft(feature_values: Dict[str, Any], model_path: str = "models/model.pkl") -> Dict[str, Any]:
    """
    Predict aircraft class from a single feature dictionary.

    Args:
        feature_values: Dictionary containing model input features
        model_path: Path to saved model artifact

    Returns:
        Dictionary containing:
        - prediction
        - probabilities
        - feature_columns
    """
    artifact = load_model(model_path)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    # Create one-row dataframe from input dictionary
    X_input = pd.DataFrame([feature_values])

    # Ensure required columns exist and are in the correct order
    missing = [col for col in feature_columns if col not in X_input.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_input = X_input[feature_columns]

    # Scale input using training scaler
    X_scaled = scaler.transform(X_input)

    # Predict class
    prediction = model.predict(X_scaled)[0]

    # Predict probabilities if supported
    probabilities = None
    # hasattr is a built-in Python function that checks if an object has a given attribute.
    if hasattr(model, "predict_proba"):
        class_probs = model.predict_proba(X_scaled)[0]
        
        probabilities = {
        label: float(prob)
        for label, prob in zip(model.classes_, class_probs)
    }

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()) if probabilities else None,
        "feature_columns": feature_columns,
    }