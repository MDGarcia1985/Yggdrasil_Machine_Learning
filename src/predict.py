"""
predict.py

Purpose:
Prediction helpers for the Yggdrasil circuit classification model.

Design:
Merge the older single-record prediction style with the current circuit
dataset pipeline.

Workflow:
- load saved model artifacts
- validate input feature dictionary/DataFrame
- preprocess/engineer input features
- encode with saved encoder
- align columns
- predict next component type
- return probabilities and confidence when supported
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from src.preprocess import preprocess_dataframe, clean_categorical_data, clean_numeric_data, engineer_features
from src.features import build_feature_matrix


MODEL_DIR = Path("Models")
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"


RAW_REQUIRED_COLUMNS = [
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


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
) -> Dict[str, Any]:
    """
    Load model artifacts saved by train.py.

    Returns:
        Dictionary containing model, encoder, and feature_columns.
    """
    model_path = Path(model_path)
    encoder_path = Path(encoder_path)
    feature_columns_path = Path(feature_columns_path)

    missing = [
        str(path)
        for path in [model_path, encoder_path, feature_columns_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s). Train the model first. "
            f"Missing: {missing}"
        )

    return {
        "model": joblib.load(model_path),
        "encoder": joblib.load(encoder_path),
        "feature_columns": joblib.load(feature_columns_path),
    }


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare unlabeled prediction data.

    Unlike preprocess_dataframe(), this does NOT create next_component_type,
    because prediction input does not have a known next label.
    """
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    prepared = df[RAW_REQUIRED_COLUMNS].copy()
    prepared = clean_categorical_data(prepared)
    prepared = clean_numeric_data(prepared)
    prepared = engineer_features(prepared)

    return prepared


def align_feature_columns(
    X_encoded: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Align encoded prediction features to the training feature order.
    """
    aligned = X_encoded.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0

    return aligned[feature_columns]


def predict_record(
    feature_values: Dict[str, Any],
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
) -> Dict[str, Any]:
    """
    Predict next_component_type from a single feature dictionary.

    Args:
        feature_values:
            Single circuit connection row as a dictionary.

    Returns:
        Dictionary with prediction, probabilities, confidence, and feature columns.
    """
    artifact = load_artifacts(model_path, encoder_path, feature_columns_path)

    model = artifact["model"]
    encoder = artifact["encoder"]
    feature_columns = artifact["feature_columns"]

    X_input = pd.DataFrame([feature_values])
    prepared = prepare_prediction_dataframe(X_input)

    X_encoded, _ = build_feature_matrix(
        prepared,
        fit_encoder=False,
        encoder=encoder,
    )
    X_encoded = align_feature_columns(X_encoded, feature_columns)

    prediction = model.predict(X_encoded)[0]

    probabilities = None
    confidence = None

    if hasattr(model, "predict_proba"):
        class_probs = model.predict_proba(X_encoded)[0]
        probabilities = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, class_probs)
        }
        confidence = max(probabilities.values()) if probabilities else None

    return {
        "prediction": str(prediction),
        "probabilities": probabilities,
        "confidence": confidence,
        "feature_columns": feature_columns,
        "input_row": feature_values,
    }


def predict_dataframe(
    df: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
) -> pd.DataFrame:
    """
    Predict next_component_type for every row in a DataFrame.
    """
    artifact = load_artifacts(model_path, encoder_path, feature_columns_path)

    model = artifact["model"]
    encoder = artifact["encoder"]
    feature_columns = artifact["feature_columns"]

    prepared = prepare_prediction_dataframe(df)

    X_encoded, _ = build_feature_matrix(
        prepared,
        fit_encoder=False,
        encoder=encoder,
    )
    X_encoded = align_feature_columns(X_encoded, feature_columns)

    predictions = model.predict(X_encoded)

    result = prepared.copy()
    result["predicted_next_component_type"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_encoded)
        result["prediction_confidence"] = probabilities.max(axis=1)
    else:
        result["prediction_confidence"] = None

    return result


def predict_model(model, X_test):
    """
    Predict labels from an already-loaded model and test feature matrix.
    """
    return model.predict(X_test)


def evaluate_predictions(predictions, y_test) -> List[Dict[str, object]]:
    """
    Format predictions against known labels for testing/reporting.
    """
    results = []

    for i, (predicted, actual) in enumerate(zip(predictions, y_test)):
        result = "Correct" if predicted == actual else "Incorrect"

        results.append(
            {
                "index": i + 1,
                "predicted": str(predicted),
                "actual": str(actual),
                "result": result,
            }
        )

    return results


def model_accuracy(predictions, y_test) -> float:
    """
    Return simple accuracy score.
    """
    total_predictions = int(len(y_test))

    if total_predictions == 0:
        return 0.0

    correct_predictions = int(sum(predictions == y_test))

    return correct_predictions / total_predictions


def main():
    """
    Manual smoke test. Requires trained artifacts.
    """
    example = {
        "circuit_name": "button_debounce",
        "description": "RC debounce with schmitt trigger",
        "ref_des": "R1",
        "component_kind": "primitive",
        "component_type": "resistor",
        "component_value": 10000,
        "component_value_type": "ohm",
        "pin_name": "A",
        "net_name": "NET_VIN",
    }

    print(predict_record(example))


if __name__ == "__main__":
    main()
