"""
predict.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

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
    Purpose:
        Load pickles produced by training, with explicit missing-file errors.

    Design:
        Pre-checks all three paths so the user gets one combined error list
        instead of failing on the first missing file.

    Workflow:
        Every prediction API (`predict_record`, `predict_dataframe`) starts here.

    Data handoff:
        Inputs: paths on disk (defaults under `Models/`).
        Outputs: dict with `model`, `encoder`, `feature_columns` for encoding
        and alignment steps.
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
    Purpose:
        Run the same cleaning + engineering steps as training, without labels.

    Design:
        Skips `add_next_component_label` because inference has no ground-truth
        "next" type to supervise against.

    Workflow:
        Internal bridge between raw UI/API rows and `build_feature_matrix`.

    Data handoff:
        Inputs: DataFrame with `RAW_REQUIRED_COLUMNS` populated.
        Outputs: engineered table; passed to `features.build_feature_matrix` with
        the saved `OneHotEncoder`.
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
    Purpose:
        Guarantee prediction matrices match the exact column order from training.

    Design:
        Missing columns (e.g. unseen categorical levels dropped by encoder) are
        zero-filled; final reindex drops any extra columns not in training.

    Workflow:
        After `build_feature_matrix`, before `model.predict`.

    Data handoff:
        Inputs: freshly encoded `X_encoded`, training's `feature_columns` list.
        Outputs: DataFrame ready for sklearn `predict` / `predict_proba`.
    """
    aligned = X_encoded.copy()

    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0

    # Column order must match training or tree models read wrong features.
    return aligned[feature_columns]


def predict_record(
    feature_values: Dict[str, Any],
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
) -> Dict[str, Any]:
    """
    Purpose:
        Score one connection row (dict) and return label + optional probabilities.

    Design:
        Wraps single-row DataFrame construction so the batching code path stays
        identical to `predict_dataframe`.

    Workflow:
        UI quick-preview, unit tests, REPL demos.

    Data handoff:
        Inputs: `feature_values` matching raw CSV schema; artifact paths.
        Outputs: dict with `prediction`, `probabilities`, `confidence`, echo of
        `feature_columns` and raw `input_row` for auditing.
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
        # classes_: label order matches columns of predict_proba output.
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
    Purpose:
        Batch scoring for many rows while preserving engineered feature columns.

    Design:
        Appends `predicted_next_component_type` and `prediction_confidence`
        (max class probability) to the prepared (engineered) frame, not the raw input.

    Workflow:
        Offline evaluation on unlabeled exports or UI batch runs.

    Data handoff:
        Inputs: DataFrame with raw required columns (see `RAW_REQUIRED_COLUMNS`).
        Outputs: copy of prepared rows plus prediction columns for CSV/UI.
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
    Purpose:
        Thin wrapper so tests and notebooks can call `.predict` via one import.

    Design:
        No alignment or encoding—caller must supply already-encoded `X_test`.

    Workflow:
        Used after manual matrix construction in tests.

    Data handoff:
        Inputs: fitted sklearn estimator, feature matrix.
        Outputs: ndarray/Series of predicted class labels.
    """
    return model.predict(X_test)


def evaluate_predictions(predictions, y_test) -> List[Dict[str, object]]:
    """
    Purpose:
        Build a per-row correctness table for human-readable test output.

    Design:
        Zips iterables positionally—callers must ensure equal lengths.

    Workflow:
        Test harnesses comparing `model.predict` outputs to held-out labels.

    Data handoff:
        Inputs: parallel sequences `predictions`, `y_test`.
        Outputs: list of dict rows with 1-based `index` for display.
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
    Purpose:
        Compute plain accuracy without importing sklearn metrics.

    Design:
        Uses elementwise equality with sum; empty inputs return 0.0.

    Workflow:
        Lightweight checks in tests or UI summaries.

    Data handoff:
        Inputs: parallel `predictions` and `y_test` (often numpy arrays).
        Outputs: scalar float in [0, 1].
    """
    total_predictions = int(len(y_test))

    if total_predictions == 0:
        return 0.0

    correct_predictions = int(sum(predictions == y_test))

    return correct_predictions / total_predictions


def main():
    """
    Purpose:
        Quick CLI smoke test verifying artifacts load and a row scores.

    Design:
        Hard-coded example dict mimics one training CSV row shape.

    Workflow:
        Run module as script after training locally.

    Data handoff:
        Reads default `Models/` pickles; prints `predict_record` dict to stdout.
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
