"""
train.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

Purpose:
Train the baseline classifier for the Yggdrasil circuit dataset.

Pipeline:
- load preprocessed circuit dataframe
- build feature matrix
- get target labels
- split train/test
- train RandomForestClassifier
- evaluate model
- report feature importance
- save model + encoder + feature columns

Primary MVP target:
- next_component_type
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocess import DEFAULT_DATA_PATHS, load_preprocessed_dataframe, split_data
from src.features import build_feature_matrix, get_target_labels
from src.feature_importance import get_model_importance, print_top_importance


MODEL_DIR = Path("Models")
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
TRAINING_REPORT_PATH = MODEL_DIR / "training_report.pkl"

def train_model(
    csv_path: str | Path | Iterable[str | Path] = DEFAULT_DATA_PATHS,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Dict[str, Any]:
    """
    Purpose:
        Fit the baseline RandomForest on encoded circuit rows and report quality.

    Design:
        End-to-end orchestration: preprocess → one-hot → random holdout split
        (via `split_data`, `stratify=None`) → balanced forest → metrics + importances.

    Workflow:
        Primary training entry; `main()` calls this then `save_artifacts`.

    Data handoff:
        Inputs: raw CSV path(s) (`preprocess.load_preprocessed_dataframe`).
        Outputs: dict with `model`, `encoder`, `feature_columns`, holdout
        predictions, and report payloads for `save_artifacts` and tests.
    """
    df = load_preprocessed_dataframe(csv_path)

    X_encoded, encoder = build_feature_matrix(df, fit_encoder=True)
    y = get_target_labels(df)

    X_train, X_test, y_train, y_test = split_data(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores for tree fitting
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # zero_division=0: silence warnings when a class has no predicted support.
    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions)

    importance = get_model_importance(
        features=X_train.columns,
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print("Training complete")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_text)

    print_top_importance(importance, top_n=15)

    return {
        "model": model,
        "encoder": encoder,
        "feature_columns": list(X_encoded.columns),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy,
        "classification_report": report_text,
        "confusion_matrix": matrix,
        "feature_importance": importance,
    }


def save_artifacts(training_result: Dict[str, Any]) -> None:
    """
    Purpose:
        Persist everything `predict.load_artifacts` needs to reproduce training inputs.

    Design:
        Uses `joblib` for sklearn objects; separate pickle files keep concerns
        clear (model vs encoder vs column order).

    Workflow:
        Run after `train_model` in CLI training; CI may skip if read-only.

    Data handoff:
        Inputs: dict returned by `train_model` (must include model, encoder,
        feature_columns, and report subfields).
        Outputs: files under `Models/` consumed by `predict` and `train.load_model`.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result["model"], MODEL_PATH)
    joblib.dump(training_result["encoder"], ENCODER_PATH)
    joblib.dump(training_result["feature_columns"], FEATURE_COLUMNS_PATH)

    report_payload = {
        "accuracy": training_result["accuracy"],
        "classification_report": training_result["classification_report"],
        "confusion_matrix": training_result["confusion_matrix"],
        "feature_importance": training_result["feature_importance"],
    }

    joblib.dump(report_payload, TRAINING_REPORT_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved encoder to: {ENCODER_PATH}")
    print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH}")
    print(f"Saved training report to: {TRAINING_REPORT_PATH}")


def load_model(
    model_path: str | Path = MODEL_PATH,
    encoder_path: str | Path = ENCODER_PATH,
    feature_columns_path: str | Path = FEATURE_COLUMNS_PATH,
) -> Dict[str, Any]:
    """
    Purpose:
        Reload model + encoder + column list from disk without running training.

    Design:
        Thin `joblib.load` trio; paths default to the same constants as save.

    Workflow:
        Preferred loader for app code that already validated paths; stricter
        checks live in `predict.load_artifacts`.

    Data handoff:
        Inputs: filesystem paths.
        Outputs: dict keys `model`, `encoder`, `feature_columns` for callers.
    """
    return {
        "model": joblib.load(model_path),
        "encoder": joblib.load(encoder_path),
        "feature_columns": joblib.load(feature_columns_path),
    }


def main() -> None:
    """
    Purpose:
        CLI entry: train on default CSVs and write artifacts to `Models/`.

    Design:
        No arguments; relies on module-level defaults for paths and hyperparams.

    Workflow:
        `python -m src.train` or equivalent.

    Data handoff:
        Reads default data paths; writes pickles via `save_artifacts`.
    """
    training_result = train_model()
    save_artifacts(training_result)


if __name__ == "__main__":
    main()
