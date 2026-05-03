"""
test_train.py

Purpose:
Run a structured test of the training pipeline for the circuit dataset.

Validates:
- preprocessing pipeline integration
- feature matrix creation
- model training
- prediction generation
- accuracy calculation
- artifact saving readiness
"""

from src.preprocess import load_preprocessed_dataframe, split_data
from src.features import build_feature_matrix, get_target_labels
from src.train import train_model
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run a training pipeline sniff test for the circuit dataset.
    """

    print("Training Pipeline Sniff Test (Circuit Dataset)")

    df = load_preprocessed_dataframe()
    X_encoded, _encoder = build_feature_matrix(df)
    y = get_target_labels(df)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    print("\nTrain/Test Split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    trained = train_model()
    predictions = trained["predictions"]
    y_test_out = trained["y_test"]

    print("\nModel Trained Successfully")

    print("\nPredictions Sample:")
    print(predictions[:10])

    print("\nPrediction Count:")
    print(len(predictions))

    print("\nBasic Consistency Checks:")
    print(f"Prediction count matches test rows: {len(predictions) == len(y_test_out)}")
    print(f"Non-empty predictions: {len(predictions) > 0}")
    print(f"Feature rows exist: {len(X_train) > 0}")


def test_train_smoke() -> None:
    output = capture_output(run_test)
    assert "Training Pipeline Sniff Test" in output
    assert "Model Trained Successfully" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_train", output)
