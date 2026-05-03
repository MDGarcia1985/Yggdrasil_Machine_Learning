"""
test_preprocess.py

Purpose:
Run a structured test of the circuit preprocessing pipeline.

Validates:
- CSV load
- cleaning
- label creation (next_component_type)
- train/test split
"""

from src.preprocess import (
    load_preprocessed_dataframe,
    load_and_split_data,
)
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run preprocessing pipeline sniff test for circuit dataset.
    """

    print("Preprocess Sniff Test (Circuit Dataset)")

    # Load full processed dataframe
    df = load_preprocessed_dataframe()

    print("\nDataset Summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column Names: {list(df.columns)}")

    print("\nUnique Circuits:")
    print(df["circuit_name"].unique())

    print("\nSample Rows:")
    print(df.head())

    # Split pipeline
    X_train, X_test, y_train, y_test, feature_columns = load_and_split_data()

    print("\nTrain/Test Split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nFeature Columns:")
    print(feature_columns)

    print("\nBasic Consistency Checks:")
    print(f"Training + Testing rows > 0: {len(X_train) + len(X_test) > 0}")
    print(f"Labels match feature rows: {len(X_train) == len(y_train)}")
    print(f"Test labels match: {len(X_test) == len(y_test)}")
    print(f"Target column exists: {'next_component_type' in df.columns}")


def test_preprocess_smoke() -> None:
    output = capture_output(run_test)
    assert "Preprocess Sniff Test" in output
    assert "next_component_type" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_preprocess", output)
