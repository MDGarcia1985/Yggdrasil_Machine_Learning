"""
test_features.py

Purpose:
Run a structured test of feature selection and feature matrix creation.
"""

from src.preprocess import (
    load_data,
    select_columns,
    clean_numeric_data,
    clean_missing_rows,
    engineer_features,
    simplify_labels,
    REQUIRED_COLUMNS,
)

from src.features import (
    MODEL_FEATURES,
    select_model_features,
    build_feature_matrix,
)

from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run a feature-preparation sniff test.
    """
    df = load_data("data/processed/aircraft_cleaned.csv")
    df = select_columns(df, REQUIRED_COLUMNS)
    df = clean_numeric_data(df)
    df = clean_missing_rows(df)
    df = engineer_features(df)
    df = simplify_labels(df)

    X, y = build_feature_matrix(df, MODEL_FEATURES, target_columns="RoleClass", drop_other=True)

    print("Feature Preparation Sniff Test")
    print("\nSelected features:")
    print(X.head())

    print("\nTarget labels:")
    print(y.head())

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFeature matrix shape:")
    print(X.shape)

    print("\nTarget vector shape:")
    print(y.shape)

    print("\nTarget counts:")
    print(y.value_counts())

    print("\nMissing values in X:")
    print(X.isna().sum())

    print(f"\nRows after feature filtering: {len(X)}")


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_features", output)