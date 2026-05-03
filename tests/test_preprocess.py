"""
test_preprocess.py

Purpose:
Run a structured test of the preprocessing pipeline and log the results.

This test currently acts as a development-stage validation script,
not a formal unit test suite.
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

from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run a preprocessing pipeline sniff test.

    This validates:
    - data loading
    - required column selection
    - numeric cleaning
    - missing-row cleanup
    - engineered feature creation
    - label simplification
    """
    df = load_data("data/processed/aircraft_cleaned.csv")
    df = select_columns(df, REQUIRED_COLUMNS)
    df = clean_numeric_data(df)
    df = clean_missing_rows(df)
    df = engineer_features(df)
    df = simplify_labels(df)

    print("Preprocess Sniff Test")
    print(df[[
        "Name",
        "PrimaryRole",
        "RoleClass",
        "NumberBuilt",
        "AspectRatio",
        "SizeIndex",
        "NumberBuilt_log"
    ]].head())

    print("\nMissing Values Summary:")
    print(df.isna().sum())

    print(f"\nRow count after cleanup: {len(df)}")

    print("\nRoleClass counts:")
    print(df["RoleClass"].value_counts())

    print("\nRole mapping sniff test:")
    print(df[["PrimaryRole", "RoleClass"]].head(10))


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_preprocess", output)