"""
Smoke tests for raw and preprocessed data loading.

Purpose:
    Confirm the MVP sample CSV is readable and preprocessing returns labeled rows
    required by the training pipeline.

Workflow role:
    Early guardrail tests for the load -> preprocess stage.
"""

from src.preprocess import (
    DEFAULT_DATA_PATHS,
    DEFAULT_DATA_PATH,
    REQUIRED_COLUMNS,
    SECONDARY_DATA_PATH,
    TARGET_COLUMN,
    load_data,
    load_preprocessed_dataframe,
)
from tests.log_utils import run_logged_test


def test_sample_csv_loads_with_required_columns():
    """
    Verify raw sample data is present and includes required schema columns.
    """
    def check():
        raw_df = load_data(DEFAULT_DATA_PATH)

        assert not raw_df.empty
        assert set(REQUIRED_COLUMNS).issubset(raw_df.columns)
        return {
            "summary": "Sample CSV loads and includes required schema.",
            "output": {
                "path": str(DEFAULT_DATA_PATH),
                "row_count": len(raw_df),
                "column_count": len(raw_df.columns),
                "required_columns_present": set(REQUIRED_COLUMNS).issubset(raw_df.columns),
            },
        }

    run_logged_test("Raw sample CSV load and required column validation.", check)


def test_default_training_data_loads_all_configured_csvs():
    """
    Verify the default training data combines both configured circuit CSV files.
    """
    def check():
        primary_df = load_data(DEFAULT_DATA_PATH)
        secondary_df = load_data(SECONDARY_DATA_PATH)
        combined_df = load_data(DEFAULT_DATA_PATHS)

        assert not combined_df.empty
        assert len(combined_df) == len(primary_df) + len(secondary_df)
        assert set(REQUIRED_COLUMNS).issubset(combined_df.columns)
        return {
            "summary": "Default training data combines primary and secondary CSV files.",
            "output": {
                "configured_paths": [str(path) for path in DEFAULT_DATA_PATHS],
                "primary_rows": len(primary_df),
                "secondary_rows": len(secondary_df),
                "combined_rows": len(combined_df),
                "combined_matches_sum": len(combined_df) == len(primary_df) + len(secondary_df),
            },
        }

    run_logged_test("Default multi-file training data load.", check)


def test_preprocessed_dataframe_contains_target_and_rows():
    """
    Verify preprocessing output contains rows and the target label column.
    """
    def check():
        processed_df = load_preprocessed_dataframe()

        assert not processed_df.empty
        assert TARGET_COLUMN in processed_df.columns
        return {
            "summary": "Preprocessed default multi-file dataframe produced labeled training rows.",
            "output": {
                "processed_row_count": len(processed_df),
                "processed_column_count": len(processed_df.columns),
                "target_column_present": TARGET_COLUMN in processed_df.columns,
            },
        }

    run_logged_test("Preprocessed dataframe has rows and target label column.", check)
