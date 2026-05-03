"""
Smoke tests for raw and preprocessed data loading.

Purpose:
    Confirm the MVP sample CSV is readable and preprocessing returns labeled rows
    required by the training pipeline.

Workflow role:
    Early guardrail tests for the load -> preprocess stage.
"""

from src.preprocess import (
    DEFAULT_DATA_PATH,
    REQUIRED_COLUMNS,
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


def test_preprocessed_dataframe_contains_target_and_rows():
    """
    Verify preprocessing output contains rows and the target label column.
    """
    def check():
        processed_df = load_preprocessed_dataframe(DEFAULT_DATA_PATH)

        assert not processed_df.empty
        assert TARGET_COLUMN in processed_df.columns
        return {
            "summary": "Preprocessed dataframe produced labeled training rows.",
            "output": {
                "processed_row_count": len(processed_df),
                "processed_column_count": len(processed_df.columns),
                "target_column_present": TARGET_COLUMN in processed_df.columns,
            },
        }

    run_logged_test("Preprocessed dataframe has rows and target label column.", check)
