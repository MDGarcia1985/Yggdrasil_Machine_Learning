"""
Path-consistency tests for MVP execution and shared constants.

Purpose:
    Confirm canonical data and entrypoint paths resolve correctly across modules.

Workflow role:
    Protects script/app startup from drift in hardcoded path assumptions.
"""

from pathlib import Path

from src.preprocess import DEFAULT_DATA_PATH
from tests.log_utils import run_logged_test
from utils.paths import (
    DEV_NOTES_PATH,
    DOCS_DIR,
    MAIN_ENTRYPOINT,
    MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_CIRCUITS_SAMPLE,
    STREAMLIT_APP_PATH,
    TEST_LOG_PATH,
)


def test_mvp_sample_path_exists():
    """
    Verify DEFAULT_DATA_PATH points to the expected sample CSV path.
    """
    def check():
        expected = Path("Data/raw/circuits_sample.csv")
        assert DEFAULT_DATA_PATH == expected
        assert DEFAULT_DATA_PATH.exists()
        return {
            "summary": "MVP sample path constant is correct and file exists.",
            "output": {
                "default_data_path": str(DEFAULT_DATA_PATH),
                "path_matches_expected": DEFAULT_DATA_PATH == expected,
                "path_exists": DEFAULT_DATA_PATH.exists(),
            },
        }

    run_logged_test("MVP sample path constant and file existence.", check)


def test_shared_path_constants_are_aligned():
    """
    Verify utils.paths constants match real project structure.
    """
    def check():
        expected_project_sample = PROJECT_ROOT / DEFAULT_DATA_PATH

        assert PROJECT_ROOT.exists()
        assert MAIN_ENTRYPOINT.exists()
        assert STREAMLIT_APP_PATH.exists()
        assert DOCS_DIR.exists()
        assert DEV_NOTES_PATH.exists()
        assert TEST_LOG_PATH.exists()
        assert RAW_CIRCUITS_SAMPLE == expected_project_sample
        assert RAW_CIRCUITS_SAMPLE.exists()
        assert MODELS_DIR == PROJECT_ROOT / "Models"
        assert MODEL_PATH == MODELS_DIR / "model.pkl"

        return {
            "summary": "Shared path constants in utils.paths point to expected project locations.",
            "output": {
                "project_root": str(PROJECT_ROOT),
                "main_entrypoint": str(MAIN_ENTRYPOINT),
                "streamlit_app_path": str(STREAMLIT_APP_PATH),
                "raw_circuits_sample": str(RAW_CIRCUITS_SAMPLE),
                "test_log_path": str(TEST_LOG_PATH),
                "paths_exist": {
                    "project_root": PROJECT_ROOT.exists(),
                    "main_entrypoint": MAIN_ENTRYPOINT.exists(),
                    "streamlit_app_path": STREAMLIT_APP_PATH.exists(),
                    "docs_dir": DOCS_DIR.exists(),
                    "dev_notes": DEV_NOTES_PATH.exists(),
                    "test_log": TEST_LOG_PATH.exists(),
                    "raw_circuits_sample": RAW_CIRCUITS_SAMPLE.exists(),
                },
            },
        }

    run_logged_test("Centralized utils.paths constants are valid and aligned.", check)
