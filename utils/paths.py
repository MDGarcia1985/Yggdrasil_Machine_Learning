"""
Stable filesystem paths used by scripts and docs-friendly entrypoints.

Why this module exists:
    Centralize path definitions so app/test scripts do not rely on the current
    working directory when executed from different shells or IDE actions.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_CIRCUITS_SAMPLE = RAW_DATA_DIR / "circuits_sample.csv"

MODELS_DIR = PROJECT_ROOT / "Models"
MODEL_PATH = MODELS_DIR / "model.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.pkl"

TESTS_DIR = PROJECT_ROOT / "tests"
TEST_LOG_PATH = TESTS_DIR / "data" / "test_data.txt"

DOCS_DIR = PROJECT_ROOT / "docs"
WORKFLOW_NOTES_PATH = DOCS_DIR / "WORKFLOW_NOTES.md"
DEV_NOTES_PATH = DOCS_DIR / "DEV_NOTES.md"

MAIN_ENTRYPOINT = PROJECT_ROOT / "main.py"
STREAMLIT_APP_PATH = PROJECT_ROOT / "ui" / "app_streamlit.py"
