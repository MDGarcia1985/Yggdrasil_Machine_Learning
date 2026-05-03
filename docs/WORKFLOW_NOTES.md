# Workflow Notes

This document explains how data moves through the MVP machine learning pipeline, from raw CSV rows to predictions.

Primary audience:
- New contributors
- Developers returning after a long break
- Anyone who wants to change pipeline behavior safely

## Big Picture

The MVP goal is:
- Load `Data/raw/circuits_sample.csv`
- Train a model that predicts `next_component_type`
- Save artifacts for later prediction
- Use those artifacts for single-row or batch inference

Core modules:
- `src/preprocess.py`
- `src/features.py`
- `src/train.py`
- `src/predict.py`
- `src/feature_importance.py`

## End-to-End Data Flow

1. Raw CSV is read from disk.
2. Required columns are validated and selected.
3. Text and numeric values are cleaned.
4. Rows missing critical identity fields are removed.
5. Engineered feature flags and counts are added.
6. Label `next_component_type` is generated.
7. Features are encoded into model-ready matrix `X`.
8. Labels are extracted into `y`.
9. Data is split into train/test sets.
10. Model is trained and evaluated.
11. Artifacts are saved to `Models/`.
12. Prediction helpers load artifacts and run inference.

## File-By-File Workflow

### 1) `src/preprocess.py`

Purpose:
- Convert raw exported rows into clean, labeled rows.

Main functions and where they fit:
- `load_data()`: reads CSV into pandas DataFrame.
- `select_columns()`: enforces the input schema used by the MVP.
- `clean_categorical_data()`: normalizes casing, trims text, fills missing text with `"none"`.
- `clean_numeric_data()`: parses numeric values (`component_value`), defaults invalids to `0`.
- `clean_missing_rows()`: removes rows missing fields required for connection identity.
- `engineer_features()`: adds boolean and count-based circuit features.
- `add_next_component_label()`: creates `next_component_type` target per circuit.
- `preprocess_dataframe()`: orchestrates the full sequence above.
- `load_preprocessed_dataframe()`: one-call load + preprocess.
- `split_data()`: validates and splits into train/test partitions.

What downstream modules use from this file:
- `src/train.py` uses `load_preprocessed_dataframe()` and `split_data()`.
- `src/predict.py` reuses cleaning/engineering helpers (without label creation).

### 2) `src/features.py`

Purpose:
- Turn preprocessed rows into model input features and labels.

Main functions:
- `build_feature_matrix()`: one-hot encodes categorical fields, appends numeric/boolean columns.
- `get_target_labels()`: extracts the target column as string labels.
- Summary/report helpers: `get_dataset_summary()`, `get_sample_rows()`, count functions.

Why this separation exists:
- Keeps feature construction logic centralized and reusable for both training and inference.

### 3) `src/train.py`

Purpose:
- Execute canonical MVP training flow.

Main workflow:
- `train_model()`:
  - loads preprocessed data
  - builds encoded features and labels
  - validates basic data quality
  - splits train/test
  - trains `RandomForestClassifier`
  - computes metrics and feature importance
  - returns full training result payload
- `save_artifacts()`:
  - writes model and metadata for inference
- `load_model()`:
  - loads artifacts for use by prediction path

Artifacts written:
- `Models/model.pkl`
- `Models/encoder.pkl`
- `Models/feature_columns.pkl`
- `Models/training_report.pkl`

### 4) `src/feature_importance.py`

Purpose:
- Provide model-agnostic and model-specific feature-importance utilities.

Main behavior:
- Uses tree-native importances when available.
- Falls back to linear coefficients.
- Falls back again to permutation importance when needed.

Who uses it:
- `src/train.py` uses `get_model_importance()` and `print_top_importance()`.

### 5) `src/predict.py`

Purpose:
- Run inference using saved artifacts.

Main workflow:
- `load_artifacts()` loads model + encoder + feature order.
- `prepare_prediction_dataframe()` applies training-compatible cleaning/engineering steps.
- `build_feature_matrix(..., fit_encoder=False)` encodes using saved encoder.
- `align_feature_columns()` ensures exact feature order expected by the model.
- `predict_record()` handles one row as dict.
- `predict_dataframe()` handles multiple rows as DataFrame.

Important design note:
- Prediction preprocessing intentionally mirrors training preprocessing as closely as possible to reduce train/serve skew.

## Legacy Compatibility Modules

- `src/train_classifier.py`:
  - Delegates to `src/train.py`
  - Exists for backward compatibility of older imports/scripts
- `src/features_classifier.py`:
  - Alternate numeric/boolean feature path for comparison models
  - Not the main MVP training path

## How To Safely Change Pipeline Behavior

When editing pipeline logic:
1. Update function docstrings/comments in the same change.
2. Keep training and prediction preprocessing aligned.
3. Re-run tests:
   - `pytest tests`
4. Check terminal + log output:
   - `tests/data/test_data.txt`
5. If you add or remove features, confirm artifact compatibility:
   - encoder columns
   - `feature_columns.pkl`

## Quick Start For New Contributors

Run the canonical MVP pipeline:
- `python main.py`

Alternative (training only):
- `python -m src.train`

Run tests:
- `pytest tests`

Try prediction manually:
- `python -m src.predict`
