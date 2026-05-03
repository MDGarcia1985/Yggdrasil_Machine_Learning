"""
Public API for the machine learning pipeline package.
"""

from src.features import MODEL_FEATURES, build_feature_matrix, select_model_features
from src.predict import predict_aircraft
from src.preprocess import (
    REQUIRED_COLUMNS,
    clean_missing_rows,
    clean_numeric_data,
    engineer_features,
    load_data,
    select_columns,
    simplify_labels,
)
from src.train import (
    evaluate_model,
    load_model,
    save_model,
    scale_features,
    split_data,
    train_model,
)

__all__ = [
    "MODEL_FEATURES",
    "REQUIRED_COLUMNS",
    "build_feature_matrix",
    "clean_missing_rows",
    "clean_numeric_data",
    "engineer_features",
    "evaluate_model",
    "load_data",
    "load_model",
    "predict_aircraft",
    "save_model",
    "scale_features",
    "select_columns",
    "select_model_features",
    "simplify_labels",
    "split_data",
    "train_model",
]
