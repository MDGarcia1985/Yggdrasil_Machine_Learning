"""Public API for the Yggdrasil circuit ML pipeline."""

from src.features import (
    TARGET_COLUMN,
    build_feature_matrix,
    get_available_feature_columns,
    get_dataset_summary,
    get_target_labels,
)
from src.predict import (
    evaluate_predictions,
    load_artifacts,
    model_accuracy,
    predict_dataframe,
    predict_model,
    predict_record,
)
from src.preprocess import (
    REQUIRED_COLUMNS,
    add_next_component_label,
    clean_categorical_data,
    clean_missing_rows,
    clean_numeric_data,
    engineer_features,
    load_data,
    load_and_split_data,
    load_preprocessed_dataframe,
    preprocess_dataframe,
    select_columns,
    split_features_and_labels,
    split_data,
)
from src.train import (
    load_model,
    main,
    save_artifacts,
    train_model,
)

try:
    from src.feature_importance import (
        format_importance_text,
        get_model_importance,
        print_top_importance,
    )
except ImportError:
    format_importance_text = None
    get_model_importance = None
    print_top_importance = None

__all__ = [
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "add_next_component_label",
    "build_feature_matrix",
    "clean_categorical_data",
    "clean_missing_rows",
    "clean_numeric_data",
    "engineer_features",
    "evaluate_predictions",
    "format_importance_text",
    "get_available_feature_columns",
    "get_dataset_summary",
    "get_model_importance",
    "get_target_labels",
    "load_and_split_data",
    "load_artifacts",
    "load_data",
    "load_model",
    "load_preprocessed_dataframe",
    "main",
    "model_accuracy",
    "predict_dataframe",
    "predict_model",
    "predict_record",
    "preprocess_dataframe",
    "print_top_importance",
    "save_artifacts",
    "select_columns",
    "split_data",
    "split_features_and_labels",
    "train_model",
]
