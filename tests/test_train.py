"""
test_train.py

Purpose:
Run a structured test of the training pipeline.
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
    build_feature_matrix,
)

from src.train import (
    split_data,
    scale_features,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run a training pipeline sniff test.
    """
    df = load_data("data/processed/aircraft_cleaned.csv")
    df = select_columns(df, REQUIRED_COLUMNS)
    df = clean_numeric_data(df)
    df = clean_missing_rows(df)
    df = engineer_features(df)
    df = simplify_labels(df)

    X, y = build_feature_matrix(df, MODEL_FEATURES, target_columns="RoleClass", drop_other=True)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_model(X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)

    save_model(model, scaler, MODEL_FEATURES)

    print("\nModel saved to models/model.pkl")

    loaded = load_model()

    print("\nLoaded model artifact keys:")
    print(list(loaded.keys()))

    print("Training Pipeline Sniff Test")
    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    print("\nMissing values in X:")
    print(X.isna().sum())

    print("\nTrain/Test split:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nAccuracy:")
    print(results["accuracy"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_train", output)