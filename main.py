"""
main.py

Purpose:
Run the full WWII aircraft classification pipeline from cleaned data
through model training, evaluation, and model persistence.

Workflow:
- Load processed aircraft data
- Preprocess and engineer features
- Build feature matrix and target labels
- Split and scale data
- Train model
- Evaluate model
- Save trained model artifact

Written by: Michael Garcia
Date: 03/23/2026
For: CSC373
License: MIT
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
)


def main() -> None:
    """
    Run the full machine learning pipeline.
    """

    # ========================
    # Stage 1: Load and preprocess data
    # ========================
    df = load_data("data/processed/aircraft_cleaned.csv")
    df = select_columns(df, REQUIRED_COLUMNS)
    df = clean_numeric_data(df)
    df = clean_missing_rows(df)
    df = engineer_features(df)
    df = simplify_labels(df)

    # ========================
    # Stage 2: Build feature matrix
    # ========================
    X, y = build_feature_matrix(
        df,
        MODEL_FEATURES,
        target_columns="RoleClass",
        drop_other=True,
    )

    # ========================
    # Stage 3: Split and scale
    # ========================
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ========================
    # Stage 4: Train and evaluate
    # ========================
    model = train_model(X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)

    # ========================
    # Stage 5: Save model
    # ========================
    save_model(model, scaler, MODEL_FEATURES)

    # ========================
    # Stage 6: Print summary
    # ========================
    print("WWII Aircraft Classification Pipeline Complete")
    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    print("\nClass counts:")
    print(y.value_counts())

    print("\nAccuracy:")
    print(results["accuracy"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nModel saved to: models/model.pkl")


if __name__ == "__main__":
    main()