"""
test_predict.py

Purpose:
Run a structured test of prediction using the trained circuit model.

Validates:
- model + encoder loading
- dataframe-based prediction
- single-row prediction
- accuracy + evaluation formatting
"""

from src.preprocess import load_preprocessed_dataframe
import pandas as pd

from src.graph import CircuitGraph
from src.train import train_model, save_artifacts
from src.predict import (
    predict_dataframe,
    predict_record,
    evaluate_predictions,
    model_accuracy,
)
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run prediction sniff test for the circuit dataset.
    """

    print("Prediction Sniff Test (Circuit Dataset)")

    df = load_preprocessed_dataframe()
    trained = train_model()
    predictions = trained["predictions"]
    y_test_out = trained["y_test"]
    save_artifacts(trained)

    print("\nPrediction Output Sample:")
    print(predictions[:10])

    results = evaluate_predictions(predictions, y_test_out)
    accuracy = model_accuracy(predictions, y_test_out)

    print("\nFirst 10 Evaluation Results:")
    for r in results[:10]:
        print(r)

    print("\nAccuracy:")
    print(accuracy)

    prediction_df = predict_dataframe(df.head(10))

    graph = CircuitGraph(
        circuit_name="graph_prediction_smoke",
        description="Graph-generated rows for prediction smoke test.",
    )
    graph.load_default()
    graph_rows = graph.to_rows()
    graph_prediction_df = predict_dataframe(pd.DataFrame(graph_rows))
    graph_single_result = predict_record(graph_rows[0])

    print("\nDataFrame Prediction Sample:")
    print(prediction_df[[
        "component_type",
        "predicted_next_component_type",
        "prediction_confidence",
    ]].head())

    example_row = df.iloc[0].to_dict()
    single_result = predict_record(example_row)

    print("\nSingle Row Prediction:")
    print(single_result)

    print("\nGraph Prediction Sample:")
    print(graph_prediction_df[[
        "component_type",
        "predicted_next_component_type",
        "prediction_confidence",
    ]].head())

    print("\nGraph Single Row Prediction:")
    print(graph_single_result)

    print("\nBasic Consistency Checks:")
    print(f"Prediction count matches test rows: {len(predictions) == len(y_test_out)}")
    print(f"Evaluation count matches predictions: {len(results) == len(predictions)}")
    print(f"Accuracy between 0 and 1: {0 <= accuracy <= 1}")
    print(f"Prediction DF has results: {'predicted_next_component_type' in prediction_df.columns}")
    print(f"Graph prediction DF has results: {'predicted_next_component_type' in graph_prediction_df.columns}")
    print(f"Graph single prediction has prediction: {'prediction' in graph_single_result}")


def test_predict_smoke() -> None:
    output = capture_output(run_test)
    assert "Prediction Sniff Test" in output
    assert "predicted_next_component_type" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_predict", output)
