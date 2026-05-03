"""
test_features.py

Purpose:
Run a structured test of feature creation for the Yggdrasil circuit dataset.

Validates:
- dataset summary
- sample row extraction
- encoded feature matrix creation
- target label extraction
- report helper functions
"""

from src.preprocess import load_preprocessed_dataframe
import pandas as pd

from src.features import (
    build_feature_matrix,
    get_circuit_component_counts,
    get_dataset_summary,
    get_net_connection_counts,
    get_sample_rows,
    get_target_labels,
)
from src.graph import CircuitGraph
from src.preprocess import preprocess_dataframe
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run feature-preparation sniff test for the circuit dataset.
    """

    print("Feature Preparation Sniff Test (Circuit Dataset)")

    df = load_preprocessed_dataframe()

    summary = get_dataset_summary(df)
    sample_rows = get_sample_rows(df, n=5)
    X_encoded, encoder = build_feature_matrix(df)
    y = get_target_labels(df)
    component_counts = get_circuit_component_counts(df)
    net_counts = get_net_connection_counts(df)

    graph = CircuitGraph(
        circuit_name="graph_feature_smoke",
        description="Graph-generated rows for feature smoke test.",
    )
    graph.load_default()
    graph_df = preprocess_dataframe(pd.DataFrame(graph.to_rows()))
    graph_X_encoded, graph_encoder = build_feature_matrix(graph_df)
    graph_y = get_target_labels(graph_df)

    print("\nDataset Summary:")
    print(summary)

    print("\nSample Rows:")
    for row in sample_rows:
        print(row)

    print("\nEncoded Feature Matrix:")
    print(f"Shape: {X_encoded.shape}")
    print(f"First 10 Columns: {list(X_encoded.columns[:10])}")

    print("\nTarget Labels:")
    print(f"Count: {len(y)}")
    print(f"Unique Labels: {sorted(y.unique().tolist())}")

    print("\nEncoder Categories:")
    for feature_name, categories in zip(encoder.feature_names_in_, encoder.categories_):
        print(f"{feature_name}: {list(categories)}")

    print("\nComponent Counts:")
    print(component_counts.head(20))

    print("\nNet Counts:")
    print(net_counts.head(20))

    print("\nGraph Feature Matrix:")
    print(f"Graph rows: {len(graph_df)}")
    print(f"Graph shape: {graph_X_encoded.shape}")
    print(f"Graph labels: {sorted(graph_y.unique().tolist())}")
    print(f"Graph encoder categories: {len(graph_encoder.categories_)}")

    print("\nBasic Consistency Checks:")
    print(f"Encoded rows match dataframe rows: {len(X_encoded) == len(df)}")
    print(f"Target rows match dataframe rows: {len(y) == len(df)}")
    print(f"Summary row count matches dataframe: {summary['num_rows'] == len(df)}")
    print(f"Sample rows default to 5 or less: {len(sample_rows) <= 5}")
    print(f"Encoded feature matrix has columns: {X_encoded.shape[1] > 0}")
    print(f"Graph feature rows match graph dataframe: {len(graph_X_encoded) == len(graph_df)}")


def test_features_smoke() -> None:
    output = capture_output(run_test)
    assert "Feature Preparation Sniff Test" in output
    assert "Encoded Feature Matrix" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_features", output)
