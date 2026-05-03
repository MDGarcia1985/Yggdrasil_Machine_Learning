"""
test_predict.py

Purpose:
Run a structured test of prediction using the saved model artifact.
"""

import numpy as np

from src.predict import predict_aircraft
from utils.run_tests import capture_output, log_test_output


def run_test() -> None:
    """
    Run a prediction sniff test using a hand-built sample.
    """

    sample = {
        "Crew": 1,
        "Length": 9.83,
        "Wingspan": 11.28,
        "Height": 3.71,
        "WingArea": 21.8,
        "MaxSpeed": 710,
        "AspectRatio": (11.28 ** 2) / 21.8,
        "SizeIndex": 9.83 * 11.28,
        "NumberBuilt_log": np.log1p(15875),
    }

    result = predict_aircraft(sample)

    print("Prediction Sniff Test")
    print("\nInput sample:")
    for key, value in sample.items():
        print(f"{key}: {value}")

    print("\nPrediction:")
    print(result["prediction"])

    print("\nProbabilities:")
    print(result["probabilities"])

    print("\nConfidence:")
    print(result["confidence"])

    print("\nExpected feature columns:")
    print(result["feature_columns"])


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_predict", output)