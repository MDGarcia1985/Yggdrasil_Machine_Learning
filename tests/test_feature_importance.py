"""
Smoke tests for feature-importance utilities.

Purpose:
    Ensure model-importance helpers return usable outputs for current training
    artifacts.

Workflow role:
    Validates explainability/reporting functions used after model training.
"""

from src.feature_importance import (
    format_importance_text,
    get_model_importance,
    permutation_model_importance,
)
from src.train import train_model
from tests.log_utils import run_logged_test


def test_tree_model_importance_smoke():
    """
    Verify tree-model importance returns sorted scores for all feature columns.
    """
    def check():
        training_result = train_model()

        importance = get_model_importance(
            training_result["X_test"].columns,
            training_result["model"],
            training_result["X_test"],
            training_result["y_test"],
        )

        assert len(importance) == len(training_result["X_test"].columns)
        assert importance[0][1] >= importance[-1][1]
        top_10 = [f"{name}: {score:.2f}" for name, score in importance[:10]]
        return {
            "summary": "Model importance returns sorted feature scores for tree model.",
            "output": {
                "feature_count": len(training_result["X_test"].columns),
                "importance_count": len(importance),
                "top_10_importance_bullets": top_10,
            },
        }

    run_logged_test("Tree model feature-importance smoke test.", check)


def test_permutation_importance_and_formatter_smoke():
    """
    Verify permutation importance and formatter produce non-empty output.
    """
    def check():
        training_result = train_model()

        importance = permutation_model_importance(
            training_result["X_test"].columns,
            training_result["model"],
            training_result["X_test"],
            training_result["y_test"],
            n_repeats=3,
        )
        formatted = format_importance_text(importance, top_n=5)

        assert len(importance) == len(training_result["X_test"].columns)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        top_10 = [f"{name}: {score:.2f}" for name, score in importance[:10]]
        return {
            "summary": "Permutation importance and formatter produce readable output.",
            "output": {
                "importance_count": len(importance),
                "top_10_importance_bullets": top_10,
                "formatted_top_5": formatted,
            },
        }

    run_logged_test("Permutation importance and formatting smoke test.", check)
