"""
feature_importance.py

Purpose:
Summarize which circuit features most influence each model.

Design:
Keep explainability separate from training so the same reporting tools can be
used by Decision Trees, Random Forests, Logistic Regression, and KNN.

Workflow:
- Use tree-native importance when available.
- Use coefficient magnitude for linear models when available.
- Use permutation importance as a fallback.
- Format importance values for console or UI display.

Michael Garcia
michael@mandedesign.studio
https://mandedesign.studio

License: MIT
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from sklearn.inspection import permutation_importance


ImportanceList = List[Tuple[str, float]]


def tree_model_importance(features: Iterable[str], model) -> ImportanceList:
    """
    Report feature importance from tree-based models.

    Works with:
    - DecisionTreeClassifier
    - RandomForestClassifier
    - ExtraTreesClassifier
    - other sklearn models exposing feature_importances_

    Args:
        features:
            Ordered feature column names.
        model:
            Fitted tree-based model.

    Returns:
        List of (feature_name, importance_score), sorted highest first.

    Workflow role:
        Called by `get_model_importance()` during training reports and tests.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    return sorted(
        zip(list(features), model.feature_importances_),
        key=lambda x: -x[1],
    )


def decision_tree_importance(features: Iterable[str], decision_tree_model) -> ImportanceList:
    """
    Backward-compatible alias for tree_model_importance().
    """
    return tree_model_importance(features, decision_tree_model)


def linear_model_importance(features: Iterable[str], model) -> ImportanceList:
    """
    Estimate importance for linear models using coefficient magnitude.

    Works with:
    - LogisticRegression
    - LinearSVC-style models exposing coef_

    For multiclass classifiers, this averages absolute coefficient magnitude
    across classes.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_.")

    coef = model.coef_

    if len(coef.shape) == 1:
        scores = abs(coef)
    else:
        # For multiclass models, average absolute effect size per feature.
        scores = abs(coef).mean(axis=0)

    return sorted(
        zip(list(features), scores),
        key=lambda x: -x[1],
    )


def permutation_model_importance(
    features: Iterable[str],
    model,
    X_test,
    y_test,
    n_repeats: int = 10,
    random_state: int = 42,
) -> ImportanceList:
    """
    Estimate feature importance with permutation importance.

    This is useful for models that do not expose built-in importance scores,
    such as KNN.

    Design note:
        Permutation importance is slower but model-agnostic, so it is the
        fallback path when no native importance attribute is available.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    return sorted(
        zip(list(features), result.importances_mean),
        key=lambda x: -x[1],
    )


def knn_importance(features, knn_model, X_test, y_test) -> ImportanceList:
    """
    Backward-compatible KNN importance helper.
    """
    return permutation_model_importance(features, knn_model, X_test, y_test)


def get_model_importance(
    features: Iterable[str],
    model,
    X_test=None,
    y_test=None,
) -> ImportanceList:
    """
    Choose the best available importance method for the supplied model.

    Priority:
    1. Tree-native importance
    2. Linear coefficient magnitude
    3. Permutation importance, if X_test and y_test are supplied

    Workflow role:
        Training code uses this to keep model explainability logic in one place.
    """
    if hasattr(model, "feature_importances_"):
        return tree_model_importance(features, model)

    if hasattr(model, "coef_"):
        return linear_model_importance(features, model)

    if X_test is not None and y_test is not None:
        return permutation_model_importance(features, model, X_test, y_test)

    raise ValueError(
        "No supported feature importance method available for this model. "
        "Provide X_test and y_test to use permutation importance."
    )


def format_importance_text(
    importance_values: ImportanceList,
    top_n: int = 15,
    as_percent: bool = False,
) -> str:
    """
    Turn feature importance tuples into compact display text.

    Args:
        importance_values:
            List of (feature_name, score).
        top_n:
            Number of top features to include.
        as_percent:
            If True, format scores as percentages.

    Returns:
        Human-readable importance string.

    Primary use:
        Console and lightweight UI text output.
    """
    selected = importance_values[:top_n]

    if as_percent:
        return "  |  ".join(
            f"{name}: {score:.0%}"
            for name, score in selected
        )

    return "  |  ".join(
        f"{name}: {score:.4f}"
        for name, score in selected
    )


def print_top_importance(
    importance_values: ImportanceList,
    top_n: int = 15,
) -> None:
    """
    Print top feature importances in a readable console format.

    Primary use:
        Training CLI output (`train.train_model`).
    """
    print(f"\nTop {top_n} Feature Importances:")

    for name, score in importance_values[:top_n]:
        print(f"{name}: {score:.4f}")
