"""
feature_importance.py

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247

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
    Purpose:
        Read sklearn's native Gini-based importances for tree ensembles.

    Design:
        Sorts descending so callers can slice `[:top_n]` without re-sorting.

    Workflow:
        Invoked from `get_model_importance` when `feature_importances_` exists.

    Data handoff:
        Inputs: parallel `features` names and fitted tree `model`.
        Outputs: `List[Tuple[str, float]]` for `format_importance_text` / UI.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    # key=lambda x: -x[1]: descending sort by importance score.
    return sorted(
        zip(list(features), model.feature_importances_),
        key=lambda x: -x[1],
    )


def decision_tree_importance(features: Iterable[str], decision_tree_model) -> ImportanceList:
    """
    Purpose:
        Preserve older call sites that referenced decision trees explicitly.

    Design:
        Pure alias to `tree_model_importance` (no extra logic).

    Workflow:
        Legacy scripts/tests importing this name.

    Data handoff:
        Same as `tree_model_importance`.
    """
    return tree_model_importance(features, decision_tree_model)


def linear_model_importance(features: Iterable[str], model) -> ImportanceList:
    """
    Purpose:
        Derive a per-feature score from linear model weights when trees are absent.

    Design:
        Uses `abs(coef)`; multiclass models average across classes so each
        feature gets one scalar (shape `(n_features,)`, not per-class).

    Workflow:
        Second branch inside `get_model_importance`.

    Data handoff:
        Inputs: feature names iterable, fitted linear classifier `model`.
        Outputs: sorted `(name, score)` tuples like tree path.
    """
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_.")

    coef = model.coef_

    if len(coef.shape) == 1:
        scores = abs(coef)
    else:
        # coef_ shape (n_classes, n_features): collapse to one score per column.
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
    Purpose:
        Score features by how much shuffling each column hurts test accuracy.

    Design:
        Wraps `sklearn.inspection.permutation_importance`; cost grows with
        `n_repeats * n_features`.

    Workflow:
        Fallback from `get_model_importance` when no native coef_/importances.

    Data handoff:
        Inputs: feature names (for zipping), fitted `model`, holdout `X_test`/`y_test`.
        Outputs: sorted mean importance per feature.
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
    Purpose:
        Expose permutation importance under a KNN-specific name for old examples.

    Design:
        Delegates entirely to `permutation_model_importance`.

    Workflow:
        Legacy comparisons where KNN has no `feature_importances_`.

    Data handoff:
        Same as `permutation_model_importance`.
    """
    return permutation_model_importance(features, knn_model, X_test, y_test)


def get_model_importance(
    features: Iterable[str],
    model,
    X_test=None,
    y_test=None,
) -> ImportanceList:
    """
    Purpose:
        Single entry point for "how important is each column?" across estimators.

    Design:
        Cascading capability checks: trees → linear coef_ → permutation (needs
        holdout data).

    Workflow:
        `train.train_model` calls this for console reporting and saved reports.

    Data handoff:
        Inputs: training feature column names, fitted `model`, optional test set
        for permutation fallback.
        Outputs: `ImportanceList` stored in training artifacts / printed by
        `print_top_importance`.
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
    Purpose:
        Render the top-N importance pairs as one line of text for logs/UI.

    Design:
        Joins with `"  |  "` delimiter; optional percent formatting rescales
        display only (does not renormalize to sum to 100%).

    Workflow:
        Any caller needing a string instead of structured tuples.

    Data handoff:
        Inputs: `ImportanceList` from `get_model_importance`.
        Outputs: single formatted `str` (no side effects).
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
    Purpose:
        Emit human-readable importance lines to stdout during training.

    Design:
        One `print` per feature to avoid huge single strings on large models.

    Workflow:
        Invoked at the end of `train.train_model` for operator feedback.

    Data handoff:
        Inputs: sorted `ImportanceList` (highest scores first).
        Outputs: console only; returns `None`.
    """
    print(f"\nTop {top_n} Feature Importances:")

    for name, score in importance_values[:top_n]:
        print(f"{name}: {score:.4f}")
