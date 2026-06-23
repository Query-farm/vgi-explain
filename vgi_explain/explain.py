"""Pure SHAP-explanation logic: no Arrow, no VGI -- just numpy + shap.

The functions here take a fitted estimator + its metadata + a feature matrix and
return plain numpy arrays. Keeping the math free of the worker plumbing makes it
directly unit-testable (additivity, known-feature ranking, multi-class shape).

Explainer selection by model type:
* tree models (RandomForest / GradientBoosting / HistGradientBoosting /
  DecisionTree / XGB*) -> ``shap.TreeExplainer`` (exact, fast)
* linear models (LinearRegression / Ridge / Lasso / LogisticRegression) ->
  ``shap.LinearExplainer``
* everything else -> ``shap.KernelExplainer`` (model-agnostic fallback)

Multi-class convention: a classifier with K>2 classes yields a SHAP value per
class. ``shap_long`` emits a ``class`` column (the class label) so every
(row, feature, class) contribution is addressable; ``base_values`` returns one
row per class. Regression and binary classification collapse to a single class
(``class`` = NULL / the positive class), one contribution per (row, feature).

Importing shap is expensive, so this module is imported once at worker load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap

# Estimator class-name fragments that select an explainer family. Matched against
# ``type(estimator).__name__`` so we don't need to import every estimator class.
_TREE_NAMES = (
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "DecisionTree",
    "XGB",  # XGBClassifier / XGBRegressor
    "LGBM",
    "CatBoost",
)
_LINEAR_NAMES = (
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "SGDClassifier",
    "SGDRegressor",
)


def explainer_kind(estimator: Any) -> str:
    """Return ``"tree"``, ``"linear"``, or ``"kernel"`` for an estimator."""
    name = type(estimator).__name__
    if any(frag in name for frag in _TREE_NAMES):
        return "tree"
    if any(frag in name for frag in _LINEAR_NAMES):
        return "linear"
    return "kernel"


def _background(x: np.ndarray, max_rows: int = 50, seed: int = 0) -> np.ndarray:
    """A small background sample for explainers that need one (linear/kernel)."""
    if x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx]


@dataclass(slots=True)
class ShapResult:
    """SHAP output normalized to a consistent (rows, features, classes) shape.

    Attributes:
        values: Array of shape ``(n_rows, n_features, n_classes)`` -- per-row,
            per-feature contribution for each output (class). ``n_classes`` is 1
            for regression and binary classification.
        base_values: Array of shape ``(n_classes,)`` -- the explainer's expected
            value per output. ``values[i, :, k].sum() + base_values[k]`` ~ the
            model output for row ``i``, output ``k`` (additivity).
        class_labels: The class label for each output column, or ``[None]`` for
            regression / single-output models.
    """

    values: np.ndarray
    base_values: np.ndarray
    class_labels: list[Any]


def _normalize(
    raw_values: Any,
    raw_base: Any,
    n_rows: int,
    n_features: int,
    class_labels: list[Any] | None,
) -> ShapResult:
    """Coerce shap's several output shapes into ``(rows, features, classes)``."""
    arr = np.asarray(raw_values, dtype=float)
    # shap returns either (rows, features) [single output] or
    # (rows, features, classes) [multi-output], or a list of (rows, features).
    if isinstance(raw_values, list):
        arr = np.stack([np.asarray(v, dtype=float) for v in raw_values], axis=-1)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # Defensive: some explainers transpose binary output to (rows, features, 2).
    arr = arr.reshape(n_rows, n_features, -1)

    base = np.atleast_1d(np.asarray(raw_base, dtype=float)).reshape(-1)
    n_classes = arr.shape[2]
    if base.shape[0] != n_classes:
        # Binary classifiers sometimes report a scalar base for a 2-output array.
        base = np.resize(base, n_classes)

    if class_labels is None or len(class_labels) != n_classes:
        labels: list[Any] = [None] if n_classes == 1 else list(range(n_classes))
    else:
        labels = list(class_labels)
    return ShapResult(values=arr, base_values=base, class_labels=labels)


def compute_shap(estimator: Any, x: np.ndarray, *, classes: list[Any] | None = None) -> ShapResult:
    """Compute SHAP values for ``x``, choosing the explainer by model type.

    Args:
        estimator: A fitted scikit-learn / XGBoost estimator.
        x: Feature matrix, shape ``(n_rows, n_features)``, columns already aligned
            to the model's fitted feature order.
        classes: Optional class labels (from model metadata) for output labelling.

    Returns:
        A :class:`ShapResult` with values normalized to ``(rows, features, classes)``.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=float))
    n_rows, n_features = x.shape
    kind = explainer_kind(estimator)

    if kind == "tree":
        explainer = shap.TreeExplainer(estimator)
        exp = explainer(x, check_additivity=False)
        return _from_explanation(exp, n_rows, n_features, classes)

    if kind == "linear":
        explainer = shap.LinearExplainer(estimator, _background(x))
        exp = explainer(x)
        return _from_explanation(exp, n_rows, n_features, classes)

    # Kernel fallback: model-agnostic, needs a prediction function + background.
    f = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    explainer = shap.KernelExplainer(f, _background(x, max_rows=20))
    raw = explainer.shap_values(x, nsamples=100, silent=True)
    return _normalize(raw, explainer.expected_value, n_rows, n_features, classes)


def _from_explanation(exp: Any, n_rows: int, n_features: int, classes: list[Any] | None) -> ShapResult:
    """Normalize a ``shap.Explanation`` object."""
    return _normalize(exp.values, exp.base_values, n_rows, n_features, classes)


def shap_long(result: ShapResult, ids: list[Any], feature_names: list[str]) -> dict[str, list[Any]]:
    """Flatten a :class:`ShapResult` into long-format columns.

    Emits one row per (input row x feature x class). For single-output models the
    ``class`` column is NULL.

    Args:
        result: The normalized SHAP result to flatten.
        ids: Per-row passthrough identifiers (falls back to the row index if empty).
        feature_names: Feature column names, aligned to the result's feature axis.

    Returns:
        Dict of column-name -> list, with keys ``id`` (the passthrough), ``feature``,
        ``class``, and ``shap_value``.
    """
    n_rows, n_features, n_classes = result.values.shape
    out_id: list[Any] = []
    out_feature: list[str] = []
    out_class: list[Any] = []
    out_value: list[float] = []
    multi = n_classes > 1
    for i in range(n_rows):
        row_id = ids[i] if ids else i
        for k in range(n_classes):
            label = result.class_labels[k] if multi else None
            for j in range(n_features):
                out_id.append(row_id)
                out_feature.append(feature_names[j])
                out_class.append(label)
                out_value.append(float(result.values[i, j, k]))
    return {"id": out_id, "feature": out_feature, "class": out_class, "shap_value": out_value}


def mean_abs_importance(result: ShapResult, feature_names: list[str]) -> list[tuple[str, float]]:
    """Global feature importance = mean(|shap|) over rows (averaged across classes).

    Returns ``(feature, importance)`` pairs sorted descending by importance.
    """
    # mean over rows and classes of |value| -> one score per feature.
    scores = np.abs(result.values).mean(axis=(0, 2))
    pairs = [(feature_names[j], float(scores[j])) for j in range(len(feature_names))]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs
