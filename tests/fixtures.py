"""Shared model/BLOB builders for the explain test suite.

Trains tiny, deterministic models on a dataset where feature ``signal`` is the
only informative column, and packs them into the same self-contained BLOB format
vgi-sklearn produces (so the tests exercise the real consume-the-BLOB path).
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor

from vgi_explain.registry import ModelMetadata, pack_model

FEATURES = ["signal", "noise_a", "noise_b"]
N = 60


def dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Deterministic dataset: ``signal`` separates classes; the rest is noise."""
    rng = np.random.default_rng(seed)
    half = N // 2
    signal = np.r_[np.zeros(half), np.ones(half)] + rng.normal(0, 0.01, N)
    noise_a = rng.normal(size=N)
    noise_b = rng.normal(size=N)
    x = np.column_stack([signal, noise_a, noise_b])
    y = np.r_[np.zeros(half), np.ones(half)].astype(int)
    return x, y, list(range(N))


def _meta(name: str, estimator: str, task: str, classes: list[int] | None) -> ModelMetadata:
    return ModelMetadata(
        name=name,
        estimator=estimator,
        task=task,
        target="target",
        feature_names=list(FEATURES),
        classes=classes,
        n_samples=N,
        n_features=len(FEATURES),
        train_score=1.0,
        sklearn_version=sklearn.__version__,
        created_at="2026-01-01T00:00:00+00:00",
    )


def rf_classifier_blob() -> bytes:
    x, y, _ = dataset()
    est = RandomForestClassifier(n_estimators=25, random_state=0).fit(x, y)
    return pack_model(est, _meta("rf_clf", "random_forest_classifier", "classification", [0, 1]))


def xgb_regressor_blob() -> bytes:
    x, y, _ = dataset()
    est = XGBRegressor(n_estimators=25, max_depth=3, random_state=0).fit(x, y.astype(float))
    return pack_model(est, _meta("xgb_reg", "xgb_regressor", "regression", None))


def xgb_multiclass_blob() -> bytes:
    """A 3-class XGBoost classifier (for multi-class output shape tests)."""
    x, y, _ = dataset()
    # Make a 3rd class out of the upper third so we have 3 distinct labels.
    y3 = y.copy()
    y3[N - N // 3 :] = 2
    est = XGBClassifier(n_estimators=25, max_depth=3, random_state=0).fit(x, y3)
    return pack_model(est, _meta("xgb_mc", "xgb_classifier", "classification", [0, 1, 2]))


def linear_regressor_blob() -> bytes:
    x, y, _ = dataset()
    est = LinearRegression().fit(x, y.astype(float))
    return pack_model(est, _meta("lin_reg", "linear_regression", "regression", None))
