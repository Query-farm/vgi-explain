"""Pure SHAP-logic tests: explainer choice, additivity, multi-class, importance."""

from __future__ import annotations

import numpy as np

from tests.fixtures import (
    FEATURES,
    dataset,
    linear_regressor_blob,
    rf_classifier_blob,
    xgb_multiclass_blob,
    xgb_regressor_blob,
)
from vgi_explain.explain import (
    compute_shap,
    explainer_kind,
    mean_abs_importance,
    shap_long,
)
from vgi_explain.registry import unpack_model


def test_explainer_kind() -> None:
    rf, _ = unpack_model(rf_classifier_blob())
    xgb, _ = unpack_model(xgb_regressor_blob())
    lin, _ = unpack_model(linear_regressor_blob())
    assert explainer_kind(rf) == "tree"
    assert explainer_kind(xgb) == "tree"
    assert explainer_kind(lin) == "linear"


def test_tree_classifier_additivity() -> None:
    est, meta = unpack_model(rf_classifier_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x[:8], classes=meta.classes)
    assert res.values.shape == (8, 3, 2)
    # sum(shap) + base ~ predict_proba, per class.
    recon = res.values.sum(axis=1) + res.base_values
    proba = est.predict_proba(x[:8])
    assert np.abs(recon - proba).max() < 1e-6


def test_xgb_regressor_additivity() -> None:
    est, meta = unpack_model(xgb_regressor_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x[:8], classes=meta.classes)
    assert res.values.shape == (8, 3, 1)
    recon = res.values.sum(axis=1)[:, 0] + res.base_values[0]
    pred = est.predict(x[:8])
    assert np.abs(recon - pred).max() < 1e-4


def test_linear_additivity() -> None:
    est, meta = unpack_model(linear_regressor_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x[:8], classes=meta.classes)
    recon = res.values.sum(axis=1)[:, 0] + res.base_values[0]
    pred = est.predict(x[:8])
    assert np.abs(recon - pred).max() < 1e-6


def test_multiclass_shape_and_long() -> None:
    est, meta = unpack_model(xgb_multiclass_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x[:4], classes=meta.classes)
    assert res.values.shape[2] == 3
    assert res.base_values.shape[0] == 3
    long = shap_long(res, ids=[10, 11, 12, 13], feature_names=FEATURES)
    # 4 rows x 3 features x 3 classes = 36 long rows, class column populated.
    assert len(long["shap_value"]) == 4 * 3 * 3
    assert set(long["class"]) == {0, 1, 2}


def test_known_feature_ranks_first() -> None:
    est, meta = unpack_model(rf_classifier_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x, classes=meta.classes)
    ranking = mean_abs_importance(res, FEATURES)
    assert ranking[0][0] == "signal"
    assert ranking[0][1] > ranking[1][1]


def test_long_uses_row_index_without_ids() -> None:
    est, meta = unpack_model(xgb_regressor_blob())
    x, _, _ = dataset()
    res = compute_shap(est, x[:3], classes=meta.classes)
    long = shap_long(res, ids=[], feature_names=FEATURES)
    # single-output -> class is None, ids fall back to 0..n-1
    assert set(long["class"]) == {None}
    assert sorted(set(long["id"])) == [0, 1, 2]
