"""Tests for consuming the model BLOB: round-trip, robustness, trusted load."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from tests.fixtures import rf_classifier_blob, xgb_regressor_blob
from vgi_explain.registry import (
    InvalidModelBlobError,
    ModelMetadata,
    pack_model,
    unpack_meta,
    unpack_model,
)


def test_pack_unpack_roundtrip() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    est = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, [0, 1])
    meta = ModelMetadata(
        name="m",
        estimator="random_forest_classifier",
        task="classification",
        target="y",
        feature_names=["a", "b"],
        classes=[0, 1],
        n_features=2,
        n_samples=2,
    )
    blob = pack_model(est, meta)

    m2 = unpack_meta(blob)
    assert m2.feature_names == ["a", "b"]
    assert m2.classes == [0, 1]

    est2, m3 = unpack_model(blob)
    assert m3.name == "m"
    assert int(est2.predict([[1.0, 1.0]])[0]) == 1


def test_consumes_sklearn_format_rf_and_xgb() -> None:
    # Both unpack via the same skops path; xgboost is in the trusted prefixes.
    _, rf_meta = unpack_model(rf_classifier_blob())
    assert rf_meta.feature_names == ["signal", "noise_a", "noise_b"]
    est, xgb_meta = unpack_model(xgb_regressor_blob())
    assert xgb_meta.task == "regression"
    assert type(est).__name__ == "XGBRegressor"


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x00\x00",
        b"garbage-not-a-blob",
        struct.pack(">I", 1000) + b"short",  # claims 1000-byte metadata, truncated
    ],
)
def test_garbage_blob_raises_invalid(blob: bytes) -> None:
    with pytest.raises(InvalidModelBlobError):
        unpack_meta(blob)


def test_valid_meta_but_corrupt_estimator_raises() -> None:
    meta = json.dumps(
        ModelMetadata(name="m", estimator="x", task="regression", target="y", feature_names=["a"]).to_dict()
    ).encode()
    blob = struct.pack(">I", len(meta)) + meta + b"not-skops-bytes"
    # metadata reads fine...
    assert unpack_meta(blob).name == "m"
    # ...but loading the estimator fails clearly rather than crashing.
    with pytest.raises(InvalidModelBlobError):
        unpack_model(blob)
