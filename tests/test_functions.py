"""In-process tests for the SQL surface (source function + bind validation).

The streaming ``shap_values`` and buffering ``feature_importance`` functions are
covered end-to-end against a real Client in test_client.py; here we drive the
source ``shap_base_value`` through the in-process harness and check argument /
feature-alignment validation that happens at bind.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from vgi.arguments import Arguments
from vgi.invocation import FunctionType
from vgi.protocol import BindRequest

from tests.fixtures import rf_classifier_blob, xgb_regressor_blob
from tests.harness import invoke_table_function
from vgi_explain.features import matrix
from vgi_explain.functions import FeatureImportance, ShapBaseValue, ShapValues


def _bind(func_cls: type, *, named: dict[str, pa.Scalar], input_schema: pa.Schema | None = None):
    bind_req = BindRequest(
        function_name=func_cls.Meta.name,
        arguments=Arguments(positional=(), named=named),
        function_type=FunctionType.TABLE,
        input_schema=input_schema,
    )
    return func_cls.bind(bind_req)


def test_shap_base_value_regression_one_row() -> None:
    table = invoke_table_function(ShapBaseValue, named={"model": pa.scalar(xgb_regressor_blob(), type=pa.binary())})
    assert table.num_rows == 1
    assert table.column("class").to_pylist() == [None]
    assert np.isfinite(table.column("base_value").to_pylist()[0])


def test_shap_base_value_classifier_per_class() -> None:
    table = invoke_table_function(ShapBaseValue, named={"model": pa.scalar(rf_classifier_blob(), type=pa.binary())})
    # A tree classifier yields a base value per class; the class column is labelled.
    assert table.num_rows >= 1
    base = table.column("base_value").to_pylist()
    assert all(np.isfinite(v) for v in base)
    if table.num_rows > 1:
        assert table.column("class").to_pylist() == ["0", "1"]


def test_shap_base_value_invalid_model_errors() -> None:
    with pytest.raises(ValueError, match="invalid model"):
        invoke_table_function(ShapBaseValue, named={"model": pa.scalar(b"garbage", type=pa.binary())})


def test_shap_base_value_missing_model_errors() -> None:
    with pytest.raises(ValueError, match="requires 'model'"):
        invoke_table_function(ShapBaseValue, named={})


def test_shap_values_bind_missing_feature_errors() -> None:
    schema = pa.schema([("id", pa.int64()), ("signal", pa.float64())])  # missing noise_a/noise_b
    with pytest.raises(ValueError, match="missing feature column"):
        _bind(
            ShapValues,
            named={"model": pa.scalar(rf_classifier_blob(), type=pa.binary()), "id": pa.scalar("id")},
            input_schema=schema,
        )


def test_shap_values_bind_shuffled_columns_ok() -> None:
    # Columns in a different order than the model's feature_names -> still binds,
    # because features are aligned by name.
    schema = pa.schema(
        [("noise_b", pa.float64()), ("id", pa.int64()), ("signal", pa.float64()), ("noise_a", pa.float64())]
    )
    resp = _bind(
        ShapValues,
        named={"model": pa.scalar(rf_classifier_blob(), type=pa.binary()), "id": pa.scalar("id")},
        input_schema=schema,
    )
    assert resp.output_schema.names == ["id", "feature", "class", "shap_value"]


def test_feature_importance_bind_invalid_model_errors() -> None:
    schema = pa.schema([("signal", pa.float64()), ("noise_a", pa.float64()), ("noise_b", pa.float64())])
    with pytest.raises(ValueError, match="invalid model"):
        _bind(FeatureImportance, named={"model": pa.scalar(b"nope", type=pa.binary())}, input_schema=schema)


def test_matrix_name_alignment() -> None:
    # A table whose columns are shuffled produces the same matrix when selected
    # by the model's feature order.
    t = pa.table({"noise_b": [1.0, 2.0], "signal": [9.0, 8.0], "noise_a": [0.0, 0.0]})
    m = matrix(t, ["signal", "noise_a", "noise_b"])
    assert m.tolist() == [[9.0, 0.0, 1.0], [8.0, 0.0, 2.0]]


def test_matrix_missing_column_errors() -> None:
    t = pa.table({"signal": [1.0]})
    with pytest.raises(ValueError, match="missing required feature"):
        matrix(t, ["signal", "noise_a"])
