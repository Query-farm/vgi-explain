"""End-to-end tests driving the worker over the real VGI client/RPC stack.

Spawns ``explain_worker.py`` as a subprocess via ``vgi.client.Client`` and
exercises the wire protocol exactly as DuckDB would after ATTACH. This is the
faithful way to drive the streaming (``shap_values``) and buffering
(``feature_importance``) phases, so we keep their coverage here.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

from tests.fixtures import FEATURES, dataset, rf_classifier_blob, xgb_regressor_blob

_WORKER = str(Path(__file__).resolve().parent.parent / "explain_worker.py")


@pytest.fixture(scope="module")
def client() -> Client:
    with Client(f"uv run --no-sync {_WORKER}") as c:
        yield c


def _feature_batch() -> pa.RecordBatch:
    x, _, ids = dataset()
    return pa.RecordBatch.from_pydict(
        {
            "id": pa.array(ids, type=pa.int64()),
            "signal": pa.array(x[:, 0], type=pa.float64()),
            "noise_a": pa.array(x[:, 1], type=pa.float64()),
            "noise_b": pa.array(x[:, 2], type=pa.float64()),
        }
    )


def test_shap_values_long_format(client: Client) -> None:
    batch = _feature_batch()
    results = list(
        client.table_in_out_function(
            function_name="shap_values",
            input=iter([batch]),
            arguments=Arguments(
                positional=(),
                named={"model": pa.scalar(xgb_regressor_blob(), type=pa.binary()), "id": pa.scalar("id")},
            ),
        )
    )
    table = pa.Table.from_batches(results)
    # n_rows x n_features long rows (single output -> one class).
    assert table.num_rows == batch.num_rows * len(FEATURES)
    assert set(table.column("feature").to_pylist()) == set(FEATURES)
    assert table.column("class").to_pylist() == [None] * table.num_rows


def test_shap_values_reconstructs_prediction(client: Client) -> None:
    """Additivity through the full RPC path: sum(shap)+base ~ model prediction."""
    batch = _feature_batch()
    blob = xgb_regressor_blob()
    results = list(
        client.table_in_out_function(
            function_name="shap_values",
            input=iter([batch]),
            arguments=Arguments(
                positional=(), named={"model": pa.scalar(blob, type=pa.binary()), "id": pa.scalar("id")}
            ),
        )
    )
    table = pa.Table.from_batches(results)
    per_row: dict[int, float] = defaultdict(float)
    for rid, val in zip(table.column("id").to_pylist(), table.column("shap_value").to_pylist(), strict=True):
        per_row[rid] += val

    base = list(
        client.table_function(
            function_name="shap_base_value",
            arguments=Arguments(positional=(), named={"model": pa.scalar(blob, type=pa.binary())}),
        )
    )
    base_value = pa.Table.from_batches(base).column("base_value").to_pylist()[0]

    from vgi_explain.registry import unpack_model

    est, _ = unpack_model(blob)
    x, _, ids = dataset()
    preds = est.predict(x)
    recon = np.array([per_row[i] + base_value for i in ids])
    assert np.abs(recon - preds).max() < 1e-3


def test_feature_importance_ranks_signal_first(client: Client) -> None:
    batch = _feature_batch()
    results = list(
        client.table_buffering_function(
            function_name="feature_importance",
            input=iter([batch]),
            arguments=Arguments(
                positional=(),
                named={"model": pa.scalar(rf_classifier_blob(), type=pa.binary()), "id": pa.scalar("id")},
            ),
        )
    )
    table = pa.Table.from_batches(results)
    feats = table.column("feature").to_pylist()
    ranks = table.column("rank").to_pylist()
    methods = set(table.column("method").to_pylist())
    assert feats[0] == "signal"
    assert ranks == sorted(ranks)
    assert methods == {"mean_abs_shap"}


def test_shap_values_invalid_model_errors(client: Client) -> None:
    batch = _feature_batch()
    with pytest.raises(Exception, match="invalid model"):
        list(
            client.table_in_out_function(
                function_name="shap_values",
                input=iter([batch]),
                arguments=Arguments(positional=(), named={"model": pa.scalar(b"garbage", type=pa.binary())}),
            )
        )
