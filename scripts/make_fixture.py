"""Generate the committed model-BLOB parquet fixture for the SQL E2E tests.

vgi-explain has no ``fit`` function -- it only *consumes* a model BLOB produced by
a sibling worker (vgi-sklearn / vgi-xgboost). To exercise the SQL functions end
to end we need such a BLOB available inside DuckDB. This script trains tiny,
deterministic models, packs each into the same self-contained BLOB format
vgi-sklearn uses (``registry.pack_model``), and writes them to
``test/fixtures/models.parquet`` as ``(name VARCHAR, model BLOB, ...)``.

The SQL tests then do:

    SET VARIABLE m = (SELECT model FROM read_parquet('test/fixtures/models.parquet')
                      WHERE name = 'rf_clf');
    SELECT * FROM explain.shap_values((SELECT * FROM read_parquet('.../features.parquet')),
                                      model := getvariable('m'), id := 'id');

Run via ``make fixture``. The output parquet files are committed so CI does not
need to retrain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from vgi_explain.registry import ModelMetadata, pack_model

_FIXDIR = Path(__file__).resolve().parent.parent / "test" / "fixtures"

# Deterministic dataset: feature "signal" separates classes / drives the target;
# "noise_a", "noise_b" are uninformative. Same shape used by the unit tests.
_FEATURES = ["signal", "noise_a", "noise_b"]
_N = 60


def _dataset() -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(0)
    half = _N // 2
    signal = np.r_[np.zeros(half), np.ones(half)] + rng.normal(0, 0.01, _N)
    noise_a = rng.normal(size=_N)
    noise_b = rng.normal(size=_N)
    x = np.column_stack([signal, noise_a, noise_b])
    y_clf = np.r_[np.zeros(half), np.ones(half)].astype(int)
    ids = list(range(_N))
    return x, y_clf, ids


def _meta(name: str, estimator: str, task: str, classes: list[int] | None) -> ModelMetadata:
    return ModelMetadata(
        name=name,
        estimator=estimator,
        task=task,
        target="target",
        feature_names=list(_FEATURES),
        classes=classes,
        n_samples=_N,
        n_features=len(_FEATURES),
        train_score=1.0,
        sklearn_version=sklearn.__version__,
        created_at="2026-01-01T00:00:00+00:00",
    )


def main() -> None:
    _FIXDIR.mkdir(parents=True, exist_ok=True)
    x, y_clf, ids = _dataset()

    rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(x, y_clf)
    rf_blob = pack_model(rf, _meta("rf_clf", "random_forest_classifier", "classification", [0, 1]))

    xgb = XGBRegressor(n_estimators=25, max_depth=3, random_state=0).fit(x, y_clf.astype(float))
    xgb_blob = pack_model(xgb, _meta("xgb_reg", "xgb_regressor", "regression", None))

    models = pa.table(
        {
            "name": pa.array(["rf_clf", "xgb_reg"], type=pa.string()),
            "estimator": pa.array(["random_forest_classifier", "xgb_regressor"], type=pa.string()),
            "model": pa.array([rf_blob, xgb_blob], type=pa.binary()),
        }
    )
    pq.write_table(models, _FIXDIR / "models.parquet")

    features = pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "signal": pa.array(x[:, 0], type=pa.float64()),
            "noise_a": pa.array(x[:, 1], type=pa.float64()),
            "noise_b": pa.array(x[:, 2], type=pa.float64()),
        }
    )
    pq.write_table(features, _FIXDIR / "features.parquet")

    print(f"wrote {_FIXDIR / 'models.parquet'} and {_FIXDIR / 'features.parquet'}")


if __name__ == "__main__":
    main()
