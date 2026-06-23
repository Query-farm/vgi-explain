"""The SQL function surface of the explain worker.

Three functions, all reading a model BLOB (as produced by vgi-sklearn /
vgi-xgboost) passed as a scalar ``model`` argument. A DuckDB table function
allows only one subquery parameter, which is reserved for the feature relation,
so the model travels as a scalar -- typically a session VARIABLE:

    SET VARIABLE m = (SELECT model FROM sklearn.fit(...));
    SELECT * FROM explain.shap_values((SELECT id, f1, f2 FROM data),
                                       model := getvariable('m'), id := 'id');

* ``shap_values`` (table-in-out) -- long format: one row per
  (input row x feature [x class]). Streams the feature relation through the
  chosen explainer (TreeExplainer / LinearExplainer / KernelExplainer).
* ``shap_base_value`` (source) -- the explainer expected value, one row per class.
* ``feature_importance`` (buffering) -- global mean(|shap|) over the passed
  relation; with no rows, falls back to the model's native importances.

Multi-class behavior: for a classifier with K>2 classes each function reports a
contribution per class via a ``class`` column (the class label); regression and
binary classification collapse to a single output (``class`` = NULL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import OutputCollector as InOutCollector
from vgi.table_in_out_function import TableInOutGenerator

from .buffering import DrainState, SinkBuffer, input_schema_of
from .explain import compute_shap, mean_abs_importance, shap_long
from .features import matrix
from .registry import InvalidModelBlobError, ModelMetadata, UntrustedModelError, unpack_meta, unpack_model
from .schema_utils import field as sfield


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


def _aligned_features(meta: ModelMetadata, input_schema: pa.Schema, id_col: str) -> list[str]:
    """Model feature names that are present in the input (name-aligned, reorder-safe).

    Prefers the model's fitted ``feature_names`` so columns are aligned by name
    regardless of the input's column order. Falls back to the input's numeric
    columns (minus ``id``) when the model carries no feature names.
    """
    if meta.feature_names:
        return list(meta.feature_names)
    return _features_excluding(input_schema, id_col)


def _load_meta_or_raise(blob: bytes) -> ModelMetadata:
    try:
        return unpack_meta(blob)
    except (InvalidModelBlobError, UntrustedModelError) as exc:
        raise ValueError(f"invalid model: {exc}") from exc


def _load_model_or_raise(blob: bytes) -> tuple[Any, ModelMetadata]:
    try:
        return unpack_model(blob)
    except (InvalidModelBlobError, UntrustedModelError) as exc:
        raise ValueError(f"invalid model: {exc}") from exc


# ===========================================================================
# shap_values  (table-in-out: feature relation in, long-format SHAP out)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ShapValuesArgs:
    data: Annotated[TableInput, Arg(0, doc="Feature relation (must contain the model's feature columns).")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through (excluded from features).")]


_SHAP_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


class ShapValues(TableInOutGenerator[ShapValuesArgs]):
    FunctionArguments: ClassVar[type] = ShapValuesArgs

    class Meta:
        name = "shap_values"
        description = "Per-row, per-feature SHAP contributions in long format (one row per row x feature [x class])"
        categories = ["explainability", "shap", "inference"]
        examples = [
            FunctionExample(
                sql=(
                    "SET VARIABLE m = (SELECT model FROM sklearn.fit(...));\n"
                    "SELECT * FROM explain.shap_values((SELECT id, f1, f2, f3 FROM data), "
                    "model := getvariable('m'), id := 'id')"
                ),
                description="Explain each row's prediction feature by feature",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ShapValuesArgs]) -> BindResponse:
        a = params.args
        if not a.model:
            raise ValueError("shap_values requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _load_meta_or_raise(a.model)
        feats = _aligned_features(meta, input_schema, a.id)
        missing = [f for f in feats if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"input is missing feature column(s) {', '.join(missing)}; "
                f"model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        else:
            fields.append(sfield("id", pa.int64(), "0-based input row index (no id column given).", nullable=False))
        fields.append(sfield("feature", pa.string(), "Feature name.", nullable=False))
        fields.append(sfield("class", pa.string(), "Class label for the output (NULL for regression / binary)."))
        fields.append(
            sfield(
                "shap_value",
                pa.float64(),
                "Signed contribution of the feature to the model output.",
                nullable=False,
            )
        )
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[ShapValuesArgs]) -> tuple[Any, ModelMetadata]:
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _SHAP_CACHE.get(key)
        if cached is None:
            cached = _load_model_or_raise(params.args.model)
            _SHAP_CACHE[key] = cached
        return cached

    @classmethod
    def process(
        cls,
        params: ProcessParams[ShapValuesArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        estimator, meta = cls._model(params)
        assert params.init_call is not None
        input_schema = params.init_call.bind_call.input_schema
        assert input_schema is not None
        feats = _aligned_features(meta, input_schema, a.id)
        x = matrix(pa.Table.from_batches([batch]), feats)

        if x.shape[0] == 0:
            out.emit(
                pa.RecordBatch.from_pydict({n: [] for n in params.output_schema.names}, schema=params.output_schema)
            )
            return

        result = compute_shap(estimator, x, classes=meta.classes)
        ids = batch.column(a.id).to_pylist() if a.id else list(range(x.shape[0]))
        long = shap_long(result, ids, feats)

        id_name = a.id or "id"
        columns: dict[str, list[Any]] = {
            id_name: long["id"],
            "feature": long["feature"],
            "class": [None if c is None else str(c) for c in long["class"]],
            "shap_value": long["shap_value"],
        }
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# shap_base_value  (source: model BLOB only, one row per class)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ShapBaseValueArgs:
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).")]


_BASE_SCHEMA = pa.schema(
    [
        sfield("class", pa.string(), "Class label for this base value (NULL for regression / binary)."),
        sfield("base_value", pa.float64(), "Explainer expected value (the SHAP base value).", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class ShapBaseValue(TableFunctionGenerator[ShapBaseValueArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _BASE_SCHEMA

    class Meta:
        name = "shap_base_value"
        description = "The SHAP base (expected) value of a model -- one row, or one row per class"
        categories = ["explainability", "shap"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM explain.shap_base_value(model := getvariable('m'))",
                description="The model's expected output, the anchor SHAP values add onto",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ShapBaseValueArgs]) -> BindResponse:
        if not params.args.model:
            raise ValueError("shap_base_value requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
        _load_meta_or_raise(params.args.model)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def cardinality(cls, params: BindParams[ShapBaseValueArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1000)

    @classmethod
    def process(cls, params: ProcessParams[ShapBaseValueArgs], state: None, out: OutputCollector) -> None:
        estimator, meta = _load_model_or_raise(params.args.model)
        # The base value comes from the explainer; SHAP needs a tiny background
        # sample, which the model BLOB does not carry. Synthesize one zero-row so
        # tree/linear explainers report their expected value. For models that
        # require real data (kernel) the base value is taken from a single zero
        # vector -- documented as an approximation.
        n_features = meta.n_features or len(meta.feature_names) or 1
        x0 = np.zeros((1, n_features), dtype=float)
        result = compute_shap(estimator, x0, classes=meta.classes)
        multi = result.base_values.shape[0] > 1
        labels = result.class_labels
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "class": [None if not multi else str(labels[k]) for k in range(result.base_values.shape[0])],
                    "base_value": [float(v) for v in result.base_values],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


# ===========================================================================
# feature_importance  (buffering: mean(|shap|) over the relation, or native)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FeatureImportanceArgs:
    data: Annotated[TableInput, Arg(0, doc="Feature relation to average SHAP magnitudes over.")]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]


_IMPORTANCE_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield(
            "importance",
            pa.float64(),
            "Global importance: mean(|SHAP|) over rows, or native importance.",
            nullable=False,
        ),
        sfield("rank", pa.int32(), "1-based rank by importance (1 = most important).", nullable=False),
        sfield("method", pa.string(), "How importance was computed: 'mean_abs_shap' or 'native'.", nullable=False),
    ]
)


def _native_importance(estimator: Any, feature_names: list[str]) -> list[tuple[str, float]]:
    """Model-native global importance (used when no data rows are provided)."""
    imp = getattr(estimator, "feature_importances_", None)
    if imp is None:
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            arr = np.abs(np.asarray(coef, dtype=float))
            imp = arr.mean(axis=0) if arr.ndim > 1 else arr
    if imp is None:
        raise ValueError(
            "feature_importance with no data rows requires a model exposing feature_importances_ or coef_; "
            "pass a non-empty feature relation to use mean(|SHAP|) instead"
        )
    imp = np.asarray(imp, dtype=float).reshape(-1)
    pairs = [(feature_names[j], float(imp[j])) for j in range(min(len(feature_names), imp.shape[0]))]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


class FeatureImportance(SinkBuffer[FeatureImportanceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = FeatureImportanceArgs

    class Meta:
        name = "feature_importance"
        description = "Global feature importance: mean(|SHAP|) over the relation (or native importances if empty)"
        categories = ["explainability", "shap"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM explain.feature_importance((SELECT id, f1, f2, f3 FROM data), "
                    "model := getvariable('m'), id := 'id')"
                ),
                description="Rank features by their average SHAP magnitude across the dataset",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FeatureImportanceArgs]) -> BindResponse:
        a = params.args
        if not a.model:
            raise ValueError("feature_importance requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _load_meta_or_raise(a.model)
        feats = _aligned_features(meta, input_schema, a.id)
        # Only require columns be present if the relation will actually be used;
        # an empty relation falls back to native importances. We can't know row
        # count at bind, so require the columns to exist (cheap and helpful).
        missing = [f for f in feats if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"input is missing feature column(s) {', '.join(missing)}; "
                f"model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )
        return BindResponse(output_schema=_IMPORTANCE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[FeatureImportanceArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FeatureImportanceArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        estimator, meta = _load_model_or_raise(a.model)
        input_schema = input_schema_of(params)
        feats = _aligned_features(meta, input_schema, a.id)
        table = cls.buffered_table(params, input_schema)

        if table is not None and table.num_rows > 0:
            x = matrix(table, feats)
            result = compute_shap(estimator, x, classes=meta.classes)
            pairs = mean_abs_importance(result, feats)
            method = "mean_abs_shap"
        else:
            pairs = _native_importance(estimator, feats)
            method = "native"

        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": [f for f, _ in pairs],
                    "importance": [v for _, v in pairs],
                    "rank": [i + 1 for i in range(len(pairs))],
                    "method": [method] * len(pairs),
                },
                schema=params.output_schema,
            )
        )


EXPLAIN_FUNCTIONS: list[type] = [ShapValues, ShapBaseValue, FeatureImportance]
