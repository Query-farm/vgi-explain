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

import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector
from vgi_rpc.rpc import OutputCollector as InOutCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .explain import compute_shap, mean_abs_importance, shap_long
from .features import matrix
from .registry import InvalidModelBlobError, ModelMetadata, UntrustedModelError, unpack_meta, unpack_model
from .schema_utils import field as sfield

# Base GitHub blob URL for source files in this repo (pinned to `main`). Each
# object's `vgi.source_url` (VGI128) points at exactly where it is implemented.
_SOURCE_BASE = "https://github.com/Query-farm/vgi-explain/blob/main/vgi_explain"


def _source_url(relative_path: str) -> str:
    """Build the implementation `vgi.source_url` for a file under `vgi_explain/`."""
    return f"{_SOURCE_BASE}/{relative_path}"


# Self-contained, catalog-qualified SQL fragments backing the executable examples.
# vgi-explain has no `fit` of its own, so a real model has to come from a sibling
# worker (vgi-sklearn / vgi-xgboost) or, as here, from the committed fixture the
# SQL E2E suite uses.
#
# DuckDB allows a table function at most ONE subquery argument (reserved for the
# feature relation) and forbids a scalar subquery argument entirely, so the model
# BLOB cannot be inlined as `model := (SELECT …)`. The idiomatic pattern — and the
# one every docstring shows — is to hold it in a session VARIABLE and reference it
# with `getvariable('m')`. Each executable example is therefore a TWO-statement
# list: a `SET VARIABLE` step followed by the query, run in order against the same
# connection, so the strict linter EXECUTES them cleanly from the repo working
# directory. The per-function `Meta.examples` use the same `getvariable('m')` form
# (it binds under the linter's EXPLAIN even before the VARIABLE is set, because the
# output schema is static and model validation is deferred to execution).
_RF_MODEL = "(SELECT model FROM read_parquet('test/fixtures/models.parquet') WHERE name = 'rf_clf')"
_FEATURES = "(SELECT id, signal, noise_a, noise_b FROM read_parquet('test/fixtures/features.parquet'))"
_SET_MODEL = f"SET VARIABLE m = {_RF_MODEL}"


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
    """Arguments for ``shap_values``: feature relation, model BLOB, optional id."""

    data: Annotated[TableInput, Arg(0, doc="Feature relation (must contain the model's feature columns).")]
    model: Annotated[
        bytes | None,
        Arg(
            "model",
            default=b"",
            arrow_type=pa.binary(),
            doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).",
        ),
    ]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through (excluded from features).")]


_SHAP_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


class ShapValues(TableInOutGenerator[ShapValuesArgs]):
    """Stream a feature relation through SHAP, emitting long-format contributions."""

    FunctionArguments: ClassVar[type] = ShapValuesArgs

    class Meta:
        """SQL-facing metadata for ``shap_values``."""

        name = "shap_values"
        description = "Per-row, per-feature SHAP contributions in long format (one row per row x feature [x class])"
        categories = ["explainability", "shap", "inference"]
        title = "SHAP Values (Per-Row Feature Contributions)"
        keywords = (
            "shap, shap values, feature contributions, local explanation, explainability, "
            "interpretability, attribution, why prediction, per-row, long format, scikit-learn, "
            "xgboost, machine learning"
        )
        description_llm = (
            "## `shap_values`\n\n"
            "Compute **per-row, per-feature SHAP contributions** for a fitted scikit-learn / "
            "XGBoost model, returned in **long format** — one output row per "
            "(input row x feature [x class]).\n\n"
            "**When to use it.** Answer *why did the model predict this for this row?* — the signed "
            "`shap_value` for each feature shows how much that feature pushed the model's output up "
            "(positive) or down (negative), relative to the model's base value "
            "(see `shap_base_value`). Summing a row's contributions and the base value reconstructs "
            "the model's raw output (SHAP additivity).\n\n"
            "**Inputs.** A feature relation as the single subquery argument; a `model` BLOB "
            "(scalar, produced by vgi-sklearn / vgi-xgboost), typically a session VARIABLE or an "
            "inline subquery; and an optional `id` column name to carry through (excluded from "
            "features). Features are aligned to the model's fitted columns **by name**, so input "
            "column order does not matter; the input must contain every feature the model expects.\n\n"
            "**Outputs.** Columns `id`, `feature`, `class`, `shap_value`. For a multi-class "
            "classifier with K>2 classes there is one row per class (the `class` column carries the "
            "label); regression and binary classification collapse to a single output with "
            "`class = NULL`.\n\n"
            "**Edge cases.** An empty input relation yields zero rows; a missing feature column or "
            "an invalid / untrusted model BLOB raises a clear `invalid model: …` error rather than "
            "crashing."
        )
        description_md = (
            "# shap_values\n\n"
            "Per-row, per-feature **SHAP contributions** for a fitted scikit-learn / XGBoost model, "
            "in **long format** (one row per input row x feature, and per class for multi-class "
            "classifiers).\n\n"
            "## Usage\n\n"
            "```sql\n"
            "SET VARIABLE m = (SELECT model FROM read_parquet('test/fixtures/models.parquet')\n"
            "                  WHERE name = 'rf_clf');\n"
            "SELECT * FROM explain.shap_values(\n"
            "  (SELECT id, signal, noise_a, noise_b FROM data),\n"
            "  model := getvariable('m'), id := 'id');\n"
            "```\n\n"
            "The model travels as the scalar `model` argument (a DuckDB table function allows only "
            "one subquery, reserved for the feature relation). Features are aligned to the model's "
            "fitted columns by name, so input order is irrelevant.\n\n"
            "## Notes\n\n"
            "- Each row's contributions plus the model base value (`shap_base_value`) reconstruct "
            "the prediction (SHAP additivity).\n"
            "- `class` is the label for multi-class classifiers, `NULL` for regression / binary.\n"
            "- A missing feature column or an invalid model BLOB raises a clear error."
        )
        tags = {
            "vgi.title": title,
            "vgi.doc_llm": description_llm,
            "vgi.doc_md": description_md,
            "vgi.keywords": keywords,
            "vgi.source_url": _source_url("functions.py"),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "| --- | --- | --- |\n"
                "| `id` | input `id` column type (else `BIGINT`) | The `id` value carried through (column "
                "named after the `id` argument), or the 0-based input row index when no `id` is given. |\n"
                "| `feature` | `VARCHAR` | Feature (column) name the contribution is for. |\n"
                "| `class` | `VARCHAR` | Class label for multi-class classifiers; `NULL` for regression / "
                "binary classification. |\n"
                "| `shap_value` | `DOUBLE` | Signed SHAP contribution of the feature to the model output. |"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    f"SELECT * FROM explain.shap_values({_FEATURES}, "
                    "model := getvariable('m'), id := 'id') ORDER BY id, feature, class LIMIT 6"
                ),
                description="Explain each row's prediction feature by feature (model held in session VARIABLE m)",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ShapValuesArgs]) -> BindResponse:
        """Return the long-format output schema; validate the model when it is resolvable.

        The output schema (``id, feature, class, shap_value``) is static — it does
        not depend on the model — so binding never needs to unpack the BLOB. When a
        concrete model BLOB is available at plan time we validate it (and the feature
        columns) for an early, friendly error; when the ``model`` argument resolves
        to NULL / empty (e.g. ``getvariable('m')`` before the session VARIABLE is
        set, as in ``EXPLAIN``-only planning) we defer that check to ``process``,
        which always requires a real model.
        """
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.model:
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
        if not params.args.model:
            raise ValueError("shap_values requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
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
        """Compute and emit SHAP contributions for one input batch."""
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
    """Arguments for ``shap_base_value``: just the model BLOB."""

    model: Annotated[
        bytes | None,
        Arg(
            "model",
            default=b"",
            arrow_type=pa.binary(),
            doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).",
        ),
    ]


_BASE_SCHEMA = pa.schema(
    [
        sfield("class", pa.string(), "Class label for this base value (NULL for regression / binary)."),
        sfield("base_value", pa.float64(), "Explainer expected value (the SHAP base value).", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class ShapBaseValue(TableFunctionGenerator[ShapBaseValueArgs]):
    """Source function emitting a model's SHAP base value, one row per class."""

    FIXED_SCHEMA: ClassVar[pa.Schema] = _BASE_SCHEMA

    class Meta:
        """SQL-facing metadata for ``shap_base_value``."""

        name = "shap_base_value"
        description = "The SHAP base (expected) value of a model -- one row, or one row per class"
        categories = ["explainability", "shap"]
        title = "SHAP Base Value (Model Expected Output)"
        keywords = (
            "shap, base value, expected value, baseline, anchor, intercept, expected output, "
            "explainability, interpretability, additivity, scikit-learn, xgboost, machine learning"
        )
        description_llm = (
            "## `shap_base_value`\n\n"
            "Return the **SHAP base (expected) value** of a fitted scikit-learn / XGBoost model — "
            "the explainer's *expected output* over the background distribution. This is the anchor "
            "that every row's per-feature SHAP contributions (`shap_values`) add onto to reconstruct "
            "that row's prediction (SHAP additivity).\n\n"
            "**When to use it.** Establish the model's baseline before reading local explanations: a "
            "row's prediction = base value + sum of its SHAP contributions. Pair it with "
            "`shap_values` whenever you present a per-row attribution.\n\n"
            "**Inputs.** Just the scalar `model` BLOB (produced by vgi-sklearn / vgi-xgboost) — no "
            "feature relation. It is typically a session VARIABLE or an inline subquery.\n\n"
            "**Outputs.** Columns `class`, `base_value`. A multi-class classifier returns one row "
            "per class (the `class` column carries the label); regression and binary classification "
            "return a single row with `class = NULL`.\n\n"
            "**Edge cases.** The model BLOB does not carry a background sample, so the base value is "
            "computed from a single zero feature vector — exact for tree / linear explainers, an "
            "approximation for models that require real background data. An invalid / untrusted "
            "BLOB raises a clear `invalid model: …` error."
        )
        description_md = (
            "# shap_base_value\n\n"
            "The **SHAP base (expected) value** of a fitted scikit-learn / XGBoost model: the "
            "explainer's expected output — the anchor that a row's SHAP contributions add onto to "
            "reach its prediction.\n\n"
            "## Usage\n\n"
            "```sql\n"
            "SET VARIABLE m = (SELECT model FROM read_parquet('test/fixtures/models.parquet')\n"
            "                  WHERE name = 'rf_clf');\n"
            "SELECT * FROM explain.shap_base_value(model := getvariable('m'));\n"
            "```\n\n"
            "Takes only the scalar `model` BLOB (no feature relation).\n\n"
            "## Notes\n\n"
            "- One row per class for multi-class classifiers; a single `class = NULL` row "
            "otherwise.\n"
            "- Computed from a zero background vector (exact for tree / linear explainers).\n"
            "- `base_value + sum(shap_value)` reconstructs a row's model output."
        )
        tags = {
            "vgi.title": title,
            "vgi.doc_llm": description_llm,
            "vgi.doc_md": description_md,
            "vgi.keywords": keywords,
            "vgi.source_url": _source_url("functions.py"),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "| --- | --- | --- |\n"
                "| `class` | `VARCHAR` | Class label for the base value; `NULL` for regression / binary "
                "classification (single row). |\n"
                "| `base_value` | `DOUBLE` | The explainer expected value — the anchor that a row's SHAP "
                "contributions add onto to reach its prediction. |"
            ),
            # VGI509: at least one guaranteed-runnable, catalog-qualified example. Each SQL is
            # self-contained (model inlined from the committed fixture) and re-runnable against
            # an attached `explain` worker. `expected_result` is omitted deliberately — the
            # linter only needs each query to execute cleanly.
            "vgi.executable_examples": json.dumps(
                [
                    {
                        "description": "The model's expected (base) output — one row per class for a classifier.",
                        "sql": [
                            _SET_MODEL,
                            "SELECT class, base_value FROM explain.shap_base_value(model := getvariable('m')) "
                            "ORDER BY class",
                        ],
                    },
                    {
                        "description": "Explain each row's prediction feature by feature (long format).",
                        "sql": [
                            _SET_MODEL,
                            f"SELECT id, feature, class, shap_value FROM explain.shap_values("
                            f"{_FEATURES}, model := getvariable('m'), id := 'id') "
                            "ORDER BY id, feature, class LIMIT 6",
                        ],
                    },
                    {
                        "description": "Rank features by their average SHAP magnitude across the dataset.",
                        "sql": [
                            _SET_MODEL,
                            f"SELECT feature, rank, method FROM explain.feature_importance("
                            f"{_FEATURES}, model := getvariable('m'), id := 'id') ORDER BY rank",
                        ],
                    },
                ]
            ),
        }
        examples = [
            FunctionExample(
                sql=("SELECT class, base_value FROM explain.shap_base_value(model := getvariable('m')) ORDER BY class"),
                description="The model's expected output, the anchor SHAP values add onto (model in VARIABLE m)",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ShapBaseValueArgs]) -> BindResponse:
        """Return the fixed base-value schema; validate the model when it is resolvable.

        The schema (``class, base_value``) is static, so binding never needs the
        BLOB. A concrete model is validated here for an early error; a NULL / empty
        ``model`` (e.g. ``getvariable('m')`` during ``EXPLAIN``-only planning, before
        the session VARIABLE is set) defers the check to ``process``, which always
        requires a real model.
        """
        if params.args.model:
            _load_meta_or_raise(params.args.model)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def cardinality(cls, params: BindParams[ShapBaseValueArgs]) -> TableCardinality:
        """Estimate the output cardinality (one row, or one per class)."""
        return TableCardinality(estimate=1, max=1000)

    @classmethod
    def process(cls, params: ProcessParams[ShapBaseValueArgs], state: None, out: OutputCollector) -> None:
        """Compute and emit the model's expected (base) value per class."""
        if not params.args.model:
            raise ValueError("shap_base_value requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
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
    """Arguments for ``feature_importance``: feature relation, model BLOB, optional id."""

    data: Annotated[TableInput, Arg(0, doc="Feature relation to average SHAP magnitudes over.")]
    model: Annotated[
        bytes | None,
        Arg(
            "model",
            default=b"",
            arrow_type=pa.binary(),
            doc="A model BLOB (as produced by vgi-sklearn / vgi-xgboost).",
        ),
    ]
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
    """Buffering function: rank features by mean(|SHAP|) over the relation."""

    FunctionArguments: ClassVar[type] = FeatureImportanceArgs

    class Meta:
        """SQL-facing metadata for ``feature_importance``."""

        name = "feature_importance"
        description = "Global feature importance: mean(|SHAP|) over the relation (or native importances if empty)"
        categories = ["explainability", "shap"]
        title = "Feature Importance (Global SHAP Ranking)"
        keywords = (
            "feature importance, global importance, mean absolute shap, ranking, top features, "
            "explainability, interpretability, feature selection, native importance, gini, "
            "coefficients, scikit-learn, xgboost, machine learning"
        )
        description_llm = (
            "## `feature_importance`\n\n"
            "Rank a fitted model's features by **global importance** — by default the "
            "**mean absolute SHAP value** (`mean(|SHAP|)`) of each feature over the feature "
            "relation you pass. This aggregates the per-row, per-feature attributions of "
            "`shap_values` into one number per feature: how much, on average, the feature moves the "
            "model's output.\n\n"
            "**When to use it.** Answer *which features matter most overall?* — for model "
            "summaries, feature selection, or dashboards. SHAP-based importance is consistent and "
            "model-agnostic, unlike split-count / Gini heuristics.\n\n"
            "**Inputs.** A feature relation (the single subquery argument) to average over; the "
            "scalar `model` BLOB; and an optional `id` column name to exclude from features. "
            "Features are aligned to the model's fitted columns by name.\n\n"
            "**Outputs.** Columns `feature`, `importance`, `rank` (1 = most important), and `method` "
            "(`'mean_abs_shap'` or `'native'`).\n\n"
            "**Edge cases.** If the passed relation has **zero rows**, the function falls back to "
            "the model's *native* importances (`feature_importances_`, or `|coef_|` for linear "
            "models) and sets `method = 'native'`; if the model exposes neither, that empty-relation "
            "call raises a clear error. An invalid / untrusted model BLOB raises "
            "`invalid model: …`."
        )
        description_md = (
            "# feature_importance\n\n"
            "**Global feature importance** for a fitted scikit-learn / XGBoost model: "
            "`mean(|SHAP|)` over the relation you pass, or the model's native importances when the "
            "relation is empty.\n\n"
            "## Usage\n\n"
            "```sql\n"
            "SET VARIABLE m = (SELECT model FROM read_parquet('test/fixtures/models.parquet')\n"
            "                  WHERE name = 'rf_clf');\n"
            "SELECT * FROM explain.feature_importance(\n"
            "  (SELECT id, signal, noise_a, noise_b FROM data),\n"
            "  model := getvariable('m'), id := 'id')\n"
            "ORDER BY rank;\n"
            "```\n\n"
            "## Notes\n\n"
            "- `method = 'mean_abs_shap'` when rows are supplied; `'native'` when the relation is "
            "empty (uses `feature_importances_` / `|coef_|`).\n"
            "- `rank` is 1-based, most important first.\n"
            "- Features are aligned to the model's fitted columns by name."
        )
        tags = {
            "vgi.title": title,
            "vgi.doc_llm": description_llm,
            "vgi.doc_md": description_md,
            "vgi.keywords": keywords,
            "vgi.source_url": _source_url("functions.py"),
            "vgi.result_columns_md": (
                "| column | type | description |\n"
                "| --- | --- | --- |\n"
                "| `feature` | `VARCHAR` | Feature (column) name. |\n"
                "| `importance` | `DOUBLE` | Global importance: mean(|SHAP|) over the passed rows, or the "
                "model's native importance when no rows are given. |\n"
                "| `rank` | `INTEGER` | 1-based rank by importance (1 = most important). |\n"
                "| `method` | `VARCHAR` | How importance was computed: `'mean_abs_shap'` or `'native'`. |"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    f"SELECT feature, rank, method FROM explain.feature_importance({_FEATURES}, "
                    "model := getvariable('m'), id := 'id') ORDER BY rank"
                ),
                description="Rank features by their average SHAP magnitude across the dataset (model in VARIABLE m)",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FeatureImportanceArgs]) -> BindResponse:
        """Return the importance schema; validate the model/feature columns when resolvable.

        The output schema (``feature, importance, rank, method``) is static, so
        binding never needs the BLOB. A concrete model is validated here (and its
        feature columns checked) for an early, friendly error; a NULL / empty
        ``model`` (e.g. ``getvariable('m')`` during ``EXPLAIN``-only planning, before
        the session VARIABLE is set) defers the check to ``finalize``, which always
        requires a real model.
        """
        a = params.args
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.model:
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
        """Start each finalize stream with a fresh (not-yet-emitted) cursor."""
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FeatureImportanceArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        """Emit ranked feature importances once (mean(|SHAP|), or native if empty)."""
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        if not a.model:
            raise ValueError("feature_importance requires 'model' (a model BLOB, e.g. model := getvariable('m'))")
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
