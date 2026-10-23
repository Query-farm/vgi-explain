"""VGI worker exposing SHAP model explanations to DuckDB/SQL.

Assembles the explain functions into a single ``explain`` catalog and provides
the process entry points. The repo-root ``explain_worker.py`` / ``serve.py`` are
thin shims over this module for ``uv run`` and containers; installed users get
the ``vgi-explain`` and ``vgi-explain-http`` console scripts.

    ATTACH 'explain' (TYPE vgi, LOCATION 'vgi-explain');
    SET VARIABLE m = (SELECT model FROM sklearn.fit(...));
    SELECT * FROM explain.shap_values((SELECT id, f1, f2 FROM data),
                                      model := getvariable('m'), id := 'id');

shap / scikit-learn / xgboost are imported at module load (via ``functions`` ->
``explain``) so the expensive initialization happens once per process.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema, View
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_explain import __version__
from vgi_explain.functions import EXPLAIN_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
# A semver *range* (not a concrete version) for the explanation/model-BLOB
# contract this catalog serves: the whole 0.x line. Surfaced as
# CatalogInfo.data_version_spec so clients can express compatibility.
DATA_VERSION_SPEC = ">=0.1.0,<0.2.0"
GIT_COMMIT = os.environ.get("VGI_EXPLAIN_GIT_COMMIT") or "unknown"

SOURCE_URL = "https://github.com/Query-farm/vgi-explain"

_CATALOG_DESCRIPTION_LLM = (
    "Compute SHAP (SHapley Additive exPlanations) for already-fitted scikit-learn / XGBoost models as "
    "DuckDB table functions. Does not train models — it interprets the self-contained model `BLOB` produced "
    "by the sibling vgi-sklearn / vgi-xgboost workers (passed as a scalar `model` argument, typically a "
    "session VARIABLE). Use to answer: why did the model predict this row? (`shap_values` — per-row, "
    "per-feature signed contributions in long format); what is the model's baseline output? "
    "(`shap_base_value` — the explainer expected value the contributions add onto); and which features "
    "matter most overall? (`feature_importance` — global mean(|SHAP|) over a relation, or native "
    "importances). Multi-class classifiers report a contribution per class."
)

_CATALOG_DESCRIPTION_MD = (
    "# SHAP Model Explanations in SQL\n\n"
    "![SHAP logo](https://raw.githubusercontent.com/shap/shap/master/docs/artwork/shap_logo.png)\n\n"
    "**Explainable AI for machine learning predictions, right inside DuckDB — compute SHAP "
    "(SHapley Additive exPlanations) feature attributions for fitted scikit-learn and XGBoost models with "
    "plain SQL.**\n\n"
    "The `explain` catalog turns model interpretability into a set of DuckDB/SQL table functions, so you can "
    "answer *why* a model made a prediction without leaving your database or writing any Python. It is built "
    "for data engineers, analysts, and ML practitioners who need transparent, auditable, per-prediction "
    "explanations — for model debugging, regulatory and fair-lending reviews, feature selection, or simply "
    "building trust in a model's output. Every explanation is computed over your data in place via Apache "
    "Arrow, so it composes naturally with joins, filters, and aggregates in ordinary SQL.\n\n"
    "Under the hood the worker is powered by [SHAP](https://github.com/shap/shap), the reference "
    "implementation of Shapley-value feature attribution "
    "([documentation](https://shap.readthedocs.io/en/latest/)), grounded in cooperative game theory's "
    "[Shapley values](https://en.wikipedia.org/wiki/Shapley_value). Importantly, vgi-explain does **not** "
    "train models — it *interprets* the self-contained *model* `BLOB` produced by the sibling `vgi-sklearn` "
    "([scikit-learn](https://scikit-learn.org/)) and `vgi-xgboost` ([XGBoost](https://xgboost.readthedocs.io/)) "
    "workers. SHAP automatically selects an appropriate explainer (e.g. TreeExplainer for tree ensembles), so "
    "explanations are exact and fast for the supported model families. Because a DuckDB table function accepts "
    "only one subquery argument (reserved for the feature relation), the model travels as a scalar `model` "
    "argument, usually held in a session variable and read with `getvariable('m')` — see the schema's and "
    "each function's example queries for ready-to-run SQL.\n\n"
    "Three functions make up the surface. `shap_values` streams a feature relation and emits per-row, "
    "per-feature signed contributions in long format (with a contribution per class for multi-class "
    "classifiers), aligning input columns to the model's fitted features by name. `shap_base_value` returns "
    "the explainer's expected value — the model's baseline output that the per-feature contributions add onto, "
    "so that base value + sum(contributions) reproduces the prediction (SHAP's additivity property). "
    "`feature_importance` aggregates `mean(|SHAP|)` across a relation to produce a global, ranked importance "
    "of which features matter most, falling back to the model's native importances when no rows are supplied. "
    "Together they cover local (single-prediction) and global (whole-dataset) interpretability for the models "
    "you already serve from VGI."
)

_SCHEMA_DESCRIPTION_LLM = (
    "SHAP explanation table functions for fitted scikit-learn / XGBoost models: per-row, per-feature "
    "contributions (`shap_values`), the model's base/expected value (`shap_base_value`), and global feature "
    "importance (`feature_importance`). Each takes a model `BLOB` as the scalar `model` argument."
)

_SCHEMA_DESCRIPTION_MD = (
    "# SHAP Explanations\n\n"
    "Model-interpretability table functions that turn a fitted scikit-learn or XGBoost model into "
    "per-prediction and dataset-wide **SHAP** feature attributions. Every explanation is computed in place "
    "over Apache Arrow, so it composes with ordinary SQL joins, filters, and aggregates.\n\n"
    "## What it covers\n\n"
    "*Local* interpretability answers **why did the model predict this row?** by attributing the output to "
    "each input feature; *global* interpretability ranks **which features matter most** across a whole "
    "dataset. Multi-class classifiers report a contribution per class, and every attribution obeys SHAP "
    "additivity — a row's base value plus its feature contributions reconstruct the model's output.\n\n"
    "## How models arrive\n\n"
    "This schema does **not** train models: it *interprets* the self-contained model `BLOB` packed by the "
    "sibling `vgi-sklearn` / `vgi-xgboost` workers. Because a DuckDB table function reserves its single "
    "subquery slot for the feature relation, the model travels as a scalar `model` argument — typically held "
    "in a session VARIABLE and read with `getvariable('m')`."
)

_CATALOG_KEYWORDS = [
    "shap",
    "explainability",
    "interpretability",
    "model explanations",
    "feature contributions",
    "feature importance",
    "attribution",
    "scikit-learn",
    "xgboost",
    "machine learning",
    "ml",
    "inference",
    "base value",
    "expected value",
]

# Catalog-qualified SQL fragments selecting a committed model BLOB / feature
# relation from the repo's test fixtures. The model BLOB cannot be inlined as a
# scalar subquery (DuckDB forbids subquery arguments to table functions), so it is
# held in a session VARIABLE and read with getvariable('m') — the idiomatic
# pattern shown throughout the docs.
_RF_MODEL = "(SELECT model FROM read_parquet('test/fixtures/models.parquet') WHERE name = 'rf_clf')"
_FEATURES = "(SELECT id, signal, noise_a, noise_b FROM read_parquet('test/fixtures/features.parquet'))"

_SET_M = f"SET VARIABLE m = {_RF_MODEL}"

# VGI515: schema-level example queries as a JSON list of {description, sql} objects
# (each description non-empty). The model BLOB cannot be inlined (DuckDB forbids
# subquery arguments to table functions), so every example opens with the same
# `SET VARIABLE m := …` step that loads the committed fixture model, then the query
# — run in order against one connection, so each executes cleanly under --execute.
_SCHEMA_EXAMPLE_QUERIES = json.dumps(
    [
        {
            "description": "Read the model's baseline output (one row per class for a binary classifier).",
            "sql": [
                _SET_M,
                "SELECT class, base_value FROM explain.main.shap_base_value(model := getvariable('m')) ORDER BY class",
            ],
        },
        {
            "description": "Explain each row's prediction feature by feature (long-format SHAP contributions).",
            "sql": [
                _SET_M,
                f"SELECT id, feature, class, shap_value FROM explain.main.shap_values({_FEATURES}, "
                "model := getvariable('m'), id := 'id') ORDER BY id, feature, class LIMIT 6",
            ],
        },
        {
            "description": "Rank the features by their dataset-wide (global) SHAP importance.",
            "sql": [
                _SET_M,
                f"SELECT feature, rank, method FROM explain.main.feature_importance({_FEATURES}, "
                "model := getvariable('m'), id := 'id') ORDER BY rank",
            ],
        },
    ]
)

# VGI413/VGI408: the schema's ordered category registry (array order = display
# order). Every function carries a `vgi.category` naming one of these slugs (wired
# up in functions.py), so no category is left empty (VGI411/VGI412).
_CATEGORIES_JSON = json.dumps(
    [
        {
            "name": "local-explanations",
            "title": "Local Explanations",
            "description": "Per-row, per-feature SHAP contributions explaining why the model predicted each row.",
        },
        {
            "name": "base-value",
            "title": "Base Value",
            "description": "The model's expected (baseline) output that per-feature contributions add onto.",
        },
        {
            "name": "global-importance",
            "title": "Global Importance",
            "description": "Dataset-wide feature importance ranked by mean absolute SHAP value (or native).",
        },
        {
            "name": "reference",
            "title": "Reference",
            "description": "Browsable, model-free reference data about the worker's supported model families.",
        },
    ]
)

# VGI152/VGI407: the agent-suitability suite (`vgi-lint simulate`). Each task is a
# public {name, prompt} object. Prompts are natural-language and self-contained,
# naming the committed fixtures the SQL E2E suite uses so an analyst can load the
# model exactly as the docs describe. VGI416: the grader-only fields
# (`reference_sql`, `ignore_column_names`) live in the private
# `vgi-agent-tests.yaml` sidecar at the repo root, never in metadata. Every answer
# is a stable string / integer, so grading compares VALUES. Facts are pinned by
# test/sql/*.test: rf_clf is a binary classifier (classes 0/1), features are
# signal/noise_a/noise_b, and `signal` is the informative feature (rank 1).
_AGENT_TEST_TASKS_JSON = json.dumps(
    [
        {
            "name": "supported-model-families",
            "prompt": (
                "This worker publishes a browsable, model-free reference of the model families it can "
                "explain. Using that reference, return the names of every supported model family, in "
                "alphabetical order."
            ),
        },
        {
            "name": "top-feature",
            "prompt": (
                "A fitted RandomForest classifier is stored as a model BLOB in the committed fixture "
                "'test/fixtures/models.parquet' (the row WHERE name = 'rf_clf'), and a matching feature "
                "matrix (columns id, signal, noise_a, noise_b) is in 'test/fixtures/features.parquet'. Load "
                "the model into a session variable, then rank the model's features by global SHAP importance "
                "and return only the name of the single most important feature."
            ),
        },
        {
            "name": "base-value-class-count",
            "prompt": (
                "The model BLOB in fixture 'test/fixtures/models.parquet' (WHERE name = 'rf_clf') is a binary "
                "classifier. Load it into a session variable and return how many rows (one per class) its SHAP "
                "base-value output has."
            ),
        },
        {
            "name": "shap-feature-names",
            "prompt": (
                "Load the model BLOB from 'test/fixtures/models.parquet' (WHERE name = 'rf_clf') into a session "
                "variable and compute per-row SHAP contributions for the feature rows in "
                "'test/fixtures/features.parquet' (columns id, signal, noise_a, noise_b). Return the distinct "
                "feature names that appear in the output, in alphabetical order."
            ),
        },
        {
            "name": "shap-class-labels",
            "prompt": (
                "Using the binary-classifier model BLOB in 'test/fixtures/models.parquet' (WHERE name = "
                "'rf_clf') and the feature rows in 'test/fixtures/features.parquet' (columns id, signal, "
                "noise_a, noise_b), return the distinct class labels present in the per-row SHAP output, "
                "ordered."
            ),
        },
        {
            "name": "importance-method",
            "prompt": (
                "Load the model BLOB from 'test/fixtures/models.parquet' (WHERE name = 'rf_clf') into a session "
                "variable and compute global feature importance over the non-empty feature relation in "
                "'test/fixtures/features.parquet' (columns id, signal, noise_a, noise_b). Return the single "
                "method label the function reports for how importance was computed."
            ),
        },
    ]
)

_CATALOG_TAGS = {
    "vgi.title": "SHAP Model Explanations",
    "vgi.keywords": json.dumps(_CATALOG_KEYWORDS),
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.source_url": SOURCE_URL,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-explain/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-explain/blob/main/README.md",
    # VGI152/VGI407: agent-suitability suite (public {name, prompt}; graders in vgi-agent-tests.yaml).
    "vgi.agent_test_tasks": _AGENT_TEST_TASKS_JSON,
}

_SCHEMA_TAGS = {
    "vgi.title": "Explain — SHAP Functions",
    "vgi.keywords": json.dumps(
        [
            "shap",
            "shap_values",
            "shap_base_value",
            "feature_importance",
            "explainability",
            "interpretability",
            "feature contributions",
            "attribution",
            "scikit-learn",
            "xgboost",
        ]
    ),
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    # VGI413/VGI408: the schema's ordered category registry.
    "vgi.categories": _CATEGORIES_JSON,
    "vgi.example_queries": _SCHEMA_EXAMPLE_QUERIES,
    # VGI123 classifying tags use BARE keys (NOT vgi.-namespaced).
    "domain": "machine-learning",
    "category": "explainability",
    "topic": "shap-model-explanations",
}


# VGI146: a table-function-only worker gives an agent nothing to SELECT before it
# knows a function's arguments (and every function here needs a model BLOB). Expose
# one model-free, credential-free REFERENCE view an agent can browse cold to learn
# which model families this worker can explain and which SHAP explainer each uses.
# This is genuine reference data (not a re-catalogue of the worker's own functions,
# which VGI327 forbids), built from an inline VALUES list so it scans with no model
# and no network. Rows mirror `vgi_explain.explain.explainer_kind`.
_SUPPORTED_MODEL_ROWS = [
    (
        "Tree ensembles & trees",
        "RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, DecisionTree, "
        "XGBoost (XGB*), LightGBM (LGBM*), CatBoost",
        "TreeExplainer",
        "exact",
        "Exact, fast Shapley values read straight from the tree structure; needs no background sample.",
    ),
    (
        "Linear models",
        "LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier, SGDRegressor",
        "LinearExplainer",
        "exact",
        "Closed-form attributions from the fitted coefficients over a small background sample.",
    ),
    (
        "Other estimators (fallback)",
        "Any fitted estimator exposing predict / predict_proba (e.g. SVM, KNN, MLP)",
        "KernelExplainer",
        "approximate",
        "Model-agnostic sampling fallback: slower and approximate, estimated over a sampled background.",
    ),
]

_SUPPORTED_MODELS_VALUES = ",\n    ".join(
    "('{}', '{}', '{}', '{}', '{}')".format(
        family.replace("'", "''"),
        estimators.replace("'", "''"),
        explainer.replace("'", "''"),
        exactness.replace("'", "''"),
        notes.replace("'", "''"),
    )
    for family, estimators, explainer, exactness, notes in _SUPPORTED_MODEL_ROWS
)

_SUPPORTED_MODELS_VIEW_SQL = (
    "SELECT model_family, estimator_examples, shap_explainer, exactness, notes FROM (VALUES\n"
    f"    {_SUPPORTED_MODELS_VALUES}\n"
    ") AS t(model_family, estimator_examples, shap_explainer, exactness, notes)"
)

_SUPPORTED_MODELS_VIEW = View(
    name="supported_models",
    definition=_SUPPORTED_MODELS_VIEW_SQL,
    comment="Model families this worker can explain and the SHAP explainer chosen for each.",
    column_comments={
        "model_family": "Family of fitted estimator (tree ensemble, linear, or fallback).",
        "estimator_examples": "Example scikit-learn / XGBoost estimator classes in this family.",
        "shap_explainer": "The SHAP explainer this worker selects for the family.",
        "exactness": "Whether the explanation is `exact` or an `approximate` (sampled) estimate.",
        "notes": "How the explainer works and any caveats for the family.",
    },
    tags={
        "vgi.title": "Supported Model Families",
        "vgi.doc_llm": (
            "A zero-argument, model-free **reference view** of the model families this worker can "
            "explain and the SHAP explainer it picks for each: tree ensembles / trees use "
            "`TreeExplainer` (exact), linear models use `LinearExplainer` (exact), and every other "
            "fitted estimator falls back to `KernelExplainer` (approximate). Query it cold — before "
            "you have a model in hand — to check whether an estimator is supported and how exact its "
            "explanations will be. Columns: `model_family`, `estimator_examples`, `shap_explainer`, "
            "`exactness`, `notes`."
        ),
        "vgi.doc_md": (
            "# supported_models\n\n"
            "A browsable, model-free reference of the estimator families this worker explains and the "
            "SHAP explainer selected for each. It takes no arguments and needs no model, so it is the "
            "cheapest way to check whether your estimator is supported before loading a model.\n\n"
            "## Columns\n\n"
            "- `model_family` (`VARCHAR`) -- tree ensemble, linear, or fallback family.\n"
            "- `estimator_examples` (`VARCHAR`) -- example estimator classes in the family.\n"
            "- `shap_explainer` (`VARCHAR`) -- the SHAP explainer chosen (`TreeExplainer` / "
            "`LinearExplainer` / `KernelExplainer`).\n"
            "- `exactness` (`VARCHAR`) -- `exact` or `approximate`.\n"
            "- `notes` (`VARCHAR`) -- how the explainer works and any caveats."
        ),
        "vgi.keywords": json.dumps(
            [
                "supported models",
                "model families",
                "explainer",
                "tree explainer",
                "linear explainer",
                "kernel explainer",
                "reference",
                "compatibility",
                "shap",
            ]
        ),
        # VGI411: assign the view to a category from the schema's vgi.categories registry.
        "vgi.category": "reference",
        # VGI123: bare (non-namespaced) classifying tags, matching the schema's.
        "domain": "machine-learning",
        "topic": "shap-supported-models",
        # VGI501/VGI511/VGI514/VGI151: fully-qualified, projected, model-free example queries that
        # scan the inline reference (no model, no network), so they run cleanly under --execute.
        "vgi.example_queries": json.dumps(
            [
                {
                    "description": "List every supported model family and the SHAP explainer chosen for it.",
                    "sql": (
                        "SELECT model_family, shap_explainer, exactness FROM explain.main.supported_models "
                        "ORDER BY model_family"
                    ),
                },
                {
                    "description": "Find which families get exact (rather than approximate) SHAP explanations.",
                    "sql": (
                        "SELECT model_family, shap_explainer FROM explain.main.supported_models "
                        "WHERE exactness = 'exact' ORDER BY model_family"
                    ),
                },
                {
                    "description": "Count the supported model families grouped by their SHAP explainer.",
                    "sql": (
                        "SELECT shap_explainer, count(*) AS family_count FROM explain.main.supported_models "
                        "GROUP BY shap_explainer ORDER BY shap_explainer"
                    ),
                },
            ]
        ),
    },
)


_EXPLAIN_CATALOG = Catalog(
    name="explain",
    default_schema="main",
    comment="SHAP explanations for scikit-learn / XGBoost models as SQL functions.",
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            path=["main"],
            comment="SHAP explanations for scikit-learn / XGBoost models as SQL functions",
            tags=_SCHEMA_TAGS,
            functions=list(EXPLAIN_FUNCTIONS),
            views=[_SUPPORTED_MODELS_VIEW],
        ),
    ],
)


class ExplainCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _EXPLAIN_CATALOG
    catalog_name = _EXPLAIN_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        """Advertise the explain catalog and its data/implementation versions."""
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=GIT_COMMIT,
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        """Resolve ATTACH, stamping the data and implementation versions."""
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=GIT_COMMIT,
        )


class ExplainWorker(Worker):
    """Worker process hosting the explain catalog."""

    catalog = _EXPLAIN_CATALOG
    catalog_interface = ExplainCatalog


def main() -> None:
    """Run the worker (stdio by default; pass ``--http`` for the HTTP server)."""
    ExplainWorker.main()


def main_http() -> None:
    """Run the worker over HTTP (injects ``--http`` into the worker CLI)."""
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    ExplainWorker.main()
