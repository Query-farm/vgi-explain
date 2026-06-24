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
import logging
import os
import sys
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
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
    "DuckDB table functions. Does not train models — it interprets the self-contained model BLOB produced "
    "by the sibling vgi-sklearn / vgi-xgboost workers (passed as a scalar `model` argument, typically a "
    "session VARIABLE). Use to answer: why did the model predict this row? (`shap_values` — per-row, "
    "per-feature signed contributions in long format); what is the model's baseline output? "
    "(`shap_base_value` — the explainer expected value the contributions add onto); and which features "
    "matter most overall? (`feature_importance` — global mean(|SHAP|) over a relation, or native "
    "importances). Multi-class classifiers report a contribution per class."
)

_CATALOG_DESCRIPTION_MD = (
    "# explain\n\n"
    "SHAP model explanations for scikit-learn / XGBoost models, as DuckDB/SQL table functions.\n\n"
    "vgi-explain does **not** train models — it interprets the self-contained *model BLOB* produced by the "
    "sibling `vgi-sklearn` / `vgi-xgboost` workers. The model travels as a scalar `model` argument (a DuckDB "
    "table function allows only one subquery, reserved for the feature relation), so it is usually held in a "
    "session variable:\n\n"
    "```sql\n"
    "SET VARIABLE m = (SELECT model FROM sklearn.fit(...));\n"
    "SELECT * FROM explain.shap_values((SELECT id, f1, f2 FROM data), model := getvariable('m'), id := 'id');\n"
    "```\n\n"
    "Functions: `shap_values` (per-row, per-feature contributions), `shap_base_value` (expected value), "
    "`feature_importance` (global ranking)."
)

_SCHEMA_DESCRIPTION_LLM = (
    "SHAP explanation table functions for fitted scikit-learn / XGBoost models: per-row, per-feature "
    "contributions (`shap_values`), the model's base/expected value (`shap_base_value`), and global feature "
    "importance (`feature_importance`). Each takes a model BLOB as the scalar `model` argument."
)

_SCHEMA_DESCRIPTION_MD = (
    "SHAP explanation functions over Apache Arrow: `shap_values`, `shap_base_value`, `feature_importance`. "
    "Each interprets a model BLOB packed by vgi-sklearn / vgi-xgboost."
)

_CATALOG_TAGS = {
    "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.description_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query.Farm",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": "https://github.com/Query-farm/vgi-explain/issues",
    "vgi.support_policy_url": "https://github.com/Query-farm/vgi-explain/blob/main/README.md",
}

_SCHEMA_TAGS = {
    "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
}

_EXPLAIN_CATALOG = Catalog(
    name="explain",
    default_schema="main",
    comment="SHAP explanations for scikit-learn / XGBoost models as SQL functions.",
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            name="main",
            comment="SHAP explanations for scikit-learn / XGBoost models as SQL functions",
            tags=_SCHEMA_TAGS,
            functions=list(EXPLAIN_FUNCTIONS),
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
