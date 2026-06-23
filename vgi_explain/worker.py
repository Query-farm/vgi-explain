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
GIT_COMMIT = os.environ.get("VGI_EXPLAIN_GIT_COMMIT") or "unknown"

_EXPLAIN_CATALOG = Catalog(
    name="explain",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="SHAP explanations for scikit-learn / XGBoost models as SQL functions",
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
                data_version_spec=DATA_VERSION,
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
