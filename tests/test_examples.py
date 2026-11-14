"""Metadata wiring for the example queries (SQL defined once, shown where agents see it)."""

from __future__ import annotations

import json

from vgi_explain import functions as fn
from vgi_explain.worker import _SCHEMA_TAGS

COMPOSED = [fn.RECONSTRUCT_PROBA_EXAMPLE, fn.GLOBAL_VS_LOCAL_EXAMPLE, fn.TOP_FEATURE_DRILLDOWN_EXAMPLE]


def _examples(cls: type) -> list[dict[str, object]]:
    tags: dict[str, str] = cls.Meta.tags  # type: ignore[attr-defined]
    return list(json.loads(tags["vgi.example_queries"]))


def test_schema_lists_every_composed_example() -> None:
    schema = json.loads(_SCHEMA_TAGS["vgi.example_queries"])
    for ex in COMPOSED:
        assert ex in schema


def test_each_composed_example_is_in_a_function_top_five() -> None:
    shown = [ex for cls in fn.EXPLAIN_FUNCTIONS for ex in _examples(cls)[:5]]
    for ex in COMPOSED:
        assert fn.as_example_query(ex) in shown, ex["description"]


def test_function_example_queries_are_single_sql_strings() -> None:
    # Only executable examples / the schema list accept a statement list; a list in a
    # function's vgi.example_queries reaches the engine as its Python repr and fails to bind.
    for cls in fn.EXPLAIN_FUNCTIONS:
        for ex in _examples(cls):
            assert isinstance(ex["sql"], str) and ex["sql"].startswith("SET VARIABLE m = ")


def test_composed_examples_call_two_functions() -> None:
    names = ("shap_values(", "shap_base_value(", "feature_importance(")
    for ex in COMPOSED:
        query = ex["sql"][-1]
        assert sum(f"explain.main.{n}" in query for n in names) >= 2, ex["description"]
