# CLAUDE.md — vgi-explain

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker that computes **SHAP explanations** for models
trained by sibling workers (`vgi-sklearn`, `vgi-xgboost`), as DuckDB table
functions in the `explain` catalog (single `main` schema). It does **not** train
models — it consumes the self-contained *model BLOB* those workers produce.
Backed by `shap` (MIT), `scikit-learn` (BSD), `xgboost` (Apache-2.0), `skops`
(MIT). Sibling style/tooling to `vgi-sklearn` / `vgi-conform`.

## Layout

```
explain_worker.py      repo-root stdio entry point; PEP 723 inline deps; main()
serve.py               repo-root HTTP entry shim (container deploys)
vgi_explain/
  registry.py          unpack the model BLOB (skops safe-load, trusted prefixes);
                       byte-compatible with vgi_sklearn.registry's pack/unpack
  explain.py           PURE SHAP logic (no Arrow/VGI): explainer selection,
                       (rows, features, classes) normalization, additivity, importance
  features.py          align an input relation to a model's fitted columns by name
  buffering.py         single-bucket sink/combine for the buffering function
  functions.py         the SQL surface: shap_values / shap_base_value / feature_importance
  schema_utils.py      pa.Field comment helper
  worker.py            assembles the `explain` catalog + entry points
tests/                 pytest: test_registry, test_explain (pure), test_functions
                       (in-proc harness), test_client (real Client RPC)
scripts/make_fixture.py   regenerates the committed model-BLOB parquet fixture
test/fixtures/*.parquet   committed model BLOB + feature matrix for the SQL E2E
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / fixture / lint
```

## The model BLOB (read first)

The BLOB is `4-byte big-endian metadata-JSON length || metadata JSON || skops
bytes` — exactly `vgi_sklearn.registry.pack_model`. `registry.unpack_model` here
replicates that and the skops trusted-load, **adding `xgboost.*` to the trusted
prefixes** so a BLOB from vgi-xgboost (which embeds an `xgboost.sklearn`
estimator) also loads. Garbage / truncated / untrusted BLOBs raise
`InvalidModelBlobError` / `UntrustedModelError`, surfaced to SQL as a clear
`invalid model: …` error — never a crash.

## Function shapes — THE core convention

A DuckDB table function takes **at most one subquery argument**, reserved for the
feature relation. The model is therefore a **scalar `model` arg**, read from a
session VARIABLE: `SET VARIABLE m = (SELECT model FROM …); … model :=
getvariable('m')`. This mirrors `vgi-sklearn`'s `predict` exactly.

- `shap_values`  — `TableInOutGenerator`: streams the feature relation, emits long
  rows. Features aligned by name (reorder-safe). Model cached per execution_id.
- `shap_base_value` — source `TableFunctionGenerator` (model BLOB only).
- `feature_importance` — buffering (`SinkBuffer`): needs all rows for
  `mean(|SHAP|)`; with **zero** buffered rows, falls back to native importances.

`explain.py` is deliberately Arrow-free so the SHAP math (additivity, multi-class
shapes, known-feature ranking) is unit-testable in isolation. shap/sklearn/xgboost
import at module load → expensive init happens **once** per worker process.

## SQL E2E gets its model BLOB from a committed fixture

vgi-explain has no `fit`, so there's no in-SQL way to mint a model. The SQL tests
`read_parquet('test/fixtures/models.parquet')` — a committed file produced by
`scripts/make_fixture.py` (trains a tiny RandomForest + XGBoost, packs with the
shared `pack_model`). Regenerate with `make fixture`.

## haybarn / SQL test gotchas

- `uv tool install haybarn-unittest`; `export PATH="$HOME/.local/bin:$PATH"`.
- `require vgi` SILENTLY SKIPS — use explicit `statement ok` + `LOAD vgi;`.
- Files `test/sql/*.test`, header `# name:` + `# group: [vgi_explain]`, GLOB.
- `require-env VGI_EXPLAIN_WORKER`; `ATTACH 'explain' AS explain (TYPE vgi,
  LOCATION '${VGI_EXPLAIN_WORKER}');`.
- Determinism: `ORDER BY` / `rowsort`; numeric asserts via `ROUND` / tolerance.
```
