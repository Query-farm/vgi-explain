<p align="center">
  <img src="docs/vgi-logo.png" alt="Vector Gateway Interface (VGI)" width="320">
</p>

<p align="center"><em>A <a href="https://query.farm">Query.Farm</a> VGI worker for DuckDB.</em></p>

# vgi-explain

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Explain machine-learning predictions in pure SQL.** `vgi-explain` exposes
[SHAP](https://shap.readthedocs.io/) to DuckDB as ordinary SQL functions. Point
it at a model that [`vgi-sklearn`](https://github.com/Query-farm/vgi-scikit-learn)
or [`vgi-xgboost`](https://github.com/Query-farm/vgi-xgboost) trained, and get
per-row, per-feature contributions — why each prediction came out the way it did —
without leaving your query.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'sklearn' (TYPE vgi, LOCATION 'vgi-sklearn');
ATTACH 'explain' (TYPE vgi, LOCATION 'vgi-explain');   -- 'uv run explain_worker.py' from a checkout

-- train a model with vgi-sklearn (or vgi-xgboost), keep the model BLOB it returns
CREATE TABLE model AS
  SELECT model FROM sklearn.fit((SELECT * FROM sklearn.iris()),
    estimator := 'random_forest_classifier', target := 'target', id := 'sample_id');

-- a table function takes only one subquery (the feature matrix), so the model
-- travels as a scalar — read it from a session VARIABLE
SET VARIABLE m = (SELECT model FROM model);

-- per-row, per-feature SHAP contributions (long format)
SELECT * FROM explain.shap_values(
  (SELECT sample_id AS id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm
   FROM sklearn.iris()),
  model := getvariable('m'), id := 'id')
ORDER BY id, feature
LIMIT 10;
```

## How it works

`vgi-explain` does **not** train models. It consumes the self-contained *model
BLOB* that `vgi-sklearn` / `vgi-xgboost` produce — a single value holding the
fitted estimator plus its metadata (estimator type, ordered feature names,
classes). The BLOB is unpacked with [skops](https://skops.readthedocs.io/) safe
loading, restricted to the scikit-learn / numpy / scipy / xgboost namespaces, so
a malformed or hostile value yields a clear error rather than executing code.

Every function follows the same SQL-friendly contract as `vgi-sklearn`:

- **Your input table is the feature matrix**, passed as the single subquery
  argument. Features are matched to the model **by name**, so column order does
  not matter (a shuffled `SELECT` still works) and extra columns are ignored.
- **The model is a scalar `model` argument** — `model := getvariable('m')` — not
  a subquery (DuckDB allows a table function only one subquery argument).
- **`id`** names a column to exclude from the features and carry through to the
  output, so you can `JOIN` results back to the source.

The explainer is chosen automatically by model type:

| Model family                                                   | Explainer            |
| -------------------------------------------------------------- | -------------------- |
| trees: RandomForest, GradientBoosting, DecisionTree, XGB, LGBM | `TreeExplainer`      |
| linear: LinearRegression, Ridge, Lasso, LogisticRegression     | `LinearExplainer`    |
| anything else                                                  | `KernelExplainer`    |

## Functions

### `shap_values((SELECT id, f1, f2, …), model := <BLOB>, id := 'id')`

Long format: **one row per (input row × feature [× class])**.

| column       | type     | meaning                                                |
| ------------ | -------- | ------------------------------------------------------ |
| *id*         | (passthrough) | your id column (or a 0-based row index if omitted) |
| `feature`    | VARCHAR  | feature name                                           |
| `class`      | VARCHAR  | class label, or NULL for regression / binary           |
| `shap_value` | DOUBLE   | signed contribution toward the model output            |

**Additivity:** for any row, `sum(shap_value) + base_value ≈ model output` (the
raw margin / score the model produces for that row).

### `shap_base_value(model := <BLOB>)`

The explainer's expected value — the anchor SHAP values add onto. One row, or
one row per class for a multi-class classifier (`class`, `base_value`).

### `feature_importance((SELECT …), model := <BLOB>, id := 'id')`

Global importance ranked descending. With a non-empty relation it returns
**`mean(|SHAP|)` across rows** (`method = 'mean_abs_shap'`); with an empty
relation it falls back to the model's **native importances**
(`feature_importances_` or `|coef_|`, `method = 'native'`). Columns: `feature`,
`importance`, `rank`, `method`.

## Multi-class behavior

For a classifier with **K > 2** classes, SHAP produces a contribution per class.
`shap_values` emits a `class` column carrying the class label, so every
`(row, feature, class)` contribution is addressable, and `shap_base_value`
returns one row per class. Regression and **binary** classification collapse to a
single output: `class` is `NULL`, one contribution per `(row, feature)`.

## Development

```sh
uv sync --extra dev
uv run --no-sync pytest -q          # unit + in-process + real-client tests
make test-sql                       # DuckDB sqllogictest E2E (haybarn-unittest)
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_explain/
make fixture                        # regenerate the committed model-BLOB fixture
```

## Licensing

- `vgi-explain`: MIT.
- [SHAP](https://github.com/shap/shap): MIT.
- [scikit-learn](https://scikit-learn.org/): BSD-3-Clause.
- [XGBoost](https://github.com/dmlc/xgboost): Apache-2.0.
- [skops](https://github.com/skops-dev/skops): MIT.
- [numpy](https://numpy.org/): BSD-3-Clause.

---

## Authorship & License

Written by [Query.Farm](https://query.farm) — every VGI worker is designed and built by Query.Farm.

Copyright 2026 Query Farm LLC - https://query.farm

