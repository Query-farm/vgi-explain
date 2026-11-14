# SHAP: one prediction and a comparison cohort

[Open the executed notebook](01_local_and_cohort_shap.ipynb).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Query-farm/vgi-explain/blob/main/notebooks/01_local_and_cohort_shap.ipynb)

Visible SQL checks the supported model family, trains a synthetic regression forest
with the released scikit-learn worker, passes its model BLOB to vgi-explain, and
checks SHAP additivity against predictions. A signed local waterfall and a cohort
mean-absolute comparison answer distinct questions in minutes.

In Colab, choose **Runtime → Run all**. The notebook installs the shared base package;
no worker checkout, companion files, API key, Drive mount or saved model is needed.
Compatible pandas, Matplotlib, IPython and kernel libraries are retained. If an older
setup installed incompatible packages, save your work and choose **Runtime → Disconnect
and delete runtime**, reconnect, then run the updated notebook.

## Run locally

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run --locked vgi-notebooks --timeout 300
uv run --locked vgi-notebooks --timeout 300 --parameters '{"RAIN_INDEX":1.0}' --output-dir artifacts/rain-1
```

On macOS, XGBoost's wheel requires OpenMP even though this example fits a sklearn
forest: install `libomp` with Homebrew if it is absent. The worker commands use
`uvx --python 3.13` and clean, literal Git release pins:
`Query-farm/vgi-scikit-learn@v0.3.0` and `Query-farm/vgi-explain@v0.4.0`.
Their releases own the model-library and serializer dependencies. The notebook
verifies the complete fit/predict/explain roundtrip; update the two release pins
together and repeat that verification when upgrading. Initial downloads and SHAP
imports can take several minutes; no SQL engine build is needed.

The local lock uses the `[runner]` extra of
[vgi-example-notebooks v0.3.6](https://github.com/Query-farm/vgi-example-notebooks),
which owns the [standard](https://github.com/Query-farm/vgi-example-notebooks/blob/v0.3.6/docs/standard.md),
runner, branding, SVG presentation and coverage checks. The Colab setup uses only
its base package. The shared runner saves successful defaults, requires a separate
destination for parameter overrides, and publishes SVG image assets under `charts/`
for GitHub alongside native SVG notebook output. Review PNGs and experiments are
ignored artifacts. This directory has no copied standard or custom runner.

To register an optional local kernel:

```sh
uv run --locked python -m ipykernel install --user --name explain-examples --display-name 'Python (SHAP examples)'
```

## The experiment and its limits

The default is `RAIN_INDEX = 0.0`; only 0.0 and 1.0 are accepted. The explained trip
has 16 km distance and two stops. Changing rain to 1.0 changes only that trip,
preserving the training grid, fitted settings and 32-row cohort. Compare the
prediction and signed contributions; the reference and cohort importance remain
unchanged. Contributions for several features can move because tree interactions
redistribute credit. Restart and run all after a parameter change.

The notebook constructs all data in SQL. Its 90 training combinations use the
invented target `12 + 3*distance_km + 2*stops + 6*rain_index +
0.4*distance_km*rain_index` in minutes. The 32-row cohort uses intermediate distances
and stop counts. Neither represents actual trips. The forest has 64 trees, depth
six, seed seven and one training thread. Its in-sample score is not validation.
No persistent model name is supplied: the returned BLOB stays in memory.

Each worker calculation is materialized before DataFrame capture so display cannot
repeat training. Predictions and contributions join by explicit input ID. The
notebook's small inline DBAPI adapter reuses one native Haybarn session because
JupySQL otherwise opens cursor scopes that lose `SET VARIABLE` values. It changes
neither SQL nor worker results. Shared connection setup uses an empty temporary SQL
secret directory with persistent secrets disabled. SQL checks every reconstruction,
recomputes cohort importance from long SHAP output, and exercises name-based feature
alignment through a reordered input projection.

The `supported_models` view explains why this RandomForest uses `TreeExplainer`.
Its `exact` label concerns model attributions, not forecast accuracy. This regression
forest's predictions, base and SHAP contributions are in minutes. Without an explicit
background, TreeExplainer uses the fitted trees' training-path counts. The base is a
reference expectation, not necessarily a prediction for an average or zero-valued
trip. See the [SHAP tree documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
Other model families can have different background semantics and output units.

Mean absolute SHAP describes only the chosen cohort. A local contribution is not
a causal effect or the prediction change from editing just that feature. The rain
experiment is a model what-if; additivity establishes neither forecasting quality
nor causal validity.

## Coverage and verification

[coverage.json](coverage.json) maps **all four public SQL surfaces** to meaningful
visible queries, returned results, interpretation and successful execution receipts:
`shap_values`, `shap_base_value`, `feature_importance` and the queryable
`supported_models` reference view. Complete native function, table and view exports
are retained under `coverage/`. Views count as logical table surfaces; overloads
and same-name aliases count once. Both released v0.4.0 and public source baseline
`1a3b0b80c28aba31133bc1fd1096d204d11b1b75` have the same four names. Example execution
uses the published release; current-interface compatibility is reviewed separately.

The saved default and a detached notebook-only rain-state experiment pass all nine
SQL checks. Rain 0→1 changes this trip’s prediction from 64.79 to 76.70 minutes;
the 54.79-minute reference and all three cohort importance values stay fixed.
Both runs’ charts were reviewed. Three runner guards preserve defaults, and an
invalid rain value is rejected before worker attachment. A fresh Python 3.12
kernel with pandas 2.2.3, Matplotlib 3.10.0, IPython 8.30.0, ipykernel 6.17.1 and
jupyter-client 7.4.9 preloaded executes the complete notebook with the base
package only, retains those versions and passes `pip check`. The shared runner rejects saved tracebacks,
unexpected warnings and machine-specific paths. A hosted Colab UI session has not
been manually exercised; fresh notebook-only and preloaded-kernel checks are
reported separately from that UI.

Source checks pass 32 pytest tests, 79 SQL assertions, Ruff, formatting, mypy and
the CI-isolated pydoclint command. The root-locked pydoclint environment has an
existing `DocstringYields` import mismatch. Current metadata lint scores 100/100
but reports one existing VGI417 size warning on `explain.main`, so its strict
warning gate remains failing. Worker source and root dependencies are unchanged.
