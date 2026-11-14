# SHAP: one prediction and a comparison cohort

[Open the executed notebook](01_local_and_cohort_shap.ipynb). Visible SQL creates
synthetic inputs, trains a forest with the real scikit-learn worker, passes its
model BLOB to vgi-explain, and checks SHAP additivity against actual predictions.
Two charts compare a signed local waterfall with cohort mean absolute SHAP.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/). On macOS, the explanation worker imports
XGBoost, whose wheel requires the OpenMP runtime even though this example fits a
scikit-learn forest:

```bash
brew install libomp  # macOS prerequisite, if not already installed
uv sync --frozen
uv run python execute.py
```

The locked notebook environment uses Python 3.13–3.14 and Haybarn 2.0.0rc11.
The visible ATTACH commands use `uvx --python 3.13` and Git releases
`Query-farm/vgi-scikit-learn@v0.3.0` and `Query-farm/vgi-explain@v0.4.0`.
Both pin `vgi-python==0.51.1`, `scikit-learn==1.9.1` and `skops==0.16.0`;
the explainer also pins `shap==0.54.1`. Matching serializer and model-library
versions are deliberate. The commands are literal SQL, not checkout paths.

The first launch downloads dependencies and imports SHAP; allow a few minutes.
The runner allows 300 seconds per cell. No local worker build, worker source
checkout, separate training environment, API key or registered kernel is needed.
Initial setup needs internet access to GitHub, package indexes and community
extensions; the analysis uses only synthetic local data.

For an editor, register and select this notebook environment:

```bash
uv run python -m ipykernel install --user --name explain-examples \
  --display-name "Python (vgi-explain notebooks)"
```

Restart and run all after changing a parameter. SQL and parameters stay visible;
presentation code is folded. This directory runs after being copied outside the
repository and includes all required helpers and branding.

## Experiment

```bash
uv run python execute.py --parameters '{"RAIN_INDEX":1.0}' \
  --output-dir validation/rain-1
```

The default is `RAIN_INDEX = 0.0`; only 0.0 and 1.0 are accepted. The explained
trip has 16 km distance and two stops. The experiment changes only its artificial
rain state, preserving the training grid, forest settings and 32-row cohort.
Compare the prediction and signed contributions; the reference and cohort
importance should remain unchanged. Local credit can change for several features
because tree interactions redistribute it. Restore 0.0 for the default.

The runner starts a fresh kernel, saves only after success, and requires a separate
destination for parameter overrides. Experiment copies include execution and
branding companions. Validation logs and extracted charts are ignored.

## Synthetic provenance and interpretation

The notebook constructs its entire dataset in SQL: 90 training combinations use
the invented target `12 + 3*distance_km + 2*stops + 6*rain_index +
0.4*distance_km*rain_index` in minutes. The 32-row comparison cohort uses intermediate
distances and stop counts. Neither represents actual trips. The forest uses 64
trees, depth six, seed seven and one training thread. Its in-sample score is not a
validation result. No persistent model name is supplied; the BLOB stays in memory.

Training and each worker calculation are materialized with `CREATE TABLE` before
local DataFrame captures, preventing display from repeating training. Predictions
and contributions join by explicit input ID. A small DBAPI adapter reuses one
native Haybarn session because JupySQL otherwise opens separate cursor scopes
and loses `SET VARIABLE` values; it does not alter SQL or worker results. SQL checks every row's reconstruction
and recomputes cohort importance from the long SHAP output. A reordered feature
projection exercises name-based alignment.

For this regression forest, output and SHAP values have units of minutes. The
worker's TreeExplainer has no explicit background and uses the trees' training
path counts. Its base is a reference expectation, not necessarily the prediction
of an average or zero-valued trip. See the [SHAP tree documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
Other model families can have different background semantics or output units;
this notebook does not generalize its reconstruction formula to those cases.

Mean absolute SHAP describes only the chosen cohort. A local contribution is not
a causal effect or the prediction change from editing just that feature. The
experiment is a model what-if, and neither it nor additivity proves forecasting
quality. To upgrade, revise the two literal Git release pins and compatible model
dependency pins together, then verify both runs. [STANDARD.md](STANDARD.md) records
the notebook conventions.
