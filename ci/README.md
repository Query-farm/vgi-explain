# CI: the vgi-explain worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-explain
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` (the DuckDB/Haybarn sqllogictest
runner, published in Haybarn's releases) and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen` into a venv. `.venv/bin/vgi-explain`
   is a self-contained PEP 723 stdio worker the extension can spawn via
   `uv run .venv/bin/vgi-explain`.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — the standalone runner links none of the extensions the
   tests gate on, so [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL <ext> FROM
   {community,core}; LOAD <ext>;`. These tests skip `require vgi` (haybarn
   silently SKIPs it) and `LOAD vgi;` directly, so the awk also injects an
   `INSTALL vgi FROM community;` right before each bare `LOAD vgi;`. `require-env`
   and everything else pass through untouched.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, resolves `VGI_EXPLAIN_WORKER` per the selected transport (below), warms
   the extension cache once, then runs the suite in a single `haybarn-unittest`
   invocation. Any failed assertion exits non-zero and fails the job.

## Transport matrix (subprocess / http / unix)

The same suite runs over all three VGI transports — the vgi extension picks the
transport from the ATTACH `LOCATION` string that `run-integration.sh` builds per
the `TRANSPORT` env var (`subprocess` | `http` | `unix`, default `subprocess`):

- **subprocess** — `VGI_EXPLAIN_WORKER` = the stdio command (`.venv/bin/vgi-explain`).
  The extension spawns the worker per query and talks Arrow IPC over stdin/stdout.
- **http** — the script boots `vgi-explain --http --port 0 --port-file <f>` (cwd =
  the stage dir), polls the port-file for the auto-selected port, and sets
  `VGI_EXPLAIN_WORKER=http://127.0.0.1:<port>` (bare scheme://host:port, no path).
  HTTP needs waitress (pulled in by the `vgi-python[http]` main dependency) and,
  on the DuckDB side, the **httpfs** extension — without it `ATTACH 'http://…'`
  fails with "VGI HTTP transport requires the httpfs extension" (which the runner
  would then *silently skip*). The script injects `INSTALL httpfs FROM core; LOAD
  httpfs;` after each `LOAD vgi;` for the http leg only.
- **unix** — the script boots `vgi-explain --unix <sock>` (cwd = the stage dir),
  polls for the socket file, and sets `VGI_EXPLAIN_WORKER=unix://<sock>`.

The run step **guards against the silent-skip fake-pass**: the sqllogictest
runner SKIPS (exit 0) any test whose error contains "HTTP"/"Unable to connect",
so a broken http setup would report "All tests were skipped" and go green having
run nothing. The script fails the leg if it sees that, surfacing the skip reason.

All three streaming/buffering functions (`shap_values` table-in-out,
`shap_base_value` source, `feature_importance` buffering) work over the stateless
HTTP transport unchanged: each drains its rows within a single `process`/
`finalize` tick and `feature_importance`'s `DrainState` already extends
`ArrowSerializableDataclass`, so the SDK round-trips state through the
continuation token correctly. No tests are gated.

The CI `integration` job is a matrix of `transport: [subprocess, http, unix]` ×
`os: [ubuntu-latest, macos-latest]`.

## Run it locally

```bash
uv sync --python 3.13                       # install the worker + deps
# point HAYBARN_UNITTEST at a haybarn-unittest binary (or a local DuckDB
# `unittest` built with the vgi extension), and the worker at the stdio command:
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
VGI_EXPLAIN_WORKER="uv run --python 3.13 .venv/bin/vgi-explain" \
  ci/run-integration.sh
```

Or use the Makefile target `make test-sql`, which installs `haybarn-unittest`
as a uv tool and points the worker at `uv run --python 3.13 .venv/bin/vgi-explain`.
