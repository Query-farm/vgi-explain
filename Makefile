# vgi-explain — dev and test targets.
#
# Usage:
#   make test          # unit/integration (pytest) + end-to-end SQL (haybarn-unittest)
#   make test-unit     # pytest only
#   make test-sql      # DuckDB sqllogictest .test files via haybarn-unittest
#   make fixture       # regenerate the committed model-BLOB parquet fixture
#
# test-sql is self-contained: it points VGI_EXPLAIN_WORKER at the worker run as a
# uv stdio subprocess (exactly how DuckDB drives it after ATTACH) and runs the
# files under test/sql/. haybarn-unittest is a uv tool:
#   uv tool install --prerelease allow 'haybarn-unittest>=2.0.0rc11,<2.1'   # installs ~/.local/bin/haybarn-unittest
# haybarn 2.0 is required: its community `vgi` extension is the one that speaks the
# worker's VGI protocol (vgi.v3 / 3.1). --keep-home lets the runner see the
# extensions installed under the real $HOME (haybarn-unittest 2.0 isolates it).

# Worker command DuckDB uses for ATTACH (overridable). Use the installed console
# script in the project venv (what CI attaches) rather than `uv run
# explain_worker.py`, whose PEP-723 script env can resolve a different SDK.
WORKER_STDIO    ?= .venv/bin/vgi-explain

# haybarn-unittest lives in the uv tools bin; keep it on PATH.
HAYBARN_BIN     ?= $(HOME)/.local/bin
TEST_DIR         = .
TEST_PATTERN     = test/sql/*

.PHONY: test test-unit test-sql lint fixture

test: test-unit test-sql

test-unit:
	uv run --no-sync pytest -q

test-sql:
	PATH="$(HAYBARN_BIN):$$PATH" \
		VGI_EXPLAIN_WORKER="$(WORKER_STDIO)" \
		haybarn-unittest --keep-home --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

fixture:
	uv run --no-sync python scripts/make_fixture.py

lint:
	uv run --no-sync ruff check .
	uv run --no-sync mypy vgi_explain/
