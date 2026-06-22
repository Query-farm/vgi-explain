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
#   uv tool install haybarn-unittest   # installs ~/.local/bin/haybarn-unittest

# Worker command DuckDB uses for ATTACH (overridable).
WORKER_STDIO    ?= uv run --no-sync explain_worker.py

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
		haybarn-unittest --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

fixture:
	uv run --no-sync python scripts/make_fixture.py

lint:
	uv run --no-sync ruff check .
	uv run --no-sync mypy vgi_explain/
