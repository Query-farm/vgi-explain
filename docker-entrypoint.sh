#!/bin/sh
# Dispatch the vgi-explain image into one of its transports:
#   http   (default) HTTP server on $PORT (vgi-explain-http: /health + VGI RPC)
#   stdio            a worker DuckDB spawns over stdio (on-host execution)
#   *                exec'd verbatim (debug escape hatch)
set -e
case "${1:-http}" in
  http)  exec vgi-explain-http --host 0.0.0.0 --port "${PORT:-8000}" ;;
  stdio) shift 2>/dev/null || true; exec vgi-explain "$@" ;;
  *)     exec "$@" ;;
esac
