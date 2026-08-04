#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$APP_DIR/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
# Explicit CLI values ensure the limit is applied regardless of the directory from which run.sh is called.
exec streamlit run "$APP_DIR/app.py" \
  --server.maxUploadSize=2048 \
  --server.maxMessageSize=2048 \
  "$@"
