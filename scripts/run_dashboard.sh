#!/usr/bin/env bash
# Launch the shared read-only Track 1 + Track 2 dashboard.
set -euo pipefail

PORT="${1:-8501}"
PYTHON_BIN="${DASHBOARD_PYTHON_BIN:-python}"
ADDRESS="${DASHBOARD_ADDRESS:-127.0.0.1}"

echo "Starting the read-only trajectory research dashboard"
echo "Open http://${ADDRESS}:${PORT}"
exec "$PYTHON_BIN" -m streamlit run src/track1_probing/dashboard.py \
    --server.port "$PORT" \
    --server.address "$ADDRESS"
