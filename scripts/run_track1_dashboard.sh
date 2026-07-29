#!/usr/bin/env bash
# Launch the read-only Track 1 dashboard inside the current shell/tmux pane.
# Usage: bash scripts/run_track1_dashboard.sh [port]
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DASHBOARD_PORT="${1:-8501}"
if [[ ! "$DASHBOARD_PORT" =~ ^[0-9]+$ ]] ||
   (( DASHBOARD_PORT < 1 || DASHBOARD_PORT > 65535 )); then
    echo "Port must be an integer from 1 through 65535." >&2
    exit 2
fi

if [[ -n "${TRACK1_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$TRACK1_PYTHON_BIN"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "traj" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x /mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python ]]; then
    PYTHON_BIN=/mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python
else
    echo "Could not find the traj Python environment." >&2
    echo "Activate it first or set TRACK1_PYTHON_BIN=/absolute/path/to/python." >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c 'import plotly, streamlit' >/dev/null 2>&1; then
    echo "Dashboard dependencies are missing." >&2
    echo "Install them with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
    exit 2
fi

echo "Starting the read-only Track 1 dashboard"
echo "URL: http://${TRACK1_DASHBOARD_ADDRESS:-127.0.0.1}:$DASHBOARD_PORT"
echo "Stop with Ctrl-C. No replay, geometry, annotation, or probe job is run."

exec "$PYTHON_BIN" -m streamlit run src/track1_probing/dashboard.py \
    --server.address "${TRACK1_DASHBOARD_ADDRESS:-127.0.0.1}" \
    --server.port "$DASHBOARD_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
