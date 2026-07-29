#!/usr/bin/env bash
# Launch the focused preliminary Track 1 dashboard.
# Usage: bash scripts/run_track1_preliminary_dashboard.sh [port] [run_id]
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DASHBOARD_PORT="${1:-8503}"
export TRACK1_PRELIM_RUN_ID="${2:-${TRACK1_PRELIM_RUN_ID:-full_track1_20260721T090331Z}}"
if [[ ! "$DASHBOARD_PORT" =~ ^[0-9]+$ ]] ||
   (( DASHBOARD_PORT < 1 || DASHBOARD_PORT > 65535 )); then
    echo "Port must be an integer from 1 to 65535." >&2
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
    exit 2
fi

echo "Starting focused preliminary Track 1 dashboard"
echo "Run: $TRACK1_PRELIM_RUN_ID"
echo "URL: http://${TRACK1_DASHBOARD_ADDRESS:-127.0.0.1}:$DASHBOARD_PORT"
exec "$PYTHON_BIN" -m streamlit run src/track1_probing/preliminary_dashboard.py \
    --server.address "${TRACK1_DASHBOARD_ADDRESS:-127.0.0.1}" \
    --server.port "$DASHBOARD_PORT" \
    --server.headless true
