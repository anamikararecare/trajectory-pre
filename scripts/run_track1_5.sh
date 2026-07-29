#!/usr/bin/env bash
# Build Track 1.5 RSMs from the completed preliminary Track 1 run.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${TRACK1_5_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$TRACK1_5_PYTHON_BIN"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "traj" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif [[ -x /mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python ]]; then
    PYTHON_BIN=/mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python
else
    echo "Activate the traj environment or set TRACK1_5_PYTHON_BIN." >&2
    exit 2
fi

REPLAY_DIR="${TRACK1_5_REPLAY_DIR:-data/track1/replayed_activations/full_track1_20260721T090331Z}"
TURN_VARIABLES="${TRACK1_5_TURN_VARIABLES:-results/track1/preliminary/full_track1_20260721T090331Z/turn_variables.csv}"
OUT_DIR="${TRACK1_5_OUT_DIR:-results/track1_5}"

"$PYTHON_BIN" -m src.track1_5_rsm.run_track1_5 \
    --replay_dir "$REPLAY_DIR" \
    --turn_variables "$TURN_VARIABLES" \
    --out_dir "$OUT_DIR" \
    --n_layers 4 \
    --turns_per_model 10 \
    --allow_unvalidated_models \
    "$@"

