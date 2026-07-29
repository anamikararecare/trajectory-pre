#!/usr/bin/env bash
# Overnight frozen-corpus Track 1 replay and analysis.
# Run inside an existing tmux session:
#   bash scripts/run_track1_refactored.sh
# Optional stable run ID:
#   bash scripts/run_track1_refactored.sh my_run_id
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRACK1_RUN_ID="${1:-full_track1_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$TRACK1_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Run ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi

TRACK1_DATA_DIR="${TRACK1_DATA_DIR:-data/track1}"
TRACK1_RESULTS_DIR="${TRACK1_RESULTS_DIR:-results/track1/refactored/$TRACK1_RUN_ID}"
TRACK1_GEOMETRY_DIR="${TRACK1_GEOMETRY_DIR:-results/track1/geometry}"
TRACK1_ANNOTATIONS="${TRACK1_ANNOTATIONS:-data/track1/annotations.csv}"
TRACK1_LOG_DIR="${TRACK1_LOG_DIR:-results/track1/logs}"
TRACK1_GATE_ID="${TRACK1_RUN_ID}_gate"
TRACK1_REPLAY_ROOT="$TRACK1_DATA_DIR/replayed_activations"
TRACK1_REPLAY_DIR="$TRACK1_REPLAY_ROOT/$TRACK1_RUN_ID"
TRACK1_GATE_DIR="$TRACK1_REPLAY_ROOT/$TRACK1_GATE_ID"

mkdir -p "$TRACK1_LOG_DIR"
TRACK1_LOG_FILE="$TRACK1_LOG_DIR/$TRACK1_RUN_ID.log"
exec > >(tee -a "$TRACK1_LOG_FILE") 2>&1

on_error() {
    local exit_code=$?
    echo
    echo "Track 1 failed with exit code $exit_code."
    echo "Log: $TRACK1_LOG_FILE"
    exit "$exit_code"
}
trap on_error ERR

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

export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Frozen-corpus Track 1 overnight run"
echo "Run ID:       $TRACK1_RUN_ID"
echo "Python:       $PYTHON_BIN"
echo "Data:         $TRACK1_DATA_DIR"
echo "Replay:       $TRACK1_REPLAY_DIR"
echo "Geometry:     $TRACK1_GEOMETRY_DIR"
echo "Results:      $TRACK1_RESULTS_DIR"
echo "Log:          $TRACK1_LOG_FILE"
echo "Started UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

echo
echo "[1/5] Validating source and refactor code"
"$PYTHON_BIN" -m py_compile \
    src/common/llm_client.py \
    src/track1_probing/cache_activations.py \
    src/track1_probing/dashboard.py \
    src/track1_probing/dashboard_data.py \
    src/track1_probing/replay.py \
    src/track1_probing/run_track1.py \
    src/track1_probing/snapshot_analysis.py \
    src/track1_probing/trajectory_geometry.py \
    src/track1_probing/variables.py
"$PYTHON_BIN" -m pytest -q

echo
echo "[2/5] Running qwen2.5-3b replay-validation gate only"
"$PYTHON_BIN" -m src.track1_probing.run_track1 replay \
    --data_dir "$TRACK1_DATA_DIR" \
    --replay_id "$TRACK1_GATE_ID" \
    --models qwen2.5-3b \
    --validation_turns 24 \
    --validation_only

TRACK1_GATE_MANIFEST="$TRACK1_GATE_DIR/manifest.json"
TRACK1_GATE_STATUS="$(
    "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["models"]["qwen2.5-3b"]["validation"]["status"])' \
        "$TRACK1_GATE_MANIFEST"
)"
if [[ "$TRACK1_GATE_STATUS" != "passed" ]]; then
    echo "Replay gate status was '$TRACK1_GATE_STATUS'; refusing corpus-wide replay." >&2
    echo "Inspect: $TRACK1_GATE_MANIFEST" >&2
    exit 1
fi
echo "Replay gate passed."

echo
echo "[3/5] Replaying all recorded turns without generation"
echo "qwen2.5-3b is gate-validated; qwen2.5-7b is stored as not_evaluated"
echo "because the frozen corpus has no original 7b activation artifact."
"$PYTHON_BIN" -m src.track1_probing.run_track1 replay \
    --data_dir "$TRACK1_DATA_DIR" \
    --replay_id "$TRACK1_RUN_ID" \
    --models qwen2.5-3b qwen2.5-7b \
    --validation_turns 24 \
    --allow_failed_validation

echo
echo "[4/5] Regenerating full-space turn geometry from frozen transcripts"
"$PYTHON_BIN" -m src.track1_probing.run_track1 geometry \
    --data_dir "$TRACK1_DATA_DIR" \
    --out_dir "$TRACK1_GEOMETRY_DIR"

echo
echo "[5/6] Running refactored experiments 1A-1I"
ANALYZE_ARGS=(
    --data_dir "$TRACK1_DATA_DIR"
    --replay_dir "$TRACK1_REPLAY_DIR"
    --geometry_turns "$TRACK1_GEOMETRY_DIR/turn_geometry.csv"
    --out_dir "$TRACK1_RESULTS_DIR"
    --experiment all
)
if [[ -s "$TRACK1_ANNOTATIONS" ]]; then
    echo "Using offline annotations: $TRACK1_ANNOTATIONS"
    ANALYZE_ARGS+=(--annotations "$TRACK1_ANNOTATIONS")
else
    echo "No annotation file found at $TRACK1_ANNOTATIONS."
    echo "External-label targets will be listed in skipped_targets.csv."
fi
"$PYTHON_BIN" -m src.track1_probing.run_track1 analyze "${ANALYZE_ARGS[@]}"

echo
echo "[6/6] Exporting refactored CSV-based figures"
"$PYTHON_BIN" -m src.track1_probing.export_dashboard_figures \
    --results_dir "$TRACK1_RESULTS_DIR" \
    --replay_dir "$TRACK1_REPLAY_DIR" \
    --geometry "$TRACK1_GEOMETRY_DIR/turn_geometry.csv"

test -s "$TRACK1_REPLAY_DIR/manifest.json"
test -s "$TRACK1_GEOMETRY_DIR/turn_geometry.csv"
test -s "$TRACK1_RESULTS_DIR/experiment_1a_measurement_audit.csv"
test -s "$TRACK1_RESULTS_DIR/variable_registry.csv"
test -e "$TRACK1_RESULTS_DIR/oof_predictions.csv"
test -e "$TRACK1_RESULTS_DIR/fold_scores.csv"
test -s "$TRACK1_RESULTS_DIR/original_replay_sensitivity.csv"
test -s "$TRACK1_RESULTS_DIR/activation_norm_summary.csv"
test -s "$TRACK1_RESULTS_DIR/activation_transition_summary.csv"
test -s "$TRACK1_RESULTS_DIR/figures/figure_index.csv"
test -s "$TRACK1_RESULTS_DIR/artifact_manifest.csv"
for experiment in 1b 1c 1d 1e 1f 1g 1h 1i; do
    test -s "$TRACK1_RESULTS_DIR/experiment_${experiment}_snapshot_probe_scores.csv"
done
for experiment in 1a 1b 1c 1d 1e 1f 1g 1h 1i; do
    test -s "$TRACK1_RESULTS_DIR/figures/experiment_summaries/experiment_${experiment}_summary.png"
done

echo
echo "============================================================"
echo "Track 1 completed successfully."
echo "Finished UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Replay:       $TRACK1_REPLAY_DIR"
echo "Results:      $TRACK1_RESULTS_DIR"
echo "Log:          $TRACK1_LOG_FILE"
echo "============================================================"
