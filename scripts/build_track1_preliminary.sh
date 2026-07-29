#!/usr/bin/env bash
# Build a non-canonical dashboard snapshot from currently completed annotations.
# Usage: bash scripts/build_track1_preliminary.sh [replay_id] [first_x_batches]
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRACK1_RUN_ID="${1:-full_track1_20260721T090331Z}"
TRACK1_PRELIM_BATCHES="${2:-${TRACK1_PRELIM_BATCHES:-}}"
if [[ ! "$TRACK1_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Run ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi

TRACK1_DATA_DIR="${TRACK1_DATA_DIR:-data/track1}"
TRACK1_REPLAY_DIR="${TRACK1_REPLAY_DIR:-$TRACK1_DATA_DIR/replayed_activations/$TRACK1_RUN_ID}"
TRACK1_JOURNAL="${TRACK1_JOURNAL:-$TRACK1_DATA_DIR/annotation_runs/$TRACK1_RUN_ID/annotation_journal.jsonl}"
TRACK1_PRELIM_ANNOTATIONS="${TRACK1_PRELIM_ANNOTATIONS:-$TRACK1_DATA_DIR/preliminary_annotations/$TRACK1_RUN_ID}"
TRACK1_PRELIM_RESULTS="${TRACK1_PRELIM_RESULTS:-results/track1/preliminary/$TRACK1_RUN_ID}"
TRACK1_GEOMETRY_FILE="${TRACK1_GEOMETRY_FILE:-results/track1/geometry/turn_geometry.csv}"
TRACK1_PRELIM_THREADS="${TRACK1_PRELIM_THREADS:-8}"
if [[ ! "$TRACK1_PRELIM_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRACK1_PRELIM_THREADS must be a positive integer." >&2
    exit 2
fi
export OMP_NUM_THREADS="$TRACK1_PRELIM_THREADS"
export OPENBLAS_NUM_THREADS="$TRACK1_PRELIM_THREADS"
export MKL_NUM_THREADS="$TRACK1_PRELIM_THREADS"
export NUMEXPR_NUM_THREADS="$TRACK1_PRELIM_THREADS"

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

test -s "$TRACK1_REPLAY_DIR/manifest.json"
test -s "$TRACK1_JOURNAL"
test -s "$TRACK1_GEOMETRY_FILE"
mkdir -p "$TRACK1_PRELIM_ANNOTATIONS" "$TRACK1_PRELIM_RESULTS"

SNAPSHOT_ARGS=(
    --data_dir "$TRACK1_DATA_DIR"
    --journal "$TRACK1_JOURNAL"
    --out_dir "$TRACK1_PRELIM_ANNOTATIONS"
)
if [[ -n "${TRACK1_PRELIM_MAX_TURNS:-}" ]]; then
    SNAPSHOT_ARGS+=(--max_turns "$TRACK1_PRELIM_MAX_TURNS")
fi
if [[ -n "$TRACK1_PRELIM_BATCHES" ]]; then
    SNAPSHOT_ARGS+=(
        --max_batches "$TRACK1_PRELIM_BATCHES"
        --batch_size "${TRACK1_ANNOTATION_BATCH_SIZE:-8}"
    )
fi
"$PYTHON_BIN" -m src.track1_probing.preliminary_annotations "${SNAPSHOT_ARGS[@]}"

ANALYZE_ARGS=(
    --data_dir "$TRACK1_DATA_DIR"
    --replay_dir "$TRACK1_REPLAY_DIR"
    --annotations "$TRACK1_PRELIM_ANNOTATIONS/annotations_raw.csv"
    --geometry_turns "$TRACK1_GEOMETRY_FILE"
    --out_dir "$TRACK1_PRELIM_RESULTS"
    --experiment "${TRACK1_PRELIM_EXPERIMENTS:-all}"
    --sample_keys "$TRACK1_PRELIM_ANNOTATIONS/selection.csv"
    --cv_group conv_id
    --skip_sensitivity
    --preliminary_fast
)
if [[ "${TRACK1_PRELIM_TEXT_EMBEDDINGS:-0}" != "1" ]]; then
    ANALYZE_ARGS+=(--no_text_embeddings)
fi
"$PYTHON_BIN" -m src.track1_probing.run_track1 analyze "${ANALYZE_ARGS[@]}"

cp "$TRACK1_PRELIM_ANNOTATIONS/preliminary_manifest.json" "$TRACK1_PRELIM_RESULTS/"
cp "$TRACK1_PRELIM_ANNOTATIONS/coverage.csv" "$TRACK1_PRELIM_RESULTS/annotation_coverage.csv"
"$PYTHON_BIN" -m src.track1_probing.export_preliminary_figures \
    --results_dir "$TRACK1_PRELIM_RESULTS" \
    --annotations_dir "$TRACK1_PRELIM_ANNOTATIONS" \
    --geometry "$TRACK1_GEOMETRY_FILE"

echo
echo "Preliminary snapshot ready."
echo "Annotations: $TRACK1_PRELIM_ANNOTATIONS/annotations_raw.csv"
echo "Results:     $TRACK1_PRELIM_RESULTS"
echo "Dashboard:   bash scripts/run_track1_dashboard.sh"
echo "Focused UI:  bash scripts/run_track1_preliminary_dashboard.sh 8503 $TRACK1_RUN_ID"
