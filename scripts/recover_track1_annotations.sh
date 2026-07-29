#!/usr/bin/env bash
# Recover annotation-dependent Track 1 results without replaying either LLM.
# Usage: bash scripts/recover_track1_annotations.sh [replay_id] [judge_model]
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

TRACK1_RUN_ID="${1:-full_track1_20260721T090331Z}"
TRACK1_JUDGE_MODEL="${2:-${TRACK1_JUDGE_MODEL:-claude}}"
if [[ ! "$TRACK1_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Run ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 2
fi

TRACK1_DATA_DIR="${TRACK1_DATA_DIR:-data/track1}"
TRACK1_REPLAY_DIR="${TRACK1_REPLAY_DIR:-$TRACK1_DATA_DIR/replayed_activations/$TRACK1_RUN_ID}"
TRACK1_GEOMETRY_FILE="${TRACK1_GEOMETRY_FILE:-results/track1/geometry/turn_geometry.csv}"
TRACK1_ANNOTATION_DIR="${TRACK1_ANNOTATION_DIR:-$TRACK1_DATA_DIR/annotation_runs/$TRACK1_RUN_ID}"
TRACK1_RAW_ANNOTATIONS="$TRACK1_ANNOTATION_DIR/annotations_raw.csv"
TRACK1_CANONICAL_ANNOTATIONS="${TRACK1_ANNOTATIONS:-$TRACK1_DATA_DIR/annotations.csv}"
TRACK1_RESULTS_DIR="${TRACK1_RECOVERY_RESULTS:-results/track1/refactored_annotated/$TRACK1_RUN_ID}"
TRACK1_LOG_DIR="${TRACK1_LOG_DIR:-results/track1/logs}"
TRACK1_ANNOTATION_PASSES="${TRACK1_ANNOTATION_PASSES:-2}"
TRACK1_ANNOTATION_BATCH_SIZE="${TRACK1_ANNOTATION_BATCH_SIZE:-8}"
TRACK1_ANNOTATION_CONTEXT_TURNS="${TRACK1_ANNOTATION_CONTEXT_TURNS:-3}"

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
test -s "$TRACK1_GEOMETRY_FILE"
case "$TRACK1_JUDGE_MODEL" in
    claude)
        [[ -n "${ANTHROPIC_API_KEY:-}" ]] || {
            echo "ANTHROPIC_API_KEY is required for judge model 'claude'." >&2
            exit 2
        }
        ;;
    gpt)
        [[ -n "${OPENAI_API_KEY:-}" ]] || {
            echo "OPENAI_API_KEY is required for judge model 'gpt'." >&2
            exit 2
        }
        ;;
esac

mkdir -p "$TRACK1_ANNOTATION_DIR" "$TRACK1_RESULTS_DIR" "$TRACK1_LOG_DIR"
TRACK1_LOG_FILE="$TRACK1_LOG_DIR/${TRACK1_RUN_ID}_annotation_recovery.log"
exec > >(tee -a "$TRACK1_LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1

echo "Annotation-only Track 1 recovery"
echo "Replay (read only): $TRACK1_REPLAY_DIR"
echo "Judge:              $TRACK1_JUDGE_MODEL"
echo "Passes:             $TRACK1_ANNOTATION_PASSES"
echo "Annotation journal: $TRACK1_ANNOTATION_DIR"
echo "Recovered results:  $TRACK1_RESULTS_DIR"

echo
echo "[1/4] Static validation"
"$PYTHON_BIN" -m py_compile \
    src/track1_probing/offline_annotations.py \
    src/track1_probing/variables.py \
    src/track1_probing/cache_activations.py \
    src/track1_probing/snapshot_analysis.py \
    src/track1_probing/run_track1.py \
    src/track1_probing/export_dashboard_figures.py
"$PYTHON_BIN" -m pytest -q

echo
echo "[2/4] Resumable offline behavioral and persona annotation"
"$PYTHON_BIN" -m src.track1_probing.offline_annotations \
    --data_dir "$TRACK1_DATA_DIR" \
    --judge_model "$TRACK1_JUDGE_MODEL" \
    --out_dir "$TRACK1_ANNOTATION_DIR" \
    --passes "$TRACK1_ANNOTATION_PASSES" \
    --batch_size "$TRACK1_ANNOTATION_BATCH_SIZE" \
    --context_turns "$TRACK1_ANNOTATION_CONTEXT_TURNS"

if [[ -e "$TRACK1_CANONICAL_ANNOTATIONS" ]] &&
   ! cmp -s "$TRACK1_RAW_ANNOTATIONS" "$TRACK1_CANONICAL_ANNOTATIONS"; then
    echo "Refusing to overwrite existing $TRACK1_CANONICAL_ANNOTATIONS" >&2
    echo "Set TRACK1_ANNOTATIONS to a new path or move the existing file." >&2
    exit 2
fi
cp "$TRACK1_RAW_ANNOTATIONS" "$TRACK1_CANONICAL_ANNOTATIONS"

echo
echo "[3/4] Analysis-only recovery using saved replay activations"
"$PYTHON_BIN" -m src.track1_probing.run_track1 analyze \
    --data_dir "$TRACK1_DATA_DIR" \
    --replay_dir "$TRACK1_REPLAY_DIR" \
    --annotations "$TRACK1_RAW_ANNOTATIONS" \
    --geometry_turns "$TRACK1_GEOMETRY_FILE" \
    --out_dir "$TRACK1_RESULTS_DIR" \
    --experiment all

"$PYTHON_BIN" -c \
    'import pandas as pd,sys; p=sys.argv[1]; d=pd.read_csv(p); assert d.empty, f"Still skipped:\n{d.to_string(index=False)}"' \
    "$TRACK1_RESULTS_DIR/skipped_targets.csv"

echo
echo "[4/4] Regenerating recovered static figures"
"$PYTHON_BIN" -m src.track1_probing.export_dashboard_figures \
    --results_dir "$TRACK1_RESULTS_DIR" \
    --replay_dir "$TRACK1_REPLAY_DIR" \
    --geometry "$TRACK1_GEOMETRY_FILE"

test -s "$TRACK1_RESULTS_DIR/snapshot_probe_scores.csv"
test -s "$TRACK1_RESULTS_DIR/oof_predictions.csv"
test -s "$TRACK1_RESULTS_DIR/fold_scores.csv"
test -s "$TRACK1_RESULTS_DIR/original_replay_sensitivity.csv"
test -s "$TRACK1_RESULTS_DIR/annotation_reliability.csv"
test -s "$TRACK1_RESULTS_DIR/figures/figure_index.csv"
test -s "$TRACK1_RESULTS_DIR/artifact_manifest.csv"
test -s "$TRACK1_RESULTS_DIR/big_five_probe_scores.csv"
test -s "$TRACK1_RESULTS_DIR/big_five_oof_predictions.csv"
test -s "$TRACK1_RESULTS_DIR/big_five_fold_scores.csv"
test -s "$TRACK1_RESULTS_DIR/big_five_turn_states.csv"
test -s "$TRACK1_RESULTS_DIR/turn_variables.csv"
test -s "$TRACK1_RESULTS_DIR/figures/big_five_snapshot_progression.png"
test -s "$TRACK1_RESULTS_DIR/figures/big_five_observed_distributions.png"
test -s "$TRACK1_RESULTS_DIR/figures/big_five_confidence_and_mixed_deviation.png"
for experiment in 1b 1c 1d 1e 1f 1g 1h 1i; do
    test -s "$TRACK1_RESULTS_DIR/experiment_${experiment}_snapshot_probe_scores.csv"
done
for experiment in 1a 1b 1c 1d 1e 1f 1g 1h 1i; do
    test -s "$TRACK1_RESULTS_DIR/figures/experiment_summaries/experiment_${experiment}_summary.png"
done

echo
echo "Recovery completed."
echo "Annotations: $TRACK1_RAW_ANNOTATIONS"
echo "Results:     $TRACK1_RESULTS_DIR"
echo "Figures:     $TRACK1_RESULTS_DIR/figures"
echo "Log:         $TRACK1_LOG_FILE"
