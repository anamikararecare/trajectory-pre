#!/usr/bin/env bash
# Run E1, E2, and E3 on the transcript/activation pairs currently available.
#
# Usage:
#   bash scripts/run_q1_available.sh [quick|current|full] [RUN_DIR] [OUT_ROOT]
#
# Useful overrides:
#   Q1_EXPERIMENTS=e1,e2
#   Q1_ANNOTATIONS=path/to/annotations.csv
#   Q1_GEOMETRY_TURNS=path/to/turn_geometry.csv
#   Q1_MODELS=qwen2.5-3b,gemma2-2b
#   Q1_CONVERSATION_PAIRS=gemma2-9b:qwen2.5-3b,llama3-8b:qwen2.5-3b
#   Q1_TOPICS=death_penalty,medical_marijuana
#   Q1_ROLE_ORDERS=supporter:opposer,opposer:supporter
#   Q1_CONDITIONS=self_play,mixed_play
#   Q1_REQUIRE_BALANCED=1
#   Q1_TURN_RANGES=00-25%,25-50%
#   Q1_TARGETS=stance_score,stance_gap
#   Q1_E1_N_JOBS=16
#   Q1_E1_BLAS_THREADS_PER_JOB=2
#   Q1_E3_FAMILIES=stance,expressed_vad
#   Q1_NO_VAD=1
#   Q1_NO_TEXT_EMBEDDINGS=1
#   Q1_SKIP_CROSS_TEMPORAL=1
#   Q1_CONDITION_SCOPES=overall,self_play,mixed_play
#   Q1_WAIT_FOR_ANNOTATIONS=1
#   Q1_WAIT_PID=12345
#   Q1_EXPECTED_ANNOTATION_ROWS=2560
set -Eeuo pipefail

q1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$q1_repo_root"

q1_profile="${1:-quick}"
q1_run_dir="${2:-data/q1_data/q1_minimum_v1}"
q1_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
q1_out_root="${3:-results/q1/available_${q1_profile}_${q1_timestamp}}"

case "$q1_profile" in
  quick)
    q1_default_targets="stance_score,stance_gap"
    q1_default_families="stance"
    q1_default_bootstrap=50
    q1_default_scopes="overall"
    q1_default_no_vad=1
    q1_default_no_text=1
    ;;
  current)
    q1_default_targets=""
    q1_default_families=""
    q1_default_bootstrap=200
    q1_default_scopes="overall,self_play,mixed_play"
    q1_default_no_vad=0
    q1_default_no_text=0
    ;;
  full)
    q1_default_targets=""
    q1_default_families=""
    q1_default_bootstrap=500
    q1_default_scopes="overall,self_play,mixed_play"
    q1_default_no_vad=0
    q1_default_no_text=0
    ;;
  *)
    echo "Unknown profile '$q1_profile'; use quick, current, or full." >&2
    exit 2
    ;;
esac

q1_experiments="${Q1_EXPERIMENTS:-e1,e2,e3}"
q1_annotations="${Q1_ANNOTATIONS:-}"
q1_geometry="${Q1_GEOMETRY_TURNS:-}"
q1_models="${Q1_MODELS:-}"
q1_conversation_pairs="${Q1_CONVERSATION_PAIRS:-}"
q1_topics="${Q1_TOPICS:-}"
q1_role_orders="${Q1_ROLE_ORDERS:-}"
q1_conditions="${Q1_CONDITIONS:-}"
q1_require_balanced="${Q1_REQUIRE_BALANCED:-0}"
q1_turn_ranges="${Q1_TURN_RANGES:-}"
q1_turn_edges="${Q1_TURN_RANGE_EDGES:-0,25,50,75,100}"
q1_targets="${Q1_TARGETS:-$q1_default_targets}"
q1_e3_families="${Q1_E3_FAMILIES:-$q1_default_families}"
q1_bootstrap="${Q1_BOOTSTRAP_SAMPLES:-$q1_default_bootstrap}"
q1_condition_scopes="${Q1_CONDITION_SCOPES:-$q1_default_scopes}"
q1_no_vad="${Q1_NO_VAD:-$q1_default_no_vad}"
q1_no_text="${Q1_NO_TEXT_EMBEDDINGS:-$q1_default_no_text}"
q1_skip_cross="${Q1_SKIP_CROSS_TEMPORAL:-0}"
q1_e1_n_jobs="${Q1_E1_N_JOBS:-16}"
q1_e1_blas_threads="${Q1_E1_BLAS_THREADS_PER_JOB:-2}"
q1_wait_for_annotations="${Q1_WAIT_FOR_ANNOTATIONS:-0}"
q1_wait_pid="${Q1_WAIT_PID:-}"
q1_expected_annotation_rows="${Q1_EXPECTED_ANNOTATION_ROWS:-}"

if [[ -n "${Q1_PYTHON_BIN:-}" ]]; then
  q1_python="$Q1_PYTHON_BIN"
elif [[ "${CONDA_PREFIX##*/}" == "traj" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
  q1_python="$CONDA_PREFIX/bin/python"
elif [[ -x /mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python ]]; then
  q1_python=/mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python
else
  echo "Could not find the traj Python environment." >&2
  echo "Activate it or set Q1_PYTHON_BIN=/absolute/path/to/python." >&2
  exit 2
fi
"$q1_python" -c 'import matplotlib, numpy, pandas, scipy, sklearn, threadpoolctl' || {
  echo "The selected Python lacks required Q1 analysis packages: $q1_python" >&2
  exit 2
}

[[ -d "$q1_run_dir" ]] || {
  echo "Q1 run directory does not exist: $q1_run_dir" >&2
  exit 2
}
if [[ "$q1_wait_for_annotations" == "1" ]]; then
  [[ -n "$q1_annotations" ]] || {
    echo "Q1_WAIT_FOR_ANNOTATIONS=1 requires Q1_ANNOTATIONS." >&2
    exit 2
  }
  if [[ -n "$q1_expected_annotation_rows" ]] &&
     ! [[ "$q1_expected_annotation_rows" =~ ^[0-9]+$ ]]; then
    echo "Q1_EXPECTED_ANNOTATION_ROWS must be a non-negative integer." >&2
    exit 2
  fi
  echo "Waiting for Q1 annotations: $q1_annotations"
  while true; do
    if [[ -n "$q1_wait_pid" ]] && kill -0 "$q1_wait_pid" 2>/dev/null; then
      echo "Annotation PID $q1_wait_pid is still running ($(date -u +%Y-%m-%dT%H:%M:%SZ))."
      sleep 30
      continue
    fi
    if [[ -s "$q1_annotations" ]]; then
      q1_annotation_rows=$(( $(wc -l < "$q1_annotations") - 1 ))
      if [[ -z "$q1_expected_annotation_rows" ||
            "$q1_annotation_rows" -eq "$q1_expected_annotation_rows" ]]; then
        echo "Annotations ready: $q1_annotation_rows data rows."
        break
      fi
      echo "Annotation output has $q1_annotation_rows rows; expected ${q1_expected_annotation_rows:-any positive count}." >&2
    else
      echo "Annotation output was not created or is empty." >&2
    fi
    if [[ -n "$q1_wait_pid" ]]; then
      echo "Annotation PID $q1_wait_pid exited without producing the expected output; experiments will not run." >&2
      exit 1
    fi
    sleep 30
  done
fi
if [[ -n "$q1_annotations" && ! -s "$q1_annotations" ]]; then
  echo "Annotation file does not exist or is empty: $q1_annotations" >&2
  exit 2
fi
if [[ -n "$q1_geometry" && ! -s "$q1_geometry" ]]; then
  echo "Geometry file does not exist or is empty: $q1_geometry" >&2
  exit 2
fi

mkdir -p "$q1_out_root/logs"
q1_log="$q1_out_root/logs/run.log"
exec > >(tee -a "$q1_log") 2>&1
export PYTHONUNBUFFERED=1

q1_common=(
  --run-dir "$q1_run_dir"
  --turn-range-edges "$q1_turn_edges"
)
q1_status=(
  --run-dir "$q1_run_dir"
  --turn-range-edges "$q1_turn_edges"
  --json-out "$q1_out_root/available_corpus.json"
)
if [[ -n "$q1_annotations" ]]; then
  q1_common+=(--annotations "$q1_annotations")
  q1_status+=(--annotations "$q1_annotations")
fi
if [[ -n "$q1_geometry" ]]; then
  q1_common+=(--geometry-turns "$q1_geometry")
  q1_status+=(--geometry-turns "$q1_geometry")
fi
if [[ -n "$q1_models" ]]; then
  q1_common+=(--models "$q1_models")
fi
if [[ -n "$q1_turn_ranges" ]]; then
  q1_common+=(--turn-ranges "$q1_turn_ranges")
fi

q1_selection=()
if [[ -n "$q1_conversation_pairs" ]]; then
  q1_selection+=(--conversation-pairs "$q1_conversation_pairs")
fi
if [[ -n "$q1_topics" ]]; then
  q1_selection+=(--topics "$q1_topics")
fi
if [[ -n "$q1_role_orders" ]]; then
  q1_selection+=(--role-orders "$q1_role_orders")
fi
if [[ -n "$q1_conditions" ]]; then
  q1_selection+=(--conditions "$q1_conditions")
fi
if [[ "$q1_require_balanced" == "1" ]]; then
  q1_selection+=(--require-balanced)
fi
q1_status+=("${q1_selection[@]}")
if ! [[ "$q1_e1_n_jobs" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$q1_e1_blas_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "Q1_E1_N_JOBS and Q1_E1_BLAS_THREADS_PER_JOB must be positive integers." >&2
  exit 2
fi
q1_speed=()
if [[ "$q1_no_vad" == "1" ]]; then
  q1_speed+=(--no-vad)
fi
if [[ "$q1_no_text" == "1" ]]; then
  q1_speed+=(--no-text-embeddings)
fi

echo "============================================================"
echo "Q1 available-data experiment run"
echo "Profile:       $q1_profile"
echo "Experiments:   $q1_experiments"
echo "Python:        $q1_python"
echo "Corpus:        $q1_run_dir"
echo "Results:       $q1_out_root"
echo "Log:           $q1_log"
echo "Annotations:   ${q1_annotations:-none}"
echo "Geometry:      ${q1_geometry:-none}"
echo "Pairs:         ${q1_conversation_pairs:-all}"
echo "Topics:        ${q1_topics:-all}"
echo "Role orders:   ${q1_role_orders:-all}"
echo "Balanced:      $q1_require_balanced"
echo "Started UTC:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

"$q1_python" -m src.q1.q1_available "${q1_status[@]}"

if [[ ",$q1_experiments," == *,e1,* ]]; then
  echo
  echo "[E1] Layerwise encoding"
  q1_e1=(
    "${q1_common[@]}" "${q1_selection[@]}" "${q1_speed[@]}"
    --n-jobs "$q1_e1_n_jobs"
    --blas-threads-per-job "$q1_e1_blas_threads"
    --out-dir "$q1_out_root/e1"
  )
  if [[ -n "$q1_targets" ]]; then
    q1_e1+=(--targets "$q1_targets")
  fi
  "$q1_python" -m src.q1.q1_cli e1 "${q1_e1[@]}"
fi

if [[ ",$q1_experiments," == *,e2,* ]]; then
  echo
  echo "[E2] Temporal manifestation"
  q1_e2=(
    "${q1_common[@]}" "${q1_selection[@]}"
    "${q1_speed[@]}"
    --condition-scopes "$q1_condition_scopes"
    --bootstrap-samples "$q1_bootstrap"
    --out-dir "$q1_out_root/e2"
  )
  if [[ -n "$q1_targets" ]]; then
    q1_e2+=(--targets "$q1_targets")
  fi
  if [[ "$q1_skip_cross" == "1" ]]; then
    q1_e2+=(--skip-cross-temporal)
  fi
  "$q1_python" -m src.q1.q1_cli e2 "${q1_e2[@]}"
fi

if [[ ",$q1_experiments," == *,e3,* ]]; then
  echo
  echo "[E3] Variable-family activation subspaces"
  q1_e3=("${q1_common[@]}" "${q1_selection[@]}" "${q1_speed[@]}" --out-dir "$q1_out_root/e3")
  if [[ -n "$q1_e3_families" ]]; then
    q1_e3+=(--families "$q1_e3_families")
  fi
  "$q1_python" -m src.q1.q1_cli e3 "${q1_e3[@]}"
fi

echo
echo "============================================================"
echo "Q1 available-data run complete"
echo "Finished UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Results:       $q1_out_root"
echo "Availability:  $q1_out_root/available_corpus.json"
echo "Log:           $q1_log"
echo "============================================================"
