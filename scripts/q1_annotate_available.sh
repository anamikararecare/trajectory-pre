#!/usr/bin/env bash
# Resumably annotate only Q1 conversations with transcripts and activations.
#
# Usage:
#   bash scripts/q1_annotate_available.sh [gpt|claude] [RUN_DIR] [OUT_DIR]
#
# Overrides:
#   Q1_ANNOTATION_PASSES=1
#   Q1_ANNOTATION_BATCH_SIZE=8
#   Q1_ANNOTATION_CONTEXT_TURNS=3
#   Q1_ANNOTATION_FIELD_GROUPS=persona,categorical
#   Q1_ANNOTATION_MAX_CONVERSATIONS=8   # pilot subset; omit for all ready
#   Q1_CONVERSATION_PAIRS=qwen2.5-3b:qwen2.5-3b,gemma2-9b:qwen2.5-3b
#   Q1_TOPICS=death_penalty,medical_marijuana
#   Q1_ROLE_ORDERS=supporter:opposer,opposer:supporter
#   Q1_CONDITIONS=self_play,mixed_play
#   Q1_REQUIRE_BALANCED=1
set -Eeuo pipefail

q1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$q1_repo_root"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

q1_judge="${1:-gpt}"
q1_run_dir="${2:-data/q1_data/q1_minimum_v1}"
q1_annotation_dir="${3:-$q1_run_dir/q1_annotations}"
q1_passes="${Q1_ANNOTATION_PASSES:-1}"
q1_batch_size="${Q1_ANNOTATION_BATCH_SIZE:-8}"
q1_context_turns="${Q1_ANNOTATION_CONTEXT_TURNS:-3}"
q1_field_groups="${Q1_ANNOTATION_FIELD_GROUPS:-persona,categorical}"
q1_max_conversations="${Q1_ANNOTATION_MAX_CONVERSATIONS:-}"
q1_conversation_pairs="${Q1_CONVERSATION_PAIRS:-}"
q1_topics="${Q1_TOPICS:-}"
q1_role_orders="${Q1_ROLE_ORDERS:-}"
q1_conditions="${Q1_CONDITIONS:-}"
q1_require_balanced="${Q1_REQUIRE_BALANCED:-0}"

if [[ -n "${Q1_PYTHON_BIN:-}" ]]; then
  q1_python="$Q1_PYTHON_BIN"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "traj" ]] && command -v python >/dev/null 2>&1; then
  q1_python="$(command -v python)"
elif [[ -x /mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python ]]; then
  q1_python=/mnt/kaliberai/aragu/miniconda3/envs/traj/bin/python
else
  echo "Could not find the traj Python environment." >&2
  exit 2
fi

case "$q1_judge" in
  gpt)
    [[ -n "${OPENAI_API_KEY:-}" ]] || {
      echo "OPENAI_API_KEY is required for judge 'gpt'." >&2
      exit 2
    }
    ;;
  claude)
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] || {
      echo "ANTHROPIC_API_KEY is required for judge 'claude'." >&2
      exit 2
    }
    ;;
  *)
    echo "Judge must be a configured registry key such as gpt or claude." >&2
    exit 2
    ;;
esac

mkdir -p "$q1_annotation_dir"
q1_log="$q1_annotation_dir/annotation.log"
exec > >(tee -a "$q1_log") 2>&1
export PYTHONUNBUFFERED=1

q1_args=(
  --run-dir "$q1_run_dir"
  --judge-model "$q1_judge"
  --out-dir "$q1_annotation_dir"
  --passes "$q1_passes"
  --batch-size "$q1_batch_size"
  --context-turns "$q1_context_turns"
  --field-groups "$q1_field_groups"
)
if [[ -n "$q1_max_conversations" ]]; then
  q1_args+=(--max-conversations "$q1_max_conversations")
fi
if [[ -n "$q1_conversation_pairs" ]]; then
  q1_args+=(--conversation-pairs "$q1_conversation_pairs")
fi
if [[ -n "$q1_topics" ]]; then
  q1_args+=(--topics "$q1_topics")
fi
if [[ -n "$q1_role_orders" ]]; then
  q1_args+=(--role-orders "$q1_role_orders")
fi
if [[ -n "$q1_conditions" ]]; then
  q1_args+=(--conditions "$q1_conditions")
fi
if [[ "$q1_require_balanced" == "1" ]]; then
  q1_args+=(--require-balanced)
fi

echo "Q1 available-data annotation"
echo "Corpus:       $q1_run_dir"
echo "Judge:        $q1_judge"
echo "Passes:       $q1_passes"
echo "Batch size:   $q1_batch_size"
echo "Field groups: $q1_field_groups"
echo "Pilot limit:  ${q1_max_conversations:-none}"
echo "Pairs:        ${q1_conversation_pairs:-all}"
echo "Topics:       ${q1_topics:-all}"
echo "Role orders:  ${q1_role_orders:-all}"
echo "Balanced:     $q1_require_balanced"
echo "Output:       $q1_annotation_dir"
echo "Log:          $q1_log"

"$q1_python" -m src.q1.q1_annotations "${q1_args[@]}"

echo "Use this analysis argument:"
echo "  --annotations $q1_annotation_dir/annotations.csv"
