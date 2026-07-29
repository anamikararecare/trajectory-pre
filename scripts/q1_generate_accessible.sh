#!/usr/bin/env bash
# Verbose Q1 generator using the checkpoint-access-adjusted registry.
set -Eeuo pipefail

q1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$q1_repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SHARD_INDEX NUM_SHARDS [RUN_ID]" >&2
  exit 2
fi

q1_shard_index="$1"
q1_num_shards="$2"
q1_run_id="${3:-q1_minimum_v1}"

[[ -n "${OPENAI_API_KEY:-}" ]] || {
  echo "OPENAI_API_KEY is not set after loading $q1_repo_root/.env." >&2
  exit 2
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo "HF_TOKEN is not set after loading $q1_repo_root/.env." >&2
  exit 2
}

python -u -m src.q1.q1_verbose_cli generate \
  --protocol configs/q1_accessible_protocol.yaml \
  --run-id "$q1_run_id" \
  --shard-index "$q1_shard_index" \
  --num-shards "$q1_num_shards" \
  --quality-judge gpt
