#!/usr/bin/env bash
# Canonical Q1 launcher with a non-blocking per-run/per-shard process lock.
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
q1_run_root="data/q1_data/$q1_run_id"
mkdir -p "$q1_run_root"

exec 9>"$q1_run_root/q1_generation_lock__shard_${q1_shard_index}.lock"
if ! flock -n 9; then
  echo "Refusing duplicate launch: shard $q1_shard_index already holds its Q1 lock." >&2
  echo "Use the monitor rather than starting another copy." >&2
  exit 3
fi

[[ -n "${OPENAI_API_KEY:-}" ]] || {
  echo "OPENAI_API_KEY is not set after loading $q1_repo_root/.env." >&2
  exit 2
}
[[ -n "${HF_TOKEN:-}" ]] || {
  echo "HF_TOKEN is not set after loading $q1_repo_root/.env." >&2
  exit 2
}

python -u -m src.q1.q1_portable_cli generate \
  --protocol configs/q1_available_protocol.yaml \
  --run-id "$q1_run_id" \
  --shard-index "$q1_shard_index" \
  --num-shards "$q1_num_shards" \
  --quality-judge gpt
