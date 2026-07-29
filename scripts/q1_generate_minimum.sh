#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SHARD_INDEX NUM_SHARDS [RUN_ID]" >&2
  exit 2
fi

q1_shard_index="$1"
q1_num_shards="$2"
q1_run_id="${3:-q1_minimum_v1}"

python -m src.q1.q1_cli generate \
  --protocol configs/q1_protocol.yaml \
  --run-id "$q1_run_id" \
  --shard-index "$q1_shard_index" \
  --num-shards "$q1_num_shards" \
  --quality-judge gpt
