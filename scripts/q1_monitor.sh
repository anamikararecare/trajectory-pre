#!/usr/bin/env bash
set -Eeuo pipefail

q1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$q1_repo_root"

q1_run_id="${1:-q1_minimum_v1}"
q1_num_shards="${2:-3}"
q1_interval="${3:-10}"

conda run --no-capture-output -n traj python -u -m src.q1.q1_monitor \
  --protocol configs/q1_available_protocol.yaml \
  --run-id "$q1_run_id" \
  --num-shards "$q1_num_shards" \
  --interval "$q1_interval"
