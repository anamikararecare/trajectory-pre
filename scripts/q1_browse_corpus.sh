#!/usr/bin/env bash
set -euo pipefail

q1_run_id="${1:-q1_minimum_v1}"
q1_port="${2:-8504}"
export Q1_RUN_DIR="data/q1_data/${q1_run_id}"

streamlit run src/q1/q1_transcript_browser.py \
  --server.port "$q1_port" \
  --server.address 127.0.0.1
