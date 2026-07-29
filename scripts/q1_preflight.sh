#!/usr/bin/env bash
set -Eeuo pipefail

q1_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$q1_repo_root"

conda run --no-capture-output -n traj python -m src.q1.q1_preflight
