#!/usr/bin/env bash
# Tiny end-to-end smoke test for both tracks. Not meant to produce
# publication-quality results -- just confirms the pipeline runs.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Track 1 smoke test ==="
python -m src.track1_probing.run_track1 generate \
    --local_model qwen2.5-3b \
    --partner_model qwen2.5-7b \
    --topics configs/topics.yaml \
    --n_topics 2 \
    --n_turns 4 \
    --out_dir data/track1_smoke \
    --load_in_4bit

python -m src.track1_probing.run_track1 geometry \
    --data_dir data/track1_smoke \
    --out_dir results/track1_smoke/geometry \
    --bootstrap_resamples 100

python -m src.track1_probing.run_track1 probe \
    --data_dir data/track1_smoke \
    --out_dir results/track1_smoke \
    --experiment all

echo "=== Track 2 smoke test ==="
echo "NOTE: this step downloads a gated HF dataset; make sure HF_TOKEN is set"
echo "and you've accepted the dataset license on huggingface.co."
python -m src.track2_human_ai.run_track2 fetch \
    --dataset lmsys-chat-1m \
    --min_turns 4 \
    --max_conversations 200 \
    --out data/track2_smoke/filtered_conversations.jsonl

python -m src.track2_human_ai.run_track2 geometry \
    --in data/track2_smoke/filtered_conversations.jsonl \
    --out_dir results/track2_smoke \
    --min_ai_turns 2 \
    --reference results/track1_smoke/geometry/self_play_reference.npz \
    --min_convs_per_model_topic 2 \
    --min_models_per_topic 2

echo "=== Done. Check results/track1_smoke/ and results/track2_smoke/ ==="
