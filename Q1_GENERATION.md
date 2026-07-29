# Q1 corpus generation

All generated artifacts for a run are isolated under:

```text
data/q1_data/<run-id>/
  q1_plan.csv
  q1_generation_journal__shard_XX.jsonl
  q1_transcripts/q1_transcript__<conversation-id>.json
  q1_activations/q1_activations__<conversation-id>.npz
```

The minimum plan contains 240 conversations: 128 self-play conversations
(8 models × 8 topics × 2 role orders) and 112 mixed-play conversations
(7 non-anchor models × 8 topics × 2 role orders against Qwen 2.5 3B).
Each conversation contains 16 turns per agent.

Create and inspect the manifest:

```bash
conda run -n traj python -m src.q1.q1_cli plan \
  --protocol configs/q1_protocol.yaml \
  --run-id q1_minimum_v1
```

For three GPUs, launch one of the following in each of three terminals:

```bash
TRACK1_DEVICE=cuda:0 conda run -n traj bash scripts/q1_generate_minimum.sh 0 3 q1_minimum_v1
TRACK1_DEVICE=cuda:1 conda run -n traj bash scripts/q1_generate_minimum.sh 1 3 q1_minimum_v1
TRACK1_DEVICE=cuda:2 conda run -n traj bash scripts/q1_generate_minimum.sh 2 3 q1_minimum_v1
```

The commands are resumable. Re-running a shard skips transcripts already
written atomically. To disable the external quality judge, call the Python CLI
directly with `--quality-judge none`; this is faster but not recommended for
the final corpus.

Audit completion:

```bash
conda run -n traj python -m src.q1.q1_cli audit \
  --protocol configs/q1_protocol.yaml \
  --run-id q1_minimum_v1
```

Browse completed transcripts:

```bash
conda run -n traj bash scripts/q1_browse_corpus.sh q1_minimum_v1 8504
```

Then visit `http://127.0.0.1:8504`.
