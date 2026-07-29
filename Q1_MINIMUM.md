# Q1 minimum corpus

The canonical minimum-corpus configuration is
`configs/q1_minimum_protocol.yaml`, with its model registry in
`configs/q1_minimum_models.yaml`.

It specifies 240 conversations and 7,680 response turns:

- 128 self-play conversations;
- 112 cross-model conversations against the Qwen 2.5 3B anchor;
- 8 topics, 2 role orders, and 16 turns per agent;
- two official checkpoints in each of Qwen, Gemma, Llama, and Mistral.

Prepare the deterministic plan:

```bash
conda run -n traj python -m src.q1.q1_cli plan \
  --protocol configs/q1_minimum_protocol.yaml \
  --run-id q1_minimum_v1
```

Launch one shard per GPU:

```bash
TRACK1_DEVICE=cuda:0 conda run -n traj bash scripts/q1_generate_minimum_official.sh 0 3 q1_minimum_v1
TRACK1_DEVICE=cuda:1 conda run -n traj bash scripts/q1_generate_minimum_official.sh 1 3 q1_minimum_v1
TRACK1_DEVICE=cuda:2 conda run -n traj bash scripts/q1_generate_minimum_official.sh 2 3 q1_minimum_v1
```

Audit:

```bash
conda run -n traj python -m src.q1.q1_cli audit \
  --protocol configs/q1_minimum_protocol.yaml \
  --run-id q1_minimum_v1
```

Browse:

```bash
conda run -n traj bash scripts/q1_browse_corpus.sh q1_minimum_v1 8504
```

Every output stays under `data/q1_data/q1_minimum_v1/`. Generation is
resumable at conversation granularity.
