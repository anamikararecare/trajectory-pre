# Quickstart

## 0. Requirements

- Python 3.10+
- A GPU is strongly recommended for Track 1 (you need to run a local HF model
  and pull hidden states every turn). Track 2 is CPU-friendly (SBERT +
  sklearn only), though downloading the conversation corpus takes a while.
- API keys as needed (see below) — Track 1's *generation* can optionally use
  an API model as one of the two debate agents (e.g. Claude via Anthropic API
  for realism/quality), but at least one agent must be a **local HF model**
  so we can extract activations from it. The probing analysis only needs
  activations from that local model.

## 1. Install

```bash
cd attractor-repo
conda env create -f environment.yml
conda activate traj
```

The environment is named `traj` and uses Python 3.11. To refresh it after
dependency changes, run `conda env update -f environment.yml --prune`.

## 2. Configure keys

```bash
cp .env.example .env
```

Fill in whichever of these you have — everything degrades gracefully if a
key is missing (scripts will tell you what's unavailable):

```
ANTHROPIC_API_KEY=...      # optional, for API-model debate partner / judge
OPENAI_API_KEY=...         # optional, alternative API partner / judge
HF_TOKEN=...               # required for gated models + LMSYS-Chat-1M/WildChat
```

Then load it in your shell (scripts also call `dotenv.load_dotenv()`
automatically):

```bash
export $(grep -v '^#' .env | xargs)
```

## 3. Track 1 — Stance/Persuasion Probing

For the frozen existing-corpus replay and refactored experiments 1A-1I, use
the Track 1 section of [`README.md`](README.md). The generation commands below are
only for creating a separate future corpus, not for refactoring the current one.

### 3a. Generate debates + cache activations

```bash
python -m src.track1_probing.run_track1 generate \
    --local_model qwen2.5-3b \
    --partner_model qwen2.5-7b \
    --topics configs/topics.yaml \
    --n_topics 5 \
    --n_turns 20 \
    --out_dir data/track1 \
    --paper_compatible
```

This runs self-play controls for both models plus their mixed-play pair,
asks the six-item stance battery after every response, and caches
residual-stream activations for the local model. Omit `--paper_compatible`
for the cheaper exploratory one-item battery.
`--n_turns` counts responses per agent, so 20 produces 40 alternating messages.
Outputs land in `data/track1/{activations,transcripts}/`; stance responses are
stored with their corresponding transcript turns.

To use an API model as the partner, pass its registry key, for example
`--partner_model claude` or `--partner_model gpt` (see `configs/models.yaml`).
Both Qwen models in the example are public Hugging Face repositories and do
not require the separate access approval needed by `llama3-8b`.

### 3b. Establish the output-space geometry baseline

```bash
python -m src.track1_probing.run_track1 geometry \
    --data_dir data/track1 \
    --out_dir results/track1/geometry
```

This must precede attractor-related interpretation of the probes. It writes
paper-style endpoint metrics and `self_play_reference.npz` for Track 2.

### 3c. Run probes

```bash
python -m src.track1_probing.run_track1 probe \
    --data_dir data/track1 \
    --experiment 1a          # or 1b, 1c, or "all"
```

This trains the concurrent / predictive / cross-agent probes described in
the README, with leave-one-topic-out cross-validation, and writes:

- `results/track1/probe_scores.csv` — per-layer, per-horizon accuracy/R²
- `results/track1/heatmap_predictive.png` — the headline figure (layer × horizon)

### 3d. Draw paper-style result figures

```bash
python -m src.track1_probing.run_track1 visualize \
    --data_dir data/track1 \
    --results_dir results/track1 \
    --out_dir results/track1/figures
```

This produces PCA trajectory/basin panels analogous to the paper's Figures 2
and 5, full-space endpoint diagnostics, stance trajectories analogous to
Figure 9, and a four-panel activation-probe summary. The PCA plots are
descriptive views; inferential geometry metrics remain computed in 384-D.

## 4. Track 2 — Human–AI Trajectories

### 4a. Pull + filter the corpus

```bash
python -m src.track2_human_ai.run_track2 fetch \
    --dataset lmsys-chat-1m \
    --min_turns 6 \
    --out data/track2/filtered_conversations.jsonl
```

This downloads the dataset (gated — accept terms on HF first), filters to
multi-turn, single-model, opinion/controversial-topic conversations using
the keyword list in `configs/topics.yaml`, and writes a flat JSONL.

### 4b. Trajectory geometry (basin separation)

```bash
python -m src.track2_human_ai.run_track2 geometry \
    --in data/track2/filtered_conversations.jsonl \
    --reference results/track1/geometry/self_play_reference.npz \
    --out_dir results/track2
```

Projects human–AI turns into the fixed Track 1 self-play basis, centers using
AI-only topic-bucket means, and reports model separation within topic buckets.
This is an observational external-validity extension, not a controlled
replication.

### 4c. Accommodation + emotion overlay

```bash
python -m src.track2_human_ai.run_track2 accommodation \
    --in data/track2/filtered_conversations.jsonl --out_dir results/track2

python -m src.track2_human_ai.run_track2 emotion \
    --in data/track2/filtered_conversations.jsonl --out_dir results/track2
```

Redraw all Track 2 figures from the saved result tables without rerunning any
embedding or emotion models:

```bash
python -m src.track2_human_ai.run_track2 visualize \
    --results_dir results/track2 \
    --out_dir results/track2 \
    --n_bins 10
```

The model-heavy plots use heatmaps or small multiples rather than placing all
model trajectories and a large legend on one axis. Model labels include their
conversation coverage so sparse observational estimates remain visible.

## 5. Run everything end-to-end (small smoke test)

```bash
bash scripts/run_all.sh
```

Runs both tracks with tiny settings (2 topics, 4 turns per agent, 2 models) to
confirm the pipeline works end-to-end before scaling up.

## Troubleshooting

- **"gated repo" error from `datasets` or `transformers`**: go to the
  dataset/model page on huggingface.co, accept the license, and make sure
  `HF_TOKEN` is set and has read access.
- **OOM loading the local model**: pass `--load_in_4bit` (uses `bitsandbytes`)
  or switch to a smaller model, e.g. `Qwen2.5-3B-Instruct`.
- **No GPU available**: Track 1 generation + activation caching will be slow
  but will still run on CPU for the small defaults; consider cutting
  `--n_turns` further.
