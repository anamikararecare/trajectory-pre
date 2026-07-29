# Attractor Interpretability

Follow-up experiments to *"Attractor States Emerge in Multi-Turn LLM Conversations"*
(Ko & Geiping, 2026), pushing the analysis from **text-embedding geometry** into
**model internals** and from **AI–AI** interaction into **human–AI** interaction.

The original paper shows that self-play debates settle into model-specific
"attractor basins" in SBERT embedding space, and that mixed-play conversations
get pulled asymmetrically toward one partner's basin. Everything in that paper
is computed on the *outputs* of the models. This repo asks two follow-up
questions:

1. **Track 1 — Stance/Persuasion Probing.** Is the attractor/persuasion
   dynamic legible *inside* the model's activations, and can we predict where
   an agent's stance is heading before it shows up in the generated text?
2. **Track 2 — Human–AI Trajectories.** Do the same attractor basins appear
   when the debate partner is a human instead of another LLM, when human–AI
   turns are projected into a fixed Track 1 self-play reference basis?

Generation and non-geometric analyses can run independently. Track 2 geometry
requires the self-play reference exported by Track 1; partner-conditioned
human–AI turns are never treated as self-play.

## Repo layout

```
attractor-repo/
├── configs/
│   ├── topics.yaml          # debate topics + pro/con seed statements
│   └── models.yaml          # model registry (API + local HF models)
├── src/
│   ├── common/
│   │   ├── debate_prompts.py   # system prompts, questionnaire templates
│   │   ├── questionnaire.py    # stance elicitation + parsing
│   │   ├── embeddings.py       # SBERT embed, topic-centering, PCA, basin score
│   │   ├── emotion.py          # GoEmotions sentence-weighted scoring
│   │   └── llm_client.py       # unified API + local-HF client (+ activation hooks)
│   ├── track1_probing/
│   │   ├── generate_debates.py     # run debates, collect stance per turn
│   │   ├── cache_activations.py    # merge immutable turns, arrays, annotations
│   │   ├── replay.py               # teacher-forced snapshots + validation gate
│   │   ├── offline_annotations.py   # resumable behavioral/persona annotation
│   │   ├── variables.py            # variable registry and derivations
│   │   ├── snapshot_analysis.py    # time-aware experiments 1A-1I
│   │   ├── probes.py               # legacy 1a-1c compatibility probes
│   │   ├── dashboard_data.py       # read-only dashboard data assembly
│   │   ├── dashboard.py            # Streamlit replay/probe dashboard
│   │   ├── export_dashboard_figures.py # static artifact figure exporter
│   │   ├── trajectory_geometry.py  # full-space basin/turn-motion geometry
│   │   └── run_track1.py           # orchestration entrypoint
│   └── track2_human_ai/
│       ├── load_corpus.py          # pull LMSYS-Chat-1M / WildChat from HF
│       ├── reference_geometry.py   # projection in Track 1 self-play basis
│       ├── accommodation.py        # drift-toward-partner metric
│       ├── emotion_overlay.py      # emotion trajectory comparison
│       └── run_track2.py           # orchestration entrypoint
├── scripts/
│   ├── run_all.sh                    # legacy two-track smoke test
│   ├── run_track1_dashboard.sh       # read-only dashboard launcher
│   ├── recover_track1_annotations.sh # annotation-only result recovery
│   └── run_track1_refactored.sh      # frozen Track 1 overnight run
├── data/           # raw/cached conversations, activations (gitignored)
├── results/        # figures, tables, probe outputs (gitignored)
├── requirements.txt
├── environment.yml    # Conda environment (`traj`)
├── .env.example
└── QUICKSTART.md
```

## Track 1 — Frozen-corpus conversational-state probing

Track 1 asks:

> What conversational state is represented before a recorded response, how
> does that representation change during response formation, and does it
> predict the speaker's or partner's later behavior beyond what is already
> observable in the transcript?

### Frozen-corpus contract

The existing files under `data/track1/transcripts/` and
`data/track1/activations/` are immutable source artifacts. The refactored
workflow:

- never samples responses or reruns debates;
- never changes recorded messages, stance reports, or original activations;
- reconstructs each speaker's own system prompt, role, opening message, and
  assistant/user history;
- teacher-forces only the exact recorded response text through the exact
  transcript-recorded model revision;
- writes recovered arrays separately under
  `data/track1/replayed_activations/<replay_id>/`.

Each replay key identifies the turn, speaker, model, layer, snapshot, and
actual token-window length. Its `manifest.json` records transcript hashes,
model/tokenizer revisions, chat-template hash, precision, quantization,
token boundaries, EOS policy, eligibility, and validation results.

### Activation snapshots

The primary snapshots are:

- **pre-generation** — residual-stream state at the final prompt token before
  any response text exists;
- **early response** — mean over up to the first 16 recorded response tokens;
- **full response** — mean over all recorded response tokens;
- **final window** — mean over up to the final eight recorded response tokens.

The final recorded-token state is retained only as a sensitivity analysis.
Response-token states are states after consuming their corresponding token;
they are not described as predicting that same token. EOS and other special
tokens are excluded from every primary replay pool.

The original extractor pooled generated EOS for responses that terminated
before the generation limit. Replay therefore constructs a separate,
non-exported legacy EOS-inclusive mean only for original-versus-replayed
validation.

### Replay-validation gate

Before corpus replay, a deterministic validation sample spans topics,
conditions, roles, response lengths, turns, and layers. Primary admission
requires:

- median cosine similarity of at least 0.999;
- overall and per-layer fifth-percentile cosine similarity of at least 0.995;
- median norm ratio between 0.98 and 1.02;
- no absolute response-length/relative-error rank correlation above 0.2.

Failed and unevaluated models are excluded by the analysis loader unless the
operator explicitly enables warned sensitivity behavior. No model is
substituted for an unavailable or mismatched revision.

The frozen corpus contains original qwen2.5-3b full-response arrays, so 3B can
pass the gate. It has no original qwen2.5-7b activation artifact. The overnight
workflow recovers 7B arrays as `not_evaluated`, but the default analysis does
not admit them to primary results.

### Variables and annotations

Transcript-internal variables include raw stance, role-aligned stance,
confidence, stance change, partnerward movement, and stance gap. Optional
offline annotations are CSV rows keyed by `conv_id,turn`; raw multi-rater
files also include `annotator_id`.

The registry includes:

- local agreement and remaining disagreement;
- affiliation, adversariality, and emotional tone;
- realized conversational moves;
- apparent objective and response-implied expected reaction;
- explicit synthesis, resolution, and closure;
- seven perceived-persona dimensions: warmth, dominance, curiosity,
  structure, emotional stability, deference/force, and epistemic humility;
- five direct observer-rated Big Five conversational-presentation dimensions:
  Extraversion, Agreeableness, Conscientiousness, Neuroticism, and Openness,
  each with a separate visible-evidence confidence rating;
- observed conflict, alignment, accommodation, exploration, and closure;
- semantic velocity and acceleration;
- basin leaning, partnerward basin velocity, and off-axis distance.

Persona variables are called perceived persona or interaction style, not
stable personality. They use trailing three-response windows and deviations
from the same model's self-play baseline. Apparent objective and
response-implied expected reaction are external reconstructions, not private
self-reports.

The direct Big Five variables use the prefix `observer_big5_` and the same
trailing-three, self-play-baseline, mixed-play-deviation, and movement
derivations. They are direct 0–4 observer judgments of visible conversational
presentation, not BFI-44 questionnaire scores. A low evidence-confidence
value identifies turns where a trait is weakly observable.

The analysis writes `variable_registry.csv`,
`annotation_reliability.csv` when annotations are supplied, and an explicit
`skipped_targets.csv` for unavailable labels.
It also writes `big_five_probe_scores.csv`, `big_five_oof_predictions.csv`,
`big_five_fold_scores.csv`, and `big_five_turn_states.csv`; the broader
`turn_variables.csv` contains every available registered scalar variable for
each immutable turn. Experiments 1B–1I each receive a dedicated
`experiment_1x_snapshot_probe_scores.csv`; 1A retains its measurement-audit
table. The figure exporter guarantees `experiment_1a_summary.png` through
`experiment_1i_summary.png` and records every CSV/PNG pair and its completion
status in `artifact_manifest.csv`.
Observed Big Five distributions, evidence confidence, mixed-play deviations,
and snapshot-resolved probe progression are exported as dedicated PNG files.

### Refactored experiments

- **1A — Measurement and validation:** transcript/report coverage, original
  and replay coverage, gate status, annotation reliability, variance,
  prevalence, and model/topic/role/condition balance.
- **1B — Snapshot-resolved concurrent decoding:** compare pre, early, full,
  and final-window representations of current outcomes.
- **1C — Current-response prediction:** use pre-generation state to predict
  the upcoming recorded response.
- **1D — Future-self prediction:** predict same-speaker outcomes and changes
  at horizons 1–4.
- **1E — Immediate partner reaction:** predict the next partner outcome.
- **1F — Apparent objective and expected reaction:** distinguish externally
  reconstructed objectives/expectations from actual reactions.
- **1G — Observable transitions:** predict escalation, de-escalation,
  accommodation, synthesis, closure, and other explicit transition labels.
- **1H — Basin movement:** predict full-space basin leaning, partnerward
  velocity, semantic motion, and off-axis displacement.
- **1I — Partner-induced transfer:** compare mixed play with same-model
  self-play behavioral and within-model activation baselines.

No HMM or latent conversational-state model is fitted.

Every snapshot comparison uses identical rows and leave-one-topic-out folds.
Reported controls include fold-matched mean/majority, shuffled labels,
model/role/topic/turn, lagged behavioral state, and time-matched text:

| Snapshot | Text available to its baseline |
|---|---|
| pre-generation | prior transcript context |
| early response | prior context plus the exact first 16 recorded tokens |
| full response | complete recorded response |
| final window/token | complete recorded response |

Continuous targets use out-of-fold R²; categorical targets use out-of-fold
balanced accuracy. The headline value is always:

```text
score(text/state + activation) - score(text/state)
```

### One-command overnight run in tmux

From the repository root, inside the tmux session you already created:

```bash
bash scripts/run_track1_refactored.sh
```

The script performs, in order:

1. compile checks and the unit-test suite;
2. a qwen2.5-3b validation-only replay gate;
3. gate-status verification and full 3B/7B deterministic replay;
4. full-space geometry regeneration from the frozen transcripts;
5. experiments 1A–1I with output-integrity checks;
6. static figure export from the completed CSV, geometry, and replay-manifest artifacts.

It uses a UTC timestamp as the run ID. To choose one:

```bash
bash scripts/run_track1_refactored.sh overnight_v1
```

Logs are written to `results/track1/logs/<run_id>.log`; analysis results go
to `results/track1/refactored/<run_id>/`. If
`data/track1/annotations.csv` exists and is non-empty, it is used
automatically. Otherwise transcript-internal analyses run and external-label
targets are reported as skipped.

Optional environment overrides are:

```bash
TRACK1_PYTHON_BIN=/absolute/path/to/python
TRACK1_DATA_DIR=data/track1
TRACK1_RESULTS_DIR=results/track1/refactored/custom
TRACK1_GEOMETRY_DIR=results/track1/geometry
TRACK1_ANNOTATIONS=data/track1/annotations.csv
TRACK1_LOG_DIR=results/track1/logs
```

The script does not create or manage tmux and does not invoke the
`generate` subcommand.

## Track 1.5 — Cross-model representational similarity

Track 1.5 explains a cross-model activation RSM with the conversational
variables registered in Track 1. For each mixed-play conversation it:

- retains 10 agent turns per model (20 total) by default;
- selects four rank-evenly-spaced replay layers for each model and pairs them
  by relative depth;
- mean-centres each model activation space using alignment-training data;
- projects the spaces to a shared PCA rank and fits an orthogonal Procrustes
  map on other conversations with the same ordered model pair;
- computes one 10 by 10 cross-model cosine RSM per layer pair and their mean;
- computes a mean-centred RBF RSM for every continuous Track 1 variable and an
  exact-match RSM for every categorical variable; and
- compares every variable RSM with every activation RSM using Spearman RSA, a
  Model-B-axis permutation test, and within-layer Benjamini-Hochberg FDR.

Run the existing preliminary corpus from the repository root:

```bash
bash scripts/run_track1_5.sh
```

Analyze just one conversation or reduce output volume:

```bash
bash scripts/run_track1_5.sh \
  --conv_id death_penalty__mixed_play__opp_sup__1b578a \
  --no_variable_plots
```

Results are written beneath `results/track1_5/<conv_id>/`. Important files
are `activation_rsms/`, `variable_rsms/`, `rsa_variable_explanations.csv`,
`alignment_diagnostics.csv`, and `manifest.json`. The manifest records the
centering and alignment design, exact layers, held-out alignment conversations,
input hashes, and replay-validation status.

The bundled replay has four layers: 3B `[6, 12, 18, 23]` and 7B
`[7, 14, 21, 27]`. A later replay containing six or more layers can be analyzed
with `--n_layers 6`. Model B is `not_evaluated` against original activations in
the bundled run, so the launcher deliberately passes
`--allow_unvalidated_models` and marks those outputs exploratory. Remove that
flag when both models have passed replay validation.

Leave-one-conversation-out alignment is the default.
`--alignment_mode within_conversation` is available only as an exploratory
fallback and is recorded as target leakage in the manifest. Snapshot, alignment
rank, permutation count, output location, and number of turns per model are all
configurable; use `python -m src.track1_5_rsm.run_track1_5 --help` for the full
interface.

### Read-only replay dashboard

After a replay/analysis run has produced artifacts, launch the dashboard from
the repository root in the tmux pane you already have:

```bash
bash scripts/run_track1_dashboard.sh
```

Pass a port as the first argument when needed, for example
`bash scripts/run_track1_dashboard.sh 8502`. Set
`TRACK1_DASHBOARD_ADDRESS=0.0.0.0` only when remote access is intended and
the host firewall is configured appropriately. The launcher does not create a
tmux session and does not run replay, geometry, annotation, or probe jobs.

The dashboard auto-discovers replay manifests and matching analysis runs. Its
pages cover run provenance and activation status, reconstructed conversation
boundaries and OOF snapshot progression, direct hidden-state PCA and norm
views, snapshot-resolved probes, partner reactions and activation transfer,
basin movement, perceived persona and direct Big Five presentation, observable transitions, the replay-quality
audit trail, and a static figure gallery. Complete corpus replays are ranked
ahead of validation-only gate manifests. Missing optional artifacts are labeled
unavailable instead of being recomputed.

The same dashboard also includes four read-only Track 2 windows: overview,
trajectory geometry, accommodation, and emotion. They load the saved tables
and figures under `results/track2/` and do not require a Track 1 replay manifest.
Launch the combined dashboard with:

```bash
bash scripts/run_dashboard.sh
```

Static figures and their machine-readable index are written beneath
`results/track1/refactored/<run_id>/figures/`. To export them from an existing
completed run without replaying models or refitting probes:

```bash
python -m src.track1_probing.export_dashboard_figures \
  --results_dir results/track1/refactored/<run_id> \
  --replay_dir data/track1/replayed_activations/<run_id> \
  --geometry results/track1/geometry/turn_geometry.csv
```

### Recover annotation-dependent results without replay

The completed replay can be reused. From an existing tmux session, run:

```bash
bash scripts/recover_track1_annotations.sh full_track1_20260721T090331Z claude
```

The script loads `ANTHROPIC_API_KEY` from `.env`, performs two resumable
annotation passes, derives the downstream indices and persona windows, reruns
analysis only, verifies that no targets remain skipped, and regenerates static
figures. It never invokes replay or debate generation. To use the configured
OpenAI judge instead, set `OPENAI_API_KEY` and replace `claude` with `gpt`.
Re-running the same command resumes from the annotation journal. Recovered
outputs are written under
`results/track1/refactored_annotated/full_track1_20260721T090331Z/`.

Annotation journals are schema-aware. A journal created before the direct Big
Five fields were added is retained as an audit trail but does not count as a
complete row for the new schema; the recovery command re-annotates those turns
without replaying either conversation model.

### Preliminary results from an in-progress annotation journal

To inspect completed annotations without waiting for both full passes:

```bash
bash scripts/build_track1_preliminary.sh full_track1_20260721T090331Z
```

To freeze exactly the first 20 inferred annotation batches while annotation
continues in another process:

```bash
bash scripts/build_track1_preliminary.sh full_track1_20260721T090331Z 20
```

The focused interpretability dashboard contains only the selected-sample
self-play/mixed-play PC1–PC2 trajectories, concurrent decoding, and
experiments 1C–1I:

```bash
bash scripts/run_track1_preliminary_dashboard.sh 8503 full_track1_20260721T090331Z
```

Open `http://127.0.0.1:8503`. Preliminary probes use
leave-one-conversation-out validation so a one-topic journal prefix can be
explored; the UI labels this clearly and it is not a replacement for the
paper-grade leave-one-topic-out final analysis.

This snapshots only complete current-schema journal rows, writes separate
non-canonical annotations under `data/track1/preliminary_annotations/`, and
runs analysis under `results/track1/preliminary/`. It never overwrites the
canonical annotation CSV or final recovered results. The snapshot includes a
manifest and coverage table with explicit descriptive and cross-topic probe
readiness gates. Future annotation batches rotate across topic and condition
strata so early snapshots become representative sooner.

Launch the normal read-only dashboard, choose the preliminary analysis run,
and set its optional annotation CSV to the printed preliminary
`annotations_raw.csv` path. Re-run the build command at any time to refresh
the snapshot. Set `TRACK1_PRELIM_MAX_TURNS` to cap a deterministic stratified
subset, or `TRACK1_PRELIM_TEXT_EMBEDDINGS=1` to include the slower text
embedding baseline. Batch-prefix snapshots use the configured annotation batch
size (8 by default) and are explicitly labeled as corpus-order-biased.

Optional controls are `TRACK1_ANNOTATION_PASSES`,
`TRACK1_ANNOTATION_BATCH_SIZE`, `TRACK1_ANNOTATION_CONTEXT_TURNS`, and
`TRACK1_RECOVERY_RESULTS`. Two passes are the default so annotation reliability
can be reported.

### Manual component commands

Gate only:

```bash
python -m src.track1_probing.run_track1 replay \
  --data_dir data/track1 \
  --replay_id gate_v1 \
  --models qwen2.5-3b \
  --validation_turns 24 \
  --validation_only
```

Full replay after a passed gate:

```bash
python -m src.track1_probing.run_track1 replay \
  --data_dir data/track1 \
  --replay_id full_v1 \
  --models qwen2.5-3b qwen2.5-7b \
  --validation_turns 24 \
  --allow_failed_validation
```

Geometry and analysis:

```bash
python -m src.track1_probing.run_track1 geometry \
  --data_dir data/track1 \
  --out_dir results/track1/geometry

python -m src.track1_probing.run_track1 analyze \
  --data_dir data/track1 \
  --replay_dir data/track1/replayed_activations/full_v1 \
  --geometry_turns results/track1/geometry/turn_geometry.csv \
  --annotations data/track1/annotations.csv \
  --out_dir results/track1/refactored/full_v1 \
  --experiment all
```

Omit `--annotations` when no annotation file is available.

Interpret results according to timing: concurrent decodability,
pre-generation prediction, future-self prediction, partner-reaction
prediction, and incremental value beyond time-matched controls have
increasing evidential strength. None establishes causal use, a true private
intention, a stable personality, or a latent regime.

## Track 2 — Human–AI Trajectories

**Hypothesis:** attractor basins are not an artifact of AI–AI interaction;
individual AI models still occupy separable regions in embedding space when
talking to humans, but the mutual convergence/affiliation dynamics seen in
AI–AI self-play may be asymmetric (one-sided) with a human partner.

Sub-experiments (see `src/track2_human_ai/`):

- **2a. Trajectory geometry** — project real human–AI logs into Track 1's
  self-play PCA basis and report Eq. 5 within topic buckets. This is an
  external-validity extension; observational model, topic, user, version, and
  deployment confounds prevent an apples-to-apples replication claim.
- **2b. Accommodation** — does the AI drift toward the human's linguistic
  style/opinion over the conversation, and does this differ by model?
- **2c. Emotion overlay** — reuse the paper's exact GoEmotions procedure
  (their Eq. 15) to compare affect trajectories for human vs. AI turns.
- **2d. (stretch) Personality layering** — lightweight personality-trait
  scoring over turns, to see if AI personality drifts toward the human's.

Uses the [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
or [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) dataset via
`datasets`. Both are gated on Hugging Face — you'll need to accept the terms
on the dataset page and set `HF_TOKEN`.

## Getting started

See [`QUICKSTART.md`](QUICKSTART.md).

## Notes / scope
The `--paper_compatible` Track 1 mode uses 20 responses per agent and a
six-item stance battery on the eight configured topics that overlap the
paper. The added item wording is authored for this repository; it is not
represented as the paper's unpublished full ProCon prompt set.

Track 2 pooled scores are labeled exploratory. Primary corpus reporting is
conditioned on topic buckets with explicit cross-model coverage thresholds.


The default mode remains an exploratory proof of concept. Direct numerical
comparison also requires matching model versions, sampling settings, topics,
and successful Track 1 geometry. Wild-corpus results remain observationally
confounded by user population, model deployment, and collection conditions.
## Q1 generated-corpus experiments

The Q1 minimum corpus has 208 32-response debates over eight topics and two
role orders. In 112 self-play debates both agents use the same model; in 96
mixed-play debates one of six models debates the `qwen2.5-3b` anchor. A fixed
supporter and opposer are prompted with a central reason, instructed to retain
their conclusion, and answer each partner turn in a concise paragraph. Four
model layers are stored as mean hidden states over each generated response.

Q1 currently implements:

- **E1:** measures how readable each conversational variable is at every
  stored layer and in each percentage quarter of the conversation.
- **E2:** measures when decoding becomes reliable, how peak layers move, and
  whether a probe trained in one conversation quarter transfers to the others.
- **E3:** tests how many shared activation directions a related variable
  family needs, how much families overlap, and whether directions transfer
  between conversation quarters.

All three experiments use the same corpus loader, have no activation-snapshot
factor, keep model activation spaces separate, and use held-out topics for
evaluation. Their default variables cover stance, agreement/conflict,
trailing-window perceived personality style, and expressed
valence/arousal/dominance.

```bash
# Layerwise variable probes and figures
conda run -n traj python -m src.q1.q1_cli e1 \
  --run-dir data/q1_data/q1_minimum_v1 \
  --out-dir results/q1/e1

# Independent-phase and cross-temporal probes, including condition contrasts
conda run -n traj python -m src.q1.q1_cli e2 \
  --run-dir data/q1_data/q1_minimum_v1 \
  --out-dir results/q1/e2

# Family rank, overlap, cross-turn transfer, and figures
conda run -n traj python -m src.q1.q1_cli e3 \
  --run-dir data/q1_data/q1_minimum_v1 \
  --out-dir results/q1/e3
```

See [Q1.md](Q1.md) for the statistical contract, restricted-run commands,
output tables, and figure definitions.

### Run Q1 on the completed subset

The analysis loader automatically skips planned conversations that do not yet
have both transcript and activation artifacts. To run E1–E3 on everything
currently ready:

```bash
bash scripts/run_q1_available.sh current \
  data/q1_data/q1_minimum_v1 \
  results/q1/available_current
```

Use the `quick` profile for a stance-only preliminary run or `full` for 500
topic-bootstrap samples. Set `Q1_EXPERIMENTS=e1`, `e2`, or `e3` to run one
experiment at a time. The launcher saves `available_corpus.json` and a combined
log. See [Q1.md](Q1.md) for all overrides and individual commands.

## Annotate the currently available Q1 conversations

Annotation is resumable and limited to conversations that currently have both
transcript and activation files. A one-pass pilot over eight balanced
conversations:

```bash
Q1_ANNOTATION_MAX_CONVERSATIONS=8 \
bash scripts/q1_annotate_available.sh gpt \
  data/q1_data/q1_minimum_v1 \
  data/q1_data/q1_minimum_v1/q1_annotations
```

Continue from that journal and annotate every currently ready conversation:

```bash
bash scripts/q1_annotate_available.sh gpt \
  data/q1_data/q1_minimum_v1 \
  data/q1_data/q1_minimum_v1/q1_annotations
```

For two independent passes, set `Q1_ANNOTATION_PASSES=2`. The default is one
pass because the current ready corpus contains thousands of response-level
items. The output used by Q1 analysis is:

```text
data/q1_data/q1_minimum_v1/q1_annotations/annotations.csv
```

The annotator writes a row-level JSONL journal after every successful item, so
reissuing the same command resumes rather than paying for completed work.
`gpt` requires `OPENAI_API_KEY`; use `claude` with `ANTHROPIC_API_KEY`.

Then run the experiments with:

```bash
Q1_ANNOTATIONS=data/q1_data/q1_minimum_v1/q1_annotations/annotations.csv \
bash scripts/run_q1_available.sh current \
  data/q1_data/q1_minimum_v1 \
  results/q1/available_current
```

