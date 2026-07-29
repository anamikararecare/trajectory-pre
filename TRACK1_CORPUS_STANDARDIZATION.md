# Track 1 Experiment Layout and Consolidated Technical Brief

## 1. Scope and source hierarchy

This is the shared contract for researchers analysing the existing Track 1
corpus. It separates:

1. the **immutable generated corpus** in `data/track1/`;
2. the **teacher-forced activation replay** derived from that corpus;
3. annotations and geometry derived from the same recorded responses; and
4. statistical analysis choices.

The existing corpus must be treated as frozen. Researchers working from the
same base corpus should use conversation IDs and turn IDs from the recorded
files, not regenerate text and not silently substitute a different model,
prompt, topic definition, activation pool, annotation release, or subset.

The principal code and data entry points are:

| Purpose | Canonical location |
|---|---|
| Generation CLI and defaults | `src/track1_probing/run_track1.py` |
| Conversation loop and assignment | `src/track1_probing/generate_debates.py` |
| Debate prompt text | `src/common/debate_prompts.py` |
| Stance-questionnaire text and aggregation | `src/common/questionnaire.py` |
| Sampling and activation extraction | `src/common/llm_client.py` |
| Topic definitions | `configs/topics.yaml` |
| Model registry used at generation | `configs/models.yaml` |
| Recorded responses and stance reports | `data/track1/transcripts/*.json` |
| Original generation-time activations | `data/track1/activations/*.npz` |
| Frozen replay and provenance | `data/track1/replayed_activations/full_track1_20260721T090331Z/` |
| Offline annotations | `data/track1/annotation_runs/` and `data/track1/annotations.csv` |
| Output-space trajectory geometry | `results/track1/geometry/` |
| Primary refactored results | `results/track1/refactored/full_track1_20260721T090331Z/` |
| Annotated results | `results/track1/refactored_annotated/full_track1_20260721T090331Z/` |
| Design and experiment descriptions | `README.md`, Track 1 section |

## 2. Experiment layout and research goals

Track 1 asks:

> What conversational state is represented before a recorded response, how
> does that representation change during response formation, and does it
> predict the speaker's or partner's later behaviour beyond what is already
> observable in the transcript?

The refactored experiment family is:

| Experiment | Goal |
|---|---|
| 1A | Audit measurement coverage, replay validity, annotation reliability, target variance, and corpus balance. |
| 1B | Decode current outcomes from pre-generation, early-response, full-response, and final-window states. |
| 1C | Test whether the pre-generation state predicts the upcoming recorded response. |
| 1D | Predict the same speaker's outcomes and changes one to four speaker turns ahead. |
| 1E | Predict the partner's immediately following reaction. |
| 1F | Separate externally reconstructed objectives and expected reactions from realised reactions. |
| 1G | Predict observable escalation, de-escalation, accommodation, synthesis, closure, and related transitions. |
| 1H | Predict basin leaning, partnerward movement, semantic motion, and off-axis displacement. |
| 1I | Compare mixed play with same-model self-play behavioural and activation baselines to estimate partner-induced transfer. |

There is no fitted HMM or latent-state model. The core statistical unit is a
recorded response turn, nested within a conversation and topic. Model
activation spaces are not pooled.

## 3. Sample design and exact counts

This section describes the files currently on disk, not an idealised future
design.

### 3.1 Canonical conversation corpus

The repository contains:

- **61 transcripts**, each containing 40 alternating responses: 20 responses
  per agent and 2,440 response rows in total;
- **41 original activation files**;
- **61 replay activation files** in replay
  `full_track1_20260721T090331Z`;
- five topics: `death_penalty`, `medical_marijuana`,
  `four_day_workweek`, `social_media`, and `electric_vehicles`;
- 41 self-play transcripts and 20 mixed-play transcripts.

The transcript distribution is:

| Condition and ordered model pair | Conversations | Response rows |
|---|---:|---:|
| self-play, Qwen 2.5 3B → Qwen 2.5 3B | 21 | 840 |
| self-play, Qwen 2.5 7B → Qwen 2.5 7B | 20 | 800 |
| mixed-play, Qwen 2.5 3B → Qwen 2.5 7B | 20 | 800: 400 per model |
| **Total** | **61** | **2,440** |

Every complete five-topic invocation of the generator creates 30
conversations: two role orders for each of two self-play conditions and the
one ordered mixed-play condition. The on-disk corpus contains two such
replications plus one extra Qwen 2.5 3B self-play conversation for the death
penalty. Conversation IDs contain random six-character UUID suffixes; they
are the authoritative replicate identifiers.

This is not a balanced set of independent integer seeds. The generator starts
from `--seed 0` and assigns `seed + run_index` in deterministic loop order.
The two full invocations therefore repeat the same assigned seeds while
producing separate stochastic samples:

- local 3B self-play uses seed pairs 0/1, 6/7, 12/13, 18/19, and 24/25;
- partner 7B self-play uses 2/3, 8/9, 14/15, 20/21, and 26/27;
- mixed play uses 4/5, 10/11, 16/17, 22/23, and 28/29.

Do not treat the 30 distinct seed values as 30 replicates per cell. The
replicate is the conversation ID; the recorded seed is a generation
covariate/provenance field.

### 3.2 Models, roles, and experimental combinations

| Registry key | Exact checkpoint | Recorded revision | Layers |
|---|---|---|---|
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 6, 12, 18, 23 |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 7, 14, 21, 27 |

In every mixed-play conversation, agent A is Qwen 2.5 3B, agent B is Qwen
2.5 7B, and agent A speaks first. The two role orders change stance assignment
but do **not** swap model position or first-speaker position:

- A supporter, B opposer;
- A opposer, B supporter.

In self-play, both agents use the same checkpoint but maintain separate
conversation histories and opposite roles. The same loaded client is used for
both agents.

### 3.3 Original generation-time activation coverage

Generation-time extraction only ran for agent A when agent A was the selected
local hookable model. Consequently:

- all 20 mixed-play activation files contain only the 20 Qwen 2.5 3B
  agent-A turns;
- 21 Qwen 2.5 3B self-play files contain only the 20 agent-A turns;
- Qwen 2.5 7B self-play transcripts have no original activation files;
- there are 820 generation-time activation-bearing response rows and four
  layer vectors per row, or 3,280 original vectors.

The absence of an original Qwen 2.5 7B array is by design of the old
extractor, not missing-at-random data.

## 4. Topics and assigned stance content

The first five entries of `configs/topics.yaml` were selected by
`load_topics(... )[:5]`. Topic order is therefore part of the protocol.

| ID | Display name | Supporter seed | Opposer seed |
|---|---|---|---|
| `death_penalty` | the death penalty | The death penalty is a just and effective deterrent for the most severe crimes. | The death penalty risks executing innocent people and does not reliably deter crime. |
| `medical_marijuana` | medical marijuana | Medical marijuana provides real relief for patients and should be widely legalized. | Medical marijuana legalization moves faster than the clinical evidence supports and risks misuse. |
| `four_day_workweek` | a four-day workweek | A four-day workweek improves wellbeing and productivity without sacrificing output. | A four-day workweek is impractical for most industries and would hurt output or pay. |
| `social_media` | social media's effect on society | Social media has been a net positive, connecting people and democratizing information. | Social media has been a net negative, harming mental health and polarizing discourse. |
| `electric_vehicles` | electric vehicle adoption | Electric vehicles are essential for reducing emissions and should be aggressively subsidized. | Electric vehicle mandates move faster than infrastructure and grid capacity can support. |

The generated corpus used `paper_compatible=true`. Each topic therefore
required exactly six questionnaire items:

- two topic-specific pro statements;
- two topic-specific con statements;
- one explicit support statement;
- one explicit oppose statement.

The exact six statements for each topic are in `configs/topics.yaml` under
`pro_items`, `con_items`, `support_item`, and `oppose_item`. Those strings
must not be paraphrased in a replication. Scores for pro items are retained;
scores for con items are reversed as `6 - score`; `stance_score` is the mean
of all six aligned values. `stance_confidence` is the unweighted mean of the
six confidence values. Both range from 1 to 5.

The speaking model answered the six-item battery after **every response**.
These questionnaire calls are not visible to the debate partner and are not
added to the visible debate history. They nevertheless consume the speaking
model's sampling RNG before its next debate response and are therefore part
of the generation procedure.

## 5. Canonical prompt protocol

The source of truth is `src/common/debate_prompts.py` and
`src/common/questionnaire.py`. The generated corpus used the paper-compatible
branches.

### 5.1 Shared system instruction and role insertion

Each agent received the following shared text:

```text
You are in a vibrant online discussion forum. Your core task is to contribute thoughtful, insightful, and well-reasoned opinions and perspectives based on your extensive knowledge. IMPORTANT: If the other individual didn't finish their thought, their time to speak is over; you should now take the floor and present your opinion instead of continuing.
```

It was followed by a blank line and one role insertion:

```text
Intrinsically, you believe in this viewpoint for this topic: {pro_seed}
```

or:

```text
Intrinsically, you believe in this viewpoint for this topic: {con_seed}
```

The role insertion assigns supporter to `pro_seed` and opposer to `con_seed`.
There is no neutral role in this corpus.

### 5.2 Neutral user opening and perspective-specific history

Both agents began with the same user message:

```text
I've been thinking about {display_name}. {pro_seed} But some say: {con_seed} What are your thoughts on this?
```

Agent A responds first. Thereafter, each agent sees its own prior responses as
`assistant` messages and the partner's responses as `user` messages. No
speaker label, XML wrapper, or additional reminder is inserted.

### 5.3 Stance questionnaire prompt

After every response, each of the six statements was asked separately:

```text
I'd like to hear your specific take on something more direct. Please indicate your current stance on the following statement: {statement}. Also, how confident are you in your stance? Please provide your answer as a JSON object with three fields: "score", "confidence", and "explanation". "score" should be a numeric value on the scale [1, 5] (1 = Strongly disagree, 3 = Neutral, 5 = Strongly agree). "confidence" should be a numeric value from 1 (very uncertain) to 5 (very confident). "explanation" should be a short (1-5 sentence) justification for the score. Return ONLY valid JSON with no extra text.
```

The model saw its ordinary history, its just-generated response as an
`assistant` message, and this questionnaire as a `user` message. Parsing first
tries full JSON, then an embedded JSON object, then the first standalone digit
1–5. Raw model output is retained in `stance_responses`.

## 6. Canonical generation and sampling configuration

All 61 transcripts record the same settings:

| Setting | Value |
|---|---|
| Responses | 20 per agent; 40 per conversation |
| Speaker order | A on even zero-based turns, B on odd turns |
| Maximum new tokens | 400 |
| Local sampling | `do_sample=true` |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Pad token | tokenizer EOS token |
| Torch seed | transcript `local_torch_seed`, set once at conversation start |
| Precision at loading | bfloat16 on CUDA, float32 on CPU |
| Quantisation | none unless `--load_in_4bit`; replay manifest says none |
| Stance judge | self |
| Questionnaire perspective | subjective |
| Retry/quality gate | none |

Generation stops naturally or at 400 tokens. There is no word-count target,
response retry, or post-generation quality gate in legacy Track 1. Some turns
hit the 400-token limit. Researchers must not remove them without declaring a
pre-registered exclusion rule and reporting its effect.

The current registry has no pinned `revision` fields, but every transcript
stores the resolved model commit. Replication and replay must use the
transcript-recorded revisions above, not whichever revision the model name
resolves to later. Tokenizer revision must match model revision.

## 7. Hidden-activation representations

### 7.1 Legacy generation-time representation

`LocalHFClient.generate_with_activations` runs a second forward pass over the
prompt plus generated token IDs and stores the mean hidden state over the
generated token slice. Keys are:

```text
{layer_index}__{zero_based_conversation_turn}
```

This legacy response pool may include a generated EOS token when generation
terminated before the cap. It has no pre-generation, early, or final-window
snapshot.

### 7.2 Canonical frozen replay representations

For the refactored analyses, use:

```text
data/track1/replayed_activations/full_track1_20260721T090331Z/
```

The replay:

- never samples or changes response text;
- reconstructs the exact per-speaker prompt and history;
- teacher-forces the recorded response through the recorded revision;
- excludes EOS and other response special tokens from primary pools;
- records bfloat16-on-CUDA/float32-on-CPU precision and no quantisation.

Primary snapshots are:

| Snapshot | Definition | Text available at that point |
|---|---|---|
| `pre_generation` | Final prompt-token state before response text | Prior transcript only |
| `early_response` | Mean of first up to 16 recorded response-token states | Prior transcript plus those tokens |
| `full_response` | Mean of all recorded response-token states | Complete response |
| `final_window` | Mean of final up to 8 response-token states | Complete response |

`final_token` is sensitivity-only. Response states are after consuming the
indexed token and must not be described as predicting that same token.

The manifest includes all 61 transcript hashes and 2,440 turn records. Qwen
2.5 3B passed validation against original arrays. Qwen 2.5 7B is
`not_evaluated` because no original 7B generation-time arrays exist. Primary
analysis must therefore follow the replay eligibility field; a warned
sensitivity analysis may include unevaluated 7B states, but must be labelled
as such.

## 8. Analysis and artifact outputs

The following are generated from the frozen text and must be versioned rather
than recomputed ad hoc:

| Artifact | Location and caution |
|---|---|
| Full annotation journal | `data/track1/annotation_runs/full_track1_20260721T090331Z/annotation_journal.jsonl` |
| Aggregated annotation table | `data/track1/annotations.csv` |
| Preliminary annotation release | `data/track1/preliminary_annotations/full_track1_20260721T090331Z/` |
| Preliminary manifest | Same directory, `preliminary_manifest.json` |
| Geometry turn table | `results/track1/geometry/turn_geometry.csv` |
| Self-play reference basis | `results/track1/geometry/self_play_reference.npz` |
| Variable definitions used by a run | `<result-run>/variable_registry.csv` |
| Joined turn-level variables | `<result-run>/turn_variables.csv` |
| Out-of-fold predictions | `<result-run>/oof_predictions.csv` |

The preliminary annotation manifest is explicitly partial: 160 selected rows,
eight conversations, one topic, no paired ratings, and no cross-topic probe
readiness. It is descriptive only. Do not present it as a complete or
reliability-validated label set.

## 9. Common analysis contract

Researchers analysing the same base corpus must freeze and report all of the
following.

### 9.1 Corpus identity

- Use the same 61 transcript IDs or publish an explicit inclusion manifest.
- Preserve zero-based `turn`, `speaker`, `role`, `model`, and `topic_id`.
- Use `conv_id` as the replicate/grouping unit; never treat individual turns
  as independent in splits or uncertainty estimates.
- Report whether the extra 3B death-penalty self-play conversation is retained.
- Do not infer sample counts from filenames alone; join transcript, replay,
  annotation, and geometry data on `conv_id, turn` and, where present,
  `speaker`.

### 9.2 Representation identity

- State whether the input is original generation-time activation or replay.
- For replay, name replay ID, snapshot, layer, model revision, tokenizer
  revision, and eligibility rule.
- Never pool raw activation dimensions across models.
- Match layer by exact index for within-model analyses; use relative depth or
  a declared alignment method for cross-model comparisons.
- Do not mix legacy EOS-inclusive means with EOS-excluding replay means.

### 9.3 Outcomes and annotations

- Use the run's `variable_registry.csv` as the definition/version record.
- Preserve direction alignment for stance and role-aligned outcomes.
- Distinguish raw turn-level values, changes, trailing-three summaries,
  self-play baselines, and mixed-play deviations.
- Treat perceived-persona and observer Big Five fields as visible interaction
  style, not stable personality.
- State annotation file, annotator set, aggregation rule, missingness rule,
  and reliability threshold.
- Do not impute unavailable external annotations into confirmatory targets.

### 9.4 Baselines and evaluation

- Use identical rows and topic folds for compared snapshots and baselines.
- Hold out complete topics; never randomly split turns from the same
  conversation across train and test.
- Pre-generation text baselines receive prior transcript only.
- Early-response baselines receive prior transcript plus the same first 16
  tokens represented by the activation.
- Full/final baselines receive the full response.
- Report fold-matched mean/majority, shuffled-label, metadata/state,
  text-only, activation-only, and combined models where applicable.
- Continuous outcomes use out-of-fold R²; categorical outcomes use
  out-of-fold balanced accuracy. The headline incremental quantity is
  `combined - text/state baseline`.
- Fit preprocessing, embeddings, PCA, regularisation, and feature selection
  inside the training fold.

### 9.5 Reproducibility record

Every researcher should save:

- transcript inclusion CSV plus SHA-256 hashes;
- replay manifest path and hash;
- topic/config/prompt/model-registry hashes;
- annotation and geometry hashes;
- model, tokenizer, chat-template, precision, quantisation, and layer details;
- code revision or a source archive when no Git revision is available;
- random seeds for folds, bootstrap, permutation, probe fitting, and any
  resampling;
- software environment and hardware;
- all skipped cells and reasons, not only successful estimates.

Current source hashes at the time this guide was written are:

```text
2eba736792cd9cfba87dc37c46aadc201db81949617db8a6a7423d4495bd0750  configs/topics.yaml
3a070ad74c8fece54edaa637f44bfc65506357d633959a9edf081746f53498d0  configs/models.yaml
4b141cdf9875878e2586a7be35c8313d527f6fb9101c079dc10976db8a0983a1  src/common/debate_prompts.py
5993d11edce5bf8dcc94607261902726f06d427dbe069229112adc9c0bdb22fb  src/common/questionnaire.py
```

Hashes document the audited state; they are not a claim that future edits are
equivalent.

## 10. Group-wide standardisation decisions

Before parallel analysis starts, record one answer for each item:

1. Exact transcript inclusion manifest: 61 files or a declared subset.
2. Extra death-penalty 3B replicate retained or excluded.
3. Replay ID and eligible-model policy.
4. Primary snapshots and sensitivity snapshots.
5. Exact layer indices or relative-depth mapping.
6. Annotation release, aggregation, and minimum reliability.
7. Geometry release and self-play reference.
8. Variable registry and confirmatory target list.
9. Turn windows/horizons and minimum sample/variance rules.
10. Topic-level split and bootstrap/permutation unit.
11. Text/state baseline features and embedding checkpoint.
12. Missing-data, truncated-response, and failed-cell handling.
13. Multiple-comparison family and correction.
14. Output schema, filenames, rounding, and reporting conventions.

If any of these differ, the analyses are not estimates from one standardised
pipeline and should be reported as separate specifications.
