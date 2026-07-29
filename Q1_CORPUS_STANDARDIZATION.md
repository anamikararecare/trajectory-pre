# Anamika V2 Experiment Layout and Consolidated Technical Brief

## 1. Scope and source hierarchy

This is a brief on the conversational corpus generation for Q1:

> To what extent are conversational state and trajectory variables organised
> across layers and activation subspaces inside language models, and how does
> that organisation manifest over turns?

This experimental track reuses Anamika's V1 variable registry during
analysis, but it is **not** the legacy V1 corpus. Raw activation spaces
must remain model-specific.

The principal entry points are:

| Purpose | Canonical location |
|---|---|
| Deterministic factorial plan | `src/q1/design.py` |
| Actual plan for this run | `data/q1_data/q1_minimum_v1/q1_plan.csv` |
| Protocol matching that plan | `configs/q1_available_protocol.yaml` |
| Model registry matching that plan | `configs/q1_available_models.yaml` |
| Topic and questionnaire definitions | `configs/q1_topics.yaml` |
| Stance-locked prompts | `src/q1/prompts.py` |
| Conversation generation | `src/q1/q1_generate.py` |
| Mechanical and external quality measurements | `src/q1/quality.py` |
| Gate wrappers used while completing the run | `src/q1/q1_soft_gate_cli.py`, `q1_resilient_cli.py`, and `q1_mechanical_gate_cli.py` |
| Current launcher with per-shard locking | `scripts/q1_generate_locked.sh` |
| Transcripts | `data/q1_data/q1_minimum_v1/q1_transcripts/` |
| Activations | `data/q1_data/q1_minimum_v1/q1_activations/` |
| Generation journals | `data/q1_data/q1_minimum_v1/q1_generation_journal__shard_*.jsonl` |
| Corpus loader | `src/q1/corpus.py` |
| E1–E3 methods | `Q1.md` and `src/q1/` |
| Existing completed-subset results | `results/q1/available_current/` |

## 2. Experiment layout and research goals

Q1 contains three related experiments.

### 2.1 E1: layerwise encoding over conversation phases

E1 fits a separate probe for each model, Track 1 variable, conversation
percentage range, and stored layer. Its goal is to locate where a variable is
decodable and whether that location/strength changes from early to late
conversation. Default ranges are `(0,25%]`, `(25,50%]`, `(50,75%]`, and
`(75,100%]`, which are turns 1–8, 9–16, 17–24, and 25–32 in a complete Q1
conversation.

### 2.2 E2: temporal manifestation and cross-phase transfer

E2 combines independent range-specific probes with cross-temporal
generalisation: train in one conversation range and test the same model/layer
probe in every range. Its goal is to distinguish stable state axes from
phase-specific representations, estimate reliable onset and peak-layer
migration, and describe mixed-play versus self-play differences. E2 estimates
`overall`, `self_play`, and `mixed_play` scopes. These condition contrasts are
descriptive, not randomised causal effects.

### 2.3 E3: subspace dimensionality and overlap

E3 tests how many activation directions related variable families need,
whether different families occupy overlapping subspaces, and whether those
directions transfer across conversation phases. It keeps model spaces
separate and compares families using within-model reduced-rank geometry and
principal-angle summaries.

All experiments use complete response text as the text baseline for the
response-token-mean activation, leave one topic out for outer evaluation, and
fit regularisation/preprocessing only on training topics.

## 3. Sample design and exact counts

### 3.1 Planned factorial design

The authoritative plan currently on disk has **208 conversations and 6,656
response rows**:

- 112 self-play conversations:
  `7 models × 8 topics × 2 role orders × 1 seed`;
- 96 mixed-play conversations:
  `6 non-anchor models × 8 topics × 2 role orders × 1 seed`;
- 16 responses per agent, 32 responses per conversation;
- four activation layers per response, or 26,624 planned activation vectors.

The plan's model set is the seven-model `q1_available` set. Older documents
that say 240 conversations or eight models refer to superseded
`q1_minimum_protocol.yaml`/`q1_generation_protocol.yaml` designs and do not
describe `q1_plan.csv`.

The design code creates deterministic conversation IDs from:

```text
topic | condition | model_a | model_b | role_a | role_b | seed
```

using the first ten hex characters of SHA-256. `group_model`, `group_index`,
and `task_index` organise generation/sharding; they are not experimental
outcomes.

### 3.2 Current completed corpus

At the time of this audit:

- **191/208 conversations** have both transcript and activation files;
- **6,112 response rows** and **24,448 layer vectors** are present;
- 96 self-play and 95 mixed-play conversations are complete;
- every completed conversation has 32 turns and four vectors per turn;
- every completed turn reports `quality_gate_mode="external_and_report"`,
  although field-level provenance shows that the effective gate changed
  during resumptions.

Missing cells are:

- all 16 Mistral 7B self-play conversations; and
- Mistral 7B vs Qwen 2.5 3B, zoos, A-supporter/B-opposer.

Therefore the available subset is not condition-balanced for Mistral 7B.
Analyses must either require all 208 cells or publish the exact 191-cell
inclusion manifest and retain missing-cell diagnostics.

#### 3.2.1 Current response counts by model

| Model | Complete conversations containing model | Response rows by that model | Conditions |
|---|---:|---:|---|
| Gemma 2 2B | 32 | 768 | self and mixed |
| Gemma 2 9B | 32 | 768 | self and mixed |
| Llama 3 8B | 32 | 768 | self and mixed |
| Mistral 7B | 15 | 240 | mixed only |
| Mistral Nemo 12B | 32 | 768 | self and mixed |
| Qwen 2.5 3B anchor | 111 | 2,032 | self and mixed |
| Qwen 2.5 7B | 32 | 768 | self and mixed |

“Conversations containing model” counts a mixed conversation once even
though only 16 of its 32 responses come from that model.

`results/q1/available_current/available_corpus.json` records an earlier
166-conversation snapshot and is now stale relative to the 191 paired files.
It must not be used as the current inclusion manifest without regeneration.

## 4. Models, roles, and experimental combinations

| Key | Exact checkpoint | Recorded revision | Family/band | Layers |
|---|---|---|---|---|
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Qwen/small, anchor | 6, 12, 18, 23 |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Qwen/large | 7, 14, 21, 27 |
| `gemma2-2b` | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | Gemma/small | 7, 13, 20, 25 |
| `gemma2-9b` | `google/gemma-2-9b-it` | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | Gemma/large | 10, 21, 31, 41 |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` | Llama/single available | 8, 16, 24, 31 |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | Mistral/small | 8, 16, 24, 31 |
| `mistral-nemo-12b` | `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | Mistral/large | 10, 20, 30, 39 |

Model and tokenizer revisions are not pinned in the current YAML, so the
resolved revisions stored in transcripts are mandatory provenance. A future
generation run must add explicit `revision` and `tokenizer_revision` fields or
save equivalent immutable resolutions before the first sample.

### 4.1 Self-play

For each model/topic/role order, both agents use the same checkpoint. They
have separate histories but share one loaded client object. Agent A speaks
first.

### 4.2 Mixed play

For each non-anchor model:

- agent A is always the non-anchor `group_model`;
- agent B is always `qwen2.5-3b`;
- agent A always speaks first;
- the two role orders are A-supporter/B-opposer and A-opposer/B-supporter.

The protocol balances stance order, but it does not counterbalance model
position or first speaker. Any “model effect” in mixed play is therefore
confounded with A/B and first/second speaker unless the analysis explicitly
models or limits the claim.

## 5. Topics and assigned stance content

All eight topics use exact strings from `configs/q1_topics.yaml`.

| ID | Display name | Supporter central reason | Opposer central reason | Fixed support statement | Fixed oppose statement |
|---|---|---|---|---|---|
| `death_penalty` | the death penalty | The death penalty is a just and effective deterrent for the most severe crimes. | The death penalty risks executing innocent people and does not reliably deter crime. | I support the death penalty. | I do not support the death penalty. |
| `medical_marijuana` | medical marijuana | Medical marijuana provides real relief for patients and should be widely legalized. | Medical marijuana legalization moves faster than the clinical evidence supports and risks misuse. | I support legal access to medical marijuana. | I do not support legal access to medical marijuana. |
| `four_day_workweek` | a four-day workweek | A four-day workweek improves wellbeing and productivity without sacrificing output. | A four-day workweek is impractical for most industries and would hurt output or pay. | I support adoption of a four-day workweek. | I do not support adoption of a four-day workweek. |
| `social_media` | social media's effect on society | Social media has been a net positive, connecting people and democratizing information. | Social media has been a net negative, harming mental health and polarizing discourse. | I believe social media has been a net positive for society. | I do not believe social media has been a net positive for society. |
| `electric_vehicles` | electric vehicle adoption | Electric vehicles are essential for reducing emissions and should be aggressively subsidized. | Electric vehicle mandates move faster than infrastructure and grid capacity can support. | I support aggressive policies to accelerate electric-vehicle adoption. | I do not support aggressive policies to accelerate electric-vehicle adoption. |
| `school_uniforms` | mandatory school uniforms | School uniforms reduce distraction and socioeconomic visibility, improving school climate. | School uniforms suppress self-expression without solid evidence of behavioral benefit. | I support mandatory school uniforms. | I do not support mandatory school uniforms. |
| `zoos` | the ethics of zoos | Zoos play an important role in conservation, education, and species protection. | Zoos confine animals for human entertainment in ways that cause lasting harm. | I support the continued operation of accredited zoos. | I do not support the continued operation of accredited zoos. |
| `free_college` | tuition-free public college | Tuition-free college reduces student debt and expands opportunity for everyone. | Tuition-free college shifts costs to taxpayers and may not target aid where it's needed most. | I support tuition-free public college. | I do not support tuition-free public college. |

The two pro, two con, support, and oppose battery items for each topic are also
in `configs/q1_topics.yaml`. They must be treated as versioned prompt content.

## 6. Canonical prompt protocol

Every transcript declares:

```text
schema = q1_transcript_v1
prompt_protocol = q1_stance_locked_response_v1
```

Unlike Track 1, Q1 sends no separate `system` message. For portability across
Qwen, Gemma, Llama, and Mistral chat templates, the system-like instructions
and opening task are concatenated into one `user` message.

### 6.1 Opening user message

For role-specific values, the first user message is exactly:

```text
You are one participant in a structured discussion. The following is your fixed global position: “{fixed_stance_statement}” You must maintain that overall conclusion throughout the conversation. A central reason available to you is: {role_seed} You may acknowledge a counterargument, but explicitly explain why it does not change your fixed conclusion. Never endorse both global positions in the same response.

Each message from the other participant is a completed turn, even if it ends abruptly. Respond to its central claim in your own words. Never finish, continue, or complete their sentence or paragraph. Do not imitate their unfinished wording.

Write exactly one self-contained paragraph of 30–80 words. Use no headings or lists. End with a complete sentence. Respond only with the discussion paragraph.

Discussion topic: {display_name}.
One position is: {pro_seed}
The opposing position is: {con_seed}

Give your opening response from your fixed position.
```

The 30–80 wording matches the generated artifacts. Superseded 60–100-word
protocols must not be used to describe this run.

### 6.2 Subsequent partner message and perspective-specific history

After the partner responds, their text is added to the other agent's history
as:

```text
The other participant's completed turn is quoted below.

<partner_turn>
{verbatim_partner_response}
</partner_turn>

Respond to that turn from your fixed global position. Do not continue or complete its wording.
```

Each agent's own accepted response is appended as an `assistant` message.
Rejected drafts and stance/quality checks are not added to the normal
conversation history. On a retry, the last user message receives:

```text
Your previous draft was rejected. Write a new draft that unambiguously maintains your fixed global position, answers rather than continues the other participant, stays within the requested length, and ends cleanly.
```

## 7. Canonical generation and retry configuration

| Setting | Value for this corpus |
|---|---|
| Seed list | `[0]` |
| Seeding | CPU Torch and all CUDA generators reset to 0 at conversation start |
| Responses | 16 per agent, 32 per conversation |
| Speaker order | A on even zero-based turns; B on odd turns |
| Max new tokens | 160 |
| Sampling | `do_sample=true` |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Pad token | tokenizer EOS token |
| Requested words | 30–80 |
| Mechanical tolerance measured by base gate | 20–95 words |
| Max attempts | 3 |
| Precision | bfloat16 on CUDA; float32 on CPU |
| Quantisation | launcher default is none |
| External judge | registry key `gpt`, `gpt-4o-mini` |
| External judge sampling | provider defaults, no deterministic seed |

Temperature and top-p come from the shared local client and are not embedded
in Q1 transcript `generation`; researchers must retain this code-level
provenance.

Resetting every conversation to seed 0 does not make the corpus a set of
independent seeded replicates, and exact regeneration is not guaranteed:
different chat templates, kernels, hardware, Transformers/PyTorch versions,
quality-judge outputs, retry paths, and intermediate self-questionnaire calls
change the RNG path.

## 8. Stance measurement protocol

### 8.1 Every response

The speaking model self-reports agreement with the topic's fixed support
statement after each candidate response. It sees its current normal history,
the candidate as an `assistant` message, and the subjective questionnaire
prompt from `src/common/questionnaire.py`. The accepted value is stored as
`stance_score` and `stance_confidence`.

A supporter is strictly role-consistent when `score > 3`; an opposer is
strictly role-consistent when `score < 3`. Whether that criterion rejected a
draft depends on the gate version described below.

### 8.2 Full battery

At each speaker's agent turns 1, 4, 8, 12, and 16, the accepted response is
also scored on the six-item topic battery. Pro items retain their 1–5 score;
con items are reversed with `6 - score`; the mean is
`stance_battery_score`. Mean item confidence is
`stance_battery_confidence`.

These self-reports and the extra battery calls consume the same local model's
sampling RNG. They are part of generation, not a purely post-hoc annotation.

## 9. Quality gates and run heterogeneity

### 9.1 Quality measurements

The local/basic gate measures:

- response token count;
- word count;
- whether token count reached the 160-token cap;
- whether the response ends in terminal punctuation;
- whether the word count is within the base 20–95 tolerance.

The GPT judge measures:

- required-role consistency;
- endorsement of both global stances;
- continuation/completion of partner wording;
- self-containedness.

The raw judge response and reason are stored. `gpt-4o-mini` uses provider
defaults and no seed.

### 9.2 Effective acceptance policies found on disk

The same run directory was resumed using progressively softer wrappers. This
is inferred directly from turn-level diagnostic fields:

| Effective provenance profile | Conversations | Acceptance behaviour |
|---|---:|---|
| Base/strict fields only | 1 | Self-report stance, external role/mixed/continuation/self-contained checks, token cap, punctuation, and length could reject. |
| Soft diagnostic fields | 16 | Stance, role, mixed stance, partner continuation, and target length were recorded but non-blocking; external self-containedness plus mechanical completion could reject. |
| Mechanical diagnostic fields | 174 | Behavioural judge values and target length were diagnostic; hard rejection was limited to empty/truncated/non-terminal mechanical incompleteness. |

Across the 6,112 accepted turns:

- 24 turns required more than one generation attempt;
- 769 are outside the originally observed 20–95-word tolerance;
- diagnostic fields mark 346 role-inconsistent, 377 mixed-stance, 286
  partner-continuation, and 159 non-self-contained judgements.

These labels are noisy judge outputs, and their categories overlap. More
importantly, acceptance policy is a corpus-generation covariate. Researchers
must either:

1. analyse all 191 conversations and include a reconstructed gate-profile
   field/sensitivity analysis; or
2. restrict to a single declared gate profile and accept the corresponding
   selection bias.

Do not interpret `quality_gate_mode="external_and_report"` as proof that all
turns passed the same hard external gate. The field was not updated by the
wrappers.

Future corpus generation must use a new run ID whenever the protocol,
checkpoint set, gate, prompt, word target, seed set, or sampling settings
change. Never resume a changed protocol into an existing run directory.

## 10. Hidden-activation representation

Each accepted turn has one vector per configured layer. Generation:

1. applies the model's native chat template with
   `add_generation_prompt=true`;
2. samples the response;
3. runs a forward pass over prompt plus generated token IDs;
4. takes the mean hidden state across the generated-token slice.

Array keys are:

```text
{layer_index}__{zero_based_conversation_turn}
```

The analysis loader calls this `generated_response_token_mean`. There is no
snapshot factor in Q1 E1–E3. Do not relabel it as pre-generation, early,
final-window, or final-token state.

The generator's slice may contain a generated EOS token when present; it does
not explicitly remove response special tokens. This differs from the Track 1
teacher-forced replay contract and must not be silently equated with it.

The accepted activation is from the accepted generation attempt. Rejected
attempt activations are discarded.

## 11. Analysis and artifact outputs

A run root contains:

```text
data/q1_data/<run-id>/
  q1_plan.csv
  q1_generation_journal__shard_XX.jsonl
  q1_transcripts/q1_transcript__<conv-id>.json
  q1_activations/q1_activations__<conv-id>.npz
```

Transcript-level required fields include:

```text
schema, prompt_protocol, conv_id, topic_id, condition,
agent_a_model, agent_b_model, agent_a_role, agent_b_role,
n_turns_per_agent, seed, generation, model_specs, turns
```

Turn-level required fields include:

```text
turn, agent_turn, speaker, model, role, text,
stance_score, stance_confidence, stance_responses,
stance_battery_score, stance_battery_confidence,
stance_battery_responses, generation_attempts,
quality_gate_status, quality_gate_mode,
response_tokens, word_count, hit_token_cap,
ends_with_terminal_punctuation, within_word_tolerance,
external_quality
```

Later files also contain `within_word_target_observed` and diagnostic judge
fields. Absence is meaningful provenance and must not be filled as “passed.”

Writes are atomic per transcript/activation pair, and reruns skip an existing
transcript. However, the generator's skip check only tests the transcript
path; corpus readiness must always require the intersection of transcript and
activation IDs. `src/q1/corpus.py` implements that intersection.

## 12. Standard analysis units and sample rules

- The observation is a response turn with the activation generated from that
  response.
- The dependence/grouping unit is the conversation.
- The outer generalisation unit is topic.
- The representation space is one model and one exact layer.
- Percentage range is based on one-indexed conversation progress:
  `100 × (turn + 1) / conversation_turns`.
- For a complete 32-turn conversation, each default quarter contains eight
  turns.
- Conversation completeness means both files exist and all intended turns and
  activation keys are present; the current loader checks file intersection
  but not every internal expected-key count, so a shared preflight should add
  that validation.

Self-play contributes 32 responses from one model per conversation.
Mixed-play contributes 16 responses from the non-anchor model and 16 from the
anchor. Sample-size tables must report both conversations and response rows;
response rows alone obscure anchor over-representation.

## 13. Common statistical contract

Default confirmatory variables in Q1 are:

- `stance_score` and `stance_gap`;
- local agreement and remaining disagreement;
- affiliation and adversariality;
- observed alignment, conflict, and accommodation indices;
- trailing-three perceived warmth, dominance, and humility;
- expressed valence, arousal, and dominance.

The text VAD checkpoint is `RobroKools/vad-bert`; scores are cached by text
SHA-256. If VAD or text embeddings are disabled, the run must state that
explicitly. Annotations are keyed by `conv_id,turn`; geometry is additionally
matched by speaker when available.

Researchers must standardise:

- the exact `variable_registry.csv` and annotation release;
- confirmatory versus exploratory target lists;
- response text embedding checkpoint/revision;
- VAD checkpoint/revision and cache;
- percentage-range boundaries;
- held-out topic folds;
- regularisation grid and inner selection;
- model/layer inclusion;
- condition scopes;
- minimum sample, class, and variance thresholds;
- bootstrap/permutation counts and random seeds;
- multiple-comparison families;
- missing/gate-diagnostic sensitivity rules.

E1 and E3 pool conditions in their primary implementation. E2 explicitly
provides overall, self-play, and mixed-play estimates. A researcher must not
describe pooled E1/E3 values as condition-specific.

## 14. Group-wide standardisation decisions

Record one group-wide answer for every item:

1. Analyse the complete 208-cell plan or the current 191-cell intersection.
2. Exact inclusion CSV and artifact hashes.
3. Treatment of the 17 missing cells and Mistral 7B condition imbalance.
4. Gate-profile reconstruction and sensitivity policy.
5. Exact transcript schema/prompt protocol admitted.
6. Exact model/tokenizer revisions and layer indices.
7. Whether response EOS can be part of the activation mean.
8. Topic strings, role assignment, A/B position, and first-speaker coding.
9. Turn-range edges and whether incomplete conversations are excluded.
10. Annotation, geometry, VAD, and text-embedding releases.
11. Target registry and confirmatory hypotheses.
12. Topic-fold, hyperparameter, bootstrap, permutation, and correction rules.
13. Whether analyses are overall or condition-specific.
14. Required inventory, skip, fold, prediction, and figure outputs.

## 15. Reproducibility record

Every analysis release should include:

- `q1_plan.csv`;
- a planned/completed/missing inventory;
- transcript and activation SHA-256 manifests;
- hashes of protocol, model registry, topic file, prompt code, generator, and
  gate wrappers;
- transcript-recorded model revisions and chat-template hashes;
- software environment, hardware, dtype, and quantisation;
- exact target/layer/range/condition selection;
- fold assignments and all random seeds;
- skipped and failed cells with reasons;
- raw fold-level and out-of-fold predictions.

Current standardisation-critical hashes are:

```text
89f4aa9da48712e0dcefeac453aadba0ea32495688fa54e14ec6a2da9fd00ed1  configs/q1_available_protocol.yaml
4d6518743841774805d34ac4bec0c4c9eb71193782100b822515164aff239105  configs/q1_available_models.yaml
2eba736792cd9cfba87dc37c46aadc201db81949617db8a6a7423d4495bd0750  configs/q1_topics.yaml
15a0a408229087bf70365f5fc64ebb2db85528d2011da9a82d9bef5194b8c079  src/q1/prompts.py
d15076369dd98b6671b78b3ba784bd219147df570b944d0dcff69abe8dbfb5c2  src/q1/q1_generate.py
5d87ba4edc7ad7d7b9cc66866f3b393eca9b20fb6a2429389e3f8cc50cc7da0d  src/q1/quality.py
2b1fe648b7b751129b7e367376defb918add6a9a4e458d49547c902efc2c08c4  src/q1/q1_soft_gate_cli.py
13c21a41c3bb6998e938b1ce2794c15514d0d8c60afce4f3530914ee345215c0  src/q1/q1_resilient_cli.py
da0d6de60b3889b5ac20aa53bd2436cb3d20feddd308d1a6e8bf5ee733504743  src/q1/q1_mechanical_gate_cli.py
009f9478efbad51bdcd94ca381494de4296d647468e3c010fde14438f69641db  data/q1_data/q1_minimum_v1/q1_plan.csv
```

Hashes identify the audited state; future edits require a new manifest and,
for any generation-affecting edit, a new run ID.
