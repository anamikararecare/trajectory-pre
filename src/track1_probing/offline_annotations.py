"""Offline, resumable annotations for frozen Track 1 transcripts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.common.llm_client import ChatMessage, build_client, load_model_registry
from src.track1_probing.variables import BIG_FIVE_TRAITS, aggregate_annotations


ANNOTATION_SCHEMA = "track1_behavioral_direct_big5_v1"
BIG_FIVE_SCORE_FIELDS = tuple(f"observer_big5_{trait}" for trait in BIG_FIVE_TRAITS)
BIG_FIVE_CONFIDENCE_FIELDS = tuple(
    f"observer_big5_{trait}_confidence" for trait in BIG_FIVE_TRAITS
)


CONTINUOUS_FIELDS = (
    "local_agreement", "remaining_disagreement", "affiliation", "adversariality",
    "perceived_persona_warmth", "perceived_persona_dominance",
    "perceived_persona_curiosity", "perceived_persona_structure",
    "perceived_persona_stability", "perceived_persona_deference",
    "perceived_persona_humility",
    *BIG_FIVE_SCORE_FIELDS,
    *BIG_FIVE_CONFIDENCE_FIELDS,
)
CATEGORICAL_FIELDS = {
    "emotional_tone": {"positive", "neutral", "negative", "mixed"},
    "realized_move": {
        "assert", "rebut", "clarify", "question", "concede", "accommodate",
        "synthesize", "resolve", "close", "other",
    },
    "apparent_objective": {
        "persuade", "defend", "challenge", "clarify", "explore",
        "accommodate", "synthesize", "resolve", "close", "other",
    },
    "response_implied_expected_reaction": {
        "agreement", "concession", "clarification", "rebuttal", "escalation",
        "accommodation", "synthesis", "closure", "uncertain",
    },
}
BINARY_FIELDS = ("explicit_synthesis", "explicit_resolution", "explicit_closure")
OUTPUT_FIELDS = (*CONTINUOUS_FIELDS, *CATEGORICAL_FIELDS, *BINARY_FIELDS)


SYSTEM_PROMPT = """You are an offline behavioral annotator for frozen debate transcripts.
Judge only evidence visible in PRIOR CONTEXT and CURRENT RESPONSE for each item.
Never infer private mental states. apparent_objective and expected_reaction are
response-implied textual reconstructions. Persona fields describe perceived
interaction style in this response, not stable personality.

The observer_big5 fields are direct observer ratings of visible conversational
presentation. They are informed by the Big Five constructs but are NOT BFI-44
scores and must not be treated as a standardized personality inventory. Use
the current response and visible prior context only. Do not infer a private or
stable trait. If evidence is weak or mixed, use score 2 and low confidence.

Return one JSON array and no prose. Preserve each item_id exactly.

Continuous fields use 0,1,2,3,4:
- local_agreement: explicit/local agreement with the partner
- remaining_disagreement: substantive unresolved disagreement
- affiliation: warmth, rapport, or cooperative orientation
- adversariality: confrontational or combative orientation
- perceived_persona_warmth: perceived warmth/agreeableness
- perceived_persona_dominance: perceived assertiveness/control
- perceived_persona_curiosity: perceived curiosity/openness
- perceived_persona_structure: perceived organization/deliberateness
- perceived_persona_stability: perceived calm versus reactivity
- perceived_persona_deference: deference (0) versus forcefulness (4)
- perceived_persona_humility: epistemic humility/qualification
- observer_big5_extraversion: energetic, outgoing, expressive, and assertive
  presentation (0 reserved/quiet; 4 strongly extraverted presentation)
- observer_big5_agreeableness: compassionate, respectful, trusting, and
  cooperative presentation (0 antagonistic; 4 strongly agreeable)
- observer_big5_conscientiousness: organized, careful, reliable, deliberate,
  and persistent presentation (0 careless/disorganized; 4 strongly conscientious)
- observer_big5_neuroticism: anxious, emotionally reactive, volatile, or
  stress-sensitive presentation (0 calm/resilient; 4 strongly reactive)
- observer_big5_openness: curious, original, imaginative, and engaged with
  abstract or novel ideas (0 closed/conventional; 4 strongly open)
- observer_big5_<trait>_confidence: evidence sufficiency for that trait rating
  (0 no visible evidence; 1 weak; 2 mixed; 3 good; 4 clear and repeated)

Categorical choices:
- emotional_tone: positive|neutral|negative|mixed
- realized_move: assert|rebut|clarify|question|concede|accommodate|synthesize|resolve|close|other
- apparent_objective: persuade|defend|challenge|clarify|explore|accommodate|synthesize|resolve|close|other
- response_implied_expected_reaction: agreement|concession|clarification|rebuttal|escalation|accommodation|synthesis|closure|uncertain

Binary integer fields use 0 or 1:
explicit_synthesis, explicit_resolution, explicit_closure.
Every output object must contain item_id and every field."""


def load_annotation_items(data_dir: str, context_turns: int = 3) -> list[dict]:
    items = []
    for path in sorted(Path(data_dir, "transcripts").glob("*.json")):
        transcript = json.loads(path.read_text())
        turns = transcript["turns"]
        for position, turn in enumerate(turns):
            prior = turns[max(0, position - context_turns):position]
            items.append({
                "item_id": f"{transcript['conv_id']}::{int(turn['turn'])}",
                "conv_id": transcript["conv_id"],
                "turn": int(turn["turn"]),
                "speaker": turn["speaker"],
                "role": turn.get("role"),
                "topic_id": transcript.get("topic_id"),
                "condition": transcript.get("condition"),
                "prior_context": [
                    {
                        "speaker": item.get("speaker"),
                        "role": item.get("role"),
                        "text": item.get("text", ""),
                    }
                    for item in prior
                ],
                "current_response": turn.get("text", ""),
            })
    return items


def leakage_safe_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    """Batch at most one turn per conversation, rotating across corpus strata."""
    by_conversation: dict[str, deque] = defaultdict(deque)
    conversation_strata = {}
    for item in items:
        conv_id = item["conv_id"]
        by_conversation[conv_id].append(item)
        conversation_strata.setdefault(
            conv_id, (str(item.get("topic_id", "")), str(item.get("condition", "")))
        )
    by_stratum: dict[tuple[str, str], deque] = defaultdict(deque)
    for conv_id, stratum in conversation_strata.items():
        by_stratum[stratum].append(conv_id)
    conversation_order = []
    strata = sorted(by_stratum)
    while any(by_stratum.values()):
        for stratum in strata:
            if by_stratum[stratum]:
                conversation_order.append(by_stratum[stratum].popleft())
    pending = []
    for _ in range(max((len(queue) for queue in by_conversation.values()), default=0)):
        for conv_id in conversation_order:
            if by_conversation[conv_id]:
                pending.append(by_conversation[conv_id].popleft())
    batches = []
    while pending:
        batch, used_conversations, keep = [], set(), []
        for item in pending:
            if len(batch) < batch_size and item["conv_id"] not in used_conversations:
                batch.append(item)
                used_conversations.add(item["conv_id"])
            else:
                keep.append(item)
        batches.append(batch)
        pending = keep
    return batches


def parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith(chr(96) * 3):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Annotator response did not contain a JSON array.")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, list):
        raise ValueError("Annotator response must be a JSON array.")
    return value


def normalize_annotation(
    row: dict,
    expected_id: str,
    output_fields: Sequence[str] = OUTPUT_FIELDS,
) -> dict:
    if row.get("item_id") != expected_id:
        raise ValueError(f"Expected item_id {expected_id}, received {row.get('item_id')}")
    normalized: dict[str, Any] = {"item_id": expected_id}
    selected = set(output_fields)
    unknown = selected.difference(OUTPUT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown annotation fields: {sorted(unknown)}")
    for field in (name for name in CONTINUOUS_FIELDS if name in selected):
        value = float(row[field])
        if not 0 <= value <= 4:
            raise ValueError(f"{field} must be between 0 and 4")
        normalized[field] = value
    for field, choices in CATEGORICAL_FIELDS.items():
        if field not in selected:
            continue
        value = str(row[field]).strip().lower()
        if value not in choices:
            raise ValueError(f"Invalid {field}: {value}")
        normalized[field] = value
    for field in (name for name in BINARY_FIELDS if name in selected):
        value = int(row[field])
        if value not in (0, 1):
            raise ValueError(f"{field} must be 0 or 1")
        normalized[field] = value
    return normalized


def annotate_batch(
    client,
    batch: list[dict],
    max_retries: int,
    output_fields: Sequence[str] = OUTPUT_FIELDS,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    items_by_id = {item["item_id"]: item for item in batch}
    if len(items_by_id) != len(batch):
        raise ValueError("Annotation batch contains duplicate item IDs.")
    unresolved = dict(items_by_id)
    annotations: dict[str, dict] = {}
    last_error = None
    for attempt in range(max_retries):
        try:
            payload = [
                {
                    "item_id": item["item_id"], "topic_id": item["topic_id"],
                    "speaker": item["speaker"], "role": item["role"],
                    "condition": item["condition"], "prior_context": item["prior_context"],
                    "current_response": item["current_response"],
                }
                for item in unresolved.values()
            ]
            expected_ids = list(unresolved)
            message = ChatMessage(
                role="user",
                content=(
                    "Annotate these independent items. Return exactly one object for each "
                    "of these item_ids and preserve them verbatim: "
                    f"{json.dumps(expected_ids, ensure_ascii=False)}\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            )
            raw = client.chat(
                [ChatMessage(role="system", content=system_prompt), message],
                max_tokens=max(2000, len(unresolved) * 700),
            )
            parsed = parse_json_array(raw)
            returned_ids = [row.get("item_id") for row in parsed]
            counts = Counter(returned_ids)
            invalid = {}
            for row in parsed:
                item_id = row.get("item_id")
                if item_id not in unresolved or counts[item_id] != 1:
                    continue
                try:
                    annotations[item_id] = normalize_annotation(
                        row, item_id, output_fields=output_fields
                    )
                except Exception as error:
                    invalid[item_id] = str(error)
            for item_id in annotations:
                unresolved.pop(item_id, None)
            if not unresolved:
                return [annotations[item["item_id"]] for item in batch]

            missing = sorted(set(unresolved) - set(returned_ids))
            extra = sorted(set(returned_ids) - set(items_by_id), key=str)
            duplicate = sorted(
                (item_id for item_id, count in counts.items() if count > 1), key=str
            )
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            if duplicate:
                details.append(f"duplicate={duplicate}")
            if invalid:
                details.append(f"invalid={invalid}")
            if not details:
                details.append(f"unresolved={sorted(unresolved)}")
            last_error = ValueError("Annotator response mismatch: " + "; ".join(details))
        except Exception as error:
            last_error = error
        if attempt + 1 < max_retries:
            time.sleep(min(2 ** attempt, 30))

    # A model can repeatedly truncate or corrupt a response for a large batch.
    # Preserve any valid rows recovered above and subdivide only the unresolved
    # items. The singleton base case still raises the original useful error.
    if len(unresolved) > 1:
        remaining = list(unresolved.values())
        midpoint = len(remaining) // 2
        for smaller_batch in (remaining[:midpoint], remaining[midpoint:]):
            for annotation in annotate_batch(
                client,
                smaller_batch,
                max_retries,
                output_fields=output_fields,
                system_prompt=system_prompt,
            ):
                annotations[annotation["item_id"]] = annotation
        return [annotations[item["item_id"]] for item in batch]
    raise RuntimeError(f"Annotation batch failed after {max_retries} attempts: {last_error}")


def read_completed(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def complete_journal_rows(
    rows: list[dict],
    output_fields: Sequence[str] = OUTPUT_FIELDS,
    annotation_schema: str = ANNOTATION_SCHEMA,
) -> list[dict]:
    """Keep the latest complete row for each item/pass under this schema."""
    required = {"item_id", "conv_id", "turn", "annotator_id", *output_fields}
    latest = {}
    for row in rows:
        if row.get("annotation_schema") != annotation_schema:
            continue
        if not required.issubset(row):
            continue
        latest[(row["item_id"], row["annotator_id"])] = row
    return list(latest.values())


def write_csvs(
    rows: list[dict],
    raw_csv: Path,
    aggregated_csv: Path,
    output_fields: Sequence[str] = OUTPUT_FIELDS,
) -> None:
    frame = pd.DataFrame(rows)
    columns = ["conv_id", "turn", "annotator_id", *output_fields]
    frame[columns].sort_values(["conv_id", "turn", "annotator_id"]).to_csv(
        raw_csv, index=False
    )
    aggregate_annotations(frame[columns]).to_csv(aggregated_csv, index=False)


def run_annotations(
    data_dir: str,
    judge_model: str,
    registry_path: str,
    out_dir: str,
    passes: int,
    batch_size: int,
    context_turns: int,
    max_retries: int,
) -> tuple[Path, Path]:
    if passes < 1 or batch_size < 1:
        raise ValueError("passes and batch_size must be positive")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "annotation_journal.jsonl"
    raw_csv = output / "annotations_raw.csv"
    aggregated_csv = output / "annotations.csv"
    items = load_annotation_items(data_dir, context_turns)
    item_lookup = {item["item_id"]: item for item in items}
    completed = complete_journal_rows(read_completed(journal))
    completed_keys = {(row["item_id"], row["annotator_id"]) for row in completed}
    registry = load_model_registry(registry_path)
    client = build_client(judge_model, registry)

    with journal.open("a") as handle:
        for pass_index in range(passes):
            annotator_id = f"{judge_model}_pass_{pass_index + 1}"
            remaining = [
                item for item in items
                if (item["item_id"], annotator_id) not in completed_keys
            ]
            batches = leakage_safe_batches(remaining, batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                annotations = annotate_batch(client, batch, max_retries)
                for annotation in annotations:
                    item_id = annotation.pop("item_id")
                    source = item_lookup[item_id]
                    row = {
                        "conv_id": source["conv_id"], "turn": source["turn"],
                        "annotator_id": annotator_id,
                        "annotation_schema": ANNOTATION_SCHEMA,
                        **annotation,
                    }
                    handle.write(json.dumps({"item_id": item_id, **row}, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.append({"item_id": item_id, **row})
                print(
                    f"{annotator_id}: batch {batch_index}/{len(batches)} "
                    f"({len(completed)} total annotations)",
                    flush=True,
                )
    write_csvs(completed, raw_csv, aggregated_csv)
    expected = len(items) * passes
    if len(completed) != expected:
        raise RuntimeError(f"Expected {expected} annotations, found {len(completed)}")
    return raw_csv, aggregated_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate frozen Track 1 responses")
    parser.add_argument("--data_dir", default="data/track1")
    parser.add_argument("--judge_model", default="claude")
    parser.add_argument("--registry", default="configs/models.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--context_turns", type=int, default=3)
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()
    raw, aggregated = run_annotations(
        args.data_dir, args.judge_model, args.registry, args.out_dir,
        args.passes, args.batch_size, args.context_turns, args.max_retries,
    )
    print(f"Raw annotations: {raw}")
    print(f"Aggregated annotations: {aggregated}")


if __name__ == "__main__":
    main()
