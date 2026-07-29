"""Resumable behavioral/persona annotation for analysis-ready Q1 turns."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from src.common.llm_client import build_client, load_model_registry
from src.q1.corpus import (
    corpus_inventory,
    filter_q1_dataset,
    load_q1_transcripts,
    validate_factorial_balance,
)
from src.track1_probing.offline_annotations import (
    ANNOTATION_SCHEMA,
    CATEGORICAL_FIELDS,
    OUTPUT_FIELDS,
    SYSTEM_PROMPT,
    annotate_batch,
    complete_journal_rows,
    leakage_safe_batches,
    read_completed,
    write_csvs,
)


Q1_PERSONA_FIELDS = (
    "perceived_persona_warmth",
    "perceived_persona_dominance",
    "perceived_persona_curiosity",
    "perceived_persona_structure",
    "perceived_persona_stability",
    "perceived_persona_deference",
    "perceived_persona_humility",
)
Q1_ANNOTATION_FIELD_GROUPS = {
    "persona": Q1_PERSONA_FIELDS,
    "categorical": tuple(CATEGORICAL_FIELDS),
}
Q1_DEFAULT_ANNOTATION_FIELD_GROUPS = ("persona", "categorical")
Q1_PERSONA_DEFINITIONS = {
    "perceived_persona_warmth": "visible warmth and agreeableness",
    "perceived_persona_dominance": "visible assertiveness and control",
    "perceived_persona_curiosity": "visible curiosity and openness",
    "perceived_persona_structure": "visible organization and deliberateness",
    "perceived_persona_stability": "visible calm versus reactivity",
    "perceived_persona_deference": "deference (0) versus forcefulness (4)",
    "perceived_persona_humility": "visible epistemic humility and qualification",
}


def resolve_q1_annotation_fields(
    groups: list[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve named Q1 judge-field groups without stance/VAD duplication."""
    selected_groups = tuple(groups or Q1_DEFAULT_ANNOTATION_FIELD_GROUPS)
    if selected_groups == ("all",):
        return selected_groups, tuple(OUTPUT_FIELDS)
    unknown = sorted(set(selected_groups) - Q1_ANNOTATION_FIELD_GROUPS.keys())
    if unknown:
        raise ValueError(
            f"Unknown Q1 annotation field groups: {unknown}. "
            f"Choose from {sorted(Q1_ANNOTATION_FIELD_GROUPS)} or all."
        )
    fields = tuple(
        dict.fromkeys(
            field
            for group in selected_groups
            for field in Q1_ANNOTATION_FIELD_GROUPS[group]
        )
    )
    return selected_groups, fields


def q1_annotation_prompt(fields: tuple[str, ...]) -> str:
    """Build a strict prompt containing only the requested judged fields."""
    if fields == tuple(OUTPUT_FIELDS):
        return SYSTEM_PROMPT
    persona = [field for field in fields if field in Q1_PERSONA_FIELDS]
    categorical = [field for field in fields if field in CATEGORICAL_FIELDS]
    sections = [
        "You are an offline behavioral annotator for frozen debate transcripts.",
        "Judge only PRIOR CONTEXT and CURRENT RESPONSE. Do not infer private "
        "mental states or stable personality.",
        "Return one JSON array and no prose. Preserve item_id exactly. Return "
        "only item_id and every requested field below.",
    ]
    if persona:
        definitions = "\n".join(
            f"- {field}: {Q1_PERSONA_DEFINITIONS[field]}"
            for field in persona
        )
        sections.append(
            "Persona fields use 0,1,2,3,4 and describe perceived interaction "
            f"style in this response:\n{definitions}"
        )
    if categorical:
        choices = "\n".join(
            f"- {field}: {'|'.join(sorted(CATEGORICAL_FIELDS[field]))}"
            for field in categorical
        )
        sections.append(f"Categorical choices:\n{choices}")
    return "\n\n".join(sections)

def _limited_ready_ids(
    inventory,
    maximum: int | None,
    conversation_pairs: list[str] | None = None,
    topics: list[str] | None = None,
    role_orders: list[str] | None = None,
    conditions: list[str] | None = None,
    require_balanced: bool = False,
) -> set[str]:
    ready = inventory[inventory["analysis_ready"]].copy()
    ready = filter_q1_dataset(
        ready,
        conversation_pairs=conversation_pairs,
        topics=topics,
        role_orders=role_orders,
        conditions=conditions,
    )
    if require_balanced:
        validate_factorial_balance(
            ready,
            expected_levels={
                key: values
                for key, values in {
                    "conversation_pair": conversation_pairs,
                    "topic_id": topics,
                    "role_order": role_orders,
                }.items()
                if values
            },
        )
    if maximum is None or maximum >= len(ready):
        return set(ready["conv_id"].astype(str))
    if maximum < 1:
        raise ValueError("--max-conversations must be positive")
    strata: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for row in ready.sort_values("conv_id").itertuples(index=False):
        strata[(str(row.topic_id), str(row.condition))].append(str(row.conv_id))
    selected = []
    ordered_strata = sorted(strata)
    while len(selected) < maximum and any(strata.values()):
        for stratum in ordered_strata:
            if strata[stratum] and len(selected) < maximum:
                selected.append(strata[stratum].popleft())
    return set(selected)


def load_q1_annotation_items(
    run_dir: str | Path,
    context_turns: int = 3,
    max_conversations: int | None = None,
    conversation_pairs: list[str] | None = None,
    topics: list[str] | None = None,
    role_orders: list[str] | None = None,
    conditions: list[str] | None = None,
    require_balanced: bool = False,
    field_groups: list[str] | None = None,
) -> list[dict]:
    """Build visible-context annotation items for analysis-ready conversations."""
    ready_ids = _limited_ready_ids(
        corpus_inventory(run_dir),
        max_conversations,
        conversation_pairs=conversation_pairs,
        topics=topics,
        role_orders=role_orders,
        conditions=conditions,
        require_balanced=require_balanced,
    )
    items = []
    for transcript in load_q1_transcripts(run_dir):
        if str(transcript["conv_id"]) not in ready_ids:
            continue
        turns = transcript["turns"]
        for position, turn in enumerate(turns):
            prior = turns[max(0, position - context_turns):position]
            items.append(
                {
                    "item_id": (
                        f"{transcript['conv_id']}::{int(turn['turn'])}"
                    ),
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
                }
            )
    return items


def run_q1_annotations(
    run_dir: str,
    judge_model: str,
    registry_path: str,
    out_dir: str,
    passes: int = 1,
    batch_size: int = 8,
    context_turns: int = 3,
    max_retries: int = 5,
    max_conversations: int | None = None,
    conversation_pairs: list[str] | None = None,
    topics: list[str] | None = None,
    role_orders: list[str] | None = None,
    conditions: list[str] | None = None,
    require_balanced: bool = False,
    field_groups: list[str] | None = None,
) -> tuple[Path, Path]:
    """Annotate current Q1 rows, resuming from a schema-versioned journal."""
    selected_groups, output_fields = resolve_q1_annotation_fields(field_groups)
    annotation_schema = (
        ANNOTATION_SCHEMA
        if selected_groups == ("all",)
        else f"{ANNOTATION_SCHEMA}__q1_{'_'.join(selected_groups)}"
    )
    system_prompt = q1_annotation_prompt(output_fields)
    if passes < 1 or batch_size < 1:
        raise ValueError("passes and batch size must be positive")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "annotation_journal.jsonl"
    raw_csv = output / "annotations_raw.csv"
    aggregated_csv = output / "annotations.csv"
    items = load_q1_annotation_items(
        run_dir,
        context_turns=context_turns,
        max_conversations=max_conversations,
        conversation_pairs=conversation_pairs,
        topics=topics,
        role_orders=role_orders,
        conditions=conditions,
        require_balanced=require_balanced,
    )
    if not items:
        raise ValueError("No analysis-ready Q1 turns are available to annotate")
    item_lookup = {item["item_id"]: item for item in items}
    journal_rows = read_completed(journal)
    completed_by_key = {}
    compatible_schemas = tuple(
        dict.fromkeys((ANNOTATION_SCHEMA, annotation_schema))
    )
    for compatible_schema in compatible_schemas:
        for row in complete_journal_rows(
            journal_rows,
            output_fields=output_fields,
            annotation_schema=compatible_schema,
        ):
            if row["item_id"] in item_lookup:
                completed_by_key[(
                    row["item_id"], row["annotator_id"]
                )] = row
    completed = list(completed_by_key.values())
    completed_keys = {
        (row["item_id"], row["annotator_id"]) for row in completed
    }
    client = build_client(
        judge_model, load_model_registry(registry_path)
    )

    with journal.open("a") as handle:
        for pass_index in range(passes):
            annotator_id = f"{judge_model}_pass_{pass_index + 1}"
            remaining = [
                item
                for item in items
                if (item["item_id"], annotator_id) not in completed_keys
            ]
            batches = leakage_safe_batches(remaining, batch_size)
            for batch_index, batch in enumerate(batches, start=1):
                annotations = annotate_batch(
                    client,
                    batch,
                    max_retries,
                    output_fields=output_fields,
                    system_prompt=system_prompt,
                )
                for annotation in annotations:
                    item_id = annotation.pop("item_id")
                    source = item_lookup[item_id]
                    row = {
                        "conv_id": source["conv_id"],
                        "turn": source["turn"],
                        "annotator_id": annotator_id,
                        "annotation_schema": annotation_schema,
                        **annotation,
                    }
                    journal_row = {"item_id": item_id, **row}
                    handle.write(
                        json.dumps(journal_row, ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    completed.append(journal_row)
                    completed_keys.add((item_id, annotator_id))
                print(
                    f"{annotator_id}: batch {batch_index}/{len(batches)} "
                    f"({len(completed)} completed rows)",
                    flush=True,
                )
    expected = len(items) * passes
    if len(completed) != expected:
        raise RuntimeError(
            f"Expected {expected} complete annotations, found {len(completed)}"
        )
    write_csvs(
        completed,
        raw_csv,
        aggregated_csv,
        output_fields=output_fields,
    )
    return raw_csv, aggregated_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate currently analysis-ready Q1 responses"
    )
    parser.add_argument(
        "--run-dir", default="data/q1_data/q1_minimum_v1"
    )
    parser.add_argument("--judge-model", default="gpt")
    parser.add_argument("--registry", default="configs/models.yaml")
    parser.add_argument(
        "--out-dir",
        default="data/q1_data/q1_minimum_v1/q1_annotations",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-turns", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Balanced deterministic subset for a pilot; omit for all ready conversations",
    )
    parser.add_argument("--conversation-pairs")
    parser.add_argument("--topics")
    parser.add_argument("--role-orders")
    parser.add_argument("--conditions")
    parser.add_argument("--require-balanced", action="store_true")
    parser.add_argument(
        "--field-groups",
        default=",".join(Q1_DEFAULT_ANNOTATION_FIELD_GROUPS),
        help="Comma-separated Q1 judge groups: persona,categorical; or all",
    )
    args = parser.parse_args()
    raw, aggregated = run_q1_annotations(
        run_dir=args.run_dir,
        judge_model=args.judge_model,
        registry_path=args.registry,
        out_dir=args.out_dir,
        passes=args.passes,
        batch_size=args.batch_size,
        context_turns=args.context_turns,
        max_retries=args.max_retries,
        max_conversations=args.max_conversations,
        conversation_pairs=(
            [item.strip() for item in args.conversation_pairs.split(",")]
            if args.conversation_pairs else None
        ),
        topics=(
            [item.strip() for item in args.topics.split(",")]
            if args.topics else None
        ),
        role_orders=(
            [item.strip() for item in args.role_orders.split(",")]
            if args.role_orders else None
        ),
        conditions=(
            [item.strip() for item in args.conditions.split(",")]
            if args.conditions else None
        ),
        require_balanced=args.require_balanced,
        field_groups=[
            item.strip() for item in args.field_groups.split(",") if item.strip()
        ],
    )
    print(f"Raw annotations: {raw}")
    print(f"Aggregated annotations: {aggregated}")


if __name__ == "__main__":
    main()
