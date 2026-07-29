"""Load and inspect generated Q1 corpora.

The Q1 generation format stores one response-pooled activation per
``layer__conversation_turn``.  This module intentionally does not introduce
activation snapshots: that pooled response representation is the sole E1
activation unit.
"""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.track1_probing.variables import (
    VARIABLES,
    add_lagged_behavioral_state,
    add_persona_baselines,
    derive_annotation_variables,
    derive_stance_variables,
    merge_annotations,
)


DEFAULT_TURN_RANGE_EDGES = (0.0, 25.0, 50.0, 75.0, 100.0)
_ACTIVATION_KEY = re.compile(r"^(?P<layer>-?\d+)__(?P<turn>\d+)$")
_VARIABLE_NAMES = {variable.name for variable in VARIABLES}


def _parse_ordered_pair(value: str, label: str) -> tuple[str, str]:
    parts = tuple(item.strip() for item in str(value).split(":"))
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{label} must use exactly one ':' separator, got {value!r}."
        )
    return parts


def filter_q1_dataset(
    frame: pd.DataFrame,
    conversation_pairs: Sequence[str] | None = None,
    topics: Sequence[str] | None = None,
    role_orders: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Filter complete conversations using explicit agent-A/agent-B metadata.

    Conversation pairs and role orders are ordered and use ``A:B`` syntax.
    Filtering at conversation level prevents a model-row filter from silently
    retaining the same model's conversations with unwanted partners.
    """
    out = frame
    filters = (
        (conversation_pairs, "conversation_pair"),
        (topics, "topic_id"),
        (role_orders, "role_order"),
        (conditions, "condition"),
    )
    for requested, column in filters:
        if not requested:
            continue
        if column not in out:
            raise ValueError(
                f"Q1 dataset cannot filter by {column!r}; column is absent."
            )
        values = list(requested)
        if column in {"conversation_pair", "role_order"}:
            values = [
                ":".join(_parse_ordered_pair(value, column))
                for value in values
            ]
        out = out[out[column].astype(str).isin(values)]
    return out.copy()


def validate_factorial_balance(
    frame: pd.DataFrame,
    factors: Sequence[str] = (
        "conversation_pair",
        "topic_id",
        "role_order",
    ),
    expected_levels: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Require equal conversation counts in every selected factorial cell."""
    columns = list(factors)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Balance factors are missing: {missing}")
    conversations = frame[["conv_id", *columns]].drop_duplicates()
    if conversations.empty:
        raise ValueError("The selected Q1 corpus contains no conversations.")
    if conversations["conv_id"].duplicated().any():
        raise ValueError(
            "A conversation maps to multiple values of a balance factor."
        )
    expected = expected_levels or {}
    levels = [
        sorted(str(value) for value in expected[column])
        if column in expected
        else sorted(conversations[column].dropna().astype(str).unique())
        for column in columns
    ]
    if any(not values for values in levels):
        raise ValueError("Every balance factor must contain at least one level.")
    cells = pd.MultiIndex.from_tuples(
        list(product(*levels)), names=columns
    )
    counts = (
        conversations.assign(
            **{
                column: conversations[column].astype(str)
                for column in columns
            }
        )
        .groupby(columns, dropna=False)
        .size()
        .reindex(cells, fill_value=0)
        .rename("n_conversations")
        .reset_index()
    )
    if counts["n_conversations"].nunique() != 1:
        deficient = counts[
            counts["n_conversations"].lt(counts["n_conversations"].max())
        ]
        examples = deficient.head(8).to_dict(orient="records")
        raise ValueError(
            "Selected Q1 conversations are not factorially balanced across "
            f"{columns}. Cell counts range from "
            f"{counts['n_conversations'].min()} to "
            f"{counts['n_conversations'].max()}; deficient examples: "
            f"{examples}"
        )
    return counts


def parse_turn_range_edges(value: str | Sequence[float]) -> tuple[float, ...]:
    """Validate percentage boundaries spanning the complete conversation."""
    if isinstance(value, str):
        edges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        edges = tuple(float(item) for item in value)
    if len(edges) < 2:
        raise ValueError("At least two turn-range boundaries are required.")
    if edges[0] != 0.0 or edges[-1] != 100.0:
        raise ValueError("Turn-range boundaries must start at 0 and end at 100.")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("Turn-range boundaries must be strictly increasing.")
    return edges


def turn_range_labels(edges: Sequence[float]) -> tuple[str, ...]:
    """Return stable human-readable labels such as ``00-25%``."""
    parsed = parse_turn_range_edges(edges)

    def format_edge(edge: float) -> str:
        return str(int(edge)) if edge.is_integer() else f"{edge:g}"

    return tuple(
        f"{format_edge(left).zfill(2)}-{format_edge(right).zfill(2)}%"
        for left, right in zip(parsed, parsed[1:])
    )


def add_turn_ranges(
    frame: pd.DataFrame,
    edges: Sequence[float] = DEFAULT_TURN_RANGE_EDGES,
) -> pd.DataFrame:
    """Add conversation-relative progress and percentage-range columns.

    Turn progress is one-indexed: in a 32-response conversation, turns 0--7
    occupy ``00-25%`` and turns 24--31 occupy ``75-100%``.
    """
    parsed = parse_turn_range_edges(edges)
    labels = turn_range_labels(parsed)
    out = frame.copy()
    if "conversation_turns" not in out:
        out["conversation_turns"] = out.groupby("conv_id")["turn"].transform(
            "count"
        )
    denominator = pd.to_numeric(out["conversation_turns"], errors="coerce")
    out["conversation_turn_number"] = pd.to_numeric(
        out["turn"], errors="coerce"
    ) + 1.0
    out["conversation_turn_pct"] = (
        100.0 * out["conversation_turn_number"] / denominator
    )
    out["turn_range"] = pd.cut(
        out["conversation_turn_pct"],
        bins=parsed,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("object")
    if out["turn_range"].isna().any():
        raise ValueError("Some turns fall outside the configured percentage ranges.")
    return out


def load_q1_transcripts(run_dir: str | Path) -> list[dict]:
    """Read generated transcripts verbatim and attach their source paths."""
    root = Path(run_dir) / "q1_transcripts"
    transcripts: list[dict] = []
    for path in sorted(root.glob("q1_transcript__*.json")):
        value = json.loads(path.read_text())
        value["_source_path"] = str(path)
        transcripts.append(value)
    return transcripts


def corpus_inventory(run_dir: str | Path) -> pd.DataFrame:
    """Describe planned, transcript-complete, and activation-complete samples."""
    root = Path(run_dir)
    plan_path = root / "q1_plan.csv"
    plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
    transcripts = {
        path.name.removeprefix("q1_transcript__").removesuffix(".json")
        for path in (root / "q1_transcripts").glob("q1_transcript__*.json")
    }
    activations = {
        path.name.removeprefix("q1_activations__").removesuffix(".npz")
        for path in (root / "q1_activations").glob("q1_activations__*.npz")
    }
    expected = set(plan["conv_id"]) if "conv_id" in plan else transcripts | activations
    rows = []
    plan_lookup = (
        plan.set_index("conv_id").to_dict(orient="index") if "conv_id" in plan else {}
    )
    for conv_id in sorted(expected | transcripts | activations):
        metadata = plan_lookup.get(conv_id, {})
        rows.append(
            {
                "conv_id": conv_id,
                "planned": conv_id in expected,
                "transcript_complete": conv_id in transcripts,
                "activation_complete": conv_id in activations,
                "analysis_ready": conv_id in transcripts and conv_id in activations,
                "topic_id": metadata.get("topic_id"),
                "condition": metadata.get("condition"),
                "group_model": metadata.get("group_model"),
                "agent_a_model": metadata.get("model_a"),
                "agent_b_model": metadata.get("model_b"),
                "agent_a_role": metadata.get("role_a"),
                "agent_b_role": metadata.get("role_b"),
                "conversation_pair": (
                    f"{metadata.get('model_a')}:{metadata.get('model_b')}"
                    if metadata.get("model_a") and metadata.get("model_b")
                    else None
                ),
                "role_order": (
                    f"{metadata.get('role_a')}:{metadata.get('role_b')}"
                    if metadata.get("role_a") and metadata.get("role_b")
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _activation_path(run_dir: Path, conv_id: str) -> Path:
    return run_dir / "q1_activations" / f"q1_activations__{conv_id}.npz"


def _merge_geometry(frame: pd.DataFrame, geometry_path: str | None) -> pd.DataFrame:
    if not geometry_path:
        return frame
    geometry = pd.read_csv(geometry_path)
    keys = ["conv_id", "turn"]
    if "speaker" in geometry and "speaker" in frame:
        keys.append("speaker")
    missing = [key for key in keys if key not in geometry]
    if missing:
        raise ValueError(f"Geometry file is missing keys: {missing}")
    columns = [
        column
        for column in geometry
        if column in keys or column in _VARIABLE_NAMES
    ]
    value_columns = [column for column in columns if column not in keys]
    if not value_columns:
        raise ValueError("Geometry file contains no registered Track 1 variables.")
    overlap = set(value_columns).intersection(frame.columns)
    if overlap:
        frame = frame.drop(columns=sorted(overlap))
    return frame.merge(
        geometry[columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def load_q1_dataset(
    run_dir: str | Path,
    annotations: str | None = None,
    geometry_path: str | None = None,
    turn_range_edges: Sequence[float] = DEFAULT_TURN_RANGE_EDGES,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Load Q1 turns, activations, and all available Track 1 variables."""
    root = Path(run_dir)
    inventory = corpus_inventory(root)
    if require_complete and (
        inventory.empty or not inventory["analysis_ready"].all()
    ):
        missing = (
            inventory.loc[~inventory["analysis_ready"], "conv_id"].tolist()
            if not inventory.empty
            else []
        )
        raise ValueError(
            f"Q1 corpus is incomplete: {len(missing)} planned conversations "
            "lack a transcript or activation file."
        )

    rows: list[dict] = []
    seen_turns: set[tuple[str, int]] = set()
    for transcript in load_q1_transcripts(root):
        conv_id = transcript["conv_id"]
        activation_path = _activation_path(root, conv_id)
        if not activation_path.exists():
            continue
        turns = transcript.get("turns", [])
        with np.load(activation_path, allow_pickle=False) as arrays:
            activation_lookup: dict[int, dict[int, np.ndarray]] = {}
            for key in arrays.files:
                match = _ACTIVATION_KEY.fullmatch(key)
                if match is None:
                    raise ValueError(
                        f"Malformed Q1 activation key {key!r} in {activation_path}"
                    )
                layer = int(match["layer"])
                turn_index = int(match["turn"])
                activation_lookup.setdefault(turn_index, {})[layer] = arrays[key]

            for turn in turns:
                turn_index = int(turn["turn"])
                identity = (conv_id, turn_index)
                if identity in seen_turns:
                    raise ValueError(f"Duplicate Q1 turn identity: {identity}")
                seen_turns.add(identity)
                speaker = str(turn["speaker"])
                model = turn.get(
                    "model", transcript.get(f"agent_{speaker}_model")
                )
                row = {
                    "conv_id": conv_id,
                    "topic_id": transcript["topic_id"],
                    "condition": transcript.get("condition", "unknown"),
                    "agent_a_model": transcript.get("agent_a_model"),
                    "agent_b_model": transcript.get("agent_b_model"),
                    "agent_a_role": transcript.get("agent_a_role"),
                    "agent_b_role": transcript.get("agent_b_role"),
                    "conversation_pair": (
                        f"{transcript.get('agent_a_model')}:"
                        f"{transcript.get('agent_b_model')}"
                    ),
                    "role_order": (
                        f"{transcript.get('agent_a_role')}:"
                        f"{transcript.get('agent_b_role')}"
                    ),
                    "seed": transcript.get("seed"),
                    "speaker": speaker,
                    "model": model,
                    "role": turn.get("role"),
                    "turn": turn_index,
                    "agent_turn": turn.get("agent_turn"),
                    "conversation_turns": len(turns),
                    "text": turn.get("text", ""),
                    "transcript_context_text": "\n".join(
                        str(prior.get("text", ""))
                        for prior in turns[:turn_index]
                    ),
                    "stance_score": turn.get("stance_score"),
                    "stance_confidence": turn.get("stance_confidence"),
                    "stance_battery_score": turn.get("stance_battery_score"),
                    "stance_battery_confidence": turn.get(
                        "stance_battery_confidence"
                    ),
                    "activation_pooling": "generated_response_token_mean",
                    "_source_path": transcript.get("_source_path"),
                }
                for layer, vector in activation_lookup.get(turn_index, {}).items():
                    row[f"layer_{layer}"] = vector
                rows.append(row)

    if not rows:
        raise ValueError(f"No analysis-ready Q1 samples found under {root}.")
    frame = derive_stance_variables(pd.DataFrame(rows))
    frame = merge_annotations(frame, annotations)
    frame = _merge_geometry(frame, geometry_path)
    frame = derive_annotation_variables(frame)
    frame = add_persona_baselines(frame)
    frame = add_lagged_behavioral_state(frame)
    return add_turn_ranges(frame, turn_range_edges)

