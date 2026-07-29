"""Audit the analysis-ready portion of a generated Q1 corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.q1.corpus import (
    corpus_inventory,
    filter_q1_dataset,
    load_q1_dataset,
    parse_turn_range_edges,
    validate_factorial_balance,
)
from src.q1.e1_layerwise import Q1_STATE_TARGETS, available_layers


def available_corpus_report(
    run_dir: str | Path,
    annotations: str | None = None,
    geometry_path: str | None = None,
    turn_range_edges: str = "0,25,50,75,100",
    conversation_pairs: list[str] | None = None,
    topics: list[str] | None = None,
    role_orders: list[str] | None = None,
    conditions: list[str] | None = None,
    require_balanced: bool = False,
) -> dict:
    """Describe exactly which currently completed artifacts will be analyzed."""
    inventory = corpus_inventory(run_dir)
    inventory = filter_q1_dataset(
        inventory,
        conversation_pairs=conversation_pairs,
        topics=topics,
        role_orders=role_orders,
        conditions=conditions,
    )
    frame = load_q1_dataset(
        run_dir,
        annotations=annotations,
        geometry_path=geometry_path,
        turn_range_edges=parse_turn_range_edges(turn_range_edges),
        require_complete=False,
    )
    frame = filter_q1_dataset(
        frame,
        conversation_pairs=conversation_pairs,
        topics=topics,
        role_orders=role_orders,
        conditions=conditions,
    )
    if frame.empty:
        raise ValueError("No Q1 rows match the requested selection.")
    if require_balanced:
        validate_factorial_balance(
            frame,
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
    model_rows = []
    for model, rows in frame.groupby("model", sort=True):
        model_rows.append(
            {
                "model": str(model),
                "turns": int(len(rows)),
                "conversations": int(rows["conv_id"].nunique()),
                "topics": int(rows["topic_id"].nunique()),
                "conditions": sorted(
                    rows["condition"].dropna().astype(str).unique().tolist()
                ),
                "layers": available_layers(frame, str(model)),
            }
        )
    available_targets = [
        target
        for target in Q1_STATE_TARGETS
        if target in frame
        and pd.to_numeric(frame[target], errors="coerce").notna().any()
    ]
    planned = int(inventory["planned"].sum()) if not inventory.empty else 0
    ready = (
        int(inventory["analysis_ready"].sum()) if not inventory.empty else 0
    )
    return {
        "run_dir": str(Path(run_dir)),
        "planned_conversations": planned,
        "analysis_ready_conversations": ready,
        "not_ready_conversations": max(0, planned - ready),
        "loaded_conversations": int(frame["conv_id"].nunique()),
        "loaded_turns": int(len(frame)),
        "topics": int(frame["topic_id"].nunique()),
        "conditions": sorted(
            frame["condition"].dropna().astype(str).unique().tolist()
        ),
        "turn_ranges": list(
            dict.fromkeys(frame["turn_range"].dropna().astype(str))
        ),
        "models": model_rows,
        "available_core_targets_before_vad": available_targets,
        "annotations": annotations,
        "geometry": geometry_path,
        "selection": {
            "conversation_pairs": conversation_pairs or [],
            "topics": topics or [],
            "role_orders": role_orders or [],
            "conditions": conditions or [],
            "factorially_balanced": require_balanced,
        },
        "loader_contract": (
            "transcript and activation intersection; incomplete planned "
            "conversations are skipped"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the currently analysis-ready Q1 corpus"
    )
    parser.add_argument(
        "--run-dir", default="data/q1_data/q1_minimum_v1"
    )
    parser.add_argument("--annotations")
    parser.add_argument("--geometry-turns")
    parser.add_argument(
        "--turn-range-edges", default="0,25,50,75,100"
    )
    parser.add_argument("--conversation-pairs")
    parser.add_argument("--topics")
    parser.add_argument("--role-orders")
    parser.add_argument("--conditions")
    parser.add_argument("--require-balanced", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()
    report = available_corpus_report(
        args.run_dir,
        annotations=args.annotations,
        geometry_path=args.geometry_turns,
        turn_range_edges=args.turn_range_edges,
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
    )
    rendered = json.dumps(report, indent=2)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
