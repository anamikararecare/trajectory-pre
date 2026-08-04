"""Command-line integration for Q1 experiment E2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _integers(value: str | None) -> list[int] | None:
    parsed = _csv(value)
    return [int(item) for item in parsed] if parsed else None


def e2_command(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.q1.corpus import (
        corpus_inventory,
        filter_q1_dataset,
        load_q1_dataset,
        parse_turn_range_edges,
        validate_factorial_balance,
    )
    from src.q1.e1_layerwise import (
        Q1_STATE_TARGETS,
        default_embedding_cache_path,
        add_response_text_embeddings,
    )
    from src.q1.e2_temporal import (
        DEFAULT_CONDITION_SCOPES,
        run_e2,
        save_e2_results,
    )
    from src.q1.text_vad import (
        VAD_COLUMNS,
        add_text_vad_scores,
        default_vad_cache_path,
    )
    from src.track1_probing.variables import registry_frame
    from src.q1.progress import ProgressReporter

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(args.progress_log or output / "e2_progress.jsonl")
    reporter.event("e2_load", status="started")

    frame = load_q1_dataset(
        args.run_dir,
        annotations=args.annotations,
        geometry_path=args.geometry_turns,
        turn_range_edges=parse_turn_range_edges(args.turn_range_edges),
        require_complete=args.require_complete,
    )
    selected_pairs = _csv(args.conversation_pairs)
    selected_topics = _csv(args.topics)
    selected_role_orders = _csv(args.role_orders)
    selected_conditions = _csv(args.conditions)
    frame = filter_q1_dataset(
        frame,
        conversation_pairs=selected_pairs,
        topics=selected_topics,
        role_orders=selected_role_orders,
        conditions=selected_conditions,
    )
    if frame.empty:
        raise ValueError("No Q1 rows match the requested E2 selection.")
    if args.require_balanced:
        validate_factorial_balance(
            frame,
            expected_levels={
                key: values
                for key, values in {
                    "conversation_pair": selected_pairs,
                    "topic_id": selected_topics,
                    "role_order": selected_role_orders,
                }.items()
                if values
            },
        )
    selected_conversations = set(frame["conv_id"].astype(str))
    requested_targets = _csv(args.targets)
    targets = (
        registry_frame()["name"].tolist()
        if requested_targets == ["all"]
        else requested_targets or list(Q1_STATE_TARGETS)
    )
    models = _csv(args.models)
    turn_ranges = _csv(args.turn_ranges)
    condition_scopes = _csv(args.condition_scopes) or list(
        DEFAULT_CONDITION_SCOPES
    )
    invalid_scopes = sorted(
        set(condition_scopes) - set(DEFAULT_CONDITION_SCOPES)
    )
    if invalid_scopes:
        raise ValueError(f"Unknown condition scopes: {invalid_scopes}")
    if models:
        frame = frame[frame["model"].isin(models)].copy()
    if turn_ranges:
        frame = frame[frame["turn_range"].isin(turn_ranges)].copy()
    if not args.no_vad and set(targets).intersection(VAD_COLUMNS):
        frame = add_text_vad_scores(
            frame,
            model_name=args.vad_model,
            cache_path=(
                args.vad_cache
                or default_vad_cache_path(args.run_dir, args.vad_model)
            ),
            batch_size=args.vad_batch_size,
            device=args.vad_device,
        )
    if not args.no_text_embeddings:
        embedding_cache = args.embedding_cache or default_embedding_cache_path(
            args.run_dir, args.embedding_model
        )
        frame = add_response_text_embeddings(
            frame, args.embedding_model, cache_path=embedding_cache
        )

    e1_results = args.e1_results
    if e1_results is None:
        candidate = output.parent / "e1"
        if (candidate / "e1_layerwise_scores.csv").is_file():
            e1_results = str(candidate)
    reporter.event(
        "e2_load", status="complete",
        e1_overall_source=e1_results or "computed",
        rows=len(frame),
    )
    results = run_e2(
        frame,
        targets=targets,
        models=models,
        turn_ranges=turn_ranges,
        layers=_integers(args.layers),
        condition_scopes=condition_scopes,
        group_column=args.cv_group,
        n_bootstrap=args.bootstrap_samples,
        run_cross_temporal_analysis=not args.skip_cross_temporal,
        e1_results_dir=e1_results,
        n_jobs=args.n_jobs,
        progress=reporter.event,
    )
    save_e2_results(results, output)
    registry_frame().to_csv(output / "e2_variable_registry.csv", index=False)
    inventory = corpus_inventory(args.run_dir)
    inventory["selected"] = inventory["conv_id"].astype(str).isin(
        selected_conversations
    )
    inventory.to_csv(output / "e2_corpus_inventory.csv", index=False)
    (
        frame.groupby(
            [
                "model", "condition", "turn_range", "topic_id",
                "conversation_pair", "role_order",
            ],
            dropna=False,
        )
        .size()
        .rename("n_turns")
        .reset_index()
        .to_csv(output / "e2_turn_range_inventory.csv", index=False)
    )
    if set(VAD_COLUMNS).issubset(frame.columns):
        frame[
            [
                "conv_id",
                "turn",
                "model",
                "text_sha256",
                *VAD_COLUMNS,
                "vad_model",
                "vad_model_revision",
            ]
        ].to_csv(output / "e2_text_vad_scores.csv", index=False)
    from src.q1.e2_figures import export_e2_figures

    figures = export_e2_figures(output)
    print(
        json.dumps(
            {
                "independent_estimates": len(results.independent_scores),
                "temporal_summary_cells": len(results.temporal_summary),
                "cross_temporal_estimates": len(
                    results.cross_temporal_scores
                ),
                "condition_contrasts": len(results.condition_contrasts),
                "figures": list(figures),
                "results": str(output),
            },
            indent=2,
        )
    )


def add_e2_parser(subparsers: argparse._SubParsersAction) -> None:
    e2 = subparsers.add_parser(
        "e2",
        help="run independent-range and cross-temporal decoding analyses",
    )
    e2.add_argument("--run-dir", default="data/q1_data/q1_minimum_v1")
    e2.add_argument("--annotations")
    e2.add_argument("--geometry-turns")
    e2.add_argument("--out-dir", default="results/q1/e2")
    e2.add_argument(
        "--e1-results",
        help="Completed E1 directory; defaults to the sibling e1 directory",
    )
    e2.add_argument("--n-jobs", type=int, default=4)
    e2.add_argument("--progress-log")
    e2.add_argument("--models")
    e2.add_argument("--conversation-pairs")
    e2.add_argument("--topics")
    e2.add_argument("--role-orders")
    e2.add_argument("--conditions")
    e2.add_argument(
        "--require-balanced",
        action="store_true",
        help=(
            "Require equal conversation counts across every selected "
            "pair × topic × role-order cell"
        ),
    )
    e2.add_argument(
        "--targets",
        help="Comma-separated variables; defaults to Q1 core; use all for registry",
    )
    e2.add_argument("--layers", help="Optional comma-separated layer numbers")
    e2.add_argument(
        "--turn-range-edges", default="0,25,50,75,100"
    )
    e2.add_argument("--turn-ranges")
    e2.add_argument(
        "--condition-scopes",
        default="overall,self_play,mixed_play",
        help="Any of overall,self_play,mixed_play",
    )
    e2.add_argument(
        "--cv-group",
        choices=["topic_id", "conv_id"],
        default="topic_id",
    )
    e2.add_argument("--bootstrap-samples", type=int, default=500)
    e2.add_argument("--skip-cross-temporal", action="store_true")
    e2.add_argument("--vad-model", default="RobroKools/vad-bert")
    e2.add_argument("--vad-cache")
    e2.add_argument("--vad-batch-size", type=int, default=32)
    e2.add_argument("--vad-device")
    e2.add_argument("--no-vad", action="store_true")
    e2.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    e2.add_argument("--embedding-cache")
    e2.add_argument("--no-text-embeddings", action="store_true")
    e2.add_argument("--require-complete", action="store_true")
    e2.set_defaults(function=e2_command)
