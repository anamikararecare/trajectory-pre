"""Q1 corpus planning, generation, audit, and E1 analysis CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.q1.design import load_protocol, write_plan


def _run_root(protocol_path: str, run_id: str) -> Path:
    protocol = load_protocol(protocol_path)
    return Path(protocol["data_root"]) / run_id


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def plan_command(args: argparse.Namespace) -> None:
    root = _run_root(args.protocol, args.run_id)
    plan = write_plan(args.protocol, str(root / "q1_plan.csv"))
    if args.smoke:
        plan = (
            plan[plan["condition"].eq("self_play")]
            .groupby("group_model", sort=False)
            .head(1)
            .reset_index(drop=True)
        )
        plan["task_index"] = range(len(plan))
        plan.to_csv(root / "q1_plan.csv", index=False)
    counts = plan["condition"].value_counts().to_dict()
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "plan": str(root / "q1_plan.csv"),
                "conversations": len(plan),
                "conditions": counts,
                "models": int(plan["group_model"].nunique()),
            },
            indent=2,
        )
    )


def generate_command(args: argparse.Namespace) -> None:
    from src.q1.q1_generate import run_generation_shard

    root = run_generation_shard(
        protocol_path=args.protocol,
        run_id=args.run_id,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        quality_judge_key=args.quality_judge,
        quality_judge_registry=args.quality_judge_registry,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"Q1 shard {args.shard_index}/{args.num_shards} complete: {root}")


def audit_command(args: argparse.Namespace) -> None:
    from src.q1.corpus import corpus_inventory

    root = _run_root(args.protocol, args.run_id)
    inventory = corpus_inventory(root)
    if inventory.empty:
        raise FileNotFoundError(f"No Q1 plan or data found under {root}")
    report = {
        "run_id": args.run_id,
        "planned_conversations": int(inventory["planned"].sum()),
        "complete_transcripts": int(inventory["transcript_complete"].sum()),
        "complete_activations": int(inventory["activation_complete"].sum()),
        "analysis_ready": int(inventory["analysis_ready"].sum()),
        "missing_transcripts": int(
            (inventory["planned"] & ~inventory["transcript_complete"]).sum()
        ),
        "missing_activations": int(
            (inventory["planned"] & ~inventory["activation_complete"]).sum()
        ),
        "unexpected_transcripts": int(
            (~inventory["planned"] & inventory["transcript_complete"]).sum()
        ),
        "unexpected_activations": int(
            (~inventory["planned"] & inventory["activation_complete"]).sum()
        ),
    }
    print(json.dumps(report, indent=2))
    if report["missing_transcripts"] or report["missing_activations"]:
        raise SystemExit(1)


def e1_command(args: argparse.Namespace) -> None:
    from src.q1.corpus import (
        corpus_inventory,
        filter_q1_dataset,
        load_q1_dataset,
        parse_turn_range_edges,
        validate_factorial_balance,
    )
    from src.q1.e1_layerwise import (
        Q1_STATE_TARGETS,
        add_response_text_embeddings,
        run_e1,
        summarize_peak_layers,
    )
    from src.track1_probing.variables import registry_frame
    from src.q1.text_vad import (
        DEFAULT_VAD_MODEL, VAD_COLUMNS, add_text_vad_scores,
        default_vad_cache_path,
    )

    edges = parse_turn_range_edges(args.turn_range_edges)
    frame = load_q1_dataset(
        args.run_dir,
        annotations=args.annotations,
        geometry_path=args.geometry_turns,
        turn_range_edges=edges,
        require_complete=args.require_complete,
    )
    selected_pairs = _parse_csv(args.conversation_pairs)
    selected_topics = _parse_csv(args.topics)
    selected_role_orders = _parse_csv(args.role_orders)
    selected_conditions = _parse_csv(args.conditions)
    frame = filter_q1_dataset(
        frame,
        conversation_pairs=selected_pairs,
        topics=selected_topics,
        role_orders=selected_role_orders,
        conditions=selected_conditions,
    )
    if frame.empty:
        raise ValueError("No Q1 rows match the requested E1 selection.")
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
    requested_targets = _parse_csv(args.targets)
    targets = (
        registry_frame()["name"].tolist()
        if requested_targets == ["all"]
        else requested_targets or list(Q1_STATE_TARGETS)
    )
    selected_models = _parse_csv(args.models)
    selected_ranges = _parse_csv(args.turn_ranges)
    if selected_models:
        frame = frame[frame["model"].isin(selected_models)].copy()
    if selected_ranges:
        frame = frame[frame["turn_range"].isin(selected_ranges)].copy()
    if not args.no_vad and set(targets).intersection(VAD_COLUMNS):
        vad_cache = args.vad_cache or default_vad_cache_path(
            args.run_dir, args.vad_model
        )
        frame = add_text_vad_scores(
            frame, model_name=args.vad_model, cache_path=vad_cache,
            batch_size=args.vad_batch_size, device=args.vad_device,
        )
    if not args.no_text_embeddings:
        frame = add_response_text_embeddings(frame, args.embedding_model)
    scores, folds, predictions, skipped = run_e1(
        frame,
        targets=targets,
        models=_parse_csv(args.models),
        turn_ranges=_parse_csv(args.turn_ranges),
        group_column=args.cv_group,
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "e1_layerwise_scores.csv", index=False)
    summarize_peak_layers(scores).to_csv(
        output / "e1_peak_layer_scores.csv", index=False
    )
    folds.to_csv(output / "e1_fold_scores.csv", index=False)
    predictions.to_csv(output / "e1_oof_predictions.csv", index=False)
    skipped.to_csv(output / "e1_skipped.csv", index=False)
    registry_frame().to_csv(output / "e1_variable_registry.csv", index=False)
    if set(VAD_COLUMNS).issubset(frame.columns):
        frame[[
            "conv_id", "turn", "model", "text_sha256", *VAD_COLUMNS,
            "vad_model", "vad_model_revision",
        ]].to_csv(output / "e1_text_vad_scores.csv", index=False)
    inventory = corpus_inventory(args.run_dir)
    inventory["selected"] = inventory["conv_id"].astype(str).isin(
        selected_conversations
    )
    inventory.to_csv(output / "e1_corpus_inventory.csv", index=False)
    (
        frame.groupby(
            [
                "model", "turn_range", "topic_id", "condition",
                "conversation_pair", "role_order",
            ],
            dropna=False,
        )
        .size()
        .rename("n_turns")
        .reset_index()
        .to_csv(output / "e1_turn_range_inventory.csv", index=False)
    )
    if not scores.empty:
        from src.q1.e1_figures import export_e1_figures

        export_e1_figures(output)
    print(
        f"Q1 E1 complete: {len(scores)} layerwise estimates across "
        f"{scores['model'].nunique() if not scores.empty else 0} model(s). "
        f"Results: {output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 corpus and experiment workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="write a deterministic Q1 plan")
    plan.add_argument("--protocol", default="configs/q1_protocol.yaml")
    plan.add_argument("--run-id", default="q1_minimum_v1")
    plan.add_argument(
        "--smoke",
        action="store_true",
        help="write one self-play conversation per model",
    )
    plan.set_defaults(function=plan_command)

    generate = subparsers.add_parser(
        "generate", help="generate one resumable model-group shard"
    )
    generate.add_argument("--protocol", default="configs/q1_protocol.yaml")
    generate.add_argument("--run-id", default="q1_minimum_v1")
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    generate.add_argument(
        "--quality-judge",
        default="gpt",
        help="Registry key for the external gate; use 'none' to disable",
    )
    generate.add_argument(
        "--quality-judge-registry", default="configs/models.yaml"
    )
    generate.add_argument("--load-in-4bit", action="store_true")
    generate.set_defaults(function=generate_command)

    audit = subparsers.add_parser("audit", help="check Q1 corpus completeness")
    audit.add_argument("--protocol", default="configs/q1_protocol.yaml")
    audit.add_argument("--run-id", default="q1_minimum_v1")
    audit.set_defaults(function=audit_command)

    e1 = subparsers.add_parser(
        "e1",
        help="run model/layer/variable probes over percentage turn ranges",
    )
    e1.add_argument(
        "--run-dir", default="data/q1_data/q1_minimum_v1"
    )
    e1.add_argument("--annotations")
    e1.add_argument(
        "--geometry-turns",
        help="Optional Q1-keyed CSV containing registered geometry variables",
    )
    e1.add_argument("--vad-model", default="RobroKools/vad-bert")
    e1.add_argument("--vad-cache")
    e1.add_argument("--vad-batch-size", type=int, default=32)
    e1.add_argument("--vad-device", help="Torch device such as cuda:0 or cpu")
    e1.add_argument(
        "--no-vad", action="store_true",
        help="Skip expressed valence/arousal/dominance scoring",
    )
    e1.add_argument("--out-dir", default="results/q1/e1")
    e1.add_argument(
        "--conversation-pairs",
        help=(
            "Comma-separated ordered agent-A:agent-B pairs; for example "
            "gemma2-9b:qwen2.5-3b"
        ),
    )
    e1.add_argument("--topics", help="Optional comma-separated topic IDs")
    e1.add_argument(
        "--role-orders",
        help="Optional comma-separated agent-A:agent-B role orders",
    )
    e1.add_argument(
        "--conditions",
        help="Optional comma-separated conditions such as self_play,mixed_play",
    )
    e1.add_argument(
        "--require-balanced",
        action="store_true",
        help=(
            "Require equal conversation counts across every selected "
            "pair × topic × role-order cell"
        ),
    )
    e1.add_argument("--models", help="Comma-separated model registry keys")
    e1.add_argument(
        "--targets",
        help="Comma-separated targets; defaults to E1 core variables; use all for registry",
    )
    e1.add_argument(
        "--turn-range-edges",
        default="0,25,50,75,100",
        help="Increasing conversation-turn percentages starting at 0 and ending at 100",
    )
    e1.add_argument(
        "--turn-ranges",
        help="Optional comma-separated subset such as 00-25%%,75-100%%",
    )
    e1.add_argument(
        "--cv-group",
        choices=["topic_id", "conv_id"],
        default="topic_id",
    )
    e1.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2"
    )
    e1.add_argument(
        "--no-text-embeddings",
        action="store_true",
        help="Use metadata and lagged state, but omit the response-text baseline",
    )
    e1.add_argument(
        "--require-complete",
        action="store_true",
        help="Refuse to run unless every planned conversation has both files",
    )
    e1.set_defaults(function=e1_command)
    from src.q1.e2_cli import add_e2_parser

    add_e2_parser(subparsers)

    from src.q1.e3_cli import add_e3_parser

    add_e3_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "num_shards", 1) <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(
        args, "num_shards", 1
    ):
        raise ValueError("--shard-index must be in [0, num-shards)")
    args.function(args)


if __name__ == "__main__":
    main()
