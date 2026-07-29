"""Command-line integration for Q1 experiment E3."""

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


def _floats(value: str | None) -> list[float] | None:
    parsed = _csv(value)
    return [float(item) for item in parsed] if parsed else None


def e3_command(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.q1.corpus import (
        corpus_inventory,
        load_q1_dataset,
        parse_turn_range_edges,
    )
    from src.q1.e1_layerwise import add_response_text_embeddings
    from src.q1.e3_subspaces import (
        E3_DEFAULT_RANKS,
        E3_RIDGE_ALPHAS,
        E3_VARIABLE_FAMILIES,
        run_e3,
        save_e3_results,
    )
    from src.q1.text_vad import (
        VAD_COLUMNS,
        add_text_vad_scores,
        default_vad_cache_path,
    )

    requested_families = _csv(args.families)
    if requested_families:
        unknown = sorted(set(requested_families) - E3_VARIABLE_FAMILIES.keys())
        if unknown:
            raise ValueError(f"Unknown E3 variable families: {unknown}")
        families = {
            name: E3_VARIABLE_FAMILIES[name] for name in requested_families
        }
    else:
        families = E3_VARIABLE_FAMILIES

    frame = load_q1_dataset(
        args.run_dir,
        annotations=args.annotations,
        geometry_path=args.geometry_turns,
        turn_range_edges=parse_turn_range_edges(args.turn_range_edges),
        require_complete=args.require_complete,
    )
    models = _csv(args.models)
    turn_ranges = _csv(args.turn_ranges)
    if models:
        frame = frame[frame["model"].isin(models)].copy()
    if turn_ranges:
        frame = frame[frame["turn_range"].isin(turn_ranges)].copy()
    needs_vad = any(
        set(targets).intersection(VAD_COLUMNS) for targets in families.values()
    )
    if needs_vad and not args.no_vad:
        vad_cache = args.vad_cache or default_vad_cache_path(
            args.run_dir, args.vad_model
        )
        frame = add_text_vad_scores(
            frame,
            model_name=args.vad_model,
            cache_path=vad_cache,
            batch_size=args.vad_batch_size,
            device=args.vad_device,
        )
    if not args.no_text_embeddings:
        frame = add_response_text_embeddings(frame, args.embedding_model)

    results = run_e3(
        frame,
        families=families,
        ranks=_integers(args.ranks) or E3_DEFAULT_RANKS,
        models=models,
        turn_ranges=turn_ranges,
        layers=_integers(args.layers),
        group_column=args.cv_group,
        alphas=_floats(args.alphas) or E3_RIDGE_ALPHAS,
        transfer_rank=args.transfer_rank,
        run_cross_turn=not args.skip_cross_turn,
    )
    output = Path(args.out_dir)
    save_e3_results(results, output)
    pd.DataFrame(
        [
            {"family": family, "variable": variable}
            for family, variables in families.items()
            for variable in variables
        ]
    ).to_csv(output / "e3_variable_families.csv", index=False)
    corpus_inventory(args.run_dir).to_csv(
        output / "e3_corpus_inventory.csv", index=False
    )
    (
        frame.groupby(
            ["model", "turn_range", "topic_id", "condition"], dropna=False
        )
        .size()
        .rename("n_turns")
        .reset_index()
        .to_csv(output / "e3_turn_range_inventory.csv", index=False)
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
        ].to_csv(output / "e3_text_vad_scores.csv", index=False)
    from src.q1.e3_figures import export_e3_figures

    figures = export_e3_figures(output)
    summary = {
        "rank_estimates": len(results.rank_scores),
        "selected_subspaces": len(results.subspace_manifest),
        "overlap_estimates": len(results.overlap),
        "cross_turn_estimates": len(results.cross_turn),
        "figures": list(figures),
        "results": str(output),
    }
    print(json.dumps(summary, indent=2))


def add_e3_parser(subparsers: argparse._SubParsersAction) -> None:
    e3 = subparsers.add_parser(
        "e3",
        help="estimate variable-family subspace rank, overlap, and turn transfer",
    )
    e3.add_argument("--run-dir", default="data/q1_data/q1_minimum_v1")
    e3.add_argument("--annotations")
    e3.add_argument("--geometry-turns")
    e3.add_argument("--out-dir", default="results/q1/e3")
    e3.add_argument("--models", help="Comma-separated model registry keys")
    e3.add_argument(
        "--families",
        help="Comma-separated families: stance, agreement_conflict, personality, expressed_vad",
    )
    e3.add_argument("--layers", help="Optional comma-separated layer numbers")
    e3.add_argument(
        "--ranks",
        default="1,2,4,8",
        help="Candidate dimensions; each family's full rank is always added",
    )
    e3.add_argument("--alphas", default="1,10,100")
    e3.add_argument(
        "--turn-range-edges",
        default="0,25,50,75,100",
    )
    e3.add_argument(
        "--turn-ranges",
        help="Optional comma-separated percentage ranges such as 00-25%%,25-50%%",
    )
    e3.add_argument(
        "--cv-group",
        choices=["topic_id", "conv_id"],
        default="topic_id",
    )
    e3.add_argument("--transfer-rank", type=int, default=2)
    e3.add_argument(
        "--skip-cross-turn",
        action="store_true",
        help="Skip the source-range to destination-range transfer analysis",
    )
    e3.add_argument("--vad-model", default="RobroKools/vad-bert")
    e3.add_argument("--vad-cache")
    e3.add_argument("--vad-batch-size", type=int, default=32)
    e3.add_argument("--vad-device")
    e3.add_argument("--no-vad", action="store_true")
    e3.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    e3.add_argument("--no-text-embeddings", action="store_true")
    e3.add_argument("--require-complete", action="store_true")
    e3.set_defaults(function=e3_command)
