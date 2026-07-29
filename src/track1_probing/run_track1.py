"""
CLI entrypoint for Track 1.

Usage:
    python -m src.track1_probing.run_track1 generate --local_model qwen2.5-3b \
        --partner_model qwen2.5-7b --topics configs/topics.yaml \
        --n_topics 5 --n_turns 20 --paper_compatible --out_dir data/track1

    python -m src.track1_probing.run_track1 probe --data_dir data/track1 \
        --experiment all --out_dir results/track1
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()


def cmd_generate(args):
    from src.common.debate_prompts import load_topics
    from src.track1_probing.generate_debates import run_debate_set

    topics = load_topics(args.topics)[: args.n_topics]
    run_debate_set(
        topics=topics,
        local_model_key=args.local_model,
        partner_model_key=args.partner_model,
        n_turns_per_agent=args.n_turns,
        out_dir=args.out_dir,
        load_in_4bit=args.load_in_4bit,
        paper_compatible=args.paper_compatible,
        stance_judge_model=None if args.stance_judge == "self" else args.stance_judge,
        seed=args.seed,
    )
    print(f"Done. Transcripts + activations written under {args.out_dir}/")


def cmd_probe(args):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from src.track1_probing.cache_activations import load_dataset
    from src.track1_probing.probes import (
        experiment_1a_concurrent,
        experiment_1b_predictive,
        experiment_1c_cross_agent,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_dataset(args.data_dir)
    print(f"Loaded {len(df)} turn-rows from {df['conv_id'].nunique()} conversations.")

    results = []
    if args.experiment in ("1a", "all"):
        r = experiment_1a_concurrent(df)
        results.append(r)
        print("\n[1a concurrent]\n", r)

    if args.experiment in ("1b", "all"):
        r = experiment_1b_predictive(df)
        results.append(r)
        print("\n[1b predictive]\n", r)

        pivot = r.pivot(index="layer", columns="horizon", values="probe_r2")
        if not pivot.empty:
            plt.figure(figsize=(6, 4))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
            plt.title("1b: predictive R^2 (probe) by layer x horizon")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "heatmap_predictive.png"), dpi=150)
            plt.close()
            print(f"Saved {args.out_dir}/heatmap_predictive.png")

    if args.experiment in ("1c", "all"):
        r = experiment_1c_cross_agent(df)
        results.append(r)
        print("\n[1c cross-agent]\n", r)

    if results:
        out_csv = os.path.join(args.out_dir, "probe_scores.csv")
        pd.concat(results, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")


def cmd_geometry(args):
    from src.track1_probing.trajectory_geometry import run_geometry_analysis

    result = run_geometry_analysis(
        transcripts_dir=os.path.join(args.data_dir, "transcripts"),
        out_dir=args.out_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(f"Saved Track 1 geometry under {args.out_dir}/")
    return result


def cmd_visualize(args):
    from src.track1_probing.visualize_results import generate_all_figures

    paths = generate_all_figures(args.data_dir, args.results_dir, args.out_dir)
    print("Saved Track 1 figures:")
    for path in paths:
        print(f"  {path}")

def cmd_replay(args):
    from src.track1_probing.replay import replay_corpus

    path = replay_corpus(
        data_dir=args.data_dir,
        out_root=args.out_root,
        replay_id=args.replay_id,
        registry_path=args.registry,
        topics_path=args.topics,
        model_keys=args.models,
        load_in_4bit=args.load_in_4bit,
        validation_turns=args.validation_turns,
        allow_failed_validation=args.allow_failed_validation,
        validation_only=args.validation_only,
    )
    print(f"Saved replayed activations and manifest under {path}")


def cmd_analyze(args):
    import pandas as pd

    from src.track1_probing.cache_activations import load_dataset
    from src.track1_probing.snapshot_analysis import (
        EXPERIMENT_TARGETS,
        add_time_matched_text_embeddings,
        measurement_audit,
        original_replay_sensitivity,
        partner_transfer_summary,
        snapshot_decode,
    )
    from src.track1_probing.variables import annotation_reliability, registry_frame

    score_columns = [
        "experiment", "target", "relation", "delta_target", "horizon", "layer",
        "snapshot", "n", "metric", "null_score", "shuffled_score",
        "baseline_score", "activation_plus_baseline_score", "incremental_score",
        "time_legal_baseline", "prospectively_predictive_eligible",
        "cv_group",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    if args.annotations:
        annotation_reliability(args.annotations).to_csv(
            os.path.join(args.out_dir, "annotation_reliability.csv"), index=False
        )
    frame = load_dataset(
        args.data_dir,
        replay_dir=args.replay_dir,
        annotations=args.annotations,
        geometry_path=args.geometry_turns,
        allow_unvalidated_replay=args.allow_unvalidated_replay,
    )
    if args.sample_keys:
        sample_keys = pd.read_csv(args.sample_keys)
        required = {"conv_id", "turn"}
        missing = required.difference(sample_keys)
        if missing:
            raise ValueError(f"Sample-key file is missing columns: {sorted(missing)}")
        keys = sample_keys[["conv_id", "turn"]].drop_duplicates()
        frame = frame.merge(keys, on=["conv_id", "turn"], how="inner", validate="one_to_one")
        if frame.empty:
            raise ValueError("Sample-key file did not match any Track 1 turns")
    analysis_cv_group = args.cv_group
    if args.preliminary_fast:
        conversation_codes, conversations = pd.factorize(
            frame["conv_id"], sort=True
        )
        fold_count = min(4, len(conversations))
        if fold_count < 2:
            raise ValueError("Preliminary fast analysis requires at least two conversations")
        frame["preliminary_fold"] = conversation_codes % fold_count
        analysis_cv_group = "preliminary_fold"
        print(
            f"Preliminary fast mode: {len(frame)} turns, "
            f"{len(conversations)} conversations, {fold_count} grouped folds; "
            "shuffled refits disabled; 1D limited to horizon 1.",
            flush=True,
        )
    if not args.no_text_embeddings:
        frame = add_time_matched_text_embeddings(frame, args.embedding_model)
    measurement_audit(frame).to_csv(
        os.path.join(args.out_dir, "experiment_1a_measurement_audit.csv"), index=False
    )
    registry = registry_frame()
    registry.to_csv(os.path.join(args.out_dir, "variable_registry.csv"), index=False)
    identity_columns = [
        column for column in (
            "conv_id", "turn", "agent_turn", "speaker", "model", "partner_model",
            "topic_id", "role", "condition",
        ) if column in frame
    ]
    registered_columns = [
        column for column in registry["name"] if column in frame
    ]
    frame[[*identity_columns, *registered_columns]].to_csv(
        os.path.join(args.out_dir, "turn_variables.csv"), index=False
    )
    big_five_columns = [
        column for column in frame if column.startswith("observer_big5_")
    ]
    frame[[*identity_columns, *big_five_columns]].to_csv(
        os.path.join(args.out_dir, "big_five_turn_states.csv"), index=False
    )

    if args.experiment == "all":
        requested = list(EXPERIMENT_TARGETS)
    else:
        requested = [item.strip().upper() for item in args.experiment.split(",") if item.strip()]
        unknown = sorted(set(requested).difference(EXPERIMENT_TARGETS))
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}")
    if "1I" in requested:
        partner_transfer_summary(frame, EXPERIMENT_TARGETS["1I"]).to_csv(
            os.path.join(args.out_dir, "experiment_1i_partner_transfer.csv"), index=False
        )
    results, skipped, sensitivity_results = [], [], []
    oof_rows, fold_rows = [], []

    def checkpoint_probe_scores(experiment: str, target: str) -> None:
        checkpoint_scores = (
            pd.concat(results, ignore_index=True)
            if results else pd.DataFrame(columns=score_columns)
        )

        def write_atomic(frame: pd.DataFrame, filename: str) -> None:
            destination = os.path.join(args.out_dir, filename)
            temporary = f"{destination}.{os.getpid()}.tmp"
            frame.to_csv(temporary, index=False)
            os.replace(temporary, destination)

        write_atomic(checkpoint_scores, "snapshot_probe_scores.csv")
        write_atomic(
            checkpoint_scores[checkpoint_scores["experiment"].eq(experiment)],
            f"experiment_{experiment.lower()}_snapshot_probe_scores.csv",
        )
        skipped_frame = pd.DataFrame(
            skipped, columns=["experiment", "target", "reason"]
        )
        write_atomic(skipped_frame, "skipped_targets.csv")
        write_atomic(
            skipped_frame[skipped_frame["experiment"].eq(experiment)],
            f"experiment_{experiment.lower()}_skipped_targets.csv",
        )
        print(
            f"Checkpointed {experiment}/{target}: "
            f"{len(checkpoint_scores)} score rows",
            flush=True,
        )

    for experiment in requested:
        relation = "current"
        horizons = [0]
        snapshots = ("pre_generation", "early_response", "full_response", "final_window")
        delta_options = [False]
        if experiment == "1C":
            snapshots = ("pre_generation",)
        elif experiment == "1D":
            relation, horizons, delta_options = "future_self", [1, 2, 3, 4], [False, True]
            if args.preliminary_fast:
                horizons = [1]
        elif experiment == "1E":
            relation, horizons = "partner_next", [1]
        experiment_targets = EXPERIMENT_TARGETS[experiment]
        for target_index, target in enumerate(experiment_targets, start=1):
            print(
                f"{experiment}: target {target_index}/{len(experiment_targets)} "
                f"{target}",
                flush=True,
            )
            if target not in frame:
                skipped.append({"experiment": experiment, "target": target, "reason": "variable unavailable"})
                checkpoint_probe_scores(experiment, target)
                continue
            labels = frame[target].notna().sum()
            if labels < 6:
                skipped.append({"experiment": experiment, "target": target, "reason": "fewer than six labels"})
                checkpoint_probe_scores(experiment, target)
                continue
            work = frame.copy()
            for horizon in horizons:
                for delta in delta_options:
                    results.append(snapshot_decode(
                        work, target, relation=relation, horizon=horizon, delta=delta,
                        snapshots=snapshots, experiment=experiment,
                        detail_rows=None if args.preliminary_fast else oof_rows,
                        fold_rows=None if args.preliminary_fast else fold_rows,
                        group_column=analysis_cv_group,
                        compute_shuffled=not args.preliminary_fast,
                    ))
                    if not args.skip_sensitivity:
                        sensitivity_results.append(original_replay_sensitivity(
                            work, target, relation=relation, horizon=horizon, delta=delta,
                            experiment=experiment,
                        ))
            checkpoint_probe_scores(experiment, target)
    scores = (
        pd.concat(results, ignore_index=True)
        if results else pd.DataFrame(columns=score_columns)
    )
    scores.to_csv(os.path.join(args.out_dir, "snapshot_probe_scores.csv"), index=False)
    sensitivity = [item for item in sensitivity_results if not item.empty]
    sensitivity_frame = (
        pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame()
    )
    sensitivity_frame.to_csv(
        os.path.join(args.out_dir, "original_replay_sensitivity.csv"), index=False
    )
    oof_frame = pd.DataFrame(oof_rows)
    fold_frame = pd.DataFrame(fold_rows)
    skipped_frame = pd.DataFrame(
        skipped, columns=["experiment", "target", "reason"]
    )
    oof_frame.to_csv(
        os.path.join(args.out_dir, "oof_predictions.csv"), index=False
    )
    fold_frame.to_csv(
        os.path.join(args.out_dir, "fold_scores.csv"), index=False
    )
    skipped_frame.to_csv(
        os.path.join(args.out_dir, "skipped_targets.csv"), index=False
    )
    for experiment in EXPERIMENT_TARGETS:
        prefix = f"experiment_{experiment.lower()}"
        scores[scores["experiment"].eq(experiment)].to_csv(
            os.path.join(args.out_dir, f"{prefix}_snapshot_probe_scores.csv"),
            index=False,
        )
        skipped_frame[skipped_frame["experiment"].eq(experiment)].to_csv(
            os.path.join(args.out_dir, f"{prefix}_skipped_targets.csv"),
            index=False,
        )
    big_five_mask = scores["target"].astype(str).str.startswith("observer_big5_")
    scores[big_five_mask].to_csv(
        os.path.join(args.out_dir, "big_five_probe_scores.csv"), index=False
    )
    if not oof_frame.empty:
        big_five_oof = oof_frame[
            oof_frame["target"].astype(str).str.startswith("observer_big5_")
        ]
    else:
        big_five_oof = oof_frame
    big_five_oof.to_csv(
        os.path.join(args.out_dir, "big_five_oof_predictions.csv"), index=False
    )
    if not fold_frame.empty:
        big_five_folds = fold_frame[
            fold_frame["target"].astype(str).str.startswith("observer_big5_")
        ]
    else:
        big_five_folds = fold_frame
    big_five_folds.to_csv(
        os.path.join(args.out_dir, "big_five_fold_scores.csv"), index=False
    )
    print(f"Saved refactored Track 1 analysis under {args.out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Track 1: stance/persuasion probing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="run debates + cache activations")
    p_gen.add_argument("--local_model", required=True, help="key in configs/models.yaml, must be hookable")
    p_gen.add_argument("--partner_model", default=None, help="key in configs/models.yaml for mixed-play")
    p_gen.add_argument("--topics", default="configs/topics.yaml")
    p_gen.add_argument("--n_topics", type=int, default=5)
    p_gen.add_argument("--n_turns", type=int, default=20, help="responses per agent")
    p_gen.add_argument("--out_dir", default="data/track1")
    p_gen.add_argument("--load_in_4bit", action="store_true")
    p_gen.add_argument("--paper_compatible", action="store_true")
    p_gen.add_argument("--stance_judge", default="self", help="'self', 'configured', or registry key")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.set_defaults(func=cmd_generate)

    p_probe = sub.add_parser("probe", help="run probing experiments on cached data")
    p_probe.add_argument("--data_dir", default="data/track1")
    p_probe.add_argument("--out_dir", default="results/track1")
    p_probe.add_argument("--experiment", choices=["1a", "1b", "1c", "all"], default="all")
    p_probe.set_defaults(func=cmd_probe)
    p_replay = sub.add_parser("replay", help="teacher-force frozen recorded turns")
    p_replay.add_argument("--data_dir", default="data/track1")
    p_replay.add_argument("--out_root", default=None)
    p_replay.add_argument("--replay_id", default=None)
    p_replay.add_argument("--registry", default="configs/models.yaml")
    p_replay.add_argument("--topics", default="configs/topics.yaml")
    p_replay.add_argument("--models", nargs="*", default=None)
    p_replay.add_argument("--load_in_4bit", action="store_true")
    p_replay.add_argument("--validation_turns", type=int, default=24)
    p_replay.add_argument("--validation_only", action="store_true")
    p_replay.add_argument(
        "--allow_failed_validation",
        action="store_true",
        help="store unmatched replay snapshots with an explicit manifest warning",
    )
    p_replay.set_defaults(func=cmd_replay)

    p_analyze = sub.add_parser("analyze", help="run refactored experiments 1A-1I")
    p_analyze.add_argument("--data_dir", default="data/track1")
    p_analyze.add_argument("--replay_dir", required=True)
    p_analyze.add_argument("--annotations", default=None)
    p_analyze.add_argument(
        "--geometry_turns", default="results/track1/geometry/turn_geometry.csv"
    )
    p_analyze.add_argument("--out_dir", default="results/track1/refactored")
    p_analyze.add_argument(
        "--experiment",
        default="all",
        help="One experiment, a comma-separated list (for example 1B,1C,1D), or all",
    )
    p_analyze.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    p_analyze.add_argument("--no_text_embeddings", action="store_true")
    p_analyze.add_argument("--allow_unvalidated_replay", action="store_true")
    p_analyze.add_argument(
        "--sample_keys",
        help="Optional CSV of conv_id,turn keys restricting all analyses to a snapshot",
    )
    p_analyze.add_argument(
        "--cv_group",
        choices=["topic_id", "conv_id"],
        default="topic_id",
        help="Leave-one-group-out unit; conv_id is intended only for preliminary exploration",
    )
    p_analyze.add_argument(
        "--skip_sensitivity",
        action="store_true",
        help="Skip unchanged original-versus-replay sensitivity calculations",
    )
    p_analyze.add_argument(
        "--preliminary_fast",
        action="store_true",
        help=(
            "Exploratory screening: four conversation fold buckets, no shuffled "
            "refits or OOF detail, and 1D horizon 1 only"
        ),
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_geometry = sub.add_parser(
        "geometry", help="establish paper-style SBERT geometry before probing"
    )
    p_geometry.add_argument("--data_dir", default="data/track1")
    p_geometry.add_argument("--out_dir", default="results/track1/geometry")
    p_geometry.add_argument("--bootstrap_resamples", type=int, default=2000)
    p_geometry.add_argument("--seed", type=int, default=0)
    p_geometry.set_defaults(func=cmd_geometry)

    p_visualize = sub.add_parser("visualize", help="draw paper-style result figures")
    p_visualize.add_argument("--data_dir", default="data/track1")
    p_visualize.add_argument("--results_dir", default="results/track1")
    p_visualize.add_argument("--out_dir", default="results/track1/figures")
    p_visualize.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
