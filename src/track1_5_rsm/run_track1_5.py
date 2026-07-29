"""Command-line entrypoint for Track 1.5 RSM analysis."""

from __future__ import annotations

import argparse

from src.track1_5_rsm.analysis import run_track1_5


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build mean-centred, aligned cross-model activation RSMs and one "
            "RSM for every available Track 1 conversational variable."
        )
    )
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--turn_variables", required=True)
    parser.add_argument("--out_dir", default="results/track1_5")
    parser.add_argument(
        "--conv_id",
        action="append",
        dest="conv_ids",
        help="Conversation to analyze; repeat for multiple. Default: all eligible.",
    )
    parser.add_argument(
        "--snapshot",
        choices=[
            "pre_generation", "early_response", "full_response",
            "final_window", "final_token",
        ],
        default="full_response",
    )
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument(
        "--turns_per_model",
        type=int,
        default=10,
        help=(
            "Agent turns retained for each model. The default 10 + 10 corresponds "
            "to the existing 20-turn preliminary conversations."
        ),
    )
    parser.add_argument(
        "--alignment_mode",
        choices=["leave_one_conversation_out", "within_conversation"],
        default="leave_one_conversation_out",
    )
    parser.add_argument("--alignment_rank", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow_unvalidated_models",
        action="store_true",
        help=(
            "Allow replay arrays whose model lacks a passed original/replay gate; "
            "the manifest will mark the results exploratory."
        ),
    )
    parser.add_argument(
        "--no_variable_plots",
        action="store_true",
        help="Write variable RSM CSV/NPZ files but skip their individual PNGs.",
    )
    args = parser.parse_args()
    output = run_track1_5(
        replay_dir=args.replay_dir,
        turn_variables_path=args.turn_variables,
        out_dir=args.out_dir,
        conv_ids=args.conv_ids,
        snapshot=args.snapshot,
        n_layers=args.n_layers,
        max_turns=args.turns_per_model,
        alignment_mode=args.alignment_mode,
        alignment_rank=args.alignment_rank,
        permutations=args.permutations,
        seed=args.seed,
        allow_unvalidated_models=args.allow_unvalidated_models,
        plot_variables=not args.no_variable_plots,
    )
    print(f"Track 1.5 results written to {output}")


if __name__ == "__main__":
    main()

