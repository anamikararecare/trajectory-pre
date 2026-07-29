"""
CLI entrypoint for Track 2.

Usage:
    python -m src.track2_human_ai.run_track2 fetch --dataset lmsys-chat-1m \
        --min_turns 6 --out data/track2/filtered_conversations.jsonl

    python -m src.track2_human_ai.run_track2 geometry \
        --in data/track2/filtered_conversations.jsonl \
        --reference results/track1/geometry/self_play_reference.npz --out_dir results/track2

    python -m src.track2_human_ai.run_track2 accommodation \
        --in data/track2/filtered_conversations.jsonl --out_dir results/track2

    python -m src.track2_human_ai.run_track2 emotion \
        --in data/track2/filtered_conversations.jsonl --out_dir results/track2
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()


def cmd_fetch(args):
    from src.track2_human_ai.load_corpus import fetch_and_filter

    fetch_and_filter(
        dataset_name=args.dataset,
        out_path=args.out,
        topics_path=args.topics,
        min_turns=args.min_turns,
        max_conversations=args.max_conversations,
    )


def cmd_geometry(args):
    from src.track2_human_ai.reference_geometry import run_reference_geometry_analysis

    run_reference_geometry_analysis(
        conversations_path=getattr(args, "in"),
        out_dir=args.out_dir,
        reference_path=args.reference,
        min_ai_turns_per_conv=args.min_ai_turns,
        min_convs_per_model_topic=args.min_convs_per_model_topic,
        min_models_per_topic=args.min_models_per_topic,
    )


def cmd_accommodation(args):
    from src.track2_human_ai.accommodation import run_accommodation_analysis

    run_accommodation_analysis(
        conversations_path=getattr(args, "in"),
        out_dir=args.out_dir,
        min_ai_turns_per_conv=args.min_ai_turns,
        n_progress_bins=args.n_bins,
    )


def cmd_emotion(args):
    from src.track2_human_ai.emotion_overlay import run_emotion_overlay

    run_emotion_overlay(
        conversations_path=getattr(args, "in"), out_dir=args.out_dir, n_bins=args.n_bins
    )


def cmd_visualize(args):
    from src.track2_human_ai.visualize_results import generate_track2_figures

    generate_track2_figures(args.results_dir, args.out_dir, n_bins=args.n_bins)
    print(f"Redrew readable Track 2 figures under {args.out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Track 2: human-AI trajectories")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download + filter a human-AI conversation corpus")
    p_fetch.add_argument("--dataset", choices=["lmsys-chat-1m", "wildchat"], default="lmsys-chat-1m")
    p_fetch.add_argument("--topics", default="configs/topics.yaml")
    p_fetch.add_argument("--min_turns", type=int, default=6)
    p_fetch.add_argument("--max_conversations", type=int, default=2000)
    p_fetch.add_argument("--out", required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_geo = sub.add_parser("geometry", help="project human-AI turns into Track 1 self-play basis")
    p_geo.add_argument("--reference", required=True, help="Track 1 self_play_reference.npz")
    p_geo.add_argument("--in", dest="in", required=True)
    p_geo.add_argument("--out_dir", default="results/track2")
    p_geo.add_argument("--min_ai_turns", type=int, default=3)
    p_geo.add_argument("--min_convs_per_model_topic", type=int, default=5)
    p_geo.add_argument("--min_models_per_topic", type=int, default=2)
    p_geo.set_defaults(func=cmd_geometry)

    p_acc = sub.add_parser("accommodation", help="run accommodation analysis (2b)")
    p_acc.add_argument("--in", dest="in", required=True)
    p_acc.add_argument("--out_dir", default="results/track2")
    p_acc.add_argument("--min_ai_turns", type=int, default=3)
    p_acc.add_argument("--n_bins", type=int, default=10)
    p_acc.set_defaults(func=cmd_accommodation)

    p_emo = sub.add_parser("emotion", help="run emotion overlay analysis (2c)")
    p_emo.add_argument("--in", dest="in", required=True)
    p_emo.add_argument("--out_dir", default="results/track2")
    p_emo.add_argument("--n_bins", type=int, default=5)
    p_emo.set_defaults(func=cmd_emotion)

    p_viz = sub.add_parser("visualize", help="redraw readable figures from saved Track 2 tables")
    p_viz.add_argument("--results_dir", default="results/track2")
    p_viz.add_argument("--out_dir", default="results/track2")
    p_viz.add_argument("--n_bins", type=int, default=10)
    p_viz.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
