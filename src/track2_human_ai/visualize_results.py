"""Redraw Track 2 figures from saved tables without recomputing models."""

from __future__ import annotations

import os

import pandas as pd

from src.track2_human_ai.accommodation import plot_accommodation_results
from src.track2_human_ai.emotion_overlay import plot_emotion_results
from src.track2_human_ai.reference_geometry import plot_reference_geometry


def generate_track2_figures(results_dir: str, out_dir: str, n_bins: int = 10) -> None:
    turns = pd.read_csv(os.path.join(results_dir, "turn_level_self_play_projections.csv"))
    endpoints = (
        turns[turns.role == "assistant"]
        .sort_values("turn_idx")
        .groupby("conv_id")
        .tail(1)
        .reset_index(drop=True)
    )
    basin = pd.read_csv(os.path.join(results_dir, "basin_separation_by_topic.csv"))
    plot_reference_geometry(endpoints, basin, out_dir)

    accommodation = pd.read_csv(os.path.join(results_dir, "accommodation_by_turn.csv"))
    plot_accommodation_results(accommodation, out_dir, n_progress_bins=n_bins)

    emotion = pd.read_csv(os.path.join(results_dir, "emotion_by_turn.csv"))
    plot_emotion_results(emotion, out_dir, n_bins=n_bins)
