"""Paper-aligned SBERT geometry for Track 1 generated debates.

The analysis first establishes the output-space effect before activation probes:
topic-centered self-play basins, mixed-play contraction, partnerward pull,
dominance, and off-axis drift. Primary metrics stay in the full embedding
space; PCA is only a shared visualization/reference basis.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src.common.embeddings import (
    basin_separation_score,
    embed_texts,
    fit_reference_pca,
    topic_center,
)


def _load_turns(transcripts_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(transcripts_dir, "*.json"))):
        with open(path) as transcript_file:
            transcript = json.load(transcript_file)
        required = {"agent_a_model", "agent_b_model", "condition", "topic_id"}
        missing = required.difference(transcript)
        if missing:
            raise ValueError(
                f"{path} predates geometry metadata and is missing {sorted(missing)}."
            )
        for turn in transcript["turns"]:
            speaker = turn["speaker"]
            model = transcript[f"agent_{speaker}_model"]
            other = "b" if speaker == "a" else "a"
            rows.append(
                {
                    "conv_id": transcript["conv_id"],
                    "topic_id": transcript["topic_id"],
                    "condition": transcript["condition"],
                    "speaker": speaker,
                    "model": model,
                    "partner_model": transcript[f"agent_{other}_model"],
                    "role": turn["role"],
                    "turn": turn["turn"],
                    "agent_turn": turn.get("agent_turn", turn["turn"] // 2 + 1),
                    "text": turn["text"],
                }
            )
    if not rows:
        raise ValueError(f"No transcript JSON files found in {transcripts_dir}.")
    return pd.DataFrame(rows)


def _endpoint_table(turns: pd.DataFrame) -> pd.DataFrame:
    return (
        turns.sort_values("turn")
        .groupby(["conv_id", "speaker"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _bootstrap_interval(values: np.ndarray, seed: int, n_resamples: int) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]))


def run_geometry_analysis(
    transcripts_dir: str,
    out_dir: str,
    sbert_model: str = "all-MiniLM-L6-v2",
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    turns = _load_turns(transcripts_dir)
    embeddings = embed_texts(turns["text"].tolist(), model_name=sbert_model)
    centered = topic_center(embeddings, turns["topic_id"].tolist())
    turns["centered_embedding"] = list(centered)

    self_turns = turns[turns.condition == "self_play"]
    if self_turns.empty:
        raise ValueError("Track 1 geometry requires self_play transcripts.")
    pca = fit_reference_pca(
        np.stack(self_turns["centered_embedding"].values), n_components=2
    )
    np.savez_compressed(
        os.path.join(out_dir, "self_play_reference.npz"),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        sbert_model=np.asarray(sbert_model),
    )

    projected = pca.transform(np.stack(turns["centered_embedding"].values))
    turns["sp_pc1"], turns["sp_pc2"] = projected[:, 0], projected[:, 1]
    turns.drop(columns=["centered_embedding"]).to_csv(
        os.path.join(out_dir, "turn_geometry.csv"), index=False
    )

    endpoints = _endpoint_table(turns)
    self_endpoints = endpoints[endpoints.condition == "self_play"]
    anchors = (
        self_endpoints.groupby(["model", "topic_id"])["centered_embedding"]
        .apply(lambda values: np.mean(np.stack(values), axis=0))
        .reset_index()
    )
    if anchors.model.nunique() < 2:
        raise ValueError(
            "At least two models with self-play endpoints are required. Generate "
            "the partner self-play control before computing mixed-play geometry."
        )

    basin = basin_separation_score(
        np.stack(anchors.centered_embedding.values), anchors.model.tolist()
    )
    pd.DataFrame(basin).T.to_csv(os.path.join(out_dir, "basin_separation.csv"))

    anchor_map = {
        (row.model, row.topic_id): row.centered_embedding
        for row in anchors.itertuples(index=False)
    }
    turns = add_turn_geometry_variables(turns, anchor_map)
    turns.drop(columns=["centered_embedding"]).to_csv(
        os.path.join(out_dir, "turn_geometry.csv"), index=False
    )
    mixed = endpoints[endpoints.condition == "mixed_play"]
    mixed_means = (
        mixed.groupby(["model", "partner_model", "topic_id"])["centered_embedding"]
        .apply(lambda values: np.mean(np.stack(values), axis=0))
        .reset_index()
    )
    mixed_map = {
        (row.model, row.partner_model, row.topic_id): row.centered_embedding
        for row in mixed_means.itertuples(index=False)
    }

    metric_rows = []
    pairs = sorted(
        {
            tuple(sorted((row.model, row.partner_model)))
            for row in mixed_means.itertuples(index=False)
            if row.model != row.partner_model
        }
    )
    for model_a, model_b in pairs:
        topics = sorted(
            set(anchors[anchors.model == model_a].topic_id)
            & set(anchors[anchors.model == model_b].topic_id)
        )
        for topic_id in topics:
            keys = (
                (model_a, topic_id),
                (model_b, topic_id),
                (model_a, model_b, topic_id),
                (model_b, model_a, topic_id),
            )
            if not all(key in (anchor_map if len(key) == 2 else mixed_map) for key in keys):
                continue
            s_a, s_b = anchor_map[keys[0]], anchor_map[keys[1]]
            m_a, m_b = mixed_map[keys[2]], mixed_map[keys[3]]
            axis = s_b - s_a
            axis_norm_sq = float(np.dot(axis, axis))
            if axis_norm_sq <= 1e-12:
                continue
            axis_norm = np.sqrt(axis_norm_sq)
            alpha_a = float(np.dot(m_a - s_a, axis) / axis_norm_sq)
            alpha_b = float(np.dot(m_b - s_b, -axis) / axis_norm_sq)
            off_a = float(np.linalg.norm((m_a - s_a) - alpha_a * axis) / axis_norm)
            off_b = float(np.linalg.norm((m_b - s_b) + alpha_b * axis) / axis_norm)
            contraction = 1 - float(np.linalg.norm(m_a - m_b) / axis_norm)
            metric_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "topic_id": topic_id,
                    "contraction": contraction,
                    "alpha_a_toward_b": alpha_a,
                    "alpha_b_toward_a": alpha_b,
                    "dominance_a_over_b": alpha_b - alpha_a,
                    "off_axis_a": off_a,
                    "off_axis_b": off_b,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(out_dir, "mixed_play_metrics_by_topic.csv"), index=False)
    summaries = []
    if not metrics.empty:
        value_columns = [
            "contraction",
            "alpha_a_toward_b",
            "alpha_b_toward_a",
            "dominance_a_over_b",
            "off_axis_a",
            "off_axis_b",
        ]
        for pair, group in metrics.groupby(["model_a", "model_b"]):
            summary = {"model_a": pair[0], "model_b": pair[1], "n_topics": len(group)}
            for offset, column in enumerate(value_columns):
                values = group[column].to_numpy(float)
                low, high = _bootstrap_interval(
                    values, seed=seed + offset, n_resamples=bootstrap_resamples
                )
                summary[f"{column}_mean"] = float(values.mean())
                summary[f"{column}_ci_low"] = low
                summary[f"{column}_ci_high"] = high
            summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, "mixed_play_metrics_summary.csv"), index=False)
    return {"basin": basin, "mixed_play": summary_df}

def add_turn_geometry_variables(
    turns: pd.DataFrame,
    anchor_map: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    """Derive full-space per-turn basin and semantic-motion variables."""
    out = turns.sort_values(["conv_id", "speaker", "turn"]).copy()
    leaning, off_axis = [], []
    for row in out.itertuples(index=False):
        own = anchor_map.get((row.model, row.topic_id))
        partner = anchor_map.get((row.partner_model, row.topic_id))
        point = row.centered_embedding
        if own is None or partner is None:
            leaning.append(np.nan)
            off_axis.append(np.nan)
            continue
        axis = partner - own
        denominator = float(np.dot(axis, axis))
        if denominator <= 1e-12:
            leaning.append(np.nan)
            off_axis.append(np.nan)
            continue
        progress = float(np.dot(point - own, axis) / denominator)
        residual = point - own - progress * axis
        leaning.append(progress)
        off_axis.append(float(np.linalg.norm(residual) / np.sqrt(denominator)))
    out["basin_leaning"] = leaning
    out["off_axis_distance"] = off_axis
    out["partnerward_basin_velocity"] = out.groupby(
        ["conv_id", "speaker"]
    )["basin_leaning"].diff()
    velocity = pd.Series(np.nan, index=out.index, dtype=float)
    for _, speaker_turns in out.groupby(["conv_id", "speaker"]):
        prior = None
        for index, vector in zip(speaker_turns.index, speaker_turns["centered_embedding"]):
            if prior is not None:
                velocity.at[index] = float(np.linalg.norm(vector - prior))
            prior = vector
    out["semantic_velocity"] = velocity
    out["semantic_acceleration"] = out.groupby(
        ["conv_id", "speaker"]
    )["semantic_velocity"].diff()
    return out
