"""Confound-aware human-AI geometry using a Track 1 self-play reference.

This is an external-validity extension, not a replication: observational
human-AI logs remain confounded by model deployment, user, and collection
conditions. The fixed self-play PCA basis and topic-conditional summaries make
that limitation visible instead of treating partner-conditioned AI turns as
self-play.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.embeddings import basin_separation_score, embed_texts


def _load_conversations(path: str) -> list[dict]:
    with open(path) as corpus_file:
        return [json.loads(line) for line in corpus_file if line.strip()]


def _flatten(conversations: list[dict]) -> pd.DataFrame:
    rows = []
    for conversation in conversations:
        bucket = conversation.get("topic_bucket")
        if bucket is None:
            buckets = conversation.get("topic_buckets") or []
            bucket = buckets[0] if buckets else "unlabeled"
        for turn_index, turn in enumerate(conversation["turns"]):
            rows.append(
                {
                    "conv_id": conversation["conv_id"],
                    "model": conversation["model"],
                    "topic_bucket": bucket,
                    "turn_idx": turn_index,
                    "role": turn["role"],
                    "text": turn["text"],
                }
            )
    return pd.DataFrame(rows)


def _load_reference(reference_path: str, requested_model: str) -> tuple[np.ndarray, np.ndarray]:
    reference = np.load(reference_path)
    reference_model = str(reference["sbert_model"].item())
    if reference_model != requested_model:
        raise ValueError(
            f"Reference uses {reference_model!r}, but Track 2 requested "
            f"{requested_model!r}. Use the same embedding model."
        )
    return reference["pca_components"], reference["pca_mean"]


def run_reference_geometry_analysis(
    conversations_path: str,
    reference_path: str,
    out_dir: str,
    min_ai_turns_per_conv: int = 3,
    min_convs_per_model_topic: int = 5,
    min_models_per_topic: int = 2,
    sbert_model: str = "all-MiniLM-L6-v2",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    components, pca_mean = _load_reference(reference_path, sbert_model)
    df = _flatten(_load_conversations(conversations_path))

    ai_counts = df[df.role == "assistant"].groupby("conv_id").size()
    df = df[df.conv_id.isin(ai_counts[ai_counts >= min_ai_turns_per_conv].index)]
    df = df[~df.topic_bucket.isin(["general", "unlabeled", "multi_topic"])]

    coverage = (
        df[df.role == "assistant"]
        .drop_duplicates("conv_id")
        .groupby(["topic_bucket", "model"])
        .size()
        .rename("n_conversations")
        .reset_index()
    )
    coverage.to_csv(os.path.join(out_dir, "model_topic_coverage.csv"), index=False)
    eligible_cells = coverage[
        coverage.n_conversations >= min_convs_per_model_topic
    ]
    eligible_topics = (
        eligible_cells.groupby("topic_bucket").model.nunique()
    )
    eligible_topics = eligible_topics[eligible_topics >= min_models_per_topic].index
    eligible_pairs = set(
        map(
            tuple,
            eligible_cells[
                eligible_cells.topic_bucket.isin(eligible_topics)
            ][["topic_bucket", "model"]].itertuples(index=False, name=None),
        )
    )
    keep_conversations = {
        row.conv_id
        for row in df.drop_duplicates("conv_id").itertuples(index=False)
        if (row.topic_bucket, row.model) in eligible_pairs
    }
    df = df[df.conv_id.isin(keep_conversations)].copy()
    if df.empty:
        raise ValueError(
            "No model-topic cells have adequate cross-model coverage. Fetch more "
            "data or lower the explicit coverage thresholds."
        )

    embeddings = embed_texts(df.text.tolist(), model_name=sbert_model)
    df["embedding"] = list(embeddings)

    # Use only AI turns to estimate each topic offset, then apply that same
    # offset to human and AI turns. Human content never defines the centering.
    topic_means = {}
    for topic, group in df[df.role == "assistant"].groupby("topic_bucket"):
        topic_means[topic] = np.mean(np.stack(group.embedding.values), axis=0)
    centered = np.stack(
        [
            embedding - topic_means[topic]
            for embedding, topic in zip(df.embedding, df.topic_bucket)
        ]
    )
    df["centered_embedding"] = list(centered)
    projected = (centered - pca_mean) @ components.T
    df["sp_pc1"], df["sp_pc2"] = projected[:, 0], projected[:, 1]

    endpoints = (
        df[df.role == "assistant"]
        .sort_values("turn_idx")
        .groupby("conv_id")
        .tail(1)
        .reset_index(drop=True)
    )

    stratified_rows = []
    stratified_scores = {}
    for topic, group in endpoints.groupby("topic_bucket"):
        if group.model.nunique() < 2:
            continue
        scores = basin_separation_score(
            np.stack(group.centered_embedding.values), group.model.tolist()
        )
        stratified_scores[topic] = scores
        for model, values in scores.items():
            stratified_rows.append({"topic_bucket": topic, "model": model, **values})
    pd.DataFrame(stratified_rows).to_csv(
        os.path.join(out_dir, "basin_separation_by_topic.csv"), index=False
    )

    overall = basin_separation_score(
        np.stack(endpoints.centered_embedding.values), endpoints.model.tolist()
    )
    pd.DataFrame(overall).T.to_csv(
        os.path.join(out_dir, "exploratory_pooled_basin_separation.csv")
    )

    plot_reference_geometry(endpoints, pd.DataFrame(stratified_rows), out_dir)

    df.drop(columns=["embedding", "centered_embedding"]).to_csv(
        os.path.join(out_dir, "turn_level_self_play_projections.csv"), index=False
    )
    return {"stratified": stratified_scores, "pooled_exploratory": overall}


def plot_reference_geometry(
    endpoints: pd.DataFrame, basin_scores: pd.DataFrame, out_dir: str
) -> None:
    """Use highlight facets and a metric heatmap instead of a crowded legend."""
    os.makedirs(out_dir, exist_ok=True)
    models = endpoints.groupby("model").conv_id.nunique().sort_values(ascending=False).index
    ncols = 4
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.1 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    colors = sns.color_palette("colorblind", min(len(models), 10))
    for index, (ax, model) in enumerate(zip(axes, models)):
        group = endpoints[endpoints.model == model]
        ax.scatter(endpoints.sp_pc1, endpoints.sp_pc2, color="0.75", alpha=0.12, s=10)
        ax.scatter(
            group.sp_pc1,
            group.sp_pc2,
            color=colors[index % len(colors)],
            alpha=0.7,
            s=18,
        )
        ax.scatter(
            group.sp_pc1.mean(),
            group.sp_pc2.mean(),
            color=colors[index % len(colors)],
            edgecolor="black",
            marker="X",
            s=90,
            zorder=4,
        )
        ax.axhline(0, color="0.88", lw=0.7)
        ax.axvline(0, color="0.88", lw=0.7)
        ax.set_title(f"{model}  (n={group.conv_id.nunique()})", fontsize=9)
    for ax in axes[len(models):]:
        ax.set_visible(False)
    fig.supxlabel("Track 1 self-play PC1")
    fig.supylabel("Track 1 self-play PC2")
    fig.suptitle(
        "Human–AI endpoints in the Track 1 basis\ncolored model highlighted; all eligible endpoints in gray",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "endpoint_scatter_self_play_basis.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if not basin_scores.empty:
        matrix = basin_scores.pivot(index="model", columns="topic_bucket", values="S_basin")
        matrix = matrix.reindex(models.intersection(matrix.index))
        fig, ax = plt.subplots(
            figsize=(max(7, 2.4 * len(matrix.columns)), max(5, 0.42 * len(matrix) + 2))
        )
        sns.heatmap(
            matrix,
            mask=matrix.isna(),
            annot=True,
            fmt=".2f",
            cmap="crest",
            vmin=1,
            cbar_kws={"label": "S_basin (>1 means locally separated)"},
            ax=ax,
        )
        ax.set_xlabel("Topic bucket")
        ax.set_ylabel("AI model")
        ax.set_title("Topic-stratified endpoint basin separation")
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, "basin_separation_heatmap.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
