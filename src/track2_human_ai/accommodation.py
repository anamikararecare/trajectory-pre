"""
Experiment 2b: accommodation. We don't have a human-analogue of "self-play"
to build a clean anchor->partner axis the way the paper does (Eq. 7-10), so
instead we use each conversation's own first AI turn as that AI's local
anchor, and ask: does the AI's embedding drift toward the human's turns over
the course of the conversation, and does the magnitude of this drift differ
by model?

Metric per conversation:
  accommodation(t) = cos_sim(ai_turn_t, human_turns_so_far_mean)
                      - cos_sim(ai_turn_0, human_turns_so_far_mean)

i.e. how much closer (in cosine similarity) the AI's current turn is to the
human's running-average turn, relative to the AI's own opening turn. This is
directly comparable across conversations regardless of absolute embedding
scale, and is conceptually close to (though simpler than) Danescu-Niculescu-
Mizil-style linguistic style accommodation, applied here to semantic
embeddings instead of function-word LSM.
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from src.common.embeddings import embed_texts
from src.track2_human_ai.trajectory_geometry import load_conversations


def run_accommodation_analysis(
    conversations_path: str,
    out_dir: str,
    min_ai_turns_per_conv: int = 3,
    sbert_model: str = "all-MiniLM-L6-v2",
    n_progress_bins: int = 10,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    convs = load_conversations(conversations_path)

    rows = []
    for c in convs:
        ai_turns = [(i, t["text"]) for i, t in enumerate(c["turns"]) if t["role"] == "assistant"]
        human_turns = [(i, t["text"]) for i, t in enumerate(c["turns"]) if t["role"] == "human"]
        if len(ai_turns) < min_ai_turns_per_conv or len(human_turns) < 1:
            continue

        ai_texts = [t for _, t in ai_turns]
        human_texts = [t for _, t in human_turns]
        ai_embs = embed_texts(ai_texts, model_name=sbert_model)
        human_embs = embed_texts(human_texts, model_name=sbert_model)

        ai_anchor = ai_embs[0]
        human_indices = [i for i, _ in human_turns]

        for k, (turn_idx, _) in enumerate(ai_turns):
            # human turns strictly before this AI turn
            human_so_far = [human_embs[j] for j, hidx in enumerate(human_indices) if hidx < turn_idx]
            if not human_so_far:
                continue
            human_mean = np.mean(human_so_far, axis=0, keepdims=True)

            sim_now = cosine_similarity(ai_embs[k : k + 1], human_mean)[0, 0]
            sim_anchor = cosine_similarity(ai_anchor.reshape(1, -1), human_mean)[0, 0]

            rows.append(
                {
                    "conv_id": c["conv_id"],
                    "model": c["model"],
                    "ai_turn_position": k,
                    "accommodation": float(sim_now - sim_anchor),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No conversations produced accommodation rows -- check filters/data.")

    df.to_csv(os.path.join(out_dir, "accommodation_by_turn.csv"), index=False)

    summary = df.groupby("model")["accommodation"].agg(["mean", "std", "count"])
    summary.to_csv(os.path.join(out_dir, "accommodation_summary_by_model.csv"))
    print("\nAccommodation summary by model (positive = AI drifts toward human over the conversation):")
    print(summary)

    plot_accommodation_results(df, out_dir, n_progress_bins=n_progress_bins)

    return df


def plot_accommodation_results(
    df: pd.DataFrame,
    out_dir: str,
    n_progress_bins: int = 10,
    max_small_multiple_models: int = 12,
) -> pd.DataFrame:
    """Draw coverage-aware accommodation views without multi-line overplotting."""
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    max_position = df.groupby("conv_id")["ai_turn_position"].transform("max")
    df["relative_progress"] = df.ai_turn_position / max_position.replace(0, 1)
    df["progress_bin"] = np.minimum(
        (df.relative_progress * n_progress_bins).astype(int), n_progress_bins - 1
    )

    # First average within each conversation/bin, so long conversations do not
    # dominate the model-level curve merely by contributing more turns.
    per_conversation = (
        df.groupby(["conv_id", "model", "progress_bin"], as_index=False)
        .accommodation.mean()
    )
    summary = (
        per_conversation.groupby(["model", "progress_bin"])
        .accommodation.agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n_conversations"})
    )
    summary.to_csv(os.path.join(out_dir, "accommodation_by_progress.csv"), index=False)

    coverage = per_conversation.groupby("model").conv_id.nunique()
    final_values = (
        summary.sort_values("progress_bin").groupby("model").tail(1).set_index("model")["mean"]
    )
    order = final_values.sort_values(ascending=False).index.tolist()
    matrix = summary.pivot(index="model", columns="progress_bin", values="mean").reindex(order)
    labels = [f"{model}  (n={coverage[model]})" for model in matrix.index]
    finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
    limit = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
    limit = max(limit, 1e-3)

    fig, ax = plt.subplots(figsize=(10, max(7, 0.38 * len(matrix) + 2)))
    sns.heatmap(
        matrix,
        mask=matrix.isna(),
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        yticklabels=labels,
        cbar_kws={"label": "Mean accommodation vs. opening AI turn"},
        ax=ax,
    )
    ax.set_xlabel("Relative conversation progress bin")
    ax.set_ylabel("AI model (conversation coverage)")
    ax.set_title("AI accommodation toward the human\nconversation-weighted; models sorted by final-bin value")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accommodation_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_models = coverage.sort_values(ascending=False).head(max_small_multiple_models).index
    ncols = 3
    nrows = math.ceil(len(top_models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    x = (np.arange(n_progress_bins) + 0.5) / n_progress_bins
    for ax, model in zip(axes, top_models):
        model_summary = summary[summary.model == model].set_index("progress_bin").reindex(range(n_progress_bins))
        mean = model_summary["mean"].to_numpy(float)
        sem = model_summary["sem"].fillna(0).to_numpy(float)
        ax.plot(x, mean, marker="o", lw=1.8, color=sns.color_palette("colorblind")[0])
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.18)
        ax.axhline(0, color="0.45", ls="--", lw=0.8)
        ax.set_title(f"{model}  (n={coverage[model]})", fontsize=9)
        ax.set_xlim(0, 1)
    for ax in axes[len(top_models):]:
        ax.set_visible(False)
    fig.supxlabel("Relative conversation progress")
    fig.supylabel("Accommodation vs. opening AI turn (mean ± 95% CI)")
    fig.suptitle("Accommodation trajectories for the best-covered models", y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "accommodation_small_multiples.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary
