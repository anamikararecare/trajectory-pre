"""
Experiment 2c: reuse the paper's exact emotion-scoring procedure (Eq. 15,
sentence-length-weighted GoEmotions) to compare affect trajectories for
human vs. AI turns, and across AI models. Direct comparison target: their
Fig. 13, which shows AI-AI self-play drifting toward more agreement/
positivity and less negativity/hedging over turns. Here we ask whether that
drift is mutual (human also drifts positive) or one-sided (only the AI
drifts while the human's affect stays flat).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.common.emotion import affiliative_index, score_message
from src.track2_human_ai.trajectory_geometry import load_conversations


def run_emotion_overlay(
    conversations_path: str,
    out_dir: str,
    max_turn_position: int | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    convs = load_conversations(conversations_path)

    rows = []
    for c in tqdm(convs, desc="scoring emotion"):
        progress_denominator = max(len(c["turns"]) - 1, 1)
        for i, t in enumerate(c["turns"]):
            if max_turn_position is not None and i > max_turn_position:
                continue
            dist = score_message(t["text"])
            rows.append(
                {
                    "conv_id": c["conv_id"],
                    "model": c["model"],
                    "role": t["role"],
                    "turn_idx": i,
                    "relative_progress": i / progress_denominator,
                    "affiliative_index": affiliative_index(dist),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No turns scored -- check filters/data.")

    # normalize turn position into relative conversation progress bins so
    # conversations of different lengths are comparable, mirroring the
    # paper's turn-indexed Fig. 13 but robust to variable-length real chats
    df["turn_bin"] = pd.cut(
        df["relative_progress"], bins=n_bins, labels=False, include_lowest=True
    )
    df.to_csv(os.path.join(out_dir, "emotion_by_turn.csv"), index=False)

    plot_emotion_results(df, out_dir, n_bins=n_bins)

    print(f"Saved emotion overlay figures + table to {out_dir}/")
    return df


def plot_emotion_results(df: pd.DataFrame, out_dir: str, n_bins: int = 5) -> None:
    """Plot matched role differences and model trends without 25-line overlays."""
    os.makedirs(out_dir, exist_ok=True)
    df = df.copy()
    df["turn_bin"] = pd.cut(
        df["relative_progress"], bins=n_bins, labels=False, include_lowest=True
    )
    per_conversation = (
        df.groupby(["conv_id", "model", "role", "turn_bin"], as_index=False)
        .affiliative_index.mean()
    )

    role_summary = (
        per_conversation.groupby(["role", "turn_bin"])
        .affiliative_index.agg(["mean", "sem", "count"])
        .reset_index()
    )
    paired = per_conversation.pivot_table(
        index=["conv_id", "model", "turn_bin"], columns="role", values="affiliative_index"
    ).dropna(subset=["assistant", "human"])
    paired["ai_minus_human"] = paired.assistant - paired.human
    difference = (
        paired.groupby("turn_bin").ai_minus_human.agg(["mean", "sem", "count"]).reset_index()
    )
    role_summary.to_csv(os.path.join(out_dir, "emotion_role_by_progress.csv"), index=False)
    difference.to_csv(os.path.join(out_dir, "emotion_ai_minus_human.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = dict(zip(["assistant", "human"], sns.color_palette("colorblind", 2)))
    for role in ["assistant", "human"]:
        group = role_summary[role_summary.role == role]
        axes[0].errorbar(
            group.turn_bin,
            group["mean"],
            yerr=1.96 * group["sem"].fillna(0),
            marker="o",
            capsize=3,
            lw=2,
            color=colors[role],
            label="AI" if role == "assistant" else "Human",
        )
    axes[0].set_title("A  Absolute affect trajectories")
    axes[0].set_xlabel("Relative conversation progress bin")
    axes[0].set_ylabel("Affiliative index (mean ± 95% CI)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)

    axes[1].errorbar(
        difference.turn_bin,
        difference["mean"],
        yerr=1.96 * difference["sem"].fillna(0),
        marker="o",
        capsize=3,
        lw=2,
        color=sns.color_palette("colorblind")[2],
    )
    axes[1].axhline(0, color="0.45", ls="--", lw=1)
    axes[1].set_title("B  Matched within-conversation contrast")
    axes[1].set_xlabel("Relative conversation progress bin")
    axes[1].set_ylabel("AI affiliative index − human index")
    fig.suptitle("Exploratory human–AI affect trajectories", y=1.04)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "emotion_trajectory_human_vs_ai.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    ai = per_conversation[per_conversation.role == "assistant"].sort_values(
        ["conv_id", "turn_bin"]
    )
    ai["opening_affect"] = ai.groupby("conv_id").affiliative_index.transform("first")
    ai["affect_change"] = ai.affiliative_index - ai.opening_affect
    model_summary = (
        ai.groupby(["model", "turn_bin"])
        .affect_change.agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n_conversations"})
    )
    model_summary.to_csv(os.path.join(out_dir, "emotion_model_change_by_progress.csv"), index=False)
    coverage = ai.groupby("model").conv_id.nunique()
    final_values = (
        model_summary.sort_values("turn_bin").groupby("model").tail(1).set_index("model")["mean"]
    )
    order = final_values.sort_values(ascending=False).index.tolist()
    matrix = model_summary.pivot(index="model", columns="turn_bin", values="mean").reindex(order)
    labels = [f"{model}  (n={coverage[model]})" for model in matrix.index]
    finite = np.abs(matrix.to_numpy()[np.isfinite(matrix.to_numpy())])
    limit = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
    limit = max(limit, 1e-3)
    fig, ax = plt.subplots(figsize=(9, max(7, 0.38 * len(matrix) + 2)))
    sns.heatmap(
        matrix,
        mask=matrix.isna(),
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        yticklabels=labels,
        cbar_kws={"label": "Change from each conversation's first AI bin"},
        ax=ax,
    )
    ax.set_xlabel("Relative conversation progress bin")
    ax.set_ylabel("AI model (conversation coverage)")
    ax.set_title("AI affect change by model\nconversation-weighted; models sorted by final-bin change")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "emotion_trajectory_by_model.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
