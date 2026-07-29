"""
Legacy corpus-loading helpers. The assistant-only PCA analysis below is
disabled because partner-conditioned human-AI turns are not self-play.
Use reference_geometry.run_reference_geometry_analysis with a Track 1
self-play reference instead.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.embeddings import basin_separation_score, embed_texts, fit_reference_pca, topic_center


def load_conversations(path: str) -> list[dict]:
    convs = []
    with open(path) as f:
        for line in f:
            convs.append(json.loads(line))
    return convs


def _flatten_turns(convs: list[dict]) -> pd.DataFrame:
    rows = []
    for c in convs:
        for i, t in enumerate(c["turns"]):
            rows.append(
                {
                    "conv_id": c["conv_id"],
                    "model": c["model"],
                    "turn_idx": i,
                    "role": t["role"],
                    "text": t["text"],
                }
            )
    return pd.DataFrame(rows)


def run_geometry_analysis(
    conversations_path: str,
    out_dir: str,
    min_ai_turns_per_conv: int = 3,
    min_convs_per_model: int = 10,
    sbert_model: str = "all-MiniLM-L6-v2",
) -> dict:
    raise RuntimeError(
        "Legacy assistant-only PCA is disabled; use run_reference_geometry_analysis "
        "with a Track 1 self-play reference."
    )
    convs = load_conversations(conversations_path)
    df = _flatten_turns(convs)

    # keep only conversations with enough AI turns, and models with enough
    # conversations to form a meaningful basin
    ai_counts = df[df.role == "assistant"].groupby("conv_id").size()
    keep_convs = ai_counts[ai_counts >= min_ai_turns_per_conv].index
    df = df[df.conv_id.isin(keep_convs)]

    model_counts = df[df.role == "assistant"].drop_duplicates("conv_id").groupby("model").size()
    keep_models = model_counts[model_counts >= min_convs_per_model].index
    df = df[df.model.isin(keep_models)]

    if df.empty:
        raise ValueError(
            "No conversations survived filtering -- lower min_ai_turns_per_conv / "
            "min_convs_per_model, or fetch more data."
        )

    print(f"Embedding {len(df)} turns from {df.conv_id.nunique()} conversations, "
          f"{df.model.nunique()} models...")
    df["embedding"] = list(embed_texts(df["text"].tolist(), model_name=sbert_model))

    # topic-center per conversation (paper's Eq. 1, using conversation as the
    # centering group instead of "topic", since we don't have repeated
    # identical topics across conversations in the wild)
    emb_matrix = np.stack(df["embedding"].values)
    centered = topic_center(emb_matrix, df["conv_id"].tolist())
    df["centered_embedding"] = list(centered)

    # fit reference PCA on AI turns only (closest analogue to "SP-PCs")
    ai_mask = df.role == "assistant"
    pca = fit_reference_pca(np.stack(df.loc[ai_mask, "centered_embedding"].values), n_components=2)

    df["pc1"] = np.nan
    df["pc2"] = np.nan
    proj_all = pca.transform(np.stack(df["centered_embedding"].values))
    df["pc1"] = proj_all[:, 0]
    df["pc2"] = proj_all[:, 1]

    # AI endpoints = last AI turn per conversation
    ai_endpoints = (
        df[ai_mask].sort_values("turn_idx").groupby("conv_id").tail(1).reset_index(drop=True)
    )
    basin_scores = basin_separation_score(
        np.stack(ai_endpoints["centered_embedding"].values), ai_endpoints["model"].tolist()
    )

    basin_df = pd.DataFrame(basin_scores).T
    basin_df.index.name = "model"
    basin_df.to_csv(os.path.join(out_dir, "basin_separation_scores.csv"))
    print("\nBasin separation scores (S_basin > 1 => locally separated from nearest rival):")
    print(basin_df)

    # plot: AI endpoints colored by model, human endpoints in grey for reference
    human_endpoints = (
        df[~ai_mask].sort_values("turn_idx").groupby("conv_id").tail(1).reset_index(drop=True)
    )

    plt.figure(figsize=(7, 6))
    for model in ai_endpoints["model"].unique():
        sub = ai_endpoints[ai_endpoints.model == model]
        plt.scatter(sub.pc1, sub.pc2, label=model, alpha=0.6, s=25)
    plt.scatter(
        human_endpoints.pc1, human_endpoints.pc2, color="grey", alpha=0.15, s=10, label="human (ref)"
    )
    plt.xlabel("PC1 (AI-turn reference basis)")
    plt.ylabel("PC2 (AI-turn reference basis)")
    plt.title("Human-AI conversation endpoints, projected onto AI-turn PCA basis")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "endpoint_scatter.png"), dpi=150)
    plt.close()

    df.drop(columns=["embedding", "centered_embedding"]).to_csv(
        os.path.join(out_dir, "turn_level_projections.csv"), index=False
    )

    print(f"\nSaved figures + tables to {out_dir}/")
    return basin_scores
