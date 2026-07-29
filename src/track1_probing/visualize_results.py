"""Publication-style visualizations for completed Track 1 runs.

The PCA panels are descriptive projections in the fixed self-play basis. The
reported basin and mixed-play metrics remain the full 384-D SBERT estimates.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns


# PCA component signs are mathematically arbitrary. Use the paper's visual
# orientation for the proof-of-concept geometry figure so its trajectories can
# be compared directly with Figure 2. This is presentation-only: the saved PCA
# reference, projected CSV, and full-space geometry metrics are unchanged.
PAPER_FIGURE_PC_SIGNS = {"sp_pc1": 1.0, "sp_pc2": -1.0}


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _variance_labels(reference_path: str) -> tuple[str, str]:
    with np.load(reference_path, allow_pickle=False) as reference:
        variance = reference["explained_variance_ratio"]
    return f"Self-play PC1 ({variance[0] * 100:.1f}%)", f"Self-play PC2 ({variance[1] * 100:.1f}%)"


def _add_covariance_ellipse(ax, points: np.ndarray, color) -> None:
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    if np.any(values <= 0):
        return
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    # Two standard deviations: a descriptive basin contour, not a confidence CI.
    width, height = 4 * np.sqrt(values)
    center = points.mean(axis=0)
    ax.add_patch(
        Ellipse(center, width, height, angle=angle, facecolor=color,
                edgecolor=color, alpha=0.12, linewidth=1.5)
    )


def plot_geometry(results_dir: str, out_path: str) -> None:
    geometry_dir = os.path.join(results_dir, "geometry")
    turns = pd.read_csv(os.path.join(geometry_dir, "turn_geometry.csv"))
    for component, sign in PAPER_FIGURE_PC_SIGNS.items():
        turns[component] *= sign
    xlabel, ylabel = _variance_labels(os.path.join(geometry_dir, "self_play_reference.npz"))
    models = sorted(turns["model"].unique())
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    self_turns = turns[turns["condition"] == "self_play"]
    for model, group in self_turns.groupby("model"):
        mean = group.groupby("agent_turn")[["sp_pc1", "sp_pc2"]].mean()
        axes[0].plot(mean.sp_pc1, mean.sp_pc2, "-o", ms=3, lw=2,
                     color=palette[model], label=model)
        axes[0].scatter(mean.sp_pc1.iloc[-1], mean.sp_pc2.iloc[-1], marker="*",
                        s=130, color=palette[model], edgecolor="black", zorder=4)
    axes[0].set_title("A  Self-play mean trajectories")

    endpoints = (
        self_turns.sort_values("turn").groupby(["conv_id", "speaker"], as_index=False).tail(1)
    )
    for model, group in endpoints.groupby("model"):
        points = group[["sp_pc1", "sp_pc2"]].to_numpy()
        axes[1].scatter(points[:, 0], points[:, 1], s=35, alpha=0.7,
                        color=palette[model], label=model)
        axes[1].scatter(*points.mean(axis=0), marker="X", s=120,
                        color=palette[model], edgecolor="black", zorder=4)
        _add_covariance_ellipse(axes[1], points, palette[model])
    axes[1].set_title("B  Self-play endpoint basins")

    for (model, condition), group in turns.groupby(["model", "condition"]):
        mean = group.groupby("agent_turn")[["sp_pc1", "sp_pc2"]].mean()
        style = "-" if condition == "self_play" else "--"
        axes[2].plot(mean.sp_pc1, mean.sp_pc2, style, lw=2,
                     color=palette[model], label=f"{model} · {condition}")
        axes[2].scatter(mean.sp_pc1.iloc[-1], mean.sp_pc2.iloc[-1], s=45,
                        color=palette[model], marker="o" if condition == "self_play" else "s")
    axes[2].set_title("C  Mixed-play attraction overlay")

    for ax in axes:
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Track 1 output geometry (2-D visualization; metrics computed in 384-D)", y=1.03)
    _save(fig, out_path)


def plot_endpoint_metrics(results_dir: str, out_path: str) -> None:
    geometry_dir = os.path.join(results_dir, "geometry")
    by_topic = pd.read_csv(os.path.join(geometry_dir, "mixed_play_metrics_by_topic.csv"))
    summary = pd.read_csv(os.path.join(geometry_dir, "mixed_play_metrics_summary.csv")).iloc[0]
    models = [summary.model_a, summary.model_b]
    colors = sns.color_palette("colorblind", 2)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    sns.stripplot(data=by_topic, x="contraction", y="topic_id", ax=axes[0], color=colors[0], size=7)
    axes[0].axvline(summary.contraction_mean, color="black", ls="--", label="mean")
    axes[0].axvline(0, color="0.6", lw=1)
    axes[0].set_title("A  Pair contraction")
    axes[0].set_xlabel("C (fraction of separation closed)")
    axes[0].set_ylabel("Topic")
    axes[0].legend(frameon=False, fontsize=8)

    pull_means = [summary.alpha_a_toward_b_mean, summary.alpha_b_toward_a_mean]
    pull_low = [summary.alpha_a_toward_b_ci_low, summary.alpha_b_toward_a_ci_low]
    pull_high = [summary.alpha_a_toward_b_ci_high, summary.alpha_b_toward_a_ci_high]
    axes[1].bar(models, pull_means, color=colors)
    axes[1].errorbar(range(2), pull_means,
                     yerr=[np.array(pull_means) - pull_low, np.array(pull_high) - pull_means],
                     fmt="none", color="black", capsize=4)
    axes[1].axhline(0, color="0.6", lw=1)
    axes[1].set_title("B  Partnerward pull")
    axes[1].set_ylabel("Mean α (95% topic bootstrap CI)")
    axes[1].tick_params(axis="x", rotation=20)

    dominance = summary.dominance_a_over_b_mean
    axes[2].errorbar([0], [dominance],
                     yerr=[[dominance - summary.dominance_a_over_b_ci_low],
                           [summary.dominance_a_over_b_ci_high - dominance]],
                     fmt="o", color=colors[0], capsize=5, markersize=8)
    axes[2].axhline(0, color="0.4", lw=1)
    axes[2].set_xticks([0], [f"{models[0]} over\n{models[1]}"])
    axes[2].set_title("C  Directional dominance")
    axes[2].set_ylabel("Δ (95% topic bootstrap CI)")

    off_means = [summary.off_axis_a_mean, summary.off_axis_b_mean]
    off_low = [summary.off_axis_a_ci_low, summary.off_axis_b_ci_low]
    off_high = [summary.off_axis_a_ci_high, summary.off_axis_b_ci_high]
    axes[3].bar(models, off_means, color=colors)
    axes[3].errorbar(range(2), off_means,
                     yerr=[np.array(off_means) - off_low, np.array(off_high) - off_means],
                     fmt="none", color="black", capsize=4)
    axes[3].set_title("D  Off-axis displacement")
    axes[3].set_ylabel("Mean normalized δ⊥ (95% CI)")
    axes[3].tick_params(axis="x", rotation=20)
    fig.suptitle("Track 1 endpoint diagnostics (full 384-D SBERT space)", y=1.03)
    _save(fig, out_path)


def _load_stances(transcripts_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(transcripts_dir, "*.json"))):
        with open(path) as handle:
            transcript = json.load(handle)
        for turn in transcript["turns"]:
            speaker = turn["speaker"]
            rows.append({
                "condition": transcript["condition"],
                "model": transcript[f"agent_{speaker}_model"],
                "role": turn["role"],
                "agent_turn": turn.get("agent_turn", turn["turn"] // 2 + 1),
                "stance_score": turn["stance_score"],
            })
    return pd.DataFrame(rows)


def plot_stances(data_dir: str, out_path: str) -> None:
    stances = _load_stances(os.path.join(data_dir, "transcripts"))
    models = sorted(stances.model.unique())
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, condition in zip(axes, ["self_play", "mixed_play"]):
        subset = stances[stances.condition == condition]
        for (model, role), group in subset.groupby(["model", "role"]):
            stats = group.groupby("agent_turn").stance_score.agg(["mean", "sem"])
            linestyle = "-" if role == "supporter" else "--"
            ax.plot(stats.index, stats["mean"], linestyle, color=palette[model],
                    lw=2, label=f"{model} · {role}")
            ax.fill_between(stats.index, stats["mean"] - stats["sem"],
                            stats["mean"] + stats["sem"], color=palette[model], alpha=0.10)
        ax.axhline(3, color="0.5", lw=1, ls=":")
        ax.set_title(condition.replace("_", " ").title())
        ax.set_xlabel("Response number")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("Mean stance (1 oppose – 5 support) ± SEM")
    fig.suptitle("Track 1 stance trajectories", y=1.02)
    _save(fig, out_path)


def plot_probes(results_dir: str, out_path: str) -> None:
    scores = pd.read_csv(os.path.join(results_dir, "probe_scores.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    concurrent = scores[scores.experiment == "1a_concurrent"]
    sns.barplot(data=concurrent, x="layer", y="r2", ax=axes[0, 0], color=sns.color_palette("colorblind")[0])
    axes[0, 0].axhline(0, color="black", lw=1)
    axes[0, 0].set_title("A  Concurrent stance decoding")
    axes[0, 0].set_ylabel("LOTO R²")

    predictive = scores[scores.experiment == "1b_predictive"].copy()
    probe = predictive.pivot(index="layer", columns="horizon", values="probe_r2")
    sns.heatmap(probe, annot=True, fmt=".2f", center=0, cmap="vlag", ax=axes[0, 1])
    axes[0, 1].set_title("B  Future stance decoding (R²)")

    predictive["baseline_lift"] = predictive.probe_r2 - predictive.baseline_r2
    lift = predictive.pivot(index="layer", columns="horizon", values="baseline_lift")
    sns.heatmap(lift, annot=True, fmt=".2f", center=0, cmap="vlag", ax=axes[1, 0])
    axes[1, 0].set_title("C  Predictive improvement over trend baseline")

    cross = scores[scores.experiment == "1c_cross_agent"].copy()
    cross["majority_lift"] = cross.acc - cross.majority_acc
    cross_lift = cross.pivot(index="layer", columns="horizon", values="majority_lift")
    sns.heatmap(cross_lift, annot=True, fmt=".2f", center=0, cmap="vlag", ax=axes[1, 1])
    axes[1, 1].set_title("D  Cross-agent accuracy above majority")
    fig.suptitle("Track 1 activation probes (leave-one-topic-out)", y=1.01)
    _save(fig, out_path)


def generate_all_figures(data_dir: str, results_dir: str, out_dir: str) -> list[str]:
    sns.set_theme(style="whitegrid", context="paper")
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(out_dir, "paper_geometry.png"),
        os.path.join(out_dir, "endpoint_metrics.png"),
        os.path.join(out_dir, "stance_trajectories.png"),
        os.path.join(out_dir, "probe_summary.png"),
    ]
    plot_geometry(results_dir, paths[0])
    plot_endpoint_metrics(results_dir, paths[1])
    plot_stances(data_dir, paths[2])
    plot_probes(results_dir, paths[3])
    return paths
