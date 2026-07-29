"""Static figures for a focused point-in-time Track 1 preliminary run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SNAPSHOT_ORDER = [
    "pre_generation", "early_response", "full_response", "final_window"
]


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_trajectories(geometry: pd.DataFrame, out_dir: Path) -> list[dict]:
    frame = geometry.copy()
    frame["paper_pc1"] = frame["sp_pc1"]
    frame["paper_pc2"] = -frame["sp_pc2"]
    models = sorted(frame["model"].dropna().unique())
    palette = dict(zip(models, sns.color_palette("colorblind", len(models))))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    self_play = frame[frame["condition"].eq("self_play")]
    for model, group in self_play.groupby("model"):
        mean = group.groupby("agent_turn")[["paper_pc1", "paper_pc2"]].mean()
        axes[0].plot(
            mean["paper_pc1"], mean["paper_pc2"], "-o",
            color=palette[model], label=model, markersize=3,
        )
        if not mean.empty:
            axes[0].scatter(
                mean["paper_pc1"].iloc[-1], mean["paper_pc2"].iloc[-1],
                marker="*", s=120, color=palette[model], edgecolor="black",
            )
    axes[0].set_title("A  Self-play mean trajectories")

    endpoints = (
        self_play.sort_values("turn")
        .groupby(["conv_id", "speaker"], as_index=False)
        .tail(1)
    )
    for model, group in endpoints.groupby("model"):
        axes[1].scatter(
            group["paper_pc1"], group["paper_pc2"],
            color=palette[model], alpha=0.75, label=model,
        )
        axes[1].scatter(
            group["paper_pc1"].mean(), group["paper_pc2"].mean(),
            color=palette[model], marker="X", s=120, edgecolor="black",
        )
    axes[1].set_title("B  Available self-play endpoints")

    for (model, condition), group in frame.groupby(["model", "condition"]):
        mean = group.groupby("agent_turn")[["paper_pc1", "paper_pc2"]].mean()
        axes[2].plot(
            mean["paper_pc1"], mean["paper_pc2"],
            "-" if condition == "self_play" else "--",
            color=palette[model], label=f"{model} · {condition}",
        )
    axes[2].set_title("C  Mixed-play attraction overlay")
    for axis in axes:
        axis.axhline(0, color="0.85", linewidth=0.8)
        axis.axvline(0, color="0.85", linewidth=0.8)
        axis.set_xlabel("Self-play PC1")
        axis.set_ylabel("Self-play PC2 (paper orientation)")
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle("Preliminary selected-batch output geometry", y=1.02)
    path = out_dir / "preliminary_pc_trajectories.png"
    _save(figure, path)
    return [{"kind": "trajectories", "path": str(path)}]


def export_geometry_metrics(geometry: pd.DataFrame, out_dir: Path) -> list[dict]:
    metrics = [
        column for column in (
            "basin_leaning", "partnerward_basin_velocity",
            "semantic_velocity", "semantic_acceleration", "off_axis_distance",
        ) if column in geometry
    ]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, metric in zip(axes.flat, metrics):
        sns.lineplot(
            data=geometry, x="agent_turn", y=metric, hue="model",
            style="condition", errorbar=None, ax=axis,
        )
        axis.set_title(metric.replace("_", " ").title())
    for axis in axes.flat[len(metrics):]:
        axis.remove()
    figure.suptitle("1H selected-batch geometry trajectories", y=1.01)
    path = out_dir / "experiment_1h_geometry_trajectories.png"
    _save(figure, path)
    return [{"kind": "geometry_metrics", "experiment": "1H", "path": str(path)}]


def export_probe_summaries(scores: pd.DataFrame, out_dir: Path) -> list[dict]:
    records = []
    finite = scores.copy()
    finite["incremental_score"] = pd.to_numeric(
        finite["incremental_score"], errors="coerce"
    )
    finite = finite.dropna(subset=["incremental_score"])
    if finite.empty:
        return records

    overview = finite.pivot_table(
        index="experiment", columns="snapshot",
        values="incremental_score", aggfunc="max",
    ).reindex(columns=[item for item in SNAPSHOT_ORDER if item in finite["snapshot"].unique()])
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(overview, cmap="vlag", center=0, annot=True, fmt=".3f", ax=axis)
    axis.set_title("Best observed incremental activation lift by experiment")
    path = out_dir / "preliminary_probe_overview.png"
    _save(figure, path)
    records.append({"kind": "probe_overview", "path": str(path)})

    for experiment in [f"1{letter}" for letter in "BCDEFGHI"]:
        subset = finite[finite["experiment"].eq(experiment)]
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="target", columns="snapshot",
            values="incremental_score", aggfunc="max",
        ).reindex(columns=[item for item in SNAPSHOT_ORDER if item in subset["snapshot"].unique()])
        height = max(5, 0.42 * len(pivot))
        figure, axis = plt.subplots(figsize=(11, height))
        sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".3f", ax=axis)
        axis.set_title(
            f"{experiment}: best exploratory incremental lift across layers/horizons"
        )
        path = out_dir / f"experiment_{experiment.lower()}_summary.png"
        _save(figure, path)
        records.append({
            "kind": "experiment_summary",
            "experiment": experiment,
            "path": str(path),
        })
    return records


def export_transfer(transfer: pd.DataFrame, out_dir: Path) -> list[dict]:
    if transfer.empty:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(16, 5))
    behavior = transfer[transfer["kind"].eq("behavior")]
    activation = transfer[transfer["kind"].eq("activation")]
    if behavior.empty:
        axes[0].text(0.5, 0.5, "Behavioral transfer unavailable", ha="center")
        axes[0].set_axis_off()
    else:
        sns.barplot(
            data=behavior, x="target", y="mixed_minus_self",
            hue="model", ax=axes[0],
        )
        axes[0].tick_params(axis="x", rotation=75)
        axes[0].set_title("Behavioral mixed-play shift")
    if activation.empty:
        axes[1].text(0.5, 0.5, "Activation transfer unavailable", ha="center")
        axes[1].set_axis_off()
    else:
        sns.lineplot(
            data=activation, x="layer", y="mixed_minus_self",
            hue="snapshot", style="model", marker="o", ax=axes[1],
        )
        axes[1].set_title("Activation displacement")
    path = out_dir / "experiment_1i_transfer.png"
    _save(figure, path)
    return [{"kind": "partner_transfer", "experiment": "1I", "path": str(path)}]


def export_preliminary_figures(
    results_dir: str, annotations_dir: str, geometry_path: str
) -> list[dict]:
    results = Path(results_dir)
    annotations = Path(annotations_dir)
    out_dir = results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(annotations / "selection.csv")
    geometry = pd.read_csv(geometry_path).merge(
        selection[["conv_id", "turn"]].drop_duplicates(),
        on=["conv_id", "turn"], how="inner", validate="one_to_one",
    )
    scores = pd.read_csv(results / "snapshot_probe_scores.csv")
    transfer_path = results / "experiment_1i_partner_transfer.csv"
    transfer = (
        pd.read_csv(transfer_path)
        if transfer_path.exists() and transfer_path.stat().st_size else pd.DataFrame()
    )
    records = []
    records.extend(export_trajectories(geometry, out_dir))
    records.extend(export_geometry_metrics(geometry, out_dir))
    records.extend(export_probe_summaries(scores, out_dir))
    records.extend(export_transfer(transfer, out_dir))
    pd.DataFrame(records).to_csv(out_dir / "figure_index.csv", index=False)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument(
        "--geometry", default="results/track1/geometry/turn_geometry.csv"
    )
    args = parser.parse_args()
    records = export_preliminary_figures(
        args.results_dir, args.annotations_dir, args.geometry
    )
    print(
        f"Exported {len(records)} preliminary figures under "
        f"{Path(args.results_dir) / 'figures'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
