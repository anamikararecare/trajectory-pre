"""Static figure export for Q1 E1 layerwise probe results."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Sequence
from src.q1.core_variables import E1_REPRESENTATIVE_CURVES

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


REPRESENTATIVE_VARIABLES = E1_REPRESENTATIVE_CURVES

TIMING_COLORS = {
    "current_response": "#3b82f6",
    "transition": "#ef4444",
    "trailing_window": "#10b981",
    "unknown": "#6b7280",
}


def _turn_range_key(value: str) -> tuple[float, str]:
    match = re.match(r"^\s*([0-9.]+)", str(value))
    return (float(match.group(1)) if match else math.inf, str(value))


def _ordered(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _target_order(
    scores: pd.DataFrame, registry: pd.DataFrame
) -> list[str]:
    available = set(scores["target"].dropna().astype(str))
    registered = (
        registry["name"].dropna().astype(str).tolist()
        if "name" in registry
        else []
    )
    ordered = [target for target in registered if target in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _with_primary_metric(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    continuous = pd.to_numeric(
        out.get("activation_only_pearson"), errors="coerce"
    )
    categorical = pd.to_numeric(
        out.get("activation_only_score"), errors="coerce"
    )
    out["primary_metric"] = np.where(
        out["task"].eq("continuous"), continuous, categorical
    )
    return out


def _metric_style(task: str) -> tuple[str, str, float, float]:
    if task == "continuous":
        return (
            "Held-topic activation Pearson correlation",
            "coolwarm",
            -1.0,
            1.0,
        )
    return ("Held-topic balanced accuracy", "viridis", 0.0, 1.0)


def _draw_matrix(
    axis: plt.Axes,
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    task: str,
    annotations: np.ndarray | None = None,
) -> None:
    metric_label, cmap, lower, upper = _metric_style(task)
    masked = np.ma.masked_invalid(values)
    color_map = plt.get_cmap(cmap).with_extremes(bad="#eeeeee")
    image = axis.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=lower,
        vmax=upper,
    )
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_xticks(np.arange(len(column_labels)))
    axis.set_xticklabels(column_labels, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=7)
    axis.tick_params(length=0)
    if annotations is not None:
        font_size = max(4.0, min(7.0, 180.0 / max(1, len(column_labels))))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                label = annotations[row, column]
                if isinstance(label, str) and label:
                    value = values[row, column]
                    color = (
                        "white"
                        if np.isfinite(value)
                        and (
                            abs(value) > 0.55
                            if task == "continuous"
                            else value > 0.62
                        )
                        else "black"
                    )
                    axis.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        color=color,
                    )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.015)
    colorbar.set_label(metric_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def export_layer_turn_heatmaps(
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    destination: Path,
) -> int:
    """Write an exhaustive multi-page model/task heatmap PDF."""
    scores = _with_primary_metric(scores)
    target_order = _target_order(scores, registry)
    pages = 0
    with PdfPages(destination) as pdf:
        for model in sorted(scores["model"].dropna().astype(str).unique()):
            model_scores = scores[scores["model"].eq(model)]
            layers = sorted(
                pd.to_numeric(model_scores["layer"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
            ranges = sorted(
                model_scores["turn_range"].dropna().astype(str).unique(),
                key=_turn_range_key,
            )
            columns = [(turn_range, layer) for turn_range in ranges for layer in layers]
            column_labels = [
                f"{turn_range}\nL{layer}" for turn_range, layer in columns
            ]
            for task in ("continuous", "categorical"):
                subset = model_scores[model_scores["task"].eq(task)]
                targets = [
                    target
                    for target in target_order
                    if target in set(subset["target"])
                ]
                if not targets:
                    continue
                pivot = subset.pivot_table(
                    index="target",
                    columns=["turn_range", "layer"],
                    values="primary_metric",
                    aggfunc="mean",
                ).reindex(index=targets, columns=pd.MultiIndex.from_tuples(columns))
                width = max(10.0, 0.36 * len(columns) + 4.0)
                height = max(4.5, 0.29 * len(targets) + 2.4)
                figure, axis = plt.subplots(figsize=(width, height))
                _draw_matrix(
                    axis,
                    pivot.to_numpy(float),
                    targets,
                    column_labels,
                    f"{model} · {task} variables",
                    task,
                )
                for boundary in range(len(layers), len(columns), len(layers)):
                    axis.axvline(boundary - 0.5, color="black", linewidth=1.0)
                figure.suptitle(
                    "E1 layer × conversation-turn organization",
                    fontsize=14,
                    y=0.995,
                )
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                pages += 1
        if pages == 0:
            figure, axis = plt.subplots(figsize=(8, 3))
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                "No finite E1 layerwise scores are available.",
                ha="center",
                va="center",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages = 1
    return pages


def export_peak_layer_table(
    peaks: pd.DataFrame,
    registry: pd.DataFrame,
    destination: Path,
) -> int:
    """Render the compact variable × model/turn-range peak score table."""
    peaks = peaks.copy()
    peaks["primary_metric"] = pd.to_numeric(
        peaks["max_activation_correlation_or_score"], errors="coerce"
    )
    target_order = _target_order(
        peaks.rename(columns={"primary_metric": "activation_only_score"}),
        registry,
    )
    models = sorted(peaks["model"].dropna().astype(str).unique())
    ranges = sorted(
        peaks["turn_range"].dropna().astype(str).unique(),
        key=_turn_range_key,
    )
    columns = [(model, turn_range) for model in models for turn_range in ranges]
    column_labels = [f"{model}\n{turn_range}" for model, turn_range in columns]
    tasks = [
        task
        for task in ("continuous", "categorical")
        if peaks["task"].eq(task).any()
    ]
    if not tasks:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(0.5, 0.5, "No E1 peak scores are available.", ha="center")
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(figure)
        return 1
    task_targets = {
        task: [
            target
            for target in target_order
            if target in set(peaks.loc[peaks["task"].eq(task), "target"])
        ]
        for task in tasks
    }
    figure_width = max(12.0, 0.7 * len(columns) + 4.0)
    figure_height = max(
        5.0, sum(max(1, len(task_targets[task])) for task in tasks) * 0.38 + 3.0
    )
    figure, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [max(1, len(task_targets[task])) for task in tasks]
        },
    )
    for task, axis in zip(tasks, axes[:, 0]):
        targets = task_targets[task]
        subset = peaks[peaks["task"].eq(task)]
        score_pivot = subset.pivot_table(
            index="target",
            columns=["model", "turn_range"],
            values="primary_metric",
            aggfunc="first",
        ).reindex(index=targets, columns=pd.MultiIndex.from_tuples(columns))
        layer_pivot = subset.pivot_table(
            index="target",
            columns=["model", "turn_range"],
            values="peak_layer",
            aggfunc="first",
        ).reindex(index=targets, columns=pd.MultiIndex.from_tuples(columns))
        annotations = np.full(score_pivot.shape, "", dtype=object)
        for row in range(score_pivot.shape[0]):
            for column in range(score_pivot.shape[1]):
                score = score_pivot.iat[row, column]
                layer = layer_pivot.iat[row, column]
                if np.isfinite(score) and np.isfinite(layer):
                    annotations[row, column] = f"{score:.2f}\nL{int(layer)}"
        _draw_matrix(
            axis,
            score_pivot.to_numpy(float),
            targets,
            column_labels,
            f"{task.capitalize()} variables",
            task,
            annotations,
        )
        for boundary in range(len(ranges), len(columns), len(ranges)):
            axis.axvline(boundary - 0.5, color="black", linewidth=1.2)
    figure.suptitle(
        "E1 maximum activation score and peak layer",
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return len(tasks)


def export_peak_layer_over_time(
    peaks: pd.DataFrame,
    scores: pd.DataFrame,
    registry: pd.DataFrame,
    destination: Path,
) -> int:
    """Plot variable peak-layer movement over percentage turn ranges."""
    if peaks.empty:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(0.5, 0.5, "No E1 peak layers are available.", ha="center")
        figure.savefig(destination, dpi=200, bbox_inches="tight")
        plt.close(figure)
        return 0
    timing = (
        registry.set_index("name")["timing"].to_dict()
        if {"name", "timing"}.issubset(registry.columns)
        else {}
    )
    models = sorted(peaks["model"].dropna().astype(str).unique())
    ranges = sorted(
        peaks["turn_range"].dropna().astype(str).unique(),
        key=_turn_range_key,
    )
    range_positions = {value: index for index, value in enumerate(ranges)}
    column_count = min(3, max(1, len(models)))
    row_count = math.ceil(len(models) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 3.7 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis, model in zip(axes.flat, models):
        subset = peaks[peaks["model"].eq(model)].copy()
        model_layers = pd.to_numeric(
            scores.loc[scores["model"].eq(model), "layer"], errors="coerce"
        ).dropna()
        maximum_layer = float(model_layers.max()) if not model_layers.empty else 1.0
        subset["relative_peak_layer"] = (
            pd.to_numeric(subset["peak_layer"], errors="coerce") / maximum_layer
        )
        for target, target_rows in subset.groupby("target", sort=False):
            target_rows = target_rows.copy()
            target_rows["x"] = target_rows["turn_range"].map(range_positions)
            target_rows = target_rows.sort_values("x")
            target_timing = str(timing.get(target, "unknown"))
            axis.plot(
                target_rows["x"],
                target_rows["relative_peak_layer"],
                color=TIMING_COLORS.get(target_timing, TIMING_COLORS["unknown"]),
                alpha=0.35,
                linewidth=1.0,
                marker="o",
                markersize=2.5,
            )
        median = (
            subset.groupby("turn_range")["relative_peak_layer"]
            .median()
            .reindex(ranges)
        )
        axis.plot(
            np.arange(len(ranges)),
            median,
            color="black",
            linewidth=2.4,
            marker="o",
            label="Median",
        )
        axis.set_title(model, loc="left", fontweight="bold")
        axis.set_xticks(np.arange(len(ranges)))
        axis.set_xticklabels(ranges, rotation=25, ha="right")
        axis.set_ylim(-0.03, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Conversation turn range")
        axis.set_ylabel("Relative peak layer")
    for axis in axes.flat[len(models) :]:
        axis.axis("off")
    legend = [
        Line2D([0], [0], color=color, lw=2, label=label.replace("_", " "))
        for label, color in TIMING_COLORS.items()
        if label != "unknown"
    ]
    legend.append(Line2D([0], [0], color="black", lw=2.5, label="Median"))
    figure.legend(handles=legend, loc="upper center", ncol=len(legend))
    figure.suptitle(
        "E1 temporal movement of peak encoding depth",
        fontsize=15,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return len(models)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def export_layer_curves(
    scores: pd.DataFrame,
    output_dir: Path,
    variables: Sequence[str],
) -> list[Path]:
    """Write one model-faceted layer curve PNG per representative variable."""
    scores = _with_primary_metric(scores)
    paths = []
    ranges = sorted(
        scores["turn_range"].dropna().astype(str).unique(),
        key=_turn_range_key,
    )
    colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 0.9, max(1, len(ranges)))
    )
    for variable in variables:
        subset = scores[
            scores["target"].eq(variable) & scores["primary_metric"].notna()
        ].copy()
        if subset.empty:
            continue
        models = sorted(subset["model"].astype(str).unique())
        column_count = min(3, max(1, len(models)))
        row_count = math.ceil(len(models) / column_count)
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(5.0 * column_count, 3.6 * row_count),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        task = str(subset["task"].iloc[0])
        metric_label, _, lower, upper = _metric_style(task)
        for axis, model in zip(axes.flat, models):
            model_rows = subset[subset["model"].eq(model)].copy()
            maximum_layer = float(
                pd.to_numeric(model_rows["layer"], errors="coerce").max()
            )
            model_rows["relative_layer"] = (
                pd.to_numeric(model_rows["layer"], errors="coerce")
                / maximum_layer
            )
            for color, turn_range in zip(colors, ranges):
                line = model_rows[model_rows["turn_range"].eq(turn_range)].sort_values(
                    "relative_layer"
                )
                if line.empty:
                    continue
                axis.plot(
                    line["relative_layer"],
                    line["primary_metric"],
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    label=turn_range,
                )
            axis.set_title(model, loc="left", fontweight="bold")
            axis.set_xlim(-0.02, 1.03)
            axis.set_ylim(lower, upper)
            axis.grid(alpha=0.25)
            axis.set_xlabel("Relative layer depth")
            axis.set_ylabel(metric_label)
            if task == "continuous":
                axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        for axis in axes.flat[len(models) :]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(len(labels), 5),
            )
        figure.suptitle(
            f"E1 layerwise encoding · {variable}",
            fontsize=15,
            y=1.01,
        )
        figure.tight_layout()
        path = output_dir / f"e1_layer_curves__{_safe_name(variable)}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def export_e1_figures(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    curve_variables: Sequence[str] = REPRESENTATIVE_VARIABLES,
) -> pd.DataFrame:
    """Export all requested E1 figures and return an artifact manifest."""
    results = Path(results_dir)
    output = Path(output_dir) if output_dir else results / "figures"
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(results / "e1_layerwise_scores.csv")
    peaks = pd.read_csv(results / "e1_peak_layer_scores.csv")
    registry_path = results / "e1_variable_registry.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    records = []

    heatmap_path = output / "e1_layer_turn_heatmaps.pdf"
    heatmap_pages = export_layer_turn_heatmaps(scores, registry, heatmap_path)
    records.append(
        {
            "artifact": heatmap_path.name,
            "kind": "layer_turn_heatmaps",
            "variable": None,
            "panels_or_pages": heatmap_pages,
            "path": str(heatmap_path),
        }
    )

    table_path = output / "e1_peak_layer_table.png"
    table_panels = export_peak_layer_table(peaks, registry, table_path)
    records.append(
        {
            "artifact": table_path.name,
            "kind": "peak_layer_table",
            "variable": None,
            "panels_or_pages": table_panels,
            "path": str(table_path),
        }
    )

    temporal_path = output / "e1_peak_layer_over_time.png"
    temporal_panels = export_peak_layer_over_time(
        peaks, scores, registry, temporal_path
    )
    records.append(
        {
            "artifact": temporal_path.name,
            "kind": "peak_layer_over_time",
            "variable": None,
            "panels_or_pages": temporal_panels,
            "path": str(temporal_path),
        }
    )

    for path in export_layer_curves(scores, output, curve_variables):
        records.append(
            {
                "artifact": path.name,
                "kind": "layer_curve",
                "variable": path.stem.removeprefix("e1_layer_curves__"),
                "panels_or_pages": int(scores["model"].nunique()),
                "path": str(path),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(output / "e1_figure_manifest.csv", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Q1 E1 result figures")
    parser.add_argument("--results-dir", default="results/q1/e1")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--curve-variables",
        default=",".join(REPRESENTATIVE_VARIABLES),
        help="Comma-separated variables for individual layer-curve figures",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variables = [
        item.strip() for item in args.curve_variables.split(",") if item.strip()
    ]
    manifest = export_e1_figures(
        args.results_dir,
        output_dir=args.output_dir,
        curve_variables=variables,
    )
    print(
        f"Exported {len(manifest)} E1 figure artifact(s) to "
        f"{Path(args.output_dir) if args.output_dir else Path(args.results_dir) / 'figures'}"
    )


if __name__ == "__main__":
    main()
