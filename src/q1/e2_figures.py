"""Static figure export for Q1 E2 temporal results."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pandas.errors import EmptyDataError


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _range_key(value: str) -> tuple[float, str]:
    match = re.match(r"^\s*([0-9.]+)", str(value))
    return (float(match.group(1)) if match else math.inf, str(value))


def _empty_pdf(path: Path, message: str) -> None:
    with PdfPages(path) as pdf:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def _empty_png(path: Path, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def export_independent_heatmaps(scores: pd.DataFrame, path: Path) -> int:
    if scores.empty:
        _empty_pdf(path, "No E2 independent-range scores are available.")
        return 1
    values = scores.copy()
    values["primary_metric"] = np.where(
        values["task"].eq("continuous"),
        pd.to_numeric(values["activation_only_pearson"], errors="coerce"),
        pd.to_numeric(values["activation_only_score"], errors="coerce"),
    )
    pages = 0
    with PdfPages(path) as pdf:
        for (scope, model, task), subset in values.groupby(
            ["condition_scope", "model", "task"], sort=False
        ):
            layers = sorted(subset["layer"].dropna().astype(int).unique())
            ranges = sorted(
                subset["turn_range"].dropna().astype(str).unique(),
                key=_range_key,
            )
            targets = list(dict.fromkeys(subset["target"].astype(str)))
            columns = [
                (turn_range, layer)
                for turn_range in ranges
                for layer in layers
            ]
            pivot = subset.pivot_table(
                index="target",
                columns=["turn_range", "layer"],
                values="primary_metric",
                aggfunc="first",
            ).reindex(index=targets, columns=pd.MultiIndex.from_tuples(columns))
            matrix = pivot.to_numpy(float)
            categorical = task == "categorical"
            figure, axis = plt.subplots(
                figsize=(
                    max(10, 0.38 * len(columns) + 4),
                    max(4, 0.35 * len(targets) + 2),
                )
            )
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                cmap=plt.get_cmap(
                    "viridis" if categorical else "coolwarm"
                ).with_extremes(bad="#eeeeee"),
                vmin=0 if categorical else -1,
                vmax=1,
            )
            axis.set_yticks(range(len(targets)), targets, fontsize=7)
            axis.set_xticks(
                range(len(columns)),
                [f"{turn_range}\nL{layer}" for turn_range, layer in columns],
                rotation=90,
                fontsize=7,
            )
            axis.set_title(
                f"E2 independent temporal decoding · {scope} · {model}",
                loc="left",
            )
            for boundary in range(len(layers), len(columns), len(layers)):
                axis.axvline(boundary - 0.5, color="black", linewidth=0.8)
            figure.colorbar(
                image,
                ax=axis,
                label=(
                    "Balanced accuracy"
                    if categorical
                    else "Held-topic Pearson correlation"
                ),
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages += 1
    return pages


def export_cross_temporal_matrices(scores: pd.DataFrame, path: Path) -> int:
    if scores.empty:
        _empty_pdf(path, "Cross-temporal E2 analysis was skipped or unavailable.")
        return 1
    values = scores.copy()
    values["primary_metric"] = np.where(
        values["task"].eq("continuous"),
        pd.to_numeric(values["activation_pearson"], errors="coerce"),
        pd.to_numeric(values["activation_score"], errors="coerce"),
    )
    pages = 0
    with PdfPages(path) as pdf:
        for (scope, model, target, task), subset in values.groupby(
            ["condition_scope", "model", "target", "task"], sort=False
        ):
            layers = sorted(subset["layer"].dropna().astype(int).unique())
            ranges = sorted(
                set(subset["source_turn_range"].astype(str))
                | set(subset["destination_turn_range"].astype(str)),
                key=_range_key,
            )
            columns = min(2, max(1, len(layers)))
            rows = math.ceil(len(layers) / columns)
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(5.4 * columns, 4.6 * rows),
                squeeze=False,
            )
            for axis, layer in zip(axes.flat, layers):
                matrix = subset[subset["layer"].eq(layer)].pivot_table(
                    index="source_turn_range",
                    columns="destination_turn_range",
                    values="primary_metric",
                    aggfunc="first",
                ).reindex(index=ranges, columns=ranges)
                image = axis.imshow(
                    np.ma.masked_invalid(matrix.to_numpy(float)),
                    cmap=plt.get_cmap(
                        "viridis" if task == "categorical" else "coolwarm"
                    ).with_extremes(bad="#eeeeee"),
                    vmin=0 if task == "categorical" else -1,
                    vmax=1,
                )
                axis.set_xticks(range(len(ranges)), ranges, rotation=45)
                axis.set_yticks(range(len(ranges)), ranges)
                axis.set_xlabel("Test range")
                axis.set_ylabel("Train range")
                axis.set_title(f"Layer {layer}", loc="left")
                figure.colorbar(image, ax=axis, fraction=0.046)
            for axis in axes.flat[len(layers):]:
                axis.axis("off")
            figure.suptitle(
                f"E2 cross-temporal generalization · {scope} · "
                f"{model} · {target}",
                fontsize=13,
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages += 1
    return pages


def export_temporal_summary(summary: pd.DataFrame, path: Path) -> int:
    if summary.empty:
        _empty_png(path, "No E2 temporal summary is available.")
        return 1
    values = summary.copy()
    values["row"] = (
        values["condition_scope"].astype(str)
        + " · "
        + values["model"].astype(str)
        + " · "
        + values["target"].astype(str)
    )
    ranges = sorted(values["turn_range"].astype(str).unique(), key=_range_key)
    rows = list(dict.fromkeys(values["row"]))
    pivot = values.pivot_table(
        index="row",
        columns="turn_range",
        values="peak_correlation_or_score",
        aggfunc="first",
    ).reindex(index=rows, columns=ranges)
    layers = values.pivot_table(
        index="row",
        columns="turn_range",
        values="peak_layer",
        aggfunc="first",
    ).reindex(index=rows, columns=ranges)
    reliable = values.pivot_table(
        index="row",
        columns="turn_range",
        values="reliable_decoding",
        aggfunc="first",
    ).reindex(index=rows, columns=ranges)
    matrix = pivot.to_numpy(float)
    figure, axis = plt.subplots(
        figsize=(max(9, 1.2 * len(ranges) + 5), max(5, 0.25 * len(rows) + 2))
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=plt.get_cmap("coolwarm").with_extremes(bad="#eeeeee"),
        vmin=-1,
        vmax=1,
    )
    axis.set_yticks(range(len(rows)), rows, fontsize=6)
    axis.set_xticks(range(len(ranges)), ranges)
    axis.set_title(
        "E2 peak temporal decoding\n"
        "cell = score and peak layer; ★ = topic-bootstrap reliable",
        loc="left",
    )
    for row in range(len(rows)):
        for column in range(len(ranges)):
            score = pivot.iat[row, column]
            layer = layers.iat[row, column]
            if np.isfinite(score) and np.isfinite(layer):
                star = "★" if bool(reliable.iat[row, column]) else ""
                axis.text(
                    column,
                    row,
                    f"{score:.2f}\nL{int(layer)}{star}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    figure.colorbar(image, ax=axis, label="Peak correlation or score")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 1


def export_peak_migration(summary: pd.DataFrame, path: Path) -> int:
    if summary.empty:
        _empty_pdf(path, "No E2 peak-layer migration is available.")
        return 1
    pages = 0
    with PdfPages(path) as pdf:
        for (scope, model), subset in summary.groupby(
            ["condition_scope", "model"], sort=False
        ):
            ranges = sorted(
                subset["turn_range"].astype(str).unique(), key=_range_key
            )
            order = {name: index for index, name in enumerate(ranges)}
            figure, axis = plt.subplots(figsize=(10, 5.5))
            for target, target_rows in subset.groupby("target", sort=False):
                target_rows = target_rows.copy()
                target_rows["_order"] = target_rows["turn_range"].map(order)
                target_rows = target_rows.sort_values("_order")
                axis.plot(
                    target_rows["_order"],
                    target_rows["peak_layer"],
                    marker="o",
                    linewidth=1.2,
                    label=target,
                )
            axis.set_xticks(range(len(ranges)), ranges)
            axis.set_xlabel("Conversation turn range")
            axis.set_ylabel("Peak activation layer")
            axis.set_title(
                f"E2 layer-of-peak migration · {scope} · {model}",
                loc="left",
            )
            axis.grid(alpha=0.2)
            axis.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=7,
            )
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages += 1
    return pages


def export_generalization_diagnostics(
    diagnostics: pd.DataFrame, path: Path
) -> int:
    if diagnostics.empty:
        _empty_png(path, "No E2 cross-temporal diagnostics are available.")
        return 1
    values = diagnostics.copy()
    patterns = {
        "stable": ("#16a34a", "o"),
        "phase_specific": ("#dc2626", "s"),
        "progressive_asymmetric": ("#7c3aed", "^"),
        "intermediate_or_weak": ("#64748b", "o"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for pattern, (color, marker) in patterns.items():
        subset = values[values["descriptive_pattern"].eq(pattern)]
        axes[0].scatter(
            subset["diagonal_mean"],
            subset["off_diagonal_mean"],
            c=color,
            marker=marker,
            alpha=0.75,
            label=pattern,
        )
        axes[1].scatter(
            subset["generalization_gap"],
            subset["forward_minus_reverse"],
            c=color,
            marker=marker,
            alpha=0.75,
            label=pattern,
        )
    limits = axes[0].get_xlim()
    lower = min(limits[0], axes[0].get_ylim()[0])
    upper = max(limits[1], axes[0].get_ylim()[1])
    axes[0].plot([lower, upper], [lower, upper], "--", color="#777777")
    axes[0].set_xlabel("Diagonal decoding")
    axes[0].set_ylabel("Off-diagonal transfer")
    axes[0].set_title("Stability across phases", loc="left")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_xlabel("Diagonal − off-diagonal")
    axes[1].set_ylabel("Early→late − late→early")
    axes[1].set_title("Phase specificity and asymmetry", loc="left")
    axes[0].legend(fontsize=7)
    figure.suptitle("E2 cross-temporal generalization diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 1


def export_condition_contrast(contrasts: pd.DataFrame, path: Path) -> int:
    subset = (
        contrasts[contrasts["contrast_type"].eq("independent_range")].copy()
        if not contrasts.empty
        else pd.DataFrame()
    )
    if subset.empty:
        _empty_png(path, "No self-play versus mixed-play E2 contrast is available.")
        return 1
    subset["row"] = subset["model"].astype(str) + " · " + subset["target"].astype(str)
    ranges = sorted(
        subset["turn_range"].dropna().astype(str).unique(), key=_range_key
    )
    rows = list(dict.fromkeys(subset["row"]))
    pivot = subset.pivot_table(
        index="row",
        columns="turn_range",
        values="mixed_minus_self_peak_metric",
        aggfunc="first",
    ).reindex(index=rows, columns=ranges)
    matrix = pivot.to_numpy(float)
    bound = max(
        0.1,
        float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.1,
    )
    figure, axis = plt.subplots(
        figsize=(max(8, len(ranges) + 5), max(4, 0.25 * len(rows) + 2))
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=plt.get_cmap("coolwarm").with_extremes(bad="#eeeeee"),
        vmin=-bound,
        vmax=bound,
    )
    axis.set_yticks(range(len(rows)), rows, fontsize=6)
    axis.set_xticks(range(len(ranges)), ranges)
    axis.set_title(
        "E2 condition contrast · mixed-play minus self-play peak decoding",
        loc="left",
    )
    figure.colorbar(image, ax=axis, label="Mixed − self")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return 1


def export_e2_figures(results_dir: str | Path) -> dict[str, int]:
    root = Path(results_dir)
    independent = _read(root / "e2_independent_scores.csv")
    temporal = _read(root / "e2_temporal_summary.csv")
    cross = _read(root / "e2_cross_temporal_scores.csv")
    diagnostics = _read(root / "e2_cross_temporal_diagnostics.csv")
    contrasts = _read(root / "e2_condition_contrasts.csv")
    return {
        "e2_independent_layer_turn_heatmaps.pdf": export_independent_heatmaps(
            independent, root / "e2_independent_layer_turn_heatmaps.pdf"
        ),
        "e2_cross_temporal_matrices.pdf": export_cross_temporal_matrices(
            cross, root / "e2_cross_temporal_matrices.pdf"
        ),
        "e2_temporal_summary.png": export_temporal_summary(
            temporal, root / "e2_temporal_summary.png"
        ),
        "e2_peak_layer_migration.pdf": export_peak_migration(
            temporal, root / "e2_peak_layer_migration.pdf"
        ),
        "e2_generalization_diagnostics.png": (
            export_generalization_diagnostics(
                diagnostics, root / "e2_generalization_diagnostics.png"
            )
        ),
        "e2_self_vs_mixed.png": export_condition_contrast(
            contrasts, root / "e2_self_vs_mixed.png"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Q1 E2 figures")
    parser.add_argument("--results-dir", default="results/q1/e2")
    args = parser.parse_args()
    print(export_e2_figures(args.results_dir))


if __name__ == "__main__":
    main()
