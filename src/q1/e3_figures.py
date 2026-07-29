"""Static figure export for Q1 E3 subspace results."""

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


def _range_key(value: str) -> tuple[float, str]:
    match = re.match(r"^\s*([0-9.]+)", str(value))
    return (float(match.group(1)) if match else math.inf, str(value))


def _empty_figure(path: Path, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()

def export_rank_performance(scores: pd.DataFrame, path: Path) -> int:
    """Write model/family rank curves, with one panel per activation layer."""
    pages = 0
    with PdfPages(path) as pdf:
        if not scores.empty:
            scores = scores.copy()
            scores["rank"] = pd.to_numeric(scores["rank"], errors="coerce")
            scores["mean_target_pearson"] = pd.to_numeric(
                scores["mean_target_pearson"], errors="coerce"
            )
            for (model, family), subset in scores.groupby(
                ["model", "family"], sort=False
            ):
                layers = sorted(subset["layer"].dropna().astype(int).unique())
                ranges = sorted(
                    subset["turn_range"].dropna().astype(str).unique(),
                    key=_range_key,
                )
                columns = min(2, max(1, len(layers)))
                rows = math.ceil(len(layers) / columns)
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(6.0 * columns, 3.8 * rows),
                    squeeze=False,
                    sharey=True,
                )
                for axis, layer in zip(axes.flat, layers):
                    layer_scores = subset[subset["layer"].eq(layer)]
                    for turn_range in ranges:
                        values = layer_scores[
                            layer_scores["turn_range"].eq(turn_range)
                        ].sort_values("rank")
                        axis.plot(
                            values["rank"],
                            values["mean_target_pearson"],
                            marker="o",
                            label=turn_range,
                        )
                    axis.axhline(0, color="#777777", linewidth=0.8)
                    axis.set_title(f"Layer {layer}", loc="left")
                    axis.set_xlabel("Retained subspace dimensions")
                    axis.set_ylabel("Held-topic mean target Pearson")
                    axis.grid(alpha=0.2)
                for axis in axes.flat[len(layers):]:
                    axis.axis("off")
                if layers:
                    axes.flat[0].legend(title="Conversation turns", fontsize=8)
                figure.suptitle(
                    f"E3 rank performance · {model} · {family}",
                    fontsize=14,
                )
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                pages += 1
        if pages == 0:
            figure, axis = plt.subplots(figsize=(8, 3))
            axis.axis("off")
            axis.text(
                0.5, 0.5, "No finite E3 rank scores are available.",
                ha="center", va="center",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages = 1
    return pages


def export_selected_dimensionality(
    selections: pd.DataFrame, path: Path
) -> int:
    """Render selected rank for every family × model/range/layer cell."""
    if selections.empty:
        _empty_figure(path, "No E3 rank selections are available.")
        return 1
    values = selections.copy()
    values["rank_90pct"] = pd.to_numeric(
        values["rank_90pct"], errors="coerce"
    )
    families = list(dict.fromkeys(values["family"].astype(str)))
    models = sorted(values["model"].astype(str).unique())
    ranges = sorted(
        values["turn_range"].astype(str).unique(), key=_range_key
    )
    layers = sorted(values["layer"].dropna().astype(int).unique())
    columns = [
        (model, turn_range, layer)
        for model in models
        for turn_range in ranges
        for layer in layers
    ]
    pivot = values.pivot_table(
        index="family",
        columns=["model", "turn_range", "layer"],
        values="rank_90pct",
        aggfunc="first",
    ).reindex(index=families, columns=pd.MultiIndex.from_tuples(columns))
    matrix = pivot.to_numpy(float)
    figure, axis = plt.subplots(
        figsize=(max(11, 0.38 * len(columns) + 4), max(4, 0.7 * len(families) + 2))
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=plt.get_cmap("viridis").with_extremes(bad="#eeeeee"),
        vmin=1,
        vmax=max(2, int(np.nanmax(matrix))) if np.isfinite(matrix).any() else 2,
    )
    axis.set_yticks(range(len(families)), families)
    axis.set_xticks(
        range(len(columns)),
        [f"{model}\n{turn_range}\nL{layer}" for model, turn_range, layer in columns],
        rotation=90,
        fontsize=7,
    )
    axis.set_title(
        "E3 selected subspace dimensionality\n"
        "smallest rank reaching 90% of best positive held-topic performance",
        loc="left",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                axis.text(
                    column, row, str(int(matrix[row, column])),
                    ha="center", va="center", fontsize=7, color="white",
                )
    figure.colorbar(image, ax=axis, label="Selected dimensions")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return 1


def export_overlap(overlap: pd.DataFrame, path: Path) -> int:
    """Render principal-angle overlap by family pair and analysis cell."""
    if overlap.empty:
        _empty_figure(path, "No cross-family E3 subspace overlap is available.")
        return 1
    values = overlap.copy()
    values["family_pair"] = (
        values["family_a"].astype(str) + " × " + values["family_b"].astype(str)
    )
    values["cell"] = (
        values["model"].astype(str)
        + "\n"
        + values["turn_range"].astype(str)
        + "\nL"
        + values["layer"].astype(int).astype(str)
    )
    pairs = list(dict.fromkeys(values["family_pair"]))
    cells = list(dict.fromkeys(values["cell"]))
    pivot = values.pivot_table(
        index="family_pair",
        columns="cell",
        values="mean_cosine_similarity",
        aggfunc="first",
    ).reindex(index=pairs, columns=cells)
    matrix = pivot.to_numpy(float)
    figure, axis = plt.subplots(
        figsize=(max(10, 0.42 * len(cells) + 4), max(4, 0.55 * len(pairs) + 2))
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=plt.get_cmap("magma").with_extremes(bad="#eeeeee"),
        vmin=0,
        vmax=1,
    )
    axis.set_yticks(range(len(pairs)), pairs, fontsize=8)
    axis.set_xticks(range(len(cells)), cells, rotation=90, fontsize=7)
    axis.set_title(
        "E3 activation-subspace overlap · mean cosine of principal angles",
        loc="left",
    )
    figure.colorbar(image, ax=axis, label="Mean cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return 1


def export_cross_turn(transfer: pd.DataFrame, path: Path) -> int:
    """Write source-range × destination-range transfer matrices."""
    pages = 0
    with PdfPages(path) as pdf:
        if not transfer.empty:
            for (model, family), subset in transfer.groupby(
                ["model", "family"], sort=False
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
                    figsize=(5.2 * columns, 4.4 * rows),
                    squeeze=False,
                )
                for axis, layer in zip(axes.flat, layers):
                    matrix = subset[subset["layer"].eq(layer)].pivot_table(
                        index="source_turn_range",
                        columns="destination_turn_range",
                        values="mean_target_pearson",
                        aggfunc="first",
                    ).reindex(index=ranges, columns=ranges)
                    image = axis.imshow(
                        np.ma.masked_invalid(matrix.to_numpy(float)),
                        cmap=plt.get_cmap("coolwarm").with_extremes(bad="#eeeeee"),
                        vmin=-1,
                        vmax=1,
                    )
                    axis.set_xticks(range(len(ranges)), ranges, rotation=45)
                    axis.set_yticks(range(len(ranges)), ranges)
                    axis.set_xlabel("Test turn range")
                    axis.set_ylabel("Train turn range")
                    axis.set_title(f"Layer {layer}", loc="left")
                    figure.colorbar(image, ax=axis, fraction=0.046)
                for axis in axes.flat[len(layers):]:
                    axis.axis("off")
                figure.suptitle(
                    f"E3 cross-turn transfer · {model} · {family}",
                    fontsize=14,
                )
                figure.tight_layout()
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                pages += 1
        if pages == 0:
            figure, axis = plt.subplots(figsize=(8, 3))
            axis.axis("off")
            axis.text(
                0.5, 0.5, "Cross-turn transfer was skipped or unavailable.",
                ha="center", va="center",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages = 1
    return pages


def export_e3_figures(results_dir: str | Path) -> dict[str, int]:
    """Generate all four standard E3 result figures from CSV outputs."""
    root = Path(results_dir)
    rank_scores = _read_csv(root / "e3_rank_scores.csv")
    selections = _read_csv(root / "e3_rank_selection.csv")
    overlap = _read_csv(root / "e3_subspace_overlap.csv")
    transfer = _read_csv(root / "e3_cross_turn_transfer.csv")
    return {
        "e3_rank_performance.pdf": export_rank_performance(
            rank_scores, root / "e3_rank_performance.pdf"
        ),
        "e3_selected_dimensionality.png": export_selected_dimensionality(
            selections, root / "e3_selected_dimensionality.png"
        ),
        "e3_subspace_overlap.png": export_overlap(
            overlap, root / "e3_subspace_overlap.png"
        ),
        "e3_cross_turn_transfer.pdf": export_cross_turn(
            transfer, root / "e3_cross_turn_transfer.pdf"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Q1 E3 figures")
    parser.add_argument("--results-dir", default="results/q1/e3")
    args = parser.parse_args()
    print(export_e3_figures(args.results_dir))


if __name__ == "__main__":
    main()
