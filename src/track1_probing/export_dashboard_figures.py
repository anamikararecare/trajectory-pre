"""Export static figures from completed refactored Track 1 artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SNAPSHOT_ORDER = [
    "pre_generation",
    "early_response",
    "full_response",
    "final_window",
    "final_token",
]
EXPERIMENT_ORDER = ["1A", "1B", "1C", "1D", "1E", "1F", "1G", "1H", "1I"]


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_probe_figures(scores: pd.DataFrame, out_dir: Path) -> list[dict]:
    records = []
    summary = (
        scores.groupby(["experiment", "snapshot"], as_index=False)["incremental_score"]
        .mean()
        .pivot(index="experiment", columns="snapshot", values="incremental_score")
        .reindex(columns=[item for item in SNAPSHOT_ORDER if item in scores["snapshot"].unique()])
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.heatmap(summary, cmap="vlag", center=0, annot=True, fmt=".3f", ax=axis)
    axis.set_title("Mean incremental activation lift by experiment and snapshot")
    path = out_dir / "probe_incremental_overview.png"
    _save(figure, path)
    records.append({"kind": "probe_overview", "path": str(path)})

    grouping = ["experiment", "target", "horizon", "delta_target"]
    for identity, subset in scores.groupby(grouping, dropna=False):
        experiment, target, horizon, delta = identity
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for axis, value, title in (
            (axes[0], "activation_plus_baseline_score", "Activation + baseline score"),
            (axes[1], "incremental_score", "Incremental activation lift"),
        ):
            pivot = subset.pivot_table(
                index="layer", columns="snapshot", values=value, aggfunc="mean"
            )
            pivot = pivot.reindex(
                columns=[item for item in SNAPSHOT_ORDER if item in pivot.columns]
            )
            sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".3f", ax=axis)
            axis.set_title(title)
        figure.suptitle(
            f"{experiment} · {target} · horizon={horizon} · delta={delta}", y=1.02
        )
        filename = (
            f"{_slug(experiment)}__{_slug(target)}__h{_slug(horizon)}"
            f"__delta_{_slug(delta)}.png"
        )
        path = out_dir / "probe_heatmaps" / filename
        _save(figure, path)
        records.append({
            "kind": "probe_heatmap",
            "experiment": experiment,
            "target": target,
            "horizon": horizon,
            "delta_target": delta,
            "path": str(path),
        })
    return records


def export_fold_figure(folds: pd.DataFrame, out_dir: Path) -> list[dict]:
    if folds.empty:
        return []
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=folds, x="snapshot", y="incremental_score",
        order=[item for item in SNAPSHOT_ORDER if item in folds["snapshot"].unique()],
        hue="experiment", ax=axis,
    )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_title("Fold-level incremental lift across snapshots")
    axis.tick_params(axis="x", rotation=20)
    path = out_dir / "paired_fold_incremental_lift.png"
    _save(figure, path)
    return [{"kind": "fold_comparison", "path": str(path)}]


def export_transfer_figure(transfer: pd.DataFrame, out_dir: Path) -> list[dict]:
    activation = transfer[transfer["kind"].eq("activation")].copy()
    if activation.empty:
        return []
    figure = sns.relplot(
        data=activation, x="layer", y="mixed_minus_self", hue="snapshot",
        col="model", kind="line", marker="o", facet_kws={"sharex": False},
        hue_order=[item for item in SNAPSHOT_ORDER if item in activation["snapshot"].unique()],
        height=4.5, aspect=1.15,
    )
    figure.set_axis_labels("Layer", "Mixed-play minus self-play activation distance")
    figure.fig.suptitle("Partner-induced activation displacement", y=1.04)
    path = out_dir / "partner_activation_transfer.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure.fig)
    return [{"kind": "partner_transfer", "path": str(path)}]


def export_replay_quality(manifest: dict, out_dir: Path) -> list[dict]:
    rows = []
    for model, details in manifest.get("models", {}).items():
        for row in details.get("validation_rows", []):
            rows.append({"model": model, **row})
    validation = pd.DataFrame(rows)
    if validation.empty:
        return []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column, title in (
        (axes[0], "cosine_similarity", "Cosine similarity"),
        (axes[1], "relative_error", "Relative error"),
        (axes[2], "norm_ratio", "Norm ratio"),
    ):
        sns.ecdfplot(data=validation, x=column, hue="layer", ax=axis)
        axis.set_title(title)
    figure.suptitle("Original versus replayed full-response validation", y=1.02)
    path = out_dir / "replay_validation_quality.png"
    _save(figure, path)
    return [{"kind": "replay_quality", "path": str(path)}]


def export_geometry(geometry: pd.DataFrame, out_dir: Path) -> list[dict]:
    metrics = [
        column for column in (
            "semantic_velocity", "semantic_acceleration", "basin_leaning",
            "partnerward_basin_velocity", "off_axis_distance",
        ) if column in geometry
    ]
    if not metrics:
        return []
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, metric in zip(axes.flat, metrics):
        sns.lineplot(
            data=geometry, x="agent_turn", y=metric, hue="model",
            style="condition", errorbar=("ci", 95), ax=axis,
        )
        axis.set_title(metric.replace("_", " ").title())
    for axis in axes.flat[len(metrics):]:
        axis.remove()
    figure.suptitle("Output-space trajectory variables", y=1.01)
    path = out_dir / "geometry_variables.png"
    _save(figure, path)
    return [{"kind": "geometry", "path": str(path)}]


def export_availability(
    audit: pd.DataFrame, skipped: pd.DataFrame, out_dir: Path
) -> list[dict]:
    records = []
    variable = audit[
        audit["section"].eq("variable") & audit["level"].eq("coverage")
    ].copy()
    if not variable.empty:
        variable["value"] = pd.to_numeric(variable["value"], errors="coerce")
        figure, axis = plt.subplots(figsize=(10, max(4, len(variable) * 0.35)))
        sns.barplot(data=variable, x="value", y="variable", color="#2563EB", ax=axis)
        axis.set_title("Available observations for refactored variables")
        path = out_dir / "variable_coverage.png"
        _save(figure, path)
        records.append({"kind": "variable_coverage", "path": str(path)})
    if not skipped.empty:
        counts = skipped.groupby(["experiment", "reason"]).size().reset_index(name="targets")
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.barplot(data=counts, x="experiment", y="targets", hue="reason", ax=axis)
        axis.set_title("Skipped targets by experiment")
        path = out_dir / "skipped_target_inventory.png"
        _save(figure, path)
        records.append({"kind": "skipped_targets", "path": str(path)})
    return records


def export_sensitivity(sensitivity: pd.DataFrame, out_dir: Path) -> list[dict]:
    if sensitivity.empty:
        return []
    long = sensitivity.melt(
        id_vars=["experiment", "target", "horizon", "layer"],
        value_vars=["original_full_response_score", "replayed_full_response_score"],
        var_name="activation_source", value_name="score",
    )
    figure = sns.relplot(
        data=long, x="layer", y="score", hue="activation_source",
        col="experiment", col_wrap=3, kind="line", marker="o",
        height=3.5, aspect=1.15,
    )
    figure.fig.suptitle("Fold-matched original versus replayed probes", y=1.02)
    path = out_dir / "original_replay_probe_sensitivity.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure.fig)
    return [{"kind": "original_replay_sensitivity", "path": str(path)}]


def export_experiment_summaries(
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    skipped: pd.DataFrame,
    out_dir: Path,
) -> list[dict]:
    """Guarantee one human-readable PNG summary for every Track 1 experiment."""
    records = []
    for experiment in EXPERIMENT_ORDER:
        figure, axis = plt.subplots(figsize=(11, 6))
        row_count = 0
        if experiment == "1A":
            subset = audit[
                audit["section"].eq("variable") & audit["level"].eq("coverage")
            ].copy()
            subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
            row_count = len(subset)
            if not subset.empty:
                sns.barplot(data=subset, x="value", y="variable", color="#2563EB", ax=axis)
                axis.set_title("1A · Measurement coverage and validation audit")
                axis.set_xlabel("Available observations")
            else:
                axis.text(0.5, 0.5, "No measurement-audit rows available", ha="center", va="center")
                axis.set_axis_off()
        else:
            subset = scores[scores["experiment"].eq(experiment)]
            row_count = len(subset)
            if not subset.empty:
                pivot = subset.pivot_table(
                    index="target", columns="snapshot", values="incremental_score",
                    aggfunc="mean",
                ).reindex(
                    columns=[item for item in SNAPSHOT_ORDER if item in subset["snapshot"].unique()]
                )
                figure.set_size_inches(11, max(5, 0.38 * len(pivot)))
                sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".3f", ax=axis)
                axis.set_title(f"{experiment} · Mean incremental activation lift")
            else:
                unavailable = skipped[skipped["experiment"].eq(experiment)]
                reasons = "\n".join(
                    f"{row.target}: {row.reason}" for row in unavailable.itertuples()
                ) or "No completed or explicitly skipped targets"
                axis.text(
                    0.5, 0.5, f"No completed probe rows\n\n{reasons}",
                    ha="center", va="center", wrap=True,
                )
                axis.set_title(f"{experiment} · Artifact availability")
                axis.set_axis_off()
        path = out_dir / "experiment_summaries" / f"experiment_{experiment.lower()}_summary.png"
        _save(figure, path)
        records.append({
            "kind": "experiment_summary", "experiment": experiment,
            "result_rows": row_count, "path": str(path),
        })
    return records


def export_big_five_figures(scores: pd.DataFrame, out_dir: Path) -> list[dict]:
    big_five = scores[
        scores["target"].astype(str).str.startswith("observer_big5_")
    ].copy()
    if big_five.empty:
        return []
    summary = (
        big_five.groupby(["experiment", "target", "snapshot"], as_index=False)
        ["incremental_score"].mean()
    )
    summary["trait"] = (
        summary["target"].str.removeprefix("observer_big5_").str.removesuffix("_trailing3")
    )
    figure = sns.relplot(
        data=summary, x="snapshot", y="incremental_score", hue="trait",
        col="experiment", col_wrap=3, kind="line", marker="o",
        col_order=[item for item in EXPERIMENT_ORDER if item in summary["experiment"].unique()],
        height=3.7, aspect=1.2,
    )
    for axis in figure.axes.flat:
        axis.axhline(0, color="#666666", linewidth=0.8)
        axis.tick_params(axis="x", rotation=25)
    figure.fig.suptitle(
        "Direct observer-rated Big Five · snapshot-resolved activation lift", y=1.02
    )
    path = out_dir / "big_five_snapshot_progression.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure.fig)
    return [{"kind": "big_five", "path": str(path)}]


def export_big_five_observed(turns: pd.DataFrame, out_dir: Path) -> list[dict]:
    if turns.empty:
        return []
    raw = [
        column for column in turns
        if column.startswith("observer_big5_")
        and not column.endswith(
            ("_confidence", "_trailing3", "_self_play_baseline", "_deviation_from_self_play", "_movement")
        )
    ]
    if not raw:
        return []
    records = []
    long = turns.melt(
        id_vars=[column for column in ("condition", "model", "speaker") if column in turns],
        value_vars=raw, var_name="trait", value_name="score",
    ).dropna(subset=["score"])
    long["trait"] = long["trait"].str.removeprefix("observer_big5_")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=long, x="trait", y="score", hue="condition", ax=axis)
    axis.set_ylim(-0.2, 4.2)
    axis.set_title("Direct observer-rated Big Five conversational presentation")
    axis.tick_params(axis="x", rotation=20)
    path = out_dir / "big_five_observed_distributions.png"
    _save(figure, path)
    records.append({"kind": "big_five_observed", "path": str(path)})

    confidence = [f"{column}_confidence" for column in raw if f"{column}_confidence" in turns]
    deviations = [
        f"{column}_deviation_from_self_play"
        for column in raw if f"{column}_deviation_from_self_play" in turns
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    if confidence:
        confidence_long = turns.melt(
            value_vars=confidence, var_name="trait", value_name="confidence"
        ).dropna(subset=["confidence"])
        confidence_long["trait"] = (
            confidence_long["trait"].str.removeprefix("observer_big5_")
            .str.removesuffix("_confidence")
        )
        sns.barplot(data=confidence_long, x="trait", y="confidence", ax=axes[0])
        axes[0].set_ylim(0, 4)
        axes[0].set_title("Mean visible-evidence confidence")
        axes[0].tick_params(axis="x", rotation=25)
    else:
        axes[0].text(0.5, 0.5, "Confidence fields unavailable", ha="center", va="center")
        axes[0].set_axis_off()
    if deviations:
        mixed = turns[turns["condition"].eq("mixed_play")] if "condition" in turns else turns
        deviation_long = mixed.melt(
            value_vars=deviations, var_name="trait", value_name="deviation"
        ).dropna(subset=["deviation"])
        deviation_long["trait"] = (
            deviation_long["trait"].str.removeprefix("observer_big5_")
            .str.removesuffix("_deviation_from_self_play")
        )
        sns.barplot(data=deviation_long, x="trait", y="deviation", ax=axes[1])
        axes[1].axhline(0, color="#666666", linewidth=0.8)
        axes[1].set_title("Mixed-play deviation from same-model self-play")
        axes[1].tick_params(axis="x", rotation=25)
    else:
        axes[1].text(0.5, 0.5, "Deviation fields unavailable", ha="center", va="center")
        axes[1].set_axis_off()
    path = out_dir / "big_five_confidence_and_mixed_deviation.png"
    _save(figure, path)
    records.append({"kind": "big_five_observed", "path": str(path)})
    return records


def write_artifact_manifest(
    results_dir: Path,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    skipped: pd.DataFrame,
    records: list[dict],
) -> None:
    summaries = {
        row["experiment"]: row["path"]
        for row in records
        if row.get("kind") == "experiment_summary"
    }
    rows = []
    for experiment in EXPERIMENT_ORDER:
        if experiment == "1A":
            csv_path = results_dir / "experiment_1a_measurement_audit.csv"
            result_rows = len(audit)
        else:
            csv_path = results_dir / f"experiment_{experiment.lower()}_snapshot_probe_scores.csv"
            result_rows = int(scores["experiment"].eq(experiment).sum())
        skipped_rows = int(skipped["experiment"].eq(experiment).sum()) if not skipped.empty else 0
        status = "complete" if result_rows else ("skipped" if skipped_rows else "unavailable")
        rows.append({
            "experiment": experiment,
            "status": status,
            "result_rows": result_rows,
            "skipped_targets": skipped_rows,
            "csv_path": str(csv_path),
            "png_path": summaries.get(experiment),
            "csv_exists": csv_path.exists(),
            "png_exists": bool(summaries.get(experiment) and Path(summaries[experiment]).exists()),
        })
    pd.DataFrame(rows).to_csv(results_dir / "artifact_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--geometry", default="results/track1/geometry/turn_geometry.csv")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    replay_dir = Path(args.replay_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    scores = pd.read_csv(results_dir / "snapshot_probe_scores.csv")
    folds = pd.read_csv(results_dir / "fold_scores.csv")
    transfer = pd.read_csv(results_dir / "experiment_1i_partner_transfer.csv")
    audit = pd.read_csv(results_dir / "experiment_1a_measurement_audit.csv")
    skipped = pd.read_csv(results_dir / "skipped_targets.csv")
    sensitivity_path = results_dir / "original_replay_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    big_five_turns_path = results_dir / "big_five_turn_states.csv"
    big_five_turns = (
        pd.read_csv(big_five_turns_path) if big_five_turns_path.exists() else pd.DataFrame()
    )
    geometry = pd.read_csv(args.geometry)
    with open(replay_dir / "manifest.json") as handle:
        manifest = json.load(handle)

    records = []
    records.extend(export_probe_figures(scores, out_dir))
    records.extend(export_fold_figure(folds, out_dir))
    records.extend(export_transfer_figure(transfer, out_dir))
    records.extend(export_replay_quality(manifest, out_dir))
    records.extend(export_geometry(geometry, out_dir))
    records.extend(export_availability(audit, skipped, out_dir))
    records.extend(export_sensitivity(sensitivity, out_dir))
    records.extend(export_experiment_summaries(scores, audit, skipped, out_dir))
    records.extend(export_big_five_figures(scores, out_dir))
    records.extend(export_big_five_observed(big_five_turns, out_dir))
    records.extend(export_activation_summaries(replay_dir, results_dir, out_dir, manifest))
    pd.DataFrame(records).to_csv(out_dir / "figure_index.csv", index=False)
    write_artifact_manifest(results_dir, scores, audit, skipped, records)
    print(f"Exported {len(records)} figures under {out_dir}")

def export_activation_summaries(
    replay_dir: Path,
    results_dir: Path,
    out_dir: Path,
    manifest: dict,
) -> list[dict]:
    """Summarize model-internal magnitude and snapshot movement without labels."""
    from src.track1_probing.dashboard_data import parse_activation_key

    validation = {
        model: details.get("validation", {}).get("status", "not_evaluated")
        for model, details in manifest.get("models", {}).items()
    }
    norm_rows = []
    transition_rows = []
    for path in sorted(replay_dir.glob("*.npz")):
        import numpy as np

        with np.load(path, allow_pickle=False) as arrays:
            grouped = {}
            for key in arrays.files:
                identity = parse_activation_key(key)
                if identity is None:
                    continue
                vector = np.asarray(arrays[key], dtype=np.float32).reshape(-1)
                base = (
                    identity["turn"], identity["speaker"],
                    identity["model"], identity["layer"],
                )
                grouped.setdefault(base, {})[identity["snapshot"]] = vector
                norm_rows.append({
                    "conv_id": path.stem,
                    **{column: identity[column] for column in (
                        "turn", "speaker", "model", "layer", "snapshot", "window"
                    )},
                    "validation_status": validation.get(identity["model"]),
                    "activation_norm": float(np.linalg.norm(vector)),
                })
            for (turn, speaker, model, layer), snapshots in grouped.items():
                available = [item for item in SNAPSHOT_ORDER if item in snapshots]
                for source, destination in zip(available, available[1:]):
                    left, right = snapshots[source], snapshots[destination]
                    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                    cosine = float(np.dot(left, right) / denominator) if denominator else float("nan")
                    transition_rows.append({
                        "conv_id": path.stem,
                        "turn": turn,
                        "speaker": speaker,
                        "model": model,
                        "layer": layer,
                        "validation_status": validation.get(model),
                        "source_snapshot": source,
                        "destination_snapshot": destination,
                        "transition": f"{source} → {destination}",
                        "euclidean_distance": float(np.linalg.norm(right - left)),
                        "cosine_distance": 1.0 - cosine,
                    })

    norms = pd.DataFrame(norm_rows)
    transitions = pd.DataFrame(transition_rows)
    norm_summary = (
        norms.groupby(
            ["model", "validation_status", "speaker", "layer", "snapshot"],
            dropna=False,
        )["activation_norm"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    transition_summary = (
        transitions.groupby(
            [
                "model", "validation_status", "speaker", "layer",
                "source_snapshot", "destination_snapshot", "transition",
            ],
            dropna=False,
        )[["euclidean_distance", "cosine_distance"]]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    transition_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else column
        for column in transition_summary.columns
    ]
    norm_path = results_dir / "activation_norm_summary.csv"
    transition_path = results_dir / "activation_transition_summary.csv"
    norm_summary.to_csv(norm_path, index=False)
    transition_summary.to_csv(transition_path, index=False)

    records = []
    figure = sns.relplot(
        data=norm_summary, x="layer", y="mean", hue="snapshot",
        col="model", row="speaker", kind="line", marker="o",
        hue_order=[item for item in SNAPSHOT_ORDER if item in norms["snapshot"].unique()],
        facet_kws={"sharex": False, "sharey": False}, height=3.2, aspect=1.25,
    )
    figure.set_axis_labels("Layer", "Mean activation norm")
    figure.fig.suptitle("Model-internal activation magnitude by layer and snapshot", y=1.01)
    path = out_dir / "activation_norms_by_layer_snapshot.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure.fig)
    records.append({"kind": "activation_norms", "path": str(path)})

    distance_column = "cosine_distance_mean"
    figure = sns.relplot(
        data=transition_summary, x="layer", y=distance_column, hue="transition",
        col="model", row="speaker", kind="line", marker="o",
        facet_kws={"sharex": False, "sharey": False}, height=3.2, aspect=1.25,
    )
    figure.set_axis_labels("Layer", "Mean cosine distance")
    figure.fig.suptitle("Within-turn hidden-state movement across snapshots", y=1.01)
    path = out_dir / "activation_snapshot_transitions.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure.fig)
    records.append({"kind": "activation_transitions", "path": str(path)})
    return records


if __name__ == "__main__":
    main()
