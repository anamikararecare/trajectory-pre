"""E2: temporal manifestation of conversational encodings across turns.

E2 reuses the exact E1 observations and probes in two complementary ways:

1. independent held-topic probes within each percentage turn range; and
2. held-topic cross-temporal probes trained in one range and evaluated in all
   ranges at the same model layer.

All activation spaces remain model-specific and there is no snapshot factor.
"""

from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from src.q1.e1_layerwise import Q1_STATE_TARGETS, run_e1


DEFAULT_CONDITION_SCOPES = ("overall", "self_play", "mixed_play")



@dataclass(frozen=True)
class E2Results:
    independent_scores: pd.DataFrame
    independent_folds: pd.DataFrame
    independent_oof: pd.DataFrame
    temporal_summary: pd.DataFrame
    variable_summary: pd.DataFrame
    cross_temporal_scores: pd.DataFrame
    cross_temporal_oof: pd.DataFrame
    cross_temporal_diagnostics: pd.DataFrame
    condition_contrasts: pd.DataFrame
    skipped: pd.DataFrame


def _range_key(value: str) -> tuple[float, float, str]:
    numbers = re.findall(r"[0-9.]+", str(value))
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1]), str(value)
    return math.inf, math.inf, str(value)


def _ordered_ranges(values: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(map(str, values)), key=_range_key)


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "overall":
        return frame
    if "condition" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["condition"].eq(scope)].copy()


def _restrict_layers(
    frame: pd.DataFrame, layers: Sequence[int] | None
) -> pd.DataFrame:
    if layers is None:
        return frame
    selected = {int(layer) for layer in layers}
    drop = [
        column
        for column in frame
        if column.startswith("layer_")
        and "__" not in column
        and int(column.removeprefix("layer_")) not in selected
    ]
    return frame.drop(columns=drop)


def _primary_metric(
    task: str, observed: np.ndarray, predicted: np.ndarray
) -> float:
    valid = pd.notna(observed) & pd.notna(predicted)
    if valid.sum() < 3:
        return np.nan
    if task == "continuous":
        x = np.asarray(observed[valid], dtype=float)
        y = np.asarray(predicted[valid], dtype=float)
        if np.unique(x).size < 2 or np.unique(y).size < 2:
            return np.nan
        return float(pearsonr(x, y).statistic)
    if np.unique(observed[valid]).size < 2:
        return np.nan
    return float(balanced_accuracy_score(observed[valid], predicted[valid]))


def _bootstrap_weights(
    rng: np.random.Generator, n_bootstrap: int, n_topics: int
) -> np.ndarray:
    draws = rng.integers(0, n_topics, size=(n_bootstrap, n_topics))
    weights = np.zeros((n_bootstrap, n_topics), dtype=np.int16)
    rows = np.repeat(np.arange(n_bootstrap), n_topics)
    np.add.at(weights, (rows, draws.ravel()), 1)
    return weights


def _bootstrap_continuous(
    observed: np.ndarray,
    predicted: np.ndarray,
    topic_codes: np.ndarray,
    weights: np.ndarray,
    n_topics: int,
) -> np.ndarray:
    valid = pd.notna(observed) & pd.notna(predicted)
    x = np.asarray(observed[valid], dtype=float)
    y = np.asarray(predicted[valid], dtype=float)
    codes = topic_codes[valid]
    stats = np.zeros((n_topics, 6), dtype=float)
    np.add.at(stats[:, 0], codes, 1.0)
    np.add.at(stats[:, 1], codes, x)
    np.add.at(stats[:, 2], codes, y)
    np.add.at(stats[:, 3], codes, x * x)
    np.add.at(stats[:, 4], codes, y * y)
    np.add.at(stats[:, 5], codes, x * y)
    aggregate = weights @ stats
    n, sx, sy, sxx, syy, sxy = aggregate.T
    covariance = sxy - sx * sy / np.maximum(n, 1.0)
    variance_x = sxx - sx * sx / np.maximum(n, 1.0)
    variance_y = syy - sy * sy / np.maximum(n, 1.0)
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
    return np.divide(
        covariance,
        denominator,
        out=np.full(len(weights), np.nan),
        where=(n >= 3) & (denominator > 0),
    )


def _bootstrap_balanced_accuracy(
    observed: np.ndarray,
    predicted: np.ndarray,
    topic_codes: np.ndarray,
    weights: np.ndarray,
    n_topics: int,
) -> np.ndarray:
    valid = pd.notna(observed) & pd.notna(predicted)
    truth = np.asarray(observed[valid]).astype(str)
    guess = np.asarray(predicted[valid]).astype(str)
    codes = topic_codes[valid]
    labels, truth_codes = np.unique(truth, return_inverse=True)
    totals = np.zeros((n_topics, len(labels)), dtype=float)
    correct = np.zeros_like(totals)
    np.add.at(totals, (codes, truth_codes), 1.0)
    np.add.at(correct, (codes, truth_codes), (truth == guess).astype(float))
    weighted_totals = weights @ totals
    weighted_correct = weights @ correct
    recalls = np.divide(
        weighted_correct,
        weighted_totals,
        out=np.full_like(weighted_correct, np.nan),
        where=weighted_totals > 0,
    )
    present = np.sum(weighted_totals > 0, axis=1)
    scores = np.nanmean(recalls, axis=1)
    scores[present < 2] = np.nan
    return scores


def _bootstrap_reliability(
    rows: pd.DataFrame,
    task: str,
    n_bootstrap: int,
    seed_key: str,
) -> dict[str, float | bool]:
    if rows.empty or rows["topic_id"].nunique() < 3:
        return {
            "reliability_ci_low": np.nan,
            "reliability_ci_high": np.nan,
            "reliable_decoding": False,
            "n_bootstrap": n_bootstrap,
        }
    topics, topic_codes = np.unique(
        rows["topic_id"].astype(str).to_numpy(), return_inverse=True
    )
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    weights = _bootstrap_weights(
        np.random.default_rng(seed), n_bootstrap, len(topics)
    )
    observed = rows["observed_target"].to_numpy()
    activation = rows["activation_prediction"].to_numpy()
    if task == "continuous":
        statistics = _bootstrap_continuous(
            observed, activation, topic_codes, weights, len(topics)
        )
    else:
        activation_scores = _bootstrap_balanced_accuracy(
            observed, activation, topic_codes, weights, len(topics)
        )
        null_scores = _bootstrap_balanced_accuracy(
            observed,
            rows["null_prediction"].to_numpy(),
            topic_codes,
            weights,
            len(topics),
        )
        statistics = activation_scores - null_scores
    statistics = statistics[np.isfinite(statistics)]
    if len(statistics) < max(20, n_bootstrap // 4):
        return {
            "reliability_ci_low": np.nan,
            "reliability_ci_high": np.nan,
            "reliable_decoding": False,
            "n_bootstrap": n_bootstrap,
        }
    low, high = np.quantile(statistics, [0.025, 0.975])
    return {
        "reliability_ci_low": float(low),
        "reliability_ci_high": float(high),
        "reliable_decoding": bool(low > 0),
        "n_bootstrap": n_bootstrap,
    }


def add_reliability(
    scores: pd.DataFrame,
    oof: pd.DataFrame,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    keys = [
        "condition_scope",
        "model",
        "target",
        "task",
        "turn_range",
        "layer",
    ]
    lookup = {
        identity: group
        for identity, group in oof.groupby(keys, dropna=False, sort=False)
    }
    rows = []
    for score in scores.to_dict(orient="records"):
        identity = tuple(score[key] for key in keys)
        reliability = _bootstrap_reliability(
            lookup.get(identity, pd.DataFrame()),
            str(score["task"]),
            n_bootstrap,
            "|".join(map(str, identity)),
        )
        rows.append({**score, **reliability})
    return pd.DataFrame(rows)


def load_e1_results(
    result_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load E1 tables that are sufficient for E2's overall-scope analysis."""
    root = Path(result_dir)
    names = (
        "e1_layerwise_scores.csv",
        "e1_fold_scores.csv",
        "e1_oof_predictions.csv",
        "e1_skipped.csv",
    )
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"E1 result directory {root} is incomplete; missing {missing}"
        )
    def read_table(name: str) -> pd.DataFrame:
        try:
            return pd.read_csv(root / name)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    scores, folds, oof, skipped = (read_table(name) for name in names)
    if not folds.empty and not oof.empty and "topic_id" in oof:
        keys = ["model", "target", "turn_range", "layer"]
        label_counts = (
            oof.groupby([*keys, "topic_id"], dropna=False)["observed_target"]
            .nunique()
            .rename("held_out_label_count")
            .reset_index()
            .rename(columns={"topic_id": "held_out_group"})
        )
        folds = folds.merge(
            label_counts, on=[*keys, "held_out_group"], how="left"
        )
        categorical = folds.get("task", pd.Series(index=folds.index)).eq(
            "categorical"
        )
        folds["single_label_test_fold"] = (
            categorical & folds["held_out_label_count"].lt(2)
        )
        score_columns = [
            column for column in (
                "null_score", "baseline_score", "activation_score",
                "combined_score",
            ) if column in folds
        ]
        folds.loc[folds["single_label_test_fold"], score_columns] = np.nan
    return scores, folds, oof, skipped


def run_independent_ranges(
    frame: pd.DataFrame,
    targets: Sequence[str],
    models: Sequence[str] | None,
    turn_ranges: Sequence[str] | None,
    layers: Sequence[int] | None,
    condition_scopes: Sequence[str],
    group_column: str,
    n_bootstrap: int,
    precomputed_overall: tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ] | None = None,
    n_jobs: int = 1,
    progress: Callable[..., None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    restricted = _restrict_layers(frame, layers)

    def compute_scope(scope: str):
        if scope == "overall" and precomputed_overall is not None:
            scores, folds, oof, skipped = (
                table.copy() for table in precomputed_overall
            )
            source = "reused_e1"
        else:
            scoped = _scope_frame(restricted, scope)
            if scoped.empty:
                skipped = pd.DataFrame(
                    [
                        {
                            "stage": "independent",
                            "condition_scope": scope,
                            "model": None,
                            "target": None,
                            "turn_range": None,
                            "layer": None,
                            "reason": "condition scope unavailable",
                        }
                    ]
                )
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), skipped, "empty"
            scores, folds, oof, skipped = run_e1(
                scoped,
                targets=targets,
                models=models,
                turn_ranges=turn_ranges,
                group_column=group_column,
                progress=progress,
                progress_stage=f"e2_independent_{scope}",
            )
            source = "computed"
        for table in (scores, folds, oof):
            if not table.empty:
                table["experiment"] = "E2_independent"
                table["condition_scope"] = scope
        if not skipped.empty:
            skipped["stage"] = "independent"
            skipped["condition_scope"] = scope
        return scores, folds, oof, skipped, source

    scopes = list(condition_scopes)
    workers = max(1, min(int(n_jobs), len(scopes)))
    completed = 0
    by_scope = {}
    if workers == 1:
        for scope in scopes:
            by_scope[scope] = compute_scope(scope)
            completed += 1
            if progress is not None:
                progress(
                    "e2_independent_scopes", completed, len(scopes),
                    condition_scope=scope, source=by_scope[scope][-1],
                )
    else:
        threads_per_worker = max(1, (os.cpu_count() or 1) // workers)
        with threadpool_limits(limits=threads_per_worker):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(compute_scope, scope): scope
                    for scope in scopes
                }
                for future in as_completed(futures):
                    scope = futures[future]
                    by_scope[scope] = future.result()
                    completed += 1
                    if progress is not None:
                        progress(
                            "e2_independent_scopes", completed, len(scopes),
                            condition_scope=scope,
                            source=by_scope[scope][-1],
                        )
    score_parts, fold_parts, oof_parts, skip_parts = [], [], [], []
    for scope in scopes:
        scores, folds, oof, skipped, _ = by_scope[scope]
        score_parts.append(scores)
        fold_parts.append(folds)
        oof_parts.append(oof)
        skip_parts.append(skipped)
    scores = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    folds = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    skipped = pd.concat(skip_parts, ignore_index=True) if skip_parts else pd.DataFrame()
    if progress is not None:
        progress("e2_bootstrap", 0, len(scores), status="started")
    reliable = add_reliability(scores, oof, n_bootstrap)
    if progress is not None:
        progress("e2_bootstrap", len(scores), len(scores), status="complete")
    return reliable, folds, oof, skipped


def summarize_independent(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    values = scores.copy()
    values["primary_metric"] = np.where(
        values["task"].eq("continuous"),
        pd.to_numeric(values["activation_only_pearson"], errors="coerce"),
        pd.to_numeric(values["activation_only_score"], errors="coerce"),
    )
    cell_keys = [
        "condition_scope",
        "model",
        "target",
        "task",
        "turn_range",
    ]
    rows = []
    for identity, group in values.groupby(cell_keys, sort=False):
        finite = group[np.isfinite(group["primary_metric"])].sort_values("layer")
        if finite.empty:
            continue
        peak = finite.loc[finite["primary_metric"].idxmax()]
        layers = finite["layer"].to_numpy(float)
        metric = finite["primary_metric"].to_numpy(float)
        if len(layers) == 1 or layers.max() == layers.min():
            signed_auc = float(metric[0])
            positive_auc = float(max(metric[0], 0.0))
        else:
            depth = (layers - layers.min()) / (layers.max() - layers.min())
            signed_auc = float(np.trapezoid(metric, depth))
            positive_auc = float(np.trapezoid(np.maximum(metric, 0), depth))
        lower = float(peak.get("reliability_ci_low", np.nan))
        rows.append(
            {
                **dict(zip(cell_keys, identity)),
                "peak_layer": int(peak["layer"]),
                "peak_correlation_or_score": float(peak["primary_metric"]),
                "peak_reliability_ci_low": lower,
                "peak_reliability_ci_high": float(
                    peak.get("reliability_ci_high", np.nan)
                ),
                "reliable_decoding": bool(
                    peak.get("reliable_decoding", False)
                ),
                "layerwise_signed_auc": signed_auc,
                "layerwise_positive_auc": positive_auc,
                "n_layers": len(finite),
                "n": int(peak["n"]),
                "n_topics": int(peak["n_topics"]),
            }
        )
    temporal = pd.DataFrame(rows)
    variable_rows = []
    variable_keys = ["condition_scope", "model", "target", "task"]
    for identity, group in temporal.groupby(variable_keys, sort=False):
        group = group.copy()
        order = {name: index for index, name in enumerate(
            _ordered_ranges(group["turn_range"].astype(str))
        )}
        group["_order"] = group["turn_range"].map(order)
        group = group.sort_values("_order")
        reliable = group[group["reliable_decoding"]]
        peak_layers = group["peak_layer"].to_numpy(float)
        x = group["_order"].to_numpy(float)
        slope = (
            float(np.polyfit(x, peak_layers, 1)[0])
            if len(group) >= 2
            else np.nan
        )
        normalized_slope = (
            slope / max(1.0, float(scores.loc[
                scores["model"].eq(identity[1]), "layer"
            ].max()))
            if np.isfinite(slope)
            else np.nan
        )
        variable_rows.append(
            {
                **dict(zip(variable_keys, identity)),
                "earliest_reliable_turn_range": (
                    reliable.iloc[0]["turn_range"] if not reliable.empty else None
                ),
                "n_reliable_turn_ranges": int(group["reliable_decoding"].sum()),
                "peak_layer_first_range": int(peak_layers[0]),
                "peak_layer_last_range": int(peak_layers[-1]),
                "net_peak_layer_migration": float(
                    peak_layers[-1] - peak_layers[0]
                ),
                "absolute_peak_layer_migration": float(
                    np.abs(np.diff(peak_layers)).sum()
                ),
                "peak_layer_slope_per_range": slope,
                "normalized_peak_layer_slope": normalized_slope,
                "mean_peak_correlation_or_score": float(
                    group["peak_correlation_or_score"].mean()
                ),
                "mean_layerwise_signed_auc": float(
                    group["layerwise_signed_auc"].mean()
                ),
                "mean_layerwise_positive_auc": float(
                    group["layerwise_positive_auc"].mean()
                ),
            }
        )
    return temporal, pd.DataFrame(variable_rows)


def cross_temporal_diagnostics(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    values = scores.copy()
    values["categorical_excess"] = (
        pd.to_numeric(values.get("activation_score"), errors="coerce")
        - pd.to_numeric(values.get("null_score"), errors="coerce")
    )
    values["primary_metric"] = np.where(
        values["task"].eq("continuous"),
        pd.to_numeric(values["activation_pearson"], errors="coerce"),
        pd.to_numeric(values["categorical_excess"], errors="coerce"),
    )
    keys = ["condition_scope", "model", "target", "task", "layer"]
    rows = []
    for identity, group in values.groupby(keys, sort=False):
        ranges = _ordered_ranges(
            set(group["source_turn_range"].astype(str))
            | set(group["destination_turn_range"].astype(str))
        )
        order = {name: index for index, name in enumerate(ranges)}
        source_order = group["source_turn_range"].map(order)
        destination_order = group["destination_turn_range"].map(order)
        diagonal = group[source_order.eq(destination_order)]["primary_metric"]
        off_diagonal = group[source_order.ne(destination_order)]["primary_metric"]
        forward = group[source_order.lt(destination_order)]["primary_metric"]
        reverse = group[source_order.gt(destination_order)]["primary_metric"]
        diagonal_mean = float(diagonal.mean())
        off_diagonal_mean = float(off_diagonal.mean())
        forward_mean = float(forward.mean())
        reverse_mean = float(reverse.mean())
        ratio = (
            off_diagonal_mean / diagonal_mean
            if np.isfinite(diagonal_mean) and diagonal_mean > 0
            else np.nan
        )
        asymmetry = forward_mean - reverse_mean
        if np.isfinite(asymmetry) and abs(asymmetry) >= 0.10:
            pattern = "progressive_asymmetric"
        elif np.isfinite(ratio) and diagonal_mean > 0 and ratio >= 0.80:
            pattern = "stable"
        elif np.isfinite(ratio) and diagonal_mean > 0 and ratio < 0.50:
            pattern = "phase_specific"
        else:
            pattern = "intermediate_or_weak"
        rows.append(
            {
                **dict(zip(keys, identity)),
                "diagonal_mean": diagonal_mean,
                "off_diagonal_mean": off_diagonal_mean,
                "off_diagonal_to_diagonal_ratio": ratio,
                "generalization_gap": diagonal_mean - off_diagonal_mean,
                "forward_early_to_late_mean": forward_mean,
                "reverse_late_to_early_mean": reverse_mean,
                "forward_minus_reverse": asymmetry,
                "descriptive_pattern": pattern,
                "n_matrix_cells": len(group),
            }
        )
    return pd.DataFrame(rows)


def condition_contrasts(
    temporal: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not temporal.empty:
        keys = ["model", "target", "task", "turn_range"]
        pivot = temporal[
            temporal["condition_scope"].isin(["self_play", "mixed_play"])
        ].pivot_table(
            index=keys,
            columns="condition_scope",
            values=[
                "peak_correlation_or_score",
                "peak_layer",
                "layerwise_signed_auc",
            ],
            aggfunc="first",
        )
        for identity, values in pivot.iterrows():
            if (
                ("peak_correlation_or_score", "self_play") not in values
                or ("peak_correlation_or_score", "mixed_play") not in values
            ):
                continue
            rows.append(
                {
                    "contrast_type": "independent_range",
                    **dict(zip(keys, identity)),
                    "layer": np.nan,
                    "mixed_minus_self_peak_metric": (
                        values[("peak_correlation_or_score", "mixed_play")]
                        - values[("peak_correlation_or_score", "self_play")]
                    ),
                    "mixed_minus_self_peak_layer": (
                        values[("peak_layer", "mixed_play")]
                        - values[("peak_layer", "self_play")]
                    ),
                    "mixed_minus_self_layerwise_auc": (
                        values[("layerwise_signed_auc", "mixed_play")]
                        - values[("layerwise_signed_auc", "self_play")]
                    ),
                    "mixed_minus_self_off_diagonal": np.nan,
                    "mixed_minus_self_asymmetry": np.nan,
                }
            )
    if not diagnostics.empty:
        keys = ["model", "target", "task", "layer"]
        subset = diagnostics[
            diagnostics["condition_scope"].isin(["self_play", "mixed_play"])
        ]
        pivot = subset.pivot_table(
            index=keys,
            columns="condition_scope",
            values=["off_diagonal_mean", "forward_minus_reverse"],
            aggfunc="first",
        )
        for identity, values in pivot.iterrows():
            if (
                ("off_diagonal_mean", "self_play") not in values
                or ("off_diagonal_mean", "mixed_play") not in values
            ):
                continue
            rows.append(
                {
                    "contrast_type": "cross_temporal",
                    "model": identity[0],
                    "target": identity[1],
                    "task": identity[2],
                    "turn_range": None,
                    "layer": identity[3],
                    "mixed_minus_self_peak_metric": np.nan,
                    "mixed_minus_self_peak_layer": np.nan,
                    "mixed_minus_self_layerwise_auc": np.nan,
                    "mixed_minus_self_off_diagonal": (
                        values[("off_diagonal_mean", "mixed_play")]
                        - values[("off_diagonal_mean", "self_play")]
                    ),
                    "mixed_minus_self_asymmetry": (
                        values[("forward_minus_reverse", "mixed_play")]
                        - values[("forward_minus_reverse", "self_play")]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_e2(
    frame: pd.DataFrame,
    targets: Sequence[str] = Q1_STATE_TARGETS,
    models: Sequence[str] | None = None,
    turn_ranges: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    condition_scopes: Sequence[str] = DEFAULT_CONDITION_SCOPES,
    group_column: str = "topic_id",
    n_bootstrap: int = 500,
    run_cross_temporal_analysis: bool = True,
    e1_results_dir: str | Path | None = None,
    n_jobs: int = 4,
    progress: Callable[..., None] | None = None,
) -> E2Results:
    precomputed_overall = (
        load_e1_results(e1_results_dir)
        if e1_results_dir is not None
        else None
    )
    scores, folds, oof, independent_skips = run_independent_ranges(
        frame,
        targets,
        models,
        turn_ranges,
        layers,
        condition_scopes,
        group_column,
        n_bootstrap,
        precomputed_overall=precomputed_overall,
        n_jobs=n_jobs,
        progress=progress,
    )
    temporal, variable = summarize_independent(scores)
    from src.q1.e2_cross_temporal import run_cross_temporal_optimized
    if run_cross_temporal_analysis:
        cross_scores, cross_oof, cross_skips = run_cross_temporal_optimized(
            frame,
            targets,
            models,
            turn_ranges,
            layers,
            condition_scopes,
            group_column,
            progress=progress,
        )
    else:
        cross_scores, cross_oof, cross_skips = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    diagnostics = cross_temporal_diagnostics(cross_scores)
    contrasts = condition_contrasts(temporal, diagnostics)
    skipped = pd.concat(
        [independent_skips, cross_skips], ignore_index=True
    )
    return E2Results(
        independent_scores=scores,
        independent_folds=folds,
        independent_oof=oof,
        temporal_summary=temporal,
        variable_summary=variable,
        cross_temporal_scores=cross_scores,
        cross_temporal_oof=cross_oof,
        cross_temporal_diagnostics=diagnostics,
        condition_contrasts=contrasts,
        skipped=skipped,
    )


def save_e2_results(results: E2Results, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "e2_independent_scores.csv": results.independent_scores,
        "e2_independent_fold_scores.csv": results.independent_folds,
        "e2_independent_oof_predictions.csv": results.independent_oof,
        "e2_temporal_summary.csv": results.temporal_summary,
        "e2_variable_temporal_summary.csv": results.variable_summary,
        "e2_cross_temporal_scores.csv": results.cross_temporal_scores,
        "e2_cross_temporal_oof_predictions.csv": results.cross_temporal_oof,
        "e2_cross_temporal_diagnostics.csv": (
            results.cross_temporal_diagnostics
        ),
        "e2_condition_contrasts.csv": results.condition_contrasts,
        "e2_skipped.csv": results.skipped,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
