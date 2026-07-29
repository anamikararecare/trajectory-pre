"""E2: self-play versus mixed-play encoding and condition transfer.

E2 preserves the E1 unit of analysis:

    model × variable × percentage turn range × response-pooled layer

Mixed play is stratified by the other model in the conversation. This is
essential for the qwen2.5-3b anchor, which appears with six mixed-play
partners but has only one self-play cell per topic and role order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.q1.core_variables import E1_CORE_TARGETS
from src.q1.e1_layerwise import (
    _fit_predict,
    _inner_parameter,
    _scale_fold,
    _score,
    _score_row,
    available_layers,
    prepare_design,
    run_e1,
    summarize_peak_layers,
)


CONDITIONS = ("self_play", "mixed_play")


@dataclass(frozen=True)
class E2Results:
    condition_scores: pd.DataFrame
    condition_deltas: pd.DataFrame
    peak_layers: pd.DataFrame
    peak_layer_shifts: pd.DataFrame
    transfer_scores: pd.DataFrame
    transfer_folds: pd.DataFrame
    fold_scores: pd.DataFrame
    oof_predictions: pd.DataFrame
    skipped: pd.DataFrame


def add_interaction_partner_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Identify the other model in every generated conversation."""
    out = frame.copy()
    models_by_conversation = (
        out.groupby("conv_id")["model"]
        .apply(lambda values: tuple(sorted(set(values.dropna().astype(str)))))
        .to_dict()
    )

    def partner(row: pd.Series) -> str | None:
        models = models_by_conversation.get(row["conv_id"], ())
        others = [model for model in models if model != str(row["model"])]
        if row.get("condition") == "self_play":
            return str(row["model"])
        return others[0] if len(others) == 1 else None

    out["interaction_partner_model"] = out.apply(partner, axis=1)
    return out


def _with_selected_layers(
    frame: pd.DataFrame, layers: Sequence[int] | None
) -> pd.DataFrame:
    if layers is None:
        return frame
    keep = {f"layer_{int(layer)}" for layer in layers}
    drop = [
        column
        for column in frame
        if column.startswith("layer_")
        and "__" not in column
        and column not in keep
    ]
    return frame.drop(columns=drop)


def _label_e1_outputs(
    outputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    condition: str,
    partner_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = []
    for table in outputs:
        value = table.copy()
        if not value.empty:
            value["experiment"] = "E2"
            value["condition"] = condition
            value["interaction_partner_model"] = partner_model
        labelled.append(value)
    return tuple(labelled)  # type: ignore[return-value]


def _condition_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    keys = ["model", "target", "task", "turn_range", "layer"]
    self_scores = scores[scores["condition"].eq("self_play")].copy()
    mixed_scores = scores[scores["condition"].eq("mixed_play")].copy()
    metrics = [
        "null_score",
        "baseline_score",
        "activation_only_score",
        "activation_only_pearson",
        "activation_plus_baseline_score",
        "incremental_score",
        "incremental_pearson",
    ]
    available = [metric for metric in metrics if metric in scores]
    left = self_scores[keys + ["n", "n_conversations", *available]]
    right = mixed_scores[
        keys
        + [
            "interaction_partner_model",
            "n",
            "n_conversations",
            *available,
        ]
    ]
    merged = right.merge(left, on=keys, suffixes=("_mixed", "_self"))
    for metric in available:
        merged[f"{metric}_delta_mixed_minus_self"] = (
            pd.to_numeric(merged[f"{metric}_mixed"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_self"], errors="coerce")
        )
    merged["primary_self"] = np.where(
        merged["task"].eq("continuous"),
        merged.get("activation_only_pearson_self"),
        merged["activation_only_score_self"],
    )
    merged["primary_mixed"] = np.where(
        merged["task"].eq("continuous"),
        merged.get("activation_only_pearson_mixed"),
        merged["activation_only_score_mixed"],
    )
    merged["primary_delta_mixed_minus_self"] = (
        merged["primary_mixed"] - merged["primary_self"]
    )
    return merged.sort_values(
        ["model", "interaction_partner_model", "target", "turn_range", "layer"]
    ).reset_index(drop=True)


def _condition_peaks(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    peak_tables = []
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (model, condition, partner), group in scores.groupby(
        ["model", "condition", "interaction_partner_model"], sort=False
    ):
        peaks = summarize_peak_layers(group)
        if peaks.empty:
            continue
        peaks["condition"] = condition
        peaks["interaction_partner_model"] = partner
        peak_tables.append(peaks)
    peak_layers = (
        pd.concat(peak_tables, ignore_index=True)
        if peak_tables
        else pd.DataFrame()
    )
    if peak_layers.empty:
        return peak_layers, pd.DataFrame()
    keys = ["model", "target", "task", "turn_range"]
    self_peaks = peak_layers[peak_layers["condition"].eq("self_play")]
    mixed_peaks = peak_layers[peak_layers["condition"].eq("mixed_play")]
    shifts = mixed_peaks[
        keys
        + [
            "interaction_partner_model",
            "peak_layer",
            "max_activation_correlation_or_score",
        ]
    ].merge(
        self_peaks[
            keys + ["peak_layer", "max_activation_correlation_or_score"]
        ],
        on=keys,
        suffixes=("_mixed", "_self"),
    )
    shifts["peak_layer_shift_mixed_minus_self"] = (
        shifts["peak_layer_mixed"] - shifts["peak_layer_self"]
    )
    shifts["peak_score_delta_mixed_minus_self"] = (
        shifts["max_activation_correlation_or_score_mixed"]
        - shifts["max_activation_correlation_or_score_self"]
    )
    return peak_layers, shifts


def _transfer_one(
    comparison: pd.DataFrame,
    model: str,
    partner_model: str,
    target: str,
    layer: int,
    turn_range: str,
    source_condition: str,
    destination_condition: str,
    group_column: str,
) -> tuple[dict, list[dict]] | None:
    prepared = prepare_design(
        comparison, model, target, layer, turn_range, group_column
    )
    if prepared is None:
        return None
    condition = prepared.frame["condition"].astype(str).to_numpy()
    predictions = {
        name: np.full(len(prepared.target), np.nan)
        for name in ("null", "baseline", "activation", "combined")
    }
    fold_rows = []
    for held_out in np.unique(prepared.groups):
        train = np.flatnonzero(
            (condition == source_condition)
            & (prepared.groups != held_out)
        )
        test = np.flatnonzero(
            (condition == destination_condition)
            & (prepared.groups == held_out)
        )
        if (
            len(train) < 8
            or len(test) < 2
            or np.unique(prepared.groups[train]).size < 2
        ):
            continue
        train_y = prepared.target[train]
        if prepared.task == "categorical" and np.unique(train_y).size < 2:
            continue
        if prepared.task == "continuous":
            predictions["null"][test] = float(np.mean(train_y))
        else:
            values, counts = np.unique(train_y, return_counts=True)
            predictions["null"][test] = values[np.argmax(counts)]
        base_train, base_test = _scale_fold(prepared.baseline, train, test)
        act_train, act_test = _scale_fold(prepared.activation, train, test)
        combined_train = np.concatenate([base_train, act_train], axis=1)
        combined_test = np.concatenate([base_test, act_test], axis=1)
        designs = {
            "baseline": (
                prepared.baseline[train],
                base_train,
                base_test,
            ),
            "activation": (
                prepared.activation[train],
                act_train,
                act_test,
            ),
            "combined": (
                np.concatenate(
                    [prepared.baseline[train], prepared.activation[train]],
                    axis=1,
                ),
                combined_train,
                combined_test,
            ),
        }
        parameters = {}
        for name, (raw_train, train_x, test_x) in designs.items():
            parameter = _inner_parameter(
                raw_train,
                train_y,
                prepared.groups[train],
                prepared.task,
            )
            parameters[name] = parameter
            predictions[name][test] = _fit_predict(
                train_x, train_y, test_x, prepared.task, parameter
            )
        fold_rows.append(
            {
                "held_out_group": str(held_out),
                "n_train": len(train),
                "n_test": len(test),
                "baseline_parameter": parameters["baseline"],
                "activation_parameter": parameters["activation"],
                "combined_parameter": parameters["combined"],
                "activation_score": _score(
                    prepared.target[test],
                    predictions["activation"][test],
                    prepared.task,
                ),
                "combined_score": _score(
                    prepared.target[test],
                    predictions["combined"][test],
                    prepared.task,
                ),
            }
        )
    valid = np.isfinite(predictions["activation"])
    if valid.sum() < 3:
        return None
    identity = {
        "experiment": "E2",
        "model": model,
        "interaction_partner_model": partner_model,
        "target": target,
        "task": prepared.task,
        "turn_range": turn_range,
        "layer": layer,
        "source_condition": source_condition,
        "destination_condition": destination_condition,
        "n_test": int(valid.sum()),
        "n_topics": int(prepared.frame.loc[valid, group_column].nunique()),
        "activation_pooling": "generated_response_token_mean",
        "cv_group": group_column,
    }
    return (
        {**identity, **_score_row(prepared, predictions)},
        [{**identity, **row} for row in fold_rows],
    )


def run_e2(
    frame: pd.DataFrame,
    targets: Sequence[str] = E1_CORE_TARGETS,
    models: Sequence[str] | None = None,
    turn_ranges: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    group_column: str = "topic_id",
) -> E2Results:
    """Run condition-specific E1 probes and self/mixed transfer."""
    required = {
        "conv_id",
        "model",
        "condition",
        "turn_range",
        group_column,
    }
    missing = required.difference(frame)
    if missing:
        raise ValueError(f"E2 dataset is missing columns: {sorted(missing)}")
    working = add_interaction_partner_model(_with_selected_layers(frame, layers))
    chosen_models = (
        list(models)
        if models is not None
        else sorted(working["model"].dropna().astype(str).unique())
    )
    chosen_ranges = (
        list(turn_ranges)
        if turn_ranges is not None
        else list(dict.fromkeys(working["turn_range"].dropna().astype(str)))
    )
    score_tables = []
    fold_tables = []
    oof_tables = []
    skip_tables = []
    transfer_rows = []
    transfer_fold_rows = []
    extra_skips = []

    for model in chosen_models:
        model_rows = working[working["model"].eq(model)]
        self_rows = model_rows[model_rows["condition"].eq("self_play")]
        partners = sorted(
            model_rows.loc[
                model_rows["condition"].eq("mixed_play"),
                "interaction_partner_model",
            ]
            .dropna()
            .astype(str)
            .unique()
        )
        if self_rows.empty or not partners:
            extra_skips.append(
                {
                    "model": model,
                    "condition": None,
                    "interaction_partner_model": None,
                    "target": None,
                    "turn_range": None,
                    "layer": None,
                    "stage": "condition_comparison",
                    "reason": "model lacks self-play or mixed-play rows",
                }
            )
            continue

        self_outputs = _label_e1_outputs(
            run_e1(
                self_rows,
                targets=targets,
                models=(model,),
                turn_ranges=chosen_ranges,
                group_column=group_column,
            ),
            "self_play",
            model,
        )
        for destination, value in zip(
            (score_tables, fold_tables, oof_tables, skip_tables), self_outputs
        ):
            destination.append(value)

        for partner_model in partners:
            mixed_rows = model_rows[
                model_rows["condition"].eq("mixed_play")
                & model_rows["interaction_partner_model"].eq(partner_model)
            ]
            mixed_outputs = _label_e1_outputs(
                run_e1(
                    mixed_rows,
                    targets=targets,
                    models=(model,),
                    turn_ranges=chosen_ranges,
                    group_column=group_column,
                ),
                "mixed_play",
                partner_model,
            )
            for destination, value in zip(
                (score_tables, fold_tables, oof_tables, skip_tables),
                mixed_outputs,
            ):
                destination.append(value)

            comparison = pd.concat([self_rows, mixed_rows], ignore_index=True)
            model_layers = available_layers(comparison, model)
            for target in targets:
                for turn_range in chosen_ranges:
                    for layer in model_layers:
                        for source in CONDITIONS:
                            for destination in CONDITIONS:
                                transfer = _transfer_one(
                                    comparison,
                                    model,
                                    partner_model,
                                    target,
                                    layer,
                                    turn_range,
                                    source,
                                    destination,
                                    group_column,
                                )
                                if transfer is None:
                                    continue
                                score, folds = transfer
                                transfer_rows.append(score)
                                transfer_fold_rows.extend(folds)

    def concatenate(tables: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [table for table in tables if not table.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    condition_scores = concatenate(score_tables)
    fold_scores = concatenate(fold_tables)
    oof_predictions = concatenate(oof_tables)
    e1_skips = concatenate(skip_tables)
    if not e1_skips.empty:
        e1_skips["stage"] = "within_condition_probe"
    skipped = pd.concat(
        [e1_skips, pd.DataFrame(extra_skips)], ignore_index=True
    )
    peak_layers, peak_shifts = _condition_peaks(condition_scores)
    return E2Results(
        condition_scores=condition_scores,
        condition_deltas=_condition_deltas(condition_scores),
        peak_layers=peak_layers,
        peak_layer_shifts=peak_shifts,
        transfer_scores=pd.DataFrame(transfer_rows),
        transfer_folds=pd.DataFrame(transfer_fold_rows),
        fold_scores=fold_scores,
        oof_predictions=oof_predictions,
        skipped=skipped,
    )


def save_e2_results(results: E2Results, output_dir: str) -> None:
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("e2_condition_scores.csv", results.condition_scores),
        ("e2_condition_deltas.csv", results.condition_deltas),
        ("e2_peak_layers.csv", results.peak_layers),
        ("e2_peak_layer_shifts.csv", results.peak_layer_shifts),
        ("e2_cross_condition_transfer.csv", results.transfer_scores),
        ("e2_cross_condition_transfer_folds.csv", results.transfer_folds),
        ("e2_fold_scores.csv", results.fold_scores),
        ("e2_oof_predictions.csv", results.oof_predictions),
        ("e2_skipped.csv", results.skipped),
    ):
        table.to_csv(output / name, index=False)
