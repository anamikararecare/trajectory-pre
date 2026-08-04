"""Efficient cross-temporal probe evaluation for Q1 E2.

Each source-range/held-topic activation probe is trained once and then applied
to every destination range. This is equivalent to fitting each matrix cell
independently but avoids repeating the same source fit for every column.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score

from src.q1.e1_layerwise import (
    _fit_predict,
    _inner_parameter,
    _is_vector,
    _scale_fold,
    _task_for_target,
    available_layers,
)


def _range_key(value: str) -> tuple[float, str]:
    import re

    match = re.match(r"^\s*([0-9.]+)", str(value))
    return (float(match.group(1)) if match else float("inf"), str(value))


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "overall":
        return frame
    return frame[frame["condition"].eq(scope)].copy()


def _metric(task: str, observed: np.ndarray, predicted: np.ndarray) -> float:
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


def run_cross_temporal_optimized(
    frame: pd.DataFrame,
    targets: Sequence[str],
    models: Sequence[str] | None,
    turn_ranges: Sequence[str] | None,
    layers: Sequence[int] | None,
    condition_scopes: Sequence[str],
    group_column: str,
    progress: Callable[..., None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen_models = (
        list(models)
        if models is not None
        else sorted(frame["model"].dropna().astype(str).unique())
    )
    chosen_ranges = (
        list(turn_ranges)
        if turn_ranges is not None
        else sorted(
            frame["turn_range"].dropna().astype(str).unique(),
            key=_range_key,
        )
    )
    score_rows: list[dict] = []
    oof_rows: list[dict] = []
    skipped: list[dict] = []
    selected_layers = set(map(int, layers)) if layers is not None else None
    total_cells = 0
    for scope in condition_scopes:
        scoped = _scope_frame(frame, scope)
        for model in chosen_models:
            count = len(available_layers(scoped, model))
            if selected_layers is not None:
                count = len(
                    set(available_layers(scoped, model)) & selected_layers
                )
            total_cells += count * len(targets)
    completed_cells = 0

    def mark(scope: str, model: str, target: str, layer: int, status: str):
        nonlocal completed_cells
        completed_cells += 1
        if progress is not None:
            progress(
                "e2_cross_temporal", completed_cells, total_cells,
                condition_scope=scope, model=model, target=target,
                layer=layer, status=status,
            )
    for scope in condition_scopes:
        scoped = _scope_frame(frame, scope)
        for model in chosen_models:
            model_layers = available_layers(scoped, model)
            if selected_layers is not None:
                model_layers = [
                    layer for layer in model_layers if layer in selected_layers
                ]
            for target in targets:
                for layer in model_layers:
                    activation_column = f"layer_{layer}"
                    selected = (
                        scoped[
                            scoped["model"].eq(model)
                            & scoped["turn_range"].isin(chosen_ranges)
                            & scoped[target].notna()
                            & scoped[activation_column].map(_is_vector)
                        ].copy()
                        if target in scoped
                        else pd.DataFrame()
                    )
                    if (
                        selected.empty
                        or selected[group_column].nunique() < 3
                    ):
                        skipped.append(
                            {
                                "stage": "cross_temporal",
                                "condition_scope": scope,
                                "model": model,
                                "target": target,
                                "turn_range": None,
                                "layer": layer,
                                "reason": (
                                    "insufficient rows, topics, classes, or "
                                    "variance"
                                ),
                            }
                        )
                        mark(scope, model, target, layer, "skipped")
                        continue
                    task = _task_for_target(selected, target)
                    labels = None
                    if task == "continuous":
                        numeric = pd.to_numeric(
                            selected[target], errors="coerce"
                        )
                        selected = selected.loc[numeric.notna()].copy()
                        target_values = pd.to_numeric(
                            selected[target], errors="raise"
                        ).to_numpy(float)
                        raw_target = target_values.astype(object)
                    else:
                        codes, labels = pd.factorize(
                            selected[target].astype(str), sort=True
                        )
                        target_values = codes.astype(int)
                        raw_target = selected[target].astype(str).to_numpy(
                            object
                        )
                    if np.unique(target_values).size < 2:
                        mark(scope, model, target, layer, "skipped")
                        continue
                    activation = np.stack(
                        selected[activation_column].to_numpy()
                    )
                    groups = selected[group_column].to_numpy()
                    pair_predictions = {
                        (source, destination): {
                            "activation": np.full(len(selected), np.nan),
                            "null": np.full(len(selected), np.nan),
                        }
                        for source in chosen_ranges
                        for destination in chosen_ranges
                    }
                    fold_counts = {source: 0 for source in chosen_ranges}
                    for source_range in chosen_ranges:
                        source_mask = selected["turn_range"].eq(
                            source_range
                        ).to_numpy()
                        if source_mask.sum() < 8:
                            continue
                        for held_out in np.unique(groups):
                            train = np.flatnonzero(
                                source_mask & (groups != held_out)
                            )
                            test_all = np.flatnonzero(groups == held_out)
                            if (
                                len(train) < 8
                                or len(test_all) < 2
                                or np.unique(groups[train]).size < 2
                            ):
                                continue
                            train_y = target_values[train]
                            if (
                                task == "categorical"
                                and np.unique(train_y).size < 2
                            ):
                                continue
                            parameter = _inner_parameter(
                                activation[train],
                                train_y,
                                groups[train],
                                task,
                            )
                            train_x, test_x = _scale_fold(
                                activation, train, test_all
                            )
                            test_prediction = _fit_predict(
                                train_x,
                                train_y,
                                test_x,
                                task,
                                parameter,
                            )
                            if task == "continuous":
                                null_value = float(np.mean(train_y))
                            else:
                                values, counts = np.unique(
                                    train_y, return_counts=True
                                )
                                null_value = values[np.argmax(counts)]
                            held_rows = selected.iloc[test_all]
                            for destination_range in chosen_ranges:
                                local = held_rows["turn_range"].eq(
                                    destination_range
                                ).to_numpy()
                                test = test_all[local]
                                if len(test) < 2:
                                    continue
                                pair = pair_predictions[
                                    (source_range, destination_range)
                                ]
                                pair["activation"][test] = test_prediction[
                                    local
                                ]
                                pair["null"][test] = null_value
                            fold_counts[source_range] += 1
                    for pair_identity, predictions in pair_predictions.items():
                        source_range, destination_range = pair_identity
                        valid = np.isfinite(predictions["activation"])
                        if valid.sum() < 3:
                            continue
                        observed = target_values[valid]
                        activation_metric = _metric(
                            task,
                            observed,
                            predictions["activation"][valid],
                        )
                        null_metric = _metric(
                            task,
                            observed,
                            predictions["null"][valid],
                        )
                        identity = {
                            "experiment": "E2_cross_temporal",
                            "condition_scope": scope,
                            "model": model,
                            "target": target,
                            "task": task,
                            "layer": layer,
                            "source_turn_range": source_range,
                            "destination_turn_range": destination_range,
                            "activation_pooling": (
                                "generated_response_token_mean"
                            ),
                            "cv_group": group_column,
                            "n_source": int(
                                selected["turn_range"].eq(
                                    source_range
                                ).sum()
                            ),
                            "n_destination": int(
                                selected["turn_range"].eq(
                                    destination_range
                                ).sum()
                            ),
                            "n_folds": fold_counts[source_range],
                            "n_test": int(valid.sum()),
                            "n_test_topics": int(
                                np.unique(groups[valid]).size
                            ),
                            "activation_score": (
                                activation_metric
                                if task == "categorical"
                                else np.nan
                            ),
                            "activation_pearson": (
                                activation_metric
                                if task == "continuous"
                                else np.nan
                            ),
                            "null_score": null_metric,
                        }
                        score_rows.append(identity)
                        for position in np.flatnonzero(valid):
                            observation = selected.iloc[position]
                            activation_value = predictions["activation"][
                                position
                            ]
                            null_value = predictions["null"][position]
                            if labels is not None:
                                activation_value = labels[
                                    int(activation_value)
                                ]
                                null_value = labels[int(null_value)]
                            oof_rows.append(
                                {
                                    **identity,
                                    "conv_id": observation["conv_id"],
                                    "turn": int(observation["turn"]),
                                    "topic_id": observation["topic_id"],
                                    "speaker": observation.get("speaker"),
                                    "role": observation.get("role"),
                                    "condition": observation.get("condition"),
                                    "observed_target": raw_target[position],
                                    "activation_prediction": activation_value,
                                    "null_prediction": null_value,
                                }
                            )
                    mark(scope, model, target, layer, "complete")
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(oof_rows),
        pd.DataFrame(skipped),
    )
