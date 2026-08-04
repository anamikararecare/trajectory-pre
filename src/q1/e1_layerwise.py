"""E1: layerwise encoding across percentage ranges of Q1 conversations.

The experiment is model-stratified and uses the single response-pooled
activation saved by Q1 generation. There is intentionally no activation
snapshot degree of freedom. Each estimate is:

    model × Track 1 variable × conversation turn range × layer
"""

from __future__ import annotations

import hashlib
import os
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.q1.core_variables import E1_CORE_TARGETS
from src.track1_probing.variables import VARIABLES


Q1_STATE_TARGETS = E1_CORE_TARGETS
VARIABLES_BY_NAME = {variable.name: variable for variable in VARIABLES}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_CS = (0.01, 0.1, 1.0, 10.0)
E1_CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class PreparedDesign:
    baseline: np.ndarray
    activation: np.ndarray
    target: np.ndarray
    raw_target: np.ndarray
    groups: np.ndarray
    frame: pd.DataFrame
    task: str
    class_labels: np.ndarray | None


def _is_vector(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0


def available_layers(frame: pd.DataFrame, model: str) -> list[int]:
    """Return response-pooled layers present for one model."""
    model_rows = frame[frame["model"].eq(model)]
    layers = []
    for column in model_rows:
        if not column.startswith("layer_") or "__" in column:
            continue
        if model_rows[column].map(_is_vector).any():
            layers.append(int(column.removeprefix("layer_")))
    return sorted(set(layers))


def _task_for_target(frame: pd.DataFrame, target: str) -> str:
    variable = VARIABLES_BY_NAME.get(target)
    if variable is not None:
        return "continuous" if variable.task == "continuous" else "categorical"
    numeric = pd.to_numeric(frame[target], errors="coerce")
    return (
        "continuous"
        if numeric.notna().sum() == frame[target].notna().sum()
        else "categorical"
    )


def _baseline_design(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Build the transcript/state baseline without activation information."""
    categorical = [
        column
        for column in ("role", "topic_id", "condition", "speaker")
        if column in frame
    ]
    numeric_candidates = (
        "conversation_turn_pct",
        "agent_turn",
        "prior_stance_score",
        "prior_stance_confidence",
        "prior_stance_gap",
        "prior_local_agreement",
        "prior_remaining_disagreement",
        "prior_affiliation",
        "prior_adversariality",
        "prior_closure_evidence",
        "prior_basin_leaning",
    )
    numeric = [
        column
        for column in numeric_candidates
        if column in frame and column != target
    ]
    numeric.extend(
        column
        for column in frame
        if column.startswith("prior_")
        and column.endswith("_trailing3")
        and column != target
        and column not in numeric
    )
    pieces: list[np.ndarray] = []
    if categorical:
        pieces.append(
            pd.get_dummies(
                frame[categorical], dummy_na=True, dtype=float
            ).to_numpy()
        )
    if numeric:
        pieces.append(
            frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
    if (
        "response_text_embedding" in frame
        and frame["response_text_embedding"].map(_is_vector).all()
    ):
        pieces.append(np.stack(frame["response_text_embedding"]))
    return np.concatenate(pieces, axis=1) if pieces else np.ones((len(frame), 1))


def add_response_text_embeddings(
    frame: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add complete-response embeddings, caching unique texts when requested."""
    from src.common.embeddings import embed_texts

    out = frame.copy()
    texts = out["text"].fillna("").astype(str)
    if "text_sha256" in out:
        hashes = out["text_sha256"].fillna("").astype(str)
        missing_hash = hashes.eq("")
        hashes.loc[missing_hash] = texts.loc[missing_hash].map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
    else:
        hashes = texts.map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
    cached: dict[str, np.ndarray] = {}
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            if str(payload["model_name"].item()) == model_name:
                cached = {
                    str(key): np.asarray(value)
                    for key, value in zip(
                        payload["text_sha256"], payload["embeddings"]
                    )
                }
    unique = (
        pd.DataFrame({"hash": hashes, "text": texts})
        .drop_duplicates("hash")
        .reset_index(drop=True)
    )
    missing = unique[~unique["hash"].isin(cached)]
    if not missing.empty:
        values = embed_texts(missing["text"].tolist(), model_name)
        cached.update(
            {
                str(key): np.asarray(value, dtype=np.float32)
                for key, value in zip(missing["hash"], values)
            }
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            ordered = sorted(cached)
            temporary = cache.with_name(
                f"{cache.name}.{os.getpid()}.tmp.npz"
            )
            np.savez_compressed(
                temporary,
                model_name=np.asarray(model_name),
                text_sha256=np.asarray(ordered),
                embeddings=np.stack([cached[key] for key in ordered]),
            )
            os.replace(temporary, cache)
    out["response_text_embedding"] = [
        cached[key] for key in hashes.astype(str)
    ]
    return out


def default_embedding_cache_path(
    run_dir: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return Path(run_dir) / "derived" / f"response_embeddings__{slug}.npz"


def prepare_design(
    frame: pd.DataFrame,
    model: str,
    target: str,
    layer: int,
    turn_range: str,
    group_column: str,
) -> PreparedDesign | None:
    activation_column = f"layer_{layer}"
    if target not in frame or activation_column not in frame:
        return None
    mask = (
        frame["model"].eq(model)
        & frame["turn_range"].eq(turn_range)
        & frame[target].notna()
        & frame[activation_column].map(_is_vector)
    )
    selected = frame.loc[mask].copy()
    if len(selected) < 8 or selected[group_column].nunique() < 2:
        return None
    task = _task_for_target(selected, target)
    labels = None
    if task == "continuous":
        numeric = pd.to_numeric(selected[target], errors="coerce")
        selected = selected.loc[numeric.notna()].copy()
        target_values = pd.to_numeric(selected[target], errors="raise").to_numpy(
            float
        )
        if np.unique(target_values).size < 2:
            return None
        raw_target = target_values.astype(object)
    else:
        codes, labels = pd.factorize(selected[target].astype(str), sort=True)
        if len(labels) < 2:
            return None
        target_values = codes.astype(int)
        raw_target = selected[target].astype(str).to_numpy(object)
    return PreparedDesign(
        baseline=_baseline_design(selected, target),
        activation=np.stack(selected[activation_column].to_numpy()),
        target=target_values,
        raw_target=raw_target,
        groups=selected[group_column].to_numpy(),
        frame=selected,
        task=task,
        class_labels=labels,
    )


def _fold_impute(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(train)
    counts = finite.sum(axis=0)
    means = np.divide(
        np.where(finite, train, 0.0).sum(axis=0),
        counts,
        out=np.zeros(train.shape[1], dtype=float),
        where=counts > 0,
    )
    return (
        np.where(np.isfinite(train), train, means),
        np.where(np.isfinite(test), test, means),
    )


def _scale_fold(
    values: np.ndarray, train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_values, test_values = _fold_impute(values[train], values[test])
    scaler = StandardScaler()
    return scaler.fit_transform(train_values), scaler.transform(test_values)


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(prediction)
    if valid.sum() < 2:
        return np.nan
    denominator = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    if denominator <= 0:
        return np.nan
    return float(
        1.0 - np.sum((y[valid] - prediction[valid]) ** 2) / denominator
    )


def _correlation(y: np.ndarray, prediction: np.ndarray, kind: str) -> float:
    valid = np.isfinite(prediction) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(y[valid]).size < 2:
        return np.nan
    function = pearsonr if kind == "pearson" else spearmanr
    return float(function(y[valid], prediction[valid]).statistic)


def _balanced_accuracy(y: np.ndarray, prediction: np.ndarray) -> float:
    """Balanced accuracy without sklearn's warning-producing edge cases."""
    valid = np.isfinite(prediction)
    if not valid.any():
        return np.nan
    observed = y[valid]
    predicted = prediction[valid].astype(int)
    labels = np.unique(observed)
    if labels.size < 2:
        return np.nan
    recalls = [
        float(np.mean(predicted[observed == label] == label))
        for label in labels
    ]
    return float(np.mean(recalls))


def _score(y: np.ndarray, prediction: np.ndarray, task: str) -> float:
    return (
        _r2(y, prediction)
        if task == "continuous"
        else _balanced_accuracy(y, prediction)
    )


def _ridge_predict_many(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    parameters: Sequence[float],
) -> dict[float, np.ndarray]:
    """Predict all ridge penalties from one dual eigendecomposition.

    This is algebraically equivalent to independent Ridge fits with an
    intercept, while avoiding one matrix factorization per alpha.
    """
    x_mean = np.mean(train_x, axis=0)
    y_mean = float(np.mean(train_y))
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    gram = centered_x @ centered_x.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ centered_y
    test_kernel = (test_x - x_mean) @ centered_x.T @ eigenvectors
    return {
        float(parameter): (
            y_mean
            + test_kernel
            @ (projected_y / (eigenvalues + float(parameter)))
        )
        for parameter in parameters
    }

def _row_space_projection(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project features onto the exact training row space."""
    gram = train_x @ train_x.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    tolerance = max(train_x.shape) * np.finfo(float).eps * max(largest, 1.0)
    keep = eigenvalues > tolerance
    if not np.any(keep):
        return np.zeros((len(train_x), 1)), np.zeros((len(test_x), 1))
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    roots = np.sqrt(values)
    projected_train = vectors * roots
    projected_test = (test_x @ train_x.T @ vectors) / roots
    return projected_train, projected_test


def _logistic_predict_many(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    parameters: Sequence[float],
) -> dict[float, np.ndarray]:
    """Fit logistic penalties in the exact low-dimensional row space."""
    if np.unique(train_y).size < 2:
        return {
            float(parameter): np.full(len(test_x), np.nan)
            for parameter in parameters
        }
    projected_train, projected_test = _row_space_projection(train_x, test_x)
    estimator = LogisticRegression(
        C=float(parameters[0]),
        max_iter=3000,
        class_weight="balanced",
        warm_start=True,
    )
    predictions: dict[float, np.ndarray] = {}
    for parameter in parameters:
        estimator.set_params(C=float(parameter))
        estimator.fit(projected_train, train_y)
        predictions[float(parameter)] = estimator.predict(projected_test)
    return predictions

def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    task: str,
    parameter: float,
) -> np.ndarray:
    if task == "continuous":
        return _ridge_predict_many(
            train_x, train_y, test_x, (parameter,)
        )[parameter]
    return _logistic_predict_many(
        train_x, train_y, test_x, (parameter,)
    )[parameter]

def _inner_parameter(
    design: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    task: str,
) -> float:
    candidates = RIDGE_ALPHAS if task == "continuous" else LOGISTIC_CS
    if np.unique(groups).size < 2:
        return candidates[len(candidates) // 2]
    predictions = {
        parameter: np.full(len(target), np.nan) for parameter in candidates
    }
    for train, test in LeaveOneGroupOut().split(design, target, groups):
        if task == "categorical" and np.unique(target[train]).size < 2:
            continue
        train_x, test_x = _scale_fold(design, train, test)
        if task == "continuous":
            fold_predictions = _ridge_predict_many(
                train_x, target[train], test_x, candidates
            )
            for parameter in candidates:
                predictions[parameter][test] = fold_predictions[parameter]
        else:
            fold_predictions = _logistic_predict_many(
                train_x, target[train], test_x, candidates
            )
            for parameter in candidates:
                predictions[parameter][test] = fold_predictions[parameter]
    scored = [
        (_score(target, predictions[parameter], task), parameter)
        for parameter in candidates
    ]
    finite = [item for item in scored if np.isfinite(item[0])]
    if not finite:
        return candidates[len(candidates) // 2]
    return max(finite, key=lambda item: (item[0], -item[1]))[1]

def nested_group_predictions(
    prepared: PreparedDesign,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Generate outer held-topic predictions with inner topic-wise tuning."""
    predictions = {
        name: np.full(len(prepared.target), np.nan)
        for name in ("null", "baseline", "activation", "combined")
    }
    fold_rows: list[dict] = []
    outer_splits = list(
        LeaveOneGroupOut().split(
            prepared.activation, prepared.target, prepared.groups
        )
    )
    for outer_index, (train, test) in enumerate(outer_splits, start=1):
        train_y = prepared.target[train]
        if prepared.task == "categorical" and np.unique(train_y).size < 2:
            if progress is not None:
                progress(outer_index, len(outer_splits))
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
        inner_groups = prepared.groups[train]
        designs = {
            "baseline": (prepared.baseline[train], base_train, base_test),
            "activation": (prepared.activation[train], act_train, act_test),
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
        for name, (raw_inner, train_x, test_x) in designs.items():
            parameter = _inner_parameter(
                raw_inner, train_y, inner_groups, prepared.task
            )
            parameters[name] = parameter
            predictions[name][test] = _fit_predict(
                train_x, train_y, test_x, prepared.task, parameter
            )
        fold_rows.append(
            {
                "held_out_group": str(prepared.groups[test][0]),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "baseline_parameter": parameters["baseline"],
                "activation_parameter": parameters["activation"],
                "combined_parameter": parameters["combined"],
                "null_score": _score(
                    prepared.target[test],
                    predictions["null"][test],
                    prepared.task,
                ),
                "baseline_score": _score(
                    prepared.target[test],
                    predictions["baseline"][test],
                    prepared.task,
                ),
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
        if progress is not None:
            progress(outer_index, len(outer_splits))
    return predictions, fold_rows


def _score_row(
    prepared: PreparedDesign, predictions: dict[str, np.ndarray]
) -> dict:
    y = prepared.target
    baseline = _score(y, predictions["baseline"], prepared.task)
    activation = _score(y, predictions["activation"], prepared.task)
    combined = _score(y, predictions["combined"], prepared.task)
    row = {
        "metric": (
            "r2" if prepared.task == "continuous" else "balanced_accuracy"
        ),
        "null_score": _score(y, predictions["null"], prepared.task),
        "baseline_score": baseline,
        "activation_only_score": activation,
        "activation_plus_baseline_score": combined,
        "incremental_score": combined - baseline,
    }
    if prepared.task == "continuous":
        row.update(
            {
                "baseline_pearson": _correlation(
                    y, predictions["baseline"], "pearson"
                ),
                "activation_only_pearson": _correlation(
                    y, predictions["activation"], "pearson"
                ),
                "combined_pearson": _correlation(
                    y, predictions["combined"], "pearson"
                ),
                "incremental_pearson": _correlation(
                    y - predictions["baseline"],
                    predictions["combined"] - predictions["baseline"],
                    "pearson",
                ),
                "baseline_spearman": _correlation(
                    y, predictions["baseline"], "spearman"
                ),
                "activation_only_spearman": _correlation(
                    y, predictions["activation"], "spearman"
                ),
                "combined_spearman": _correlation(
                    y, predictions["combined"], "spearman"
                ),
            }
        )
    return row


def summarize_peak_layers(scores: pd.DataFrame) -> pd.DataFrame:
    """Select the maximum held-topic activation result per final table cell."""
    if scores.empty:
        return pd.DataFrame()
    ranked = scores.copy()
    continuous_statistic = ranked.get(
        "activation_only_pearson",
        pd.Series(np.nan, index=ranked.index, dtype=float),
    )
    categorical_statistic = ranked.get(
        "activation_only_score",
        pd.Series(np.nan, index=ranked.index, dtype=float),
    )
    ranked["peak_statistic"] = np.where(
        ranked["task"].eq("continuous"),
        continuous_statistic,
        categorical_statistic,
    )
    ranked = ranked.loc[ranked["peak_statistic"].notna()]
    if ranked.empty:
        return pd.DataFrame()
    keys = ["model", "target", "task", "turn_range"]
    positions = ranked.groupby(keys, sort=False)["peak_statistic"].idxmax()
    peaks = ranked.loc[positions].copy().rename(
        columns={
            "layer": "peak_layer",
            "peak_statistic": "max_activation_correlation_or_score",
        }
    )
    peaks["metric"] = np.where(
        peaks["task"].eq("continuous"), "pearson", "balanced_accuracy"
    )
    return peaks[
        [
            *keys,
            "metric",
            "peak_layer",
            "max_activation_correlation_or_score",
            "n",
            "n_conversations",
            "n_topics",
            "activation_pooling",
            "cv_group",
        ]
    ].sort_values(keys).reset_index(drop=True)


def _evaluate_e1_cell(
    frame: pd.DataFrame,
    model: str,
    target: str,
    layer: int,
    turn_range: str,
    group_column: str,
    fold_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Evaluate one independent E1 model/range/target/layer cell."""
    prepared = prepare_design(
        frame, model, target, layer, turn_range, group_column
    )
    if prepared is None:
        return [], [], [], [{
            "model": model,
            "target": target,
            "turn_range": turn_range,
            "layer": layer,
            "reason": "insufficient rows, topics, classes, or variance",
        }]
    predictions, folds = nested_group_predictions(
        prepared, progress=fold_progress
    )
    identity = {
        "experiment": "E1",
        "model": model,
        "target": target,
        "task": prepared.task,
        "turn_range": turn_range,
        "layer": layer,
        "n": len(prepared.frame),
        "n_conversations": prepared.frame["conv_id"].nunique(),
        "n_topics": prepared.frame["topic_id"].nunique(),
        "activation_pooling": "generated_response_token_mean",
        "cv_group": group_column,
    }
    score_rows = [{**identity, **_score_row(prepared, predictions)}]
    fold_rows = [{**identity, **fold} for fold in folds]
    oof_rows: list[dict] = []
    for position, (_, observation) in enumerate(prepared.frame.iterrows()):
        row = {
            **identity,
            "conv_id": observation["conv_id"],
            "turn": int(observation["turn"]),
            "conversation_turn_pct": observation["conversation_turn_pct"],
            "agent_turn": observation.get("agent_turn"),
            "speaker": observation["speaker"],
            "topic_id": observation["topic_id"],
            "role": observation.get("role"),
            "condition": observation.get("condition"),
            "observed_target": prepared.raw_target[position],
        }
        for name, values in predictions.items():
            value = values[position]
            if prepared.class_labels is not None and np.isfinite(value):
                row[f"{name}_prediction"] = prepared.class_labels[int(value)]
            else:
                row[f"{name}_prediction"] = value
        oof_rows.append(row)
    return score_rows, fold_rows, oof_rows, []


def run_e1(
    frame: pd.DataFrame,
    targets: Sequence[str] = Q1_STATE_TARGETS,
    models: Sequence[str] | None = None,
    turn_ranges: Sequence[str] | None = None,
    group_column: str = "topic_id",
    progress: Callable[..., None] | None = None,
    progress_stage: str = "e1",
    n_jobs: int = 1,
    blas_threads_per_job: int = 2,
    checkpoint_dir: str | Path | None = None,
    checkpoint_key: str = "",
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run E1 with bounded cell parallelism and granular progress."""
    required = {"model", "turn_range", group_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"E1 dataset is missing columns: {sorted(missing)}")
    chosen_models = (
        list(models)
        if models is not None
        else sorted(frame.loc[frame["model"].notna(), "model"].unique())
    )
    chosen_ranges = (
        list(turn_ranges)
        if turn_ranges is not None
        else list(dict.fromkeys(frame["turn_range"].dropna().astype(str)))
    )
    model_layers = {
        model: available_layers(frame, model) for model in chosen_models
    }
    total_cells = sum(
        len(model_layers[model]) * len(chosen_ranges) * len(targets)
        for model in chosen_models
    )
    tasks: list[tuple[str, str, str, int]] = []
    skipped: list[dict] = []
    completed_cells = 0
    for model in chosen_models:
        layers = model_layers[model]
        if not layers:
            skipped.append({
                "model": model,
                "target": None,
                "turn_range": None,
                "layer": None,
                "reason": "no response-pooled activation arrays",
            })
            continue
        for turn_range in chosen_ranges:
            for target in targets:
                if target not in frame:
                    skipped.append({
                        "model": model,
                        "target": target,
                        "turn_range": turn_range,
                        "layer": None,
                        "reason": "variable unavailable",
                    })
                    completed_cells += len(layers)
                    if progress is not None:
                        progress(
                            progress_stage, completed_cells, total_cells,
                            model=model, turn_range=turn_range,
                            target=target, layer=None, status="skipped",
                        )
                    continue
                tasks.extend(
                    (model, turn_range, target, layer) for layer in layers
                )

    def evaluate(task: tuple[str, str, str, int]):
        checkpoint = None
        if checkpoint_dir is not None:
            digest = hashlib.sha256(
                repr((E1_CHECKPOINT_VERSION, checkpoint_key, task)).encode()
            ).hexdigest()[:24]
            checkpoint = Path(checkpoint_dir) / f"e1_cell__{digest}.pkl"
            if resume and checkpoint.is_file():
                payload = pd.read_pickle(checkpoint)
                if (
                    payload.get("version") == E1_CHECKPOINT_VERSION
                    and payload.get("checkpoint_key") == checkpoint_key
                    and tuple(payload.get("task", ())) == task
                ):
                    if progress is not None:
                        progress(
                            "e1_checkpoint", model=task[0],
                            turn_range=task[1], target=task[2],
                            layer=task[3], status="resumed",
                        )
                    return payload["result"]
        model, turn_range, target, layer = task
        fold_callback = None
        if progress is not None:
            cell_stage = (
                f"{progress_stage}_folds:{model}:{turn_range}:"
                f"{target}:layer-{layer}"
            )
            fold_callback = lambda done, total: progress(
                cell_stage, done, total, model=model,
                turn_range=turn_range, target=target, layer=layer,
            )
        result = _evaluate_e1_cell(
            frame, model, target, layer, turn_range, group_column,
            fold_progress=fold_callback,
        )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".tmp")
            pd.to_pickle(
                {
                    "version": E1_CHECKPOINT_VERSION,
                    "checkpoint_key": checkpoint_key,
                    "task": task,
                    "result": result,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
            if progress is not None:
                progress(
                    "e1_checkpoint", model=model,
                    turn_range=turn_range, target=target,
                    layer=layer, status="saved",
                )
        return result

    workers = max(1, min(int(n_jobs), len(tasks) or 1))
    threads = max(1, int(blas_threads_per_job))
    ordered_results: dict[int, tuple[list[dict], list[dict], list[dict], list[dict]]] = {}
    with threadpool_limits(limits=threads):
        if workers == 1:
            for index, task in enumerate(tasks):
                ordered_results[index] = evaluate(task)
                completed_cells += 1
                if progress is not None:
                    model, turn_range, target, layer = task
                    status = "skipped" if ordered_results[index][3] else "complete"
                    progress(
                        progress_stage, completed_cells, total_cells,
                        model=model, turn_range=turn_range,
                        target=target, layer=layer, status=status,
                    )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluate, task): (index, task)
                    for index, task in enumerate(tasks)
                }
                for future in as_completed(futures):
                    index, task = futures[future]
                    ordered_results[index] = future.result()
                    completed_cells += 1
                    if progress is not None:
                        model, turn_range, target, layer = task
                        status = "skipped" if ordered_results[index][3] else "complete"
                        progress(
                            progress_stage, completed_cells, total_cells,
                            model=model, turn_range=turn_range,
                            target=target, layer=layer, status=status,
                        )
    score_rows: list[dict] = []
    fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    for index in sorted(ordered_results):
        scores, folds, oof, cell_skips = ordered_results[index]
        score_rows.extend(scores)
        fold_rows.extend(folds)
        oof_rows.extend(oof)
        skipped.extend(cell_skips)
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(oof_rows),
        pd.DataFrame(skipped),
    )
