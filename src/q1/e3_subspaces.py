"""E3: low-rank organization of conversational-variable activation subspaces.

For every model × variable family × percentage turn range × layer, E3:

1. residualizes the multivariate target against the E1 metadata/state/text
   baseline inside each held-topic fold;
2. fits ridge probes from activations to those residuals;
3. projects probe predictions through candidate reduced ranks;
4. selects the smallest rank reaching 90% of the best positive held-topic
   mean target correlation;
5. estimates descriptive principal-angle overlap between selected family
   subspaces; and
6. evaluates fixed-rank cross-turn transfer with held-out topics.

Raw activation spaces are never combined across models.
"""

from __future__ import annotations

import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.q1.e1_layerwise import _baseline_design, _fold_impute, _is_vector


E3_VARIABLE_FAMILIES: dict[str, tuple[str, ...]] = {
    "stance": (
        "stance_score",
        "stance_gap",
    ),
    "agreement_conflict": (
        "local_agreement",
        "remaining_disagreement",
        "affiliation",
        "adversariality",
        "observed_alignment_index",
        "observed_conflict_index",
        "observed_accommodation_index",
    ),
    "personality": (
        "perceived_persona_warmth_trailing3",
        "perceived_persona_dominance_trailing3",
        "perceived_persona_curiosity_trailing3",
        "perceived_persona_structure_trailing3",
        "perceived_persona_stability_trailing3",
        "perceived_persona_deference_trailing3",
        "perceived_persona_humility_trailing3",
    ),
    "expressed_vad": (
        "expressed_valence",
        "expressed_arousal",
        "expressed_dominance",
    ),
}

E3_RIDGE_ALPHAS = (1.0, 10.0, 100.0)
E3_DEFAULT_RANKS = (1, 2, 4, 8)
E3_BASELINE_ALPHA = 10.0
E3_TRANSFER_ALPHA = 10.0


@dataclass(frozen=True)
class PreparedFamily:
    baseline: np.ndarray
    activation: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    frame: pd.DataFrame
    target_names: tuple[str, ...]


@dataclass(frozen=True)
class FoldPrediction:
    observed_residual: np.ndarray
    predicted_residual: np.ndarray
    activation_basis: np.ndarray
    effective_rank: int


@dataclass(frozen=True)
class E3Results:
    rank_scores: pd.DataFrame
    fold_scores: pd.DataFrame
    target_scores: pd.DataFrame
    rank_selection: pd.DataFrame
    overlap: pd.DataFrame
    cross_turn: pd.DataFrame
    subspace_manifest: pd.DataFrame
    skipped: pd.DataFrame
    bases: dict[str, np.ndarray]


def available_layers(frame: pd.DataFrame, model: str) -> list[int]:
    rows = frame[frame["model"].eq(model)]
    layers = []
    for column in rows:
        if not column.startswith("layer_") or "__" in column:
            continue
        if rows[column].map(_is_vector).any():
            layers.append(int(column.removeprefix("layer_")))
    return sorted(set(layers))


def _available_family_targets(
    frame: pd.DataFrame,
    requested: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        target
        for target in requested
        if target in frame
        and pd.to_numeric(frame[target], errors="coerce").notna().any()
    )


def prepare_family(
    frame: pd.DataFrame,
    model: str,
    family_targets: Sequence[str],
    layer: int,
    turn_range: str,
    group_column: str = "topic_id",
) -> PreparedFamily | None:
    activation_column = f"layer_{layer}"
    targets = _available_family_targets(frame, family_targets)
    if len(targets) < 2 or activation_column not in frame:
        return None
    mask = (
        frame["model"].eq(model)
        & frame["turn_range"].eq(turn_range)
        & frame[activation_column].map(_is_vector)
    )
    numeric_targets = frame.loc[mask, list(targets)].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = numeric_targets.notna().all(axis=1)
    selected = frame.loc[numeric_targets.index[complete]].copy()
    if (
        len(selected) < 12
        or selected[group_column].nunique() < 3
    ):
        return None
    target = selected[list(targets)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    varying = np.nanstd(target, axis=0) > 0
    targets = tuple(np.asarray(targets)[varying])
    target = target[:, varying]
    if len(targets) < 2:
        return None
    return PreparedFamily(
        baseline=_baseline_design(selected, "__e3_family__"),
        activation=np.stack(selected[activation_column].to_numpy()),
        target=target,
        groups=selected[group_column].to_numpy(),
        frame=selected,
        target_names=targets,
    )


def candidate_ranks(
    requested: Sequence[int],
    target_count: int,
) -> tuple[int, ...]:
    ranks = {
        int(rank)
        for rank in requested
        if int(rank) > 0 and int(rank) <= target_count
    }
    ranks.add(int(target_count))
    return tuple(sorted(ranks))


def _scale_matrix(
    values: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_values, test_values = _fold_impute(values[train], values[test])
    scaler = StandardScaler()
    return scaler.fit_transform(train_values), scaler.transform(test_values)


def _standardize_targets(
    values: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = values[train].mean(axis=0)
    scale = values[train].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values[train] - mean) / scale, (values[test] - mean) / scale


@dataclass(frozen=True)
class FoldData:
    observed_residual: np.ndarray
    activation_train: np.ndarray
    activation_test: np.ndarray
    train_residual: np.ndarray


@dataclass(frozen=True)
class FoldCore:
    observed_residual: np.ndarray
    raw_prediction: np.ndarray
    prediction_center: np.ndarray
    right_vectors: np.ndarray
    coefficient: np.ndarray


def _prepare_fold_data(
    prepared: PreparedFamily, train: np.ndarray, test: np.ndarray
) -> FoldData:
    baseline_train, baseline_test = _scale_matrix(
        prepared.baseline, train, test
    )
    activation_train, activation_test = _scale_matrix(
        prepared.activation, train, test
    )
    target_train, target_test = _standardize_targets(
        prepared.target, train, test
    )
    baseline_model = Ridge(alpha=E3_BASELINE_ALPHA).fit(
        baseline_train, target_train
    )
    train_residual = target_train - baseline_model.predict(baseline_train)
    return FoldData(
        observed_residual=target_test - baseline_model.predict(baseline_test),
        activation_train=activation_train,
        activation_test=activation_test,
        train_residual=train_residual,
    )


def _fit_fold_core(data: FoldData, activation_alpha: float) -> FoldCore:
    activation_model = Ridge(alpha=activation_alpha).fit(
        data.activation_train, data.train_residual
    )
    train_prediction = activation_model.predict(data.activation_train)
    raw_prediction = activation_model.predict(data.activation_test)
    center = train_prediction.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(
        train_prediction - center, full_matrices=False
    )
    return FoldCore(
        observed_residual=data.observed_residual,
        raw_prediction=raw_prediction,
        prediction_center=center,
        right_vectors=right,
        coefficient=activation_model.coef_.T,
    )


def _project_fold(core: FoldCore, rank: int) -> FoldPrediction:
    effective_rank = min(
        int(rank), core.right_vectors.shape[0], core.raw_prediction.shape[1]
    )
    target_basis = core.right_vectors[:effective_rank].T
    reduced = core.prediction_center + (
        (core.raw_prediction - core.prediction_center)
        @ target_basis
        @ target_basis.T
    )
    raw_basis = core.coefficient @ target_basis
    if raw_basis.size:
        activation_basis, _ = np.linalg.qr(raw_basis)
        activation_basis = activation_basis[:, :effective_rank]
    else:
        activation_basis = np.empty((core.coefficient.shape[0], 0))
    return FoldPrediction(
        observed_residual=core.observed_residual,
        predicted_residual=reduced,
        activation_basis=activation_basis,
        effective_rank=effective_rank,
    )


def _fit_fold(
    prepared: PreparedFamily,
    train: np.ndarray,
    test: np.ndarray,
    rank: int,
    activation_alpha: float,
    baseline_alpha: float = E3_BASELINE_ALPHA,
) -> FoldPrediction:
    if baseline_alpha != E3_BASELINE_ALPHA:
        raise ValueError("E3 baseline alpha must match E3_BASELINE_ALPHA")
    return _project_fold(
        _fit_fold_core(_prepare_fold_data(prepared, train, test), activation_alpha),
        rank,
    )


def _target_pearsons(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    correlations = np.full(observed.shape[1], np.nan)
    for target in range(observed.shape[1]):
        x, y = observed[:, target], predicted[:, target]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3 and np.unique(x[valid]).size >= 2:
            correlations[target] = float(pearsonr(x[valid], y[valid]).statistic)
    return correlations


def _target_r2(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    values = np.full(observed.shape[1], np.nan)
    for target in range(observed.shape[1]):
        x, y = observed[:, target], predicted[:, target]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        denominator = np.sum((x[valid] - x[valid].mean()) ** 2)
        if denominator > 0:
            values[target] = 1.0 - np.sum((x[valid] - y[valid]) ** 2) / denominator
    return values


def _mean_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _median_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _inner_alphas(
    prepared: PreparedFamily,
    outer_train: np.ndarray,
    ranks: Sequence[int],
    alphas: Sequence[float],
) -> dict[int, float]:
    local_groups = prepared.groups[outer_train]
    if np.unique(local_groups).size < 3:
        middle = float(alphas[len(alphas) // 2])
        return {rank: middle for rank in ranks}
    observed = np.full(
        (len(outer_train), prepared.target.shape[1]), np.nan
    )
    predicted = {
        (rank, float(alpha)): np.full_like(observed, np.nan)
        for rank in ranks for alpha in alphas
    }
    for inner_train_local, inner_test_local in LeaveOneGroupOut().split(
        outer_train, groups=local_groups
    ):
        inner_train = outer_train[inner_train_local]
        inner_test = outer_train[inner_test_local]
        data = _prepare_fold_data(prepared, inner_train, inner_test)
        observed[inner_test_local] = data.observed_residual
        for alpha in alphas:
            core = _fit_fold_core(data, float(alpha))
            for rank in ranks:
                predicted[(rank, float(alpha))][inner_test_local] = (
                    _project_fold(core, rank).predicted_residual
                )
    selected = {}
    for rank in ranks:
        scored = [
            (
                _mean_finite(
                    _target_pearsons(observed, predicted[(rank, float(alpha))])
                ),
                float(alpha),
            )
            for alpha in alphas
        ]
        finite = [item for item in scored if np.isfinite(item[0])]
        selected[rank] = (
            max(finite, key=lambda item: (item[0], -item[1]))[1]
            if finite else float(alphas[len(alphas) // 2])
        )
    return selected


def evaluate_ranks(
    prepared: PreparedFamily,
    ranks: Sequence[int],
    alphas: Sequence[float] = E3_RIDGE_ALPHAS,
) -> dict[int, tuple[dict, list[dict], list[dict], np.ndarray, np.ndarray]]:
    chosen = tuple(dict.fromkeys(int(rank) for rank in ranks))
    observed = {
        rank: np.full_like(prepared.target, np.nan, dtype=float)
        for rank in chosen
    }
    predicted = {rank: np.full_like(value, np.nan) for rank, value in observed.items()}
    fold_rows = {rank: [] for rank in chosen}
    for train, test in LeaveOneGroupOut().split(
        prepared.activation, groups=prepared.groups
    ):
        selected = _inner_alphas(prepared, train, chosen, alphas)
        data = _prepare_fold_data(prepared, train, test)
        cores = {
            alpha: _fit_fold_core(data, alpha)
            for alpha in set(selected.values())
        }
        for rank in chosen:
            result = _project_fold(cores[selected[rank]], rank)
            observed[rank][test] = result.observed_residual
            predicted[rank][test] = result.predicted_residual
            correlations = _target_pearsons(
                result.observed_residual, result.predicted_residual
            )
            fold_rows[rank].append(
                {
                    "held_out_group": str(prepared.groups[test][0]),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "selected_alpha": selected[rank],
                    "effective_rank": result.effective_rank,
                    "mean_target_pearson": _mean_finite(correlations),
                    "mean_target_r2": _mean_finite(
                        _target_r2(result.observed_residual, result.predicted_residual)
                    ),
                }
            )
    results = {}
    for rank in chosen:
        correlations = _target_pearsons(observed[rank], predicted[rank])
        r2_values = _target_r2(observed[rank], predicted[rank])
        target_rows = [
            {"target": target, "pearson": correlations[index], "r2": r2_values[index]}
            for index, target in enumerate(prepared.target_names)
        ]
        summary = {
            "mean_target_pearson": _mean_finite(correlations),
            "median_target_pearson": _median_finite(correlations),
            "mean_target_r2": _mean_finite(r2_values),
            "n_targets": len(prepared.target_names),
            "targets": "|".join(prepared.target_names),
        }
        results[rank] = (
            summary, fold_rows[rank], target_rows,
            observed[rank], predicted[rank],
        )
    return results


def evaluate_rank(
    prepared: PreparedFamily,
    rank: int,
    alphas: Sequence[float] = E3_RIDGE_ALPHAS,
) -> tuple[dict, list[dict], list[dict], np.ndarray, np.ndarray]:
    return evaluate_ranks(prepared, [rank], alphas)[int(rank)]


def select_ranks(rank_scores: pd.DataFrame) -> pd.DataFrame:
    """Select smallest rank reaching 90% of best positive OOF correlation."""
    if rank_scores.empty:
        return pd.DataFrame()
    keys = ["model", "family", "turn_range", "layer"]
    rows = []
    for identity, group in rank_scores.groupby(keys, sort=False):
        group = group.sort_values("rank")
        finite = group[np.isfinite(group["mean_target_pearson"])]
        if finite.empty:
            continue
        best_row = finite.loc[finite["mean_target_pearson"].idxmax()]
        best_score = float(best_row["mean_target_pearson"])
        threshold = 0.9 * best_score if best_score > 0 else np.nan
        eligible = (
            finite[finite["mean_target_pearson"].ge(threshold)]
            if np.isfinite(threshold)
            else finite.iloc[0:0]
        )
        rank_90 = (
            int(eligible["rank"].min()) if not eligible.empty else np.nan
        )
        full_rank_row = group.loc[group["rank"].idxmax()]
        rows.append(
            {
                **dict(zip(keys, identity)),
                "n_targets": int(group["n_targets"].max()),
                "targets": group["targets"].iloc[0],
                "best_rank": int(best_row["rank"]),
                "best_mean_target_pearson": best_score,
                "rank_90pct": rank_90,
                "rank_90pct_threshold": threshold,
                "full_rank": int(full_rank_row["rank"]),
                "full_rank_mean_target_pearson": float(
                    full_rank_row["mean_target_pearson"]
                ),
                "n": int(group["n"].max()),
                "n_conversations": int(group["n_conversations"].max()),
                "n_topics": int(group["n_topics"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _fit_full_basis(
    prepared: PreparedFamily,
    rank: int,
    activation_alpha: float,
) -> np.ndarray:
    indices = np.arange(len(prepared.frame))
    baseline, _ = _scale_matrix(prepared.baseline, indices, indices)
    activation_values, _ = _fold_impute(prepared.activation, prepared.activation)
    activation_scaler = StandardScaler()
    activation = activation_scaler.fit_transform(activation_values)
    target, _ = _standardize_targets(prepared.target, indices, indices)
    baseline_model = Ridge(alpha=E3_BASELINE_ALPHA).fit(baseline, target)
    residual = target - baseline_model.predict(baseline)
    activation_model = Ridge(alpha=activation_alpha).fit(activation, residual)
    prediction = activation_model.predict(activation)
    centered = prediction - prediction.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, right.shape[0], residual.shape[1])
    target_basis = right[:effective_rank].T
    raw_coefficient = activation_model.coef_.T / np.where(
        activation_scaler.scale_ > 1e-12, activation_scaler.scale_, 1.0
    )[:, None]
    raw_basis = raw_coefficient @ target_basis
    basis, _ = np.linalg.qr(raw_basis)
    return basis[:, :effective_rank]


def _basis_key(
    model: str,
    family: str,
    turn_range: str,
    layer: int,
) -> str:
    raw = f"{model}__{family}__{turn_range}__layer_{layer}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def estimate_selected_bases(
    frame: pd.DataFrame,
    selections: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    fold_scores: pd.DataFrame,
    group_column: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict]]:
    bases = {}
    manifest = []
    skipped = []
    for row in selections.itertuples(index=False):
        if not np.isfinite(row.rank_90pct):
            skipped.append(
                {
                    "stage": "basis",
                    "model": row.model,
                    "family": row.family,
                    "turn_range": row.turn_range,
                    "layer": row.layer,
                    "reason": "no positive rank-90% selection",
                }
            )
            continue
        prepared = prepare_family(
            frame,
            row.model,
            families[row.family],
            int(row.layer),
            row.turn_range,
            group_column,
        )
        if prepared is None:
            continue
        matching_folds = fold_scores[
            fold_scores["model"].eq(row.model)
            & fold_scores["family"].eq(row.family)
            & fold_scores["turn_range"].eq(row.turn_range)
            & fold_scores["layer"].eq(row.layer)
            & fold_scores["rank"].eq(int(row.rank_90pct))
        ]
        alpha = (
            float(matching_folds["selected_alpha"].median())
            if not matching_folds.empty
            else E3_RIDGE_ALPHAS[1]
        )
        basis = _fit_full_basis(
            prepared, int(row.rank_90pct), activation_alpha=alpha
        )
        key = _basis_key(
            row.model, row.family, row.turn_range, int(row.layer)
        )
        bases[key] = basis.astype(np.float32)
        manifest.append(
            {
                "basis_key": key,
                "model": row.model,
                "family": row.family,
                "turn_range": row.turn_range,
                "layer": int(row.layer),
                "selected_rank": int(row.rank_90pct),
                "activation_alpha": alpha,
                "activation_width": int(basis.shape[0]),
                "basis_width": int(basis.shape[1]),
                "targets": "|".join(prepared.target_names),
                "basis_space": "raw_activation_coordinates",
            }
        )
    return bases, pd.DataFrame(manifest), skipped


def subspace_overlap(
    bases: Mapping[str, np.ndarray],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    rows = []
    cell_keys = ["model", "turn_range", "layer"]
    for identity, group in manifest.groupby(cell_keys, sort=False):
        for left, right in combinations(group.itertuples(index=False), 2):
            left_basis = bases[left.basis_key]
            right_basis = bases[right.basis_key]
            if left_basis.shape[0] != right_basis.shape[0]:
                continue
            angles = subspace_angles(left_basis, right_basis)
            cosine = np.cos(angles)
            rows.append(
                {
                    **dict(zip(cell_keys, identity)),
                    "family_a": left.family,
                    "family_b": right.family,
                    "rank_a": left.basis_width,
                    "rank_b": right.basis_width,
                    "n_principal_angles": len(angles),
                    "mean_angle_degrees": float(
                        np.degrees(angles).mean()
                    ),
                    "max_angle_degrees": float(
                        np.degrees(angles).max()
                    ),
                    "mean_cosine_similarity": float(cosine.mean()),
                    "max_cosine_similarity": float(cosine.max()),
                }
            )
    return pd.DataFrame(rows)


def _prepare_cross_turn(
    frame: pd.DataFrame,
    model: str,
    family_targets: Sequence[str],
    layer: int,
    source_range: str,
    destination_range: str,
    group_column: str,
) -> tuple[PreparedFamily, np.ndarray, np.ndarray] | None:
    activation_column = f"layer_{layer}"
    targets = _available_family_targets(frame, family_targets)
    if len(targets) < 2 or activation_column not in frame:
        return None
    mask = (
        frame["model"].eq(model)
        & frame["turn_range"].isin([source_range, destination_range])
        & frame[activation_column].map(_is_vector)
    )
    candidate = frame.loc[mask].copy()
    numeric = candidate[list(targets)].apply(pd.to_numeric, errors="coerce")
    candidate = candidate.loc[numeric.notna().all(axis=1)].copy()
    if candidate.empty:
        return None
    target = candidate[list(targets)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    varying = np.std(target, axis=0) > 0
    targets = tuple(np.asarray(targets)[varying])
    target = target[:, varying]
    if len(targets) < 2:
        return None
    prepared = PreparedFamily(
        baseline=_baseline_design(candidate, "__e3_family__"),
        activation=np.stack(candidate[activation_column].to_numpy()),
        target=target,
        groups=candidate[group_column].to_numpy(),
        frame=candidate,
        target_names=targets,
    )
    source = candidate["turn_range"].eq(source_range).to_numpy()
    destination = candidate["turn_range"].eq(destination_range).to_numpy()
    return prepared, source, destination


def _cross_turn_cell(
    frame: pd.DataFrame,
    model: str,
    layer: int,
    family: str,
    requested_targets: Sequence[str],
    turn_ranges: Sequence[str],
    group_column: str,
    transfer_rank: int,
    activation_alpha: float,
) -> list[dict]:
    activation_column = f"layer_{layer}"
    targets = _available_family_targets(frame, requested_targets)
    if len(targets) < 2 or activation_column not in frame:
        return []
    mask = (
        frame["model"].eq(model)
        & frame["turn_range"].isin(turn_ranges)
        & frame[activation_column].map(_is_vector)
    )
    numeric = frame.loc[mask, list(targets)].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = numeric.notna().all(axis=1)
    selected = frame.loc[numeric.index[complete]].copy()
    if len(selected) < 12 or selected[group_column].nunique() < 3:
        return []
    target = selected[list(targets)].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    varying = np.std(target, axis=0) > 0
    targets = tuple(np.asarray(targets)[varying])
    target = target[:, varying]
    if len(targets) < 2:
        return []
    prepared = PreparedFamily(
        baseline=_baseline_design(selected, "__e3_family__"),
        activation=np.stack(selected[activation_column].to_numpy()),
        target=target,
        groups=selected[group_column].to_numpy(),
        frame=selected,
        target_names=targets,
    )
    rank = min(transfer_rank, len(targets))
    pairs = {
        (source, destination): {
            "observed": np.full_like(target, np.nan),
            "predicted": np.full_like(target, np.nan),
            "topics": 0,
        }
        for source in turn_ranges for destination in turn_ranges
    }
    groups = prepared.groups
    for source in turn_ranges:
        source_mask = selected["turn_range"].eq(source).to_numpy()
        for held_out in np.unique(groups):
            train = np.flatnonzero(source_mask & (groups != held_out))
            test_all = np.flatnonzero(groups == held_out)
            if (
                len(train) < 8 or len(test_all) < 2
                or np.unique(groups[train]).size < 2
            ):
                continue
            data = _prepare_fold_data(prepared, train, test_all)
            result = _project_fold(
                _fit_fold_core(data, activation_alpha), rank
            )
            held_ranges = selected.iloc[test_all]["turn_range"].to_numpy()
            for destination in turn_ranges:
                local = held_ranges == destination
                if local.sum() < 2:
                    continue
                test = test_all[local]
                pair = pairs[(source, destination)]
                pair["observed"][test] = result.observed_residual[local]
                pair["predicted"][test] = result.predicted_residual[local]
                pair["topics"] += 1
    rows = []
    for (source, destination), pair in pairs.items():
        valid = np.isfinite(pair["predicted"]).all(axis=1)
        if valid.sum() < 3:
            continue
        correlations = _target_pearsons(
            pair["observed"][valid], pair["predicted"][valid]
        )
        rows.append(
            {
                "model": model,
                "family": family,
                "layer": int(layer),
                "source_turn_range": source,
                "destination_turn_range": destination,
                "rank": rank,
                "activation_alpha": activation_alpha,
                "mean_target_pearson": _mean_finite(correlations),
                "median_target_pearson": _median_finite(correlations),
                "n_targets": len(targets),
                "targets": "|".join(targets),
                "n_test": int(valid.sum()),
                "n_topics": int(pair["topics"]),
            }
        )
    return rows


def cross_turn_transfer(
    frame: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    models: Sequence[str],
    layers_by_model: Mapping[str, Sequence[int]],
    turn_ranges: Sequence[str],
    group_column: str = "topic_id",
    transfer_rank: int = 2,
    activation_alpha: float = E3_TRANSFER_ALPHA,
    n_jobs: int = 4,
    progress: Callable[..., None] | None = None,
) -> pd.DataFrame:
    tasks = [
        (model, int(layer), family, targets)
        for model in models
        for layer in layers_by_model.get(model, ())
        for family, targets in families.items()
    ]
    workers = max(1, min(int(n_jobs), len(tasks) or 1))
    rows: list[dict] = []
    completed = 0

    def compute(task):
        model, layer, family, targets = task
        return _cross_turn_cell(
            frame, model, layer, family, targets, turn_ranges,
            group_column, transfer_rank, activation_alpha,
        )

    if workers == 1:
        results = ((task, compute(task)) for task in tasks)
        for task, cell_rows in results:
            rows.extend(cell_rows)
            completed += 1
            if progress is not None:
                progress(
                    "e3_cross_turn", completed, len(tasks),
                    model=task[0], layer=task[1], family=task[2],
                )
    else:
        threads_per_worker = max(1, (os.cpu_count() or 1) // workers)
        with threadpool_limits(limits=threads_per_worker):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(compute, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    rows.extend(future.result())
                    completed += 1
                    if progress is not None:
                        progress(
                            "e3_cross_turn", completed, len(tasks),
                            model=task[0], layer=task[1], family=task[2],
                        )
    return pd.DataFrame(rows)


def run_e3(
    frame: pd.DataFrame,
    families: Mapping[str, Sequence[str]] = E3_VARIABLE_FAMILIES,
    ranks: Sequence[int] = E3_DEFAULT_RANKS,
    models: Sequence[str] | None = None,
    turn_ranges: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    group_column: str = "topic_id",
    alphas: Sequence[float] = E3_RIDGE_ALPHAS,
    transfer_rank: int = 2,
    run_cross_turn: bool = True,
    n_jobs: int = 4,
    progress: Callable[..., None] | None = None,
) -> E3Results:
    chosen_models = (
        list(models)
        if models is not None
        else sorted(frame["model"].dropna().astype(str).unique())
    )
    chosen_ranges = (
        list(turn_ranges)
        if turn_ranges is not None
        else list(dict.fromkeys(frame["turn_range"].dropna().astype(str)))
    )
    layers_by_model = {
        model: [
            layer
            for layer in available_layers(frame, model)
            if layers is None or layer in set(int(value) for value in layers)
        ]
        for model in chosen_models
    }
    rank_rows, fold_rows, target_rows, skipped = [], [], [], []
    tasks = []
    for model in chosen_models:
        if not layers_by_model[model]:
            skipped.append(
                {
                    "stage": "rank_curve", "model": model,
                    "family": None, "turn_range": None, "layer": None,
                    "reason": "no selected activation layers",
                }
            )
            continue
        for family, family_targets in families.items():
            if len(_available_family_targets(frame, family_targets)) < 2:
                skipped.append(
                    {
                        "stage": "rank_curve", "model": model,
                        "family": family, "turn_range": None, "layer": None,
                        "reason": "fewer than two family variables available",
                    }
                )
                continue
            tasks.extend(
                (model, family, family_targets, turn_range, int(layer))
                for turn_range in chosen_ranges
                for layer in layers_by_model[model]
            )

    def compute_rank_cell(task):
        model, family, family_targets, turn_range, layer = task
        prepared = prepare_family(
            frame, model, family_targets, layer, turn_range, group_column
        )
        if prepared is None:
            return [], [], [], {
                "stage": "rank_curve", "model": model, "family": family,
                "turn_range": turn_range, "layer": layer,
                "reason": (
                    "insufficient complete rows, topics, or varying targets"
                ),
            }
        identity = {
            "experiment": "E3", "model": model, "family": family,
            "turn_range": turn_range, "layer": layer,
            "n": len(prepared.frame),
            "n_conversations": prepared.frame["conv_id"].nunique(),
            "n_topics": prepared.frame[group_column].nunique(),
            "cv_group": group_column,
            "target_space": "baseline_residual_standardized",
            "activation_pooling": "generated_response_token_mean",
        }
        chosen_ranks = candidate_ranks(ranks, len(prepared.target_names))
        evaluated = evaluate_ranks(prepared, chosen_ranks, alphas)
        cell_ranks, cell_folds, cell_targets = [], [], []
        for rank in chosen_ranks:
            summary, folds_for_rank, targets_for_rank, _, _ = evaluated[rank]
            cell_ranks.append({**identity, "rank": rank, **summary})
            cell_folds.extend(
                {**identity, "rank": rank, **fold}
                for fold in folds_for_rank
            )
            cell_targets.extend(
                {**identity, "rank": rank, **target}
                for target in targets_for_rank
            )
        return cell_ranks, cell_folds, cell_targets, None

    workers = max(1, min(int(n_jobs), len(tasks) or 1))
    completed = 0
    if workers == 1:
        result_stream = ((task, compute_rank_cell(task)) for task in tasks)
        for task, result in result_stream:
            cell_ranks, cell_folds, cell_targets, cell_skip = result
            rank_rows.extend(cell_ranks)
            fold_rows.extend(cell_folds)
            target_rows.extend(cell_targets)
            if cell_skip is not None:
                skipped.append(cell_skip)
            completed += 1
            if progress is not None:
                progress(
                    "e3_rank_cells", completed, len(tasks),
                    model=task[0], family=task[1],
                    turn_range=task[3], layer=task[4],
                )
    else:
        threads_per_worker = max(1, (os.cpu_count() or 1) // workers)
        with threadpool_limits(limits=threads_per_worker):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(compute_rank_cell, task): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    cell_ranks, cell_folds, cell_targets, cell_skip = (
                        future.result()
                    )
                    rank_rows.extend(cell_ranks)
                    fold_rows.extend(cell_folds)
                    target_rows.extend(cell_targets)
                    if cell_skip is not None:
                        skipped.append(cell_skip)
                    completed += 1
                    if progress is not None:
                        progress(
                            "e3_rank_cells", completed, len(tasks),
                            model=task[0], family=task[1],
                            turn_range=task[3], layer=task[4],
                        )
    rank_scores = pd.DataFrame(rank_rows)
    folds = pd.DataFrame(fold_rows)
    target_scores = pd.DataFrame(target_rows)
    selections = select_ranks(rank_scores)
    if progress is not None:
        progress("e3_bases", 0, len(selections), status="started")
    bases, manifest, basis_skips = estimate_selected_bases(
        frame,
        selections,
        families,
        folds,
        group_column,
    )
    skipped.extend(basis_skips)
    if progress is not None:
        progress(
            "e3_bases", len(selections), len(selections), status="complete"
        )
    overlap = subspace_overlap(bases, manifest)
    cross_turn = (
        cross_turn_transfer(
            frame,
            families,
            chosen_models,
            layers_by_model,
            chosen_ranges,
            group_column=group_column,
            transfer_rank=transfer_rank,
            n_jobs=n_jobs,
            progress=progress,
        )
        if run_cross_turn
        else pd.DataFrame()
    )
    return E3Results(
        rank_scores=rank_scores,
        fold_scores=folds,
        target_scores=target_scores,
        rank_selection=selections,
        overlap=overlap,
        cross_turn=cross_turn,
        subspace_manifest=manifest,
        skipped=pd.DataFrame(
            skipped,
            columns=[
                "stage",
                "model",
                "family",
                "turn_range",
                "layer",
                "reason",
            ],
        ),
        bases=bases,
    )


def save_e3_results(results: E3Results, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.rank_scores.to_csv(output / "e3_rank_scores.csv", index=False)
    results.fold_scores.to_csv(output / "e3_fold_scores.csv", index=False)
    results.target_scores.to_csv(
        output / "e3_target_scores.csv", index=False
    )
    results.rank_selection.to_csv(
        output / "e3_rank_selection.csv", index=False
    )
    results.overlap.to_csv(output / "e3_subspace_overlap.csv", index=False)
    results.cross_turn.to_csv(
        output / "e3_cross_turn_transfer.csv", index=False
    )
    results.subspace_manifest.to_csv(
        output / "e3_subspace_manifest.csv", index=False
    )
    results.skipped.to_csv(output / "e3_skipped.csv", index=False)
    np.savez_compressed(
        output / "e3_subspace_bases.npz",
        **results.bases,
    )

