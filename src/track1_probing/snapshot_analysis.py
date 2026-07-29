"""Snapshot-resolved, time-aware analyses for refactored Track 1."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.track1_probing.cache_activations import get_snapshots
from src.track1_probing.variables import OBSERVER_BIG_FIVE_FIELDS, VARIABLES

PRIMARY_SNAPSHOTS = ("pre_generation", "early_response", "full_response", "final_window")
TIME_MATCHED_EMBEDDINGS = {
    "pre_generation": ("prompt_embedding", "transcript_context_embedding"),
    "early_response": ("prompt_embedding", "early_response_embedding"),
    "full_response": ("response_embedding",),
    "final_window": ("response_embedding",),
    "final_token": ("response_embedding",),
}
VARIABLES_BY_NAME = {variable.name: variable for variable in VARIABLES}


def measurement_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Experiment 1A coverage, balance, variance, and prevalence table."""
    rows = []
    group_columns = ["model", "topic_id", "role", "condition"]
    for column in group_columns:
        if column in df:
            for value, count in df[column].value_counts(dropna=False).items():
                rows.append({"section": "balance", "variable": column, "level": value, "value": count})
    for column in df:
        if column.startswith("layer_"):
            rows.append({
                "section": "activation_coverage", "variable": column, "level": "present",
                "value": int(df[column].map(_is_vector).sum()),
            })
    for variable in VARIABLES:
        column = variable.name
        if column not in df:
            continue
        values = df[column]
        rows.append({
            "section": "variable", "variable": column, "level": "coverage",
            "value": int(values.notna().sum()),
        })
        if variable.task == "continuous":
            numeric = pd.to_numeric(values, errors="coerce")
            rows.append({
                "section": "variable", "variable": column, "level": "variance",
                "value": float(numeric.var()),
            })
        else:
            for level, prevalence in values.value_counts(normalize=True, dropna=True).items():
                rows.append({
                    "section": "class_prevalence", "variable": column,
                    "level": level, "value": float(prevalence),
                })
    if "replay_validation_status" in df:
        for value, count in df["replay_validation_status"].value_counts(dropna=False).items():
            rows.append({"section": "replay_gate", "variable": "status", "level": value, "value": count})
    return pd.DataFrame(rows)


def _is_vector(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0


def _layer_ids(df: pd.DataFrame, snapshots: Iterable[str]) -> list[int]:
    sets = []
    for snapshot in snapshots:
        sets.append({
            int(column.split("_", 1)[1].split("__", 1)[0])
            for column in df
            if column.endswith(f"__{snapshot}")
        })
    return sorted(set.intersection(*sets)) if sets else []


def _metadata_design(
    frame: pd.DataFrame, snapshot: str, target: str, relation: str
) -> np.ndarray:
    categorical = [column for column in ("model", "role", "topic_id", "condition") if column in frame]
    numeric = [
        column for column in (
            "turn", "prior_stance_score", "prior_stance_confidence", "prior_stance_gap",
            "prior_local_agreement", "prior_remaining_disagreement",
            "prior_affiliation", "prior_adversariality",
            "prior_closure_evidence", "prior_basin_leaning",
        ) if column in frame and column != target
    ]
    numeric.extend(
        column for column in frame
        if column.startswith("prior_") and column.endswith("_trailing3")
        and column != target and column not in numeric
    )
    pieces = []
    if relation != "current" and snapshot in (
        "full_response", "final_window", "final_token"
    ):
        current_state = (
            "stance_score", "stance_confidence", "stance_gap", "local_agreement",
            "remaining_disagreement", "affiliation", "adversariality",
            "closure_evidence", "basin_leaning",
        )
        numeric.extend(
            column for column in current_state
            if column in frame
        )
        numeric.extend(
            column for column in frame
            if column.endswith("_trailing3") and not column.startswith("prior_")
            and column != target and column not in numeric
        )
    if categorical:
        pieces.append(pd.get_dummies(frame[categorical], dummy_na=True, dtype=float).to_numpy())
    if numeric:
        values = frame[numeric].apply(pd.to_numeric, errors="coerce")
        pieces.append(values.to_numpy(dtype=float))
    for column in TIME_MATCHED_EMBEDDINGS[snapshot]:
        if column in frame and frame[column].map(_is_vector).all():
            pieces.append(np.stack(frame[column]))
    return np.concatenate(pieces, axis=1) if pieces else np.ones((len(frame), 1))


def _fold_predictions(
    baseline: np.ndarray,
    activation: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_oof = np.full(len(target), np.nan)
    combined_oof = np.full(len(target), np.nan)
    for train, test in LeaveOneGroupOut().split(baseline, target, groups):
        if not len(test):
            continue
        base_scaler, activation_scaler = StandardScaler(), StandardScaler()
        base_train, base_test = _fold_impute(baseline[train], baseline[test])
        base_train = base_scaler.fit_transform(base_train)
        base_test = base_scaler.transform(base_test)
        activation_train = activation_scaler.fit_transform(activation[train])
        activation_test = activation_scaler.transform(activation[test])
        base_model = Ridge(alpha=10.0).fit(base_train, target[train])
        combined_model = Ridge(alpha=10.0).fit(
            np.concatenate([base_train, activation_train], axis=1), target[train]
        )
        base_oof[test] = base_model.predict(base_test)
        combined_oof[test] = combined_model.predict(
            np.concatenate([base_test, activation_test], axis=1)
        )
    return base_oof, combined_oof


def _fold_impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute from training-fold means without looking at the held-out topic."""
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


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(prediction)
    if valid.sum() < 2:
        return float("nan")
    denominator = np.sum((target[valid] - target[valid].mean()) ** 2)
    return float(1 - np.sum((target[valid] - prediction[valid]) ** 2) / denominator) if denominator else float("nan")


def align_target(
    df: pd.DataFrame,
    target: str,
    relation: str = "current",
    horizon: int = 1,
    delta: bool = False,
) -> pd.Series:
    """Align current, future-self, or immediate-partner outcomes to source rows."""
    ordered = df.sort_values(["conv_id", "turn"])
    current = ordered[target]
    if relation == "current":
        aligned = current
    elif relation == "future_self":
        aligned = ordered.groupby(["conv_id", "speaker"])[target].shift(-horizon)
        aligned = aligned
    elif relation == "partner_next":
        aligned = pd.Series(None, index=ordered.index, dtype=object)
        for _, conversation in ordered.groupby("conv_id"):
            indices = list(conversation.index)
            for position, index in enumerate(indices):
                speaker = ordered.at[index, "speaker"]
                later = [candidate for candidate in indices[position + 1:] if ordered.at[candidate, "speaker"] != speaker]
                if later:
                    aligned.at[index] = ordered.at[later[0], target]
    else:
        raise ValueError(f"Unknown relation: {relation}")
    if delta:
        aligned = pd.to_numeric(aligned, errors="coerce") - pd.to_numeric(current, errors="coerce")
    return aligned.reindex(df.index)


def snapshot_decode(
    df: pd.DataFrame,
    target: str,
    relation: str = "current",
    horizon: int = 0,
    delta: bool = False,
    snapshots: Iterable[str] = PRIMARY_SNAPSHOTS,
    experiment: str = "1B",
    detail_rows: list[dict] | None = None,
    fold_rows: list[dict] | None = None,
    group_column: str = "topic_id",
    compute_shuffled: bool = True,
) -> pd.DataFrame:
    """Compare snapshots on identical rows and folds; report incremental OOF R2."""
    snapshots = tuple(snapshot for snapshot in snapshots if snapshot in get_snapshots(df))
    rows = []
    target_values = align_target(df, target, relation, max(1, horizon), delta)
    for layer in _layer_ids(df, snapshots):
        activation_columns = [f"layer_{layer}__{snapshot}" for snapshot in snapshots]
        common = target_values.notna()
        for column in activation_columns:
            common &= df[column].map(_is_vector)
        common_frame = df.loc[common]
        numeric_target = pd.to_numeric(target_values.loc[common], errors="coerce")
        variable = VARIABLES_BY_NAME.get(target)
        is_regression = (
            variable.task == "continuous"
            if variable is not None
            else numeric_target.notna().sum() == int(common.sum())
        )
        if is_regression:
            y = numeric_target.to_numpy(dtype=float)
            metric = "r2"
            class_labels = None
        else:
            y, class_labels = pd.factorize(
                target_values.loc[common].astype(str), sort=True
            )
            metric = "balanced_accuracy"
        if group_column not in common_frame:
            raise ValueError(f"Cross-validation group column unavailable: {group_column}")
        groups = common_frame[group_column].to_numpy()
        if len(common_frame) < 6 or len(np.unique(groups)) < 2:
            for snapshot in snapshots:
                rows.append({
                    "experiment": experiment, "target": target, "relation": relation,
                    "delta_target": delta, "horizon": horizon, "layer": layer,
                    "snapshot": snapshot, "n": len(common_frame),
                    "metric": metric, "null_score": np.nan, "shuffled_score": np.nan,
                    "baseline_score": np.nan,
                    "activation_plus_baseline_score": np.nan,
                    "incremental_score": np.nan,
                    "time_legal_baseline": True,
                    "cv_group": group_column,
                    "prospectively_predictive_eligible": (
                        _prospective_snapshot(common_frame, snapshot)
                    ),
                })
            continue
        shuffled_y = (
            np.random.default_rng(0).permutation(y) if compute_shuffled else None
        )
        null_oof = _fold_null_predictions(y, groups, not is_regression)
        null_score = _r2(y, null_oof) if is_regression else _balanced_accuracy(y, null_oof)
        for snapshot in snapshots:
            baseline = _metadata_design(common_frame, snapshot, target, relation)
            activation = np.stack(common_frame[f"layer_{layer}__{snapshot}"])
            if is_regression:
                base_oof, combined_oof = _fold_predictions(baseline, activation, y, groups)
                base_score, combined_score = _r2(y, base_oof), _r2(y, combined_oof)
                if compute_shuffled:
                    _, shuffled_oof = _fold_predictions(
                        baseline, activation, shuffled_y, groups
                    )
                    shuffled_score = _r2(y, shuffled_oof)
                else:
                    shuffled_score = np.nan
            else:
                base_oof, combined_oof = _fold_classification_predictions(
                    baseline, activation, y, groups
                )
                base_score = _balanced_accuracy(y, base_oof)
                combined_score = _balanced_accuracy(y, combined_oof)
                if compute_shuffled:
                    _, shuffled_oof = _fold_classification_predictions(
                        baseline, activation, shuffled_y, groups
                    )
                    shuffled_score = _balanced_accuracy(y, shuffled_oof)
                else:
                    shuffled_score = np.nan
            prospective = _prospective_snapshot(common_frame, snapshot)
            rows.append({
                "experiment": experiment, "target": target, "relation": relation,
                "delta_target": delta, "horizon": horizon, "layer": layer,
                "snapshot": snapshot, "n": len(common_frame),
                "metric": metric, "null_score": null_score,
                "shuffled_score": shuffled_score, "baseline_score": base_score,
                "activation_plus_baseline_score": combined_score,
                "incremental_score": combined_score - base_score,
                "time_legal_baseline": True,
                "cv_group": group_column,
                "prospectively_predictive_eligible": prospective,
            })
            _record_oof_audit(
                detail_rows, fold_rows, common_frame, target_values.loc[common],
                y, base_oof, combined_oof, groups, class_labels, experiment,
                target, relation, delta, horizon, layer, snapshot, metric,
            )
    return pd.DataFrame(rows)


PERSONA_TARGETS = tuple(
    f"perceived_persona_{dimension}_trailing3"
    for dimension in (
        "warmth", "dominance", "curiosity", "structure", "stability",
        "deference", "humility",
    )
)
BIG_FIVE_TARGETS = tuple(f"{field}_trailing3" for field in OBSERVER_BIG_FIVE_FIELDS)
ANNOTATED_STATE_TARGETS = (
    "local_agreement", "remaining_disagreement", "emotional_tone",
    "affiliation", "adversariality", "realized_move",
    "observed_conflict_index", "observed_alignment_index",
    "observed_exploration_index", "closure_evidence",
)

EXPERIMENT_TARGETS = {
    "1B": (
        "stance_score", "stance_confidence", "basin_leaning",
        *ANNOTATED_STATE_TARGETS, *PERSONA_TARGETS, *BIG_FIVE_TARGETS,
    ),
    "1C": (
        "stance_score", "stance_change", "partnerward_stance_movement",
        "basin_leaning", "apparent_objective",
        *ANNOTATED_STATE_TARGETS, *PERSONA_TARGETS, *BIG_FIVE_TARGETS,
    ),
    "1D": (
        "stance_score", "basin_leaning", "observed_accommodation_index",
        *ANNOTATED_STATE_TARGETS, *PERSONA_TARGETS, *BIG_FIVE_TARGETS,
    ),
    "1E": (
        "stance_change", "partnerward_stance_movement", "apparent_objective",
        "response_implied_expected_reaction", "observed_accommodation_index",
        *ANNOTATED_STATE_TARGETS, *PERSONA_TARGETS, *BIG_FIVE_TARGETS,
    ),
    "1F": ("apparent_objective", "response_implied_expected_reaction"),
    "1G": (
        "observable_transition", "explicit_synthesis", "explicit_resolution",
        "explicit_closure", "closure_evidence", "observed_accommodation_index",
    ),
    "1H": (
        "basin_leaning", "partnerward_basin_velocity", "semantic_velocity",
        "semantic_acceleration", "off_axis_distance",
    ),
    "1I": (
        "stance_score", "basin_leaning", "observed_accommodation_index",
        *ANNOTATED_STATE_TARGETS, *PERSONA_TARGETS, *BIG_FIVE_TARGETS,
    ),
}


def _fold_classification_predictions(
    baseline: np.ndarray,
    activation: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_oof = np.full(len(target), np.nan)
    combined_oof = np.full(len(target), np.nan)
    for train, test in LeaveOneGroupOut().split(baseline, target, groups):
        if len(np.unique(target[train])) < 2:
            continue
        base_scaler, activation_scaler = StandardScaler(), StandardScaler()
        base_train, base_test = _fold_impute(baseline[train], baseline[test])
        base_train = base_scaler.fit_transform(base_train)
        base_test = base_scaler.transform(base_test)
        activation_train = activation_scaler.fit_transform(activation[train])
        activation_test = activation_scaler.transform(activation[test])
        base_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            base_train, target[train]
        )
        combined_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            np.concatenate([base_train, activation_train], axis=1), target[train]
        )
        base_oof[test] = base_model.predict(base_test)
        combined_oof[test] = combined_model.predict(
            np.concatenate([base_test, activation_test], axis=1)
        )
    return base_oof, combined_oof


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(prediction)
    return float(balanced_accuracy_score(target[valid], prediction[valid])) if valid.any() else float("nan")


def add_time_matched_text_embeddings(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Embed only text available at each snapshot's time boundary."""
    from src.common.embeddings import embed_texts

    out = df.copy()
    context = out["transcript_context_text"].fillna("").astype(str).tolist()
    response = out["text"].fillna("").astype(str).tolist()
    early_text = (
        out.get("early_response_text", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
    )
    out["transcript_context_embedding"] = list(embed_texts(context, model_name))
    out["response_embedding"] = list(embed_texts(response, model_name))
    early_inputs = [
        f"{prior}\n{early}" if early else prior
        for prior, early in zip(context, early_text)
    ]
    out["early_response_embedding"] = list(embed_texts(early_inputs, model_name))
    return out


def partner_transfer_summary(df: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    """Experiment 1I mixed-play minus same-model self-play summaries."""
    rows = []
    for model, model_rows in df.groupby("model"):
        self_rows = model_rows[model_rows["condition"].eq("self_play")]
        mixed_rows = model_rows[model_rows["condition"].eq("mixed_play")]
        for target in targets:
            if target not in df:
                continue
            self_values = pd.to_numeric(self_rows[target], errors="coerce")
            mixed_values = pd.to_numeric(mixed_rows[target], errors="coerce")
            if self_values.notna().any() and mixed_values.notna().any():
                self_mean, mixed_mean = float(self_values.mean()), float(mixed_values.mean())
                rows.append({
                    "kind": "behavior", "model": model, "target": target,
                    "layer": np.nan, "snapshot": None,
                    "self_play_mean": self_mean, "mixed_play_mean": mixed_mean,
                    "mixed_minus_self": mixed_mean - self_mean,
                    "n_self": int(self_values.notna().sum()),
                    "n_mixed": int(mixed_values.notna().sum()),
                })
        for snapshot in get_snapshots(model_rows):
            for layer in _layer_ids(model_rows, (snapshot,)):
                column = f"layer_{layer}__{snapshot}"
                self_vectors = [value for value in self_rows[column] if _is_vector(value)]
                mixed_vectors = [value for value in mixed_rows[column] if _is_vector(value)]
                if not self_vectors or not mixed_vectors:
                    continue
                displacement = np.mean(np.stack(mixed_vectors), axis=0) - np.mean(
                    np.stack(self_vectors), axis=0
                )
                rows.append({
                    "kind": "activation", "model": model, "target": None,
                    "layer": layer, "snapshot": snapshot,
                    "self_play_mean": np.nan, "mixed_play_mean": np.nan,
                    "mixed_minus_self": float(np.linalg.norm(displacement)),
                    "n_self": len(self_vectors), "n_mixed": len(mixed_vectors),
                })
    return pd.DataFrame(rows)
def _fold_null_predictions(
    target: np.ndarray,
    groups: np.ndarray,
    classification: bool,
) -> np.ndarray:
    predictions = np.full(len(target), np.nan)
    placeholder = np.zeros((len(target), 1))
    for train, test in LeaveOneGroupOut().split(placeholder, target, groups):
        if classification:
            values, counts = np.unique(target[train], return_counts=True)
            prediction = values[np.argmax(counts)]
        else:
            prediction = float(np.mean(target[train]))
        predictions[test] = prediction
    return predictions
def _record_oof_audit(
    detail_rows: list[dict] | None,
    fold_rows: list[dict] | None,
    frame: pd.DataFrame,
    raw_target: pd.Series,
    y: np.ndarray,
    baseline_oof: np.ndarray,
    combined_oof: np.ndarray,
    groups: np.ndarray,
    class_labels: np.ndarray | None,
    experiment: str,
    target: str,
    relation: str,
    delta: bool,
    horizon: int,
    layer: int,
    snapshot: str,
    metric: str,
) -> None:
    prospective = _prospective_snapshot(frame, snapshot)
    if detail_rows is not None:
        for position, (index, row) in enumerate(frame.iterrows()):
            observed_label = raw_target.loc[index]
            base_value = baseline_oof[position]
            combined_value = combined_oof[position]
            if class_labels is not None:
                base_label = (
                    class_labels[int(base_value)]
                    if np.isfinite(base_value) else None
                )
                combined_label = (
                    class_labels[int(combined_value)]
                    if np.isfinite(combined_value) else None
                )
            else:
                base_label = base_value
                combined_label = combined_value
            detail_rows.append({
                "conv_id": row["conv_id"], "turn": row["turn"],
                "speaker": row["speaker"], "model": row.get("model"),
                "topic_id": row["topic_id"], "role": row.get("role"),
                "condition": row.get("condition"), "experiment": experiment,
                "target": target, "relation": relation, "delta_target": delta,
                "horizon": horizon, "layer": layer, "snapshot": snapshot,
                "metric": metric, "observed_target": observed_label,
                "observed_code": y[position],
                "baseline_prediction": base_label,
                "baseline_prediction_code": base_value,
                "activation_prediction": combined_label,
                "activation_prediction_code": combined_value,
                "incremental_prediction": combined_value - base_value
                if metric == "r2" and np.isfinite(combined_value) and np.isfinite(base_value)
                else np.nan,
                "prediction_changed": bool(combined_value != base_value)
                if np.isfinite(combined_value) and np.isfinite(base_value)
                else None,
                "held_out_topic": groups[position],
                "time_legal_baseline": True,
                "prospectively_predictive_eligible": prospective,
            })
    if fold_rows is not None:
        for held_out_topic in np.unique(groups):
            mask = groups == held_out_topic
            if metric == "r2":
                baseline_score = _r2(y[mask], baseline_oof[mask])
                combined_score = _r2(y[mask], combined_oof[mask])
            else:
                baseline_score = _balanced_accuracy(y[mask], baseline_oof[mask])
                combined_score = _balanced_accuracy(y[mask], combined_oof[mask])
            fold_rows.append({
                "experiment": experiment, "target": target,
                "relation": relation, "delta_target": delta,
                "horizon": horizon, "layer": layer, "snapshot": snapshot,
                "metric": metric, "held_out_topic": held_out_topic,
                "n_test": int(mask.sum()), "baseline_score": baseline_score,
                "activation_plus_baseline_score": combined_score,
                "incremental_score": combined_score - baseline_score,
                "time_legal_baseline": True,
                "prospectively_predictive_eligible": prospective,
            })


def _prospective_snapshot(frame: pd.DataFrame, snapshot: str) -> bool:
    """True only when the snapshot and its text baseline precede later text."""
    if snapshot == "pre_generation":
        return True
    if snapshot != "early_response":
        return False
    if not {"early_response_window", "response_token_count"}.issubset(frame.columns):
        return False
    early = pd.to_numeric(frame["early_response_window"], errors="coerce")
    total = pd.to_numeric(frame["response_token_count"], errors="coerce")
    return bool((early.notna() & total.notna() & early.lt(total)).all())

def original_replay_sensitivity(
    df: pd.DataFrame,
    target: str,
    relation: str = "current",
    horizon: int = 0,
    delta: bool = False,
    experiment: str = "1B",
) -> pd.DataFrame:
    """Fold-matched original versus replayed full-response probe comparison."""
    rows = []
    target_values = align_target(df, target, relation, max(1, horizon), delta)
    replay_layers = _layer_ids(df, ("full_response",))
    for layer in replay_layers:
        original_column = f"layer_{layer}"
        replay_column = f"layer_{layer}__full_response"
        if original_column not in df:
            continue
        common = (
            target_values.notna()
            & df[original_column].map(_is_vector)
            & df[replay_column].map(_is_vector)
        )
        frame = df.loc[common]
        if len(frame) < 6 or frame["topic_id"].nunique() < 2:
            continue
        numeric_target = pd.to_numeric(target_values.loc[common], errors="coerce")
        variable = VARIABLES_BY_NAME.get(target)
        is_regression = (
            variable.task == "continuous"
            if variable is not None
            else numeric_target.notna().sum() == int(common.sum())
        )
        if is_regression:
            y = numeric_target.to_numpy(dtype=float)
            metric = "r2"
        else:
            y, _ = pd.factorize(target_values.loc[common].astype(str), sort=True)
            metric = "balanced_accuracy"
        groups = frame["topic_id"].to_numpy()
        baseline = _metadata_design(frame, "full_response", target, relation)
        original = np.stack(frame[original_column])
        replayed = np.stack(frame[replay_column])
        if is_regression:
            baseline_oof, original_oof = _fold_predictions(
                baseline, original, y, groups
            )
            _, replayed_oof = _fold_predictions(baseline, replayed, y, groups)
            score = _r2
        else:
            baseline_oof, original_oof = _fold_classification_predictions(
                baseline, original, y, groups
            )
            _, replayed_oof = _fold_classification_predictions(
                baseline, replayed, y, groups
            )
            score = _balanced_accuracy
        baseline_score = score(y, baseline_oof)
        original_score = score(y, original_oof)
        replayed_score = score(y, replayed_oof)
        rows.append({
            "experiment": experiment,
            "target": target,
            "relation": relation,
            "delta_target": delta,
            "horizon": horizon,
            "layer": layer,
            "metric": metric,
            "n": len(frame),
            "baseline_score": baseline_score,
            "original_full_response_score": original_score,
            "replayed_full_response_score": replayed_score,
            "replayed_minus_original": replayed_score - original_score,
            "same_rows_and_folds": True,
        })
    return pd.DataFrame(rows)
