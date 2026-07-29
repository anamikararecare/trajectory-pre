"""
Experiments 1a/1b/1c from the README:

  1a. Concurrent decodability: activation(t) -> stance(t), per layer.
  1b. Predictive decodability: activation(t) -> stance(t+k), per layer/horizon,
      compared against a naive baseline that just extrapolates the current
      stance trend.
  1c. Cross-agent influence: agent A's activation(t) -> agent B's future
      stance delta (only meaningful for mixed_play conversations).

All experiments use leave-one-topic-out cross-validation so we're not just
fitting topic identity, and report both R^2 (continuous) and accuracy on a
binarized "moved toward more supportive / more opposed" label, since raw
Likert R^2 can be noisy with this little data.

NOTE: this is a proof-of-concept scale (a few hundred turns at most). Results
should be read as "is there a signal worth scaling up", not as a final claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut

from src.track1_probing.cache_activations import get_layer_columns


def _is_activation(value) -> bool:
    """True only for a saved, stackable activation vector.

    Rows from conversations where agent A is not the probed local model have
    no activation file. Pandas fills their layer columns with scalar NaN,
    which must not be mixed with vector-valued rows in ``np.stack``.
    """
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 0


def _stack_layer(df: pd.DataFrame, layer_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Drops rows with missing activation/stance for this layer; returns
    (X, valid_index) so callers can align back to the original dataframe."""
    mask = df[layer_col].map(_is_activation) & df["stance_score"].notna()
    idx = df.index[mask]
    X = np.stack(df.loc[idx, layer_col].values)
    return X, idx


def _logo_score(X: np.ndarray, y: np.ndarray, groups: np.ndarray, task: str) -> float:
    """Leave-one-topic-out CV. task='regress' -> mean R^2; task='classify' ->
    mean accuracy on sign(y) (binarized around 0, e.g. a stance delta)."""
    logo = LeaveOneGroupOut()
    scores = []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(groups[train_idx])) < 1 or len(test_idx) == 0:
            continue
        if task == "regress":
            model = Ridge(alpha=10.0)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            ss_res = np.sum((y[test_idx] - pred) ** 2)
            ss_tot = np.sum((y[test_idx] - y[train_idx].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            scores.append(r2)
        elif task == "classify":
            y_bin_train = (y[train_idx] > 0).astype(int)
            if len(np.unique(y_bin_train)) < 2:
                continue
            model = LogisticRegression(max_iter=1000)
            model.fit(X[train_idx], y_bin_train)
            y_bin_test = (y[test_idx] > 0).astype(int)
            acc = model.score(X[test_idx], y_bin_test)
            scores.append(acc)
    return float(np.nanmean(scores)) if scores else float("nan")


def _logo_regression_with_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    baseline_predictions: np.ndarray,
) -> tuple[float, float]:
    """Score a ridge probe and baseline on identical LOTO folds."""
    logo = LeaveOneGroupOut()
    probe_scores, baseline_scores = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(test_idx) == 0:
            continue
        model = Ridge(alpha=10.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        null = y[train_idx].mean()
        ss_tot = np.sum((y[test_idx] - null) ** 2)
        if ss_tot <= 0:
            continue
        probe_scores.append(1 - np.sum((y[test_idx] - pred) ** 2) / ss_tot)
        baseline_scores.append(
            1
            - np.sum((y[test_idx] - baseline_predictions[test_idx]) ** 2)
            / ss_tot
        )
    return (
        float(np.nanmean(probe_scores)) if probe_scores else float("nan"),
        float(np.nanmean(baseline_scores)) if baseline_scores else float("nan"),
    )


def _logo_classification_with_majority(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    """LOTO accuracy with a training-fold majority-class baseline."""
    logo = LeaveOneGroupOut()
    probe_scores, baseline_scores = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        y_train = y[train_idx].astype(int)
        y_test = y[test_idx].astype(int)
        if len(np.unique(y_train)) < 2 or len(test_idx) == 0:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train_idx], y_train)
        probe_scores.append(model.score(X[test_idx], y_test))
        majority = int(np.mean(y_train) >= 0.5)
        baseline_scores.append(float(np.mean(y_test == majority)))
    return (
        float(np.nanmean(probe_scores)) if probe_scores else float("nan"),
        float(np.nanmean(baseline_scores)) if baseline_scores else float("nan"),
    )


def experiment_1a_concurrent(df: pd.DataFrame) -> pd.DataFrame:
    """activation(t) -> stance(t), per layer. Sanity check / control."""
    rows = []
    for layer_col in get_layer_columns(df):
        X, idx = _stack_layer(df, layer_col)
        y = df.loc[idx, "stance_score"].values.astype(float)
        groups = df.loc[idx, "topic_id"].values
        r2 = _logo_score(X, y, groups, task="regress")
        rows.append({"experiment": "1a_concurrent", "layer": layer_col, "horizon": 0, "r2": r2})
    return pd.DataFrame(rows)


def experiment_1b_predictive(df: pd.DataFrame, horizons: list[int] = (1, 2, 3, 4)) -> pd.DataFrame:
    """activation(t) -> stance(t+k), per layer/horizon, plus a naive
    trend-extrapolation baseline for comparison."""
    rows = []
    # only agent_a's own turns carry activations; build per-speaker sequences
    df = df.sort_values(["conv_id", "turn"])

    for horizon in horizons:
        for layer_col in get_layer_columns(df):
            X_list, y_list, base_pred_list, group_list = [], [], [], []

            for conv_id, g in df.groupby("conv_id"):
                g = g.reset_index(drop=True)
                a_rows = g[g["speaker"] == "a"].reset_index(drop=True)
                for i in range(len(a_rows) - horizon):
                    cur = a_rows.iloc[i]
                    fut = a_rows.iloc[i + horizon]
                    if not _is_activation(cur[layer_col]) or pd.isna(cur["stance_score"]) or pd.isna(fut["stance_score"]):
                        continue
                    X_list.append(cur[layer_col])
                    y_list.append(fut["stance_score"])
                    # naive baseline: hold current stance constant (or use
                    # local trend if we have a previous point)
                    if i > 0 and not pd.isna(a_rows.iloc[i - 1]["stance_score"]):
                        trend = cur["stance_score"] - a_rows.iloc[i - 1]["stance_score"]
                    else:
                        trend = 0.0
                    base_pred_list.append(cur["stance_score"] + trend)
                    group_list.append(cur["topic_id"])

            if len(X_list) < 6:
                rows.append(
                    {
                        "experiment": "1b_predictive",
                        "layer": layer_col,
                        "horizon": horizon,
                        "probe_r2": np.nan,
                        "baseline_r2": np.nan,
                        "n": len(X_list),
                    }
                )
                continue

            X = np.stack(X_list)
            y = np.array(y_list, dtype=float)
            base_pred = np.array(base_pred_list, dtype=float)
            groups = np.array(group_list)

            probe_r2, baseline_r2 = _logo_regression_with_baseline(
                X, y, groups, base_pred
            )

            rows.append(
                {
                    "experiment": "1b_predictive",
                    "layer": layer_col,
                    "horizon": horizon,
                    "probe_r2": probe_r2,
                    "baseline_r2": baseline_r2,
                    "n": len(X_list),
                }
            )
    return pd.DataFrame(rows)


def experiment_1c_cross_agent(
    df: pd.DataFrame,
    horizons: list[int] = (1, 2, 3),
    zero_delta_epsilon: float = 1e-8,
) -> pd.DataFrame:
    """agent_a's activation(t) -> agent_b's stance delta over the next k
    turns (their turns, not raw turn index). Only defined for mixed_play
    conversations with two distinct speakers."""
    rows = []
    df = df[df["condition"] == "mixed_play"].sort_values(["conv_id", "turn"])

    for horizon in horizons:
        for layer_col in get_layer_columns(df):
            X_list, y_list, group_list = [], [], []

            for conv_id, g in df.groupby("conv_id"):
                g = g.reset_index(drop=True)
                a_rows = g[g["speaker"] == "a"].reset_index(drop=True)
                b_rows = g[g["speaker"] == "b"].reset_index(drop=True)
                if len(b_rows) < horizon + 1:
                    continue
                for i in range(len(a_rows)):
                    a_turn = a_rows.iloc[i]
                    if not _is_activation(a_turn[layer_col]):
                        continue
                    # find B's next stance reading after this A turn, and k
                    # B-turns further ahead
                    later_b = b_rows[b_rows["turn"] > a_turn["turn"]].reset_index(drop=True)
                    if len(later_b) < horizon + 1:
                        continue
                    b_now, b_future = later_b.iloc[0], later_b.iloc[horizon]
                    if pd.isna(b_now["stance_score"]) or pd.isna(b_future["stance_score"]):
                        continue
                    X_list.append(a_turn[layer_col])
                    y_list.append(b_future["stance_score"] - b_now["stance_score"])
                    group_list.append(a_turn["topic_id"])

            if len(X_list) < 6:
                rows.append(
                    {
                        "experiment": "1c_cross_agent",
                        "layer": layer_col,
                        "horizon": horizon,
                        "acc": np.nan,
                        "majority_acc": np.nan,
                        "n": len(X_list),
                        "n_zero_dropped": 0,
                    }
                )
                continue

            X = np.stack(X_list)
            deltas = np.array(y_list, dtype=float)
            groups = np.array(group_list)
            nonzero = np.abs(deltas) >= zero_delta_epsilon
            X, deltas, groups = X[nonzero], deltas[nonzero], groups[nonzero]
            if len(X) < 6 or len(np.unique(groups)) < 2:
                rows.append(
                    {
                        "experiment": "1c_cross_agent",
                        "layer": layer_col,
                        "horizon": horizon,
                        "acc": np.nan,
                        "majority_acc": np.nan,
                        "n": len(X),
                        "n_zero_dropped": int((~nonzero).sum()),
                    }
                )
                continue
            y = (deltas > 0).astype(int)
            acc, majority_acc = _logo_classification_with_majority(X, y, groups)
            rows.append(
                {
                    "experiment": "1c_cross_agent",
                    "layer": layer_col,
                    "horizon": horizon,
                    "acc": acc,
                    "majority_acc": majority_acc,
                    "n": len(X),
                    "n_zero_dropped": int((~nonzero).sum()),
                }
            )
    return pd.DataFrame(rows)
