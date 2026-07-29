"""Cross-model activation and conversational-variable RSMs for Track 1.5.

The activation analysis compares the two speakers in a mixed-play conversation.
For every relative-depth layer pair, model-specific activations are mean-centred,
reduced to a common PCA rank, and aligned with an orthogonal Procrustes map.
The default alignment is learned from other conversations with the same ordered
model pair, so the target conversation remains held out.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr

from src.track1_probing.variables import VARIABLES


ACTIVATION_KEY = re.compile(
    r"^turn_(?P<turn>\d+)__speaker_(?P<speaker>[^_]+)"
    r"__model_(?P<model>.+?)__layer_(?P<layer>-?\d+)"
    r"__snapshot_(?P<snapshot>.+?)__window_(?P<window>\d+)$"
)


@dataclass(frozen=True)
class Alignment:
    mean_a: np.ndarray
    mean_b: np.ndarray
    basis_a: np.ndarray
    basis_b: np.ndarray
    rotation: np.ndarray
    rank: int
    n_train_pairs: int

    def transform_a(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_a) @ self.basis_a @ self.rotation

    def transform_b(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_b) @ self.basis_b


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(values: list[int], count: int) -> list[int]:
    """Select `count` rank-evenly-spaced values, including both endpoints."""
    ordered = sorted(set(int(value) for value in values))
    if count < 1:
        raise ValueError("Layer count must be positive.")
    if len(ordered) < count:
        raise ValueError(
            f"Requested {count} layers, but only {len(ordered)} are available: {ordered}"
        )
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    return [ordered[index] for index in indices]


def fit_alignment(
    values_a: np.ndarray,
    values_b: np.ndarray,
    max_rank: int = 32,
) -> Alignment:
    """Fit centred PCA bases and an A-to-B orthogonal Procrustes map."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Alignment inputs must be 2-D with equal row counts.")
    if a.shape[0] < 3:
        raise ValueError("At least three paired observations are required for alignment.")
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    centered_a, centered_b = a - mean_a, b - mean_b
    rank = min(max_rank, a.shape[0] - 1, a.shape[1], b.shape[1])
    if rank < 1:
        raise ValueError("The common alignment rank is zero.")
    _, singular_a, vt_a = np.linalg.svd(centered_a, full_matrices=False)
    _, singular_b, vt_b = np.linalg.svd(centered_b, full_matrices=False)
    effective_a = int(np.sum(singular_a > singular_a[0] * 1e-10)) if singular_a[0] else 0
    effective_b = int(np.sum(singular_b > singular_b[0] * 1e-10)) if singular_b[0] else 0
    rank = min(rank, effective_a, effective_b)
    if rank < 1:
        raise ValueError("Centred alignment inputs have no non-constant dimensions.")
    basis_a = vt_a[:rank].T
    basis_b = vt_b[:rank].T
    scores_a = centered_a @ basis_a
    scores_b = centered_b @ basis_b
    rotation, _ = orthogonal_procrustes(scores_a, scores_b)
    return Alignment(
        mean_a=mean_a,
        mean_b=mean_b,
        basis_a=basis_a,
        basis_b=basis_b,
        rotation=rotation,
        rank=rank,
        n_train_pairs=a.shape[0],
    )


def cosine_cross_similarity(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Return the row-by-row cross-set cosine-similarity matrix."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    safe_a = np.divide(a, norm_a, out=np.zeros_like(a), where=norm_a > 0)
    safe_b = np.divide(b, norm_b, out=np.zeros_like(b), where=norm_b > 0)
    matrix = safe_a @ safe_b.T
    invalid_a = norm_a[:, 0] == 0
    invalid_b = norm_b[:, 0] == 0
    matrix[np.ix_(invalid_a, np.ones(len(b), dtype=bool))] = np.nan
    matrix[np.ix_(np.ones(len(a), dtype=bool), invalid_b)] = np.nan
    return matrix


def continuous_rsm(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mean-centre a scalar variable and return an RBF cross-model RSM."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    pooled = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if pooled.size < 2:
        return np.full((len(a), len(b)), np.nan), {
            "centre": np.nan, "scale": np.nan, "constant": True
        }
    centre = float(pooled.mean())
    scale = float(pooled.std(ddof=1))
    constant = not np.isfinite(scale) or scale <= 1e-12
    if constant:
        scale = 1.0
    centered_a = (a - centre) / scale
    centered_b = (b - centre) / scale
    squared = (centered_a[:, None] - centered_b[None, :]) ** 2
    similarity = np.exp(-0.5 * squared)
    similarity[~np.isfinite(squared)] = np.nan
    return similarity, {"centre": centre, "scale": scale, "constant": constant}


def categorical_rsm(values_a: np.ndarray, values_b: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return exact-match similarity for a categorical conversational variable."""
    a = pd.Series(values_a, dtype="object")
    b = pd.Series(values_b, dtype="object")
    matrix = (a.to_numpy()[:, None] == b.to_numpy()[None, :]).astype(float)
    missing = a.isna().to_numpy()[:, None] | b.isna().to_numpy()[None, :]
    matrix[missing] = np.nan
    levels = sorted({str(value) for value in pd.concat([a, b]).dropna().unique()})
    return matrix, {"levels": levels, "constant": len(levels) < 2}


def rsa_correlation(
    activation_rsm: np.ndarray,
    variable_rsm: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Spearman RSA with a Model-B-axis permutation test."""
    activation = np.asarray(activation_rsm, dtype=float)
    variable = np.asarray(variable_rsm, dtype=float)
    mask = np.isfinite(activation) & np.isfinite(variable)
    n = int(mask.sum())
    if n < 8:
        return np.nan, np.nan, n
    if (
        np.unique(activation[mask]).size < 2
        or np.unique(variable[mask]).size < 2
    ):
        return np.nan, np.nan, n
    observed = float(spearmanr(activation[mask], variable[mask]).statistic)
    if not np.isfinite(observed) or permutations <= 0:
        return observed, np.nan, n
    exceed = 0
    for _ in range(permutations):
        permuted = variable[:, rng.permutation(variable.shape[1])]
        perm_mask = np.isfinite(activation) & np.isfinite(permuted)
        if perm_mask.sum() < 8:
            continue
        statistic = float(spearmanr(
            activation[perm_mask], permuted[perm_mask]
        ).statistic)
        if np.isfinite(statistic) and abs(statistic) >= abs(observed):
            exceed += 1
    return observed, (exceed + 1) / (permutations + 1), n


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving missing entries."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _save_matrix(
    matrix: np.ndarray,
    path: Path,
    row_labels: list[int],
    column_labels: list[int],
) -> None:
    frame = pd.DataFrame(
        matrix,
        index=[f"a_agent_turn_{value}" for value in row_labels],
        columns=[f"b_agent_turn_{value}" for value in column_labels],
    )
    frame.to_csv(path, index_label="model_a_agent_turn")


def _plot_matrix(
    matrix: np.ndarray,
    path: Path,
    title: str,
    row_labels: list[int],
    column_labels: list[int],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("Model B agent turn")
    axis.set_ylabel("Model A agent turn")
    ticks_x = np.arange(0, len(column_labels), max(1, len(column_labels) // 5))
    ticks_y = np.arange(0, len(row_labels), max(1, len(row_labels) // 5))
    axis.set_xticks(ticks_x, [column_labels[index] for index in ticks_x])
    axis.set_yticks(ticks_y, [row_labels[index] for index in ticks_y])
    figure.colorbar(image, ax=axis, label="Similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_activation_index(
    replay_dir: str,
    snapshot: str,
) -> tuple[dict, dict, dict]:
    """Load replay activations and model validation metadata."""
    root = Path(replay_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Replay manifest not found: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    index: dict[tuple[str, int, str, str, int], np.ndarray] = {}
    layers: dict[str, set[int]] = {}
    for archive_path in sorted(root.glob("*.npz")):
        conv_id = archive_path.stem
        with np.load(archive_path) as archive:
            for key in archive.files:
                match = ACTIVATION_KEY.match(key)
                if not match or match["snapshot"] != snapshot:
                    continue
                model = match["model"]
                layer = int(match["layer"])
                identity = (
                    conv_id, int(match["turn"]), match["speaker"], model, layer
                )
                index[identity] = np.asarray(archive[key], dtype=np.float32)
                layers.setdefault(model, set()).add(layer)
    validations = {
        model: details.get("validation", {}).get("status", "not_evaluated")
        for model, details in manifest.get("models", {}).items()
    }
    return index, layers, {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "validations": validations,
    }


def _conversation_models(frame: pd.DataFrame) -> tuple[str, str]:
    models_a = frame.loc[frame["speaker"].eq("a"), "model"].dropna().unique()
    models_b = frame.loc[frame["speaker"].eq("b"), "model"].dropna().unique()
    if len(models_a) != 1 or len(models_b) != 1:
        raise ValueError("Each conversation must have exactly one model per speaker.")
    return str(models_a[0]), str(models_b[0])


def _model_turns(
    frame: pd.DataFrame,
    speaker: str,
    max_turns: int,
) -> pd.DataFrame:
    selected = (
        frame[frame["speaker"].eq(speaker)]
        .sort_values("agent_turn")
        .drop_duplicates("agent_turn")
    )
    selected = selected[selected["agent_turn"].between(1, max_turns)].copy()
    if len(selected) != max_turns:
        raise ValueError(
            f"Speaker {speaker} has {len(selected)} usable agent turns; "
            f"expected {max_turns}."
        )
    return selected


def _activation_rows(
    index: dict,
    rows: pd.DataFrame,
    model: str,
    layer: int,
) -> np.ndarray:
    vectors = []
    for record in rows.itertuples(index=False):
        key = (record.conv_id, int(record.turn), record.speaker, model, layer)
        if key not in index:
            raise KeyError(f"Missing activation: {key}")
        vectors.append(index[key])
    return np.stack(vectors)


def _training_pairs(
    all_turns: pd.DataFrame,
    activation_index: dict,
    target_conv_id: str,
    model_a: str,
    model_b: str,
    layer_a: int,
    layer_b: int,
    max_turns: int,
    alignment_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if alignment_mode == "within_conversation":
        candidate_ids = [target_conv_id]
    else:
        candidate_ids = [
            conv_id for conv_id in all_turns["conv_id"].dropna().unique()
            if conv_id != target_conv_id
        ]
    arrays_a, arrays_b, used = [], [], []
    for conv_id in candidate_ids:
        frame = all_turns[all_turns["conv_id"].eq(conv_id)]
        try:
            candidate_a, candidate_b = _conversation_models(frame)
            if (candidate_a, candidate_b) != (model_a, model_b):
                continue
            rows_a = _model_turns(frame, "a", max_turns)
            rows_b = _model_turns(frame, "b", max_turns)
            values_a = _activation_rows(
                activation_index, rows_a, model_a, layer_a
            )
            values_b = _activation_rows(
                activation_index, rows_b, model_b, layer_b
            )
        except (KeyError, ValueError):
            continue
        arrays_a.append(values_a)
        arrays_b.append(values_b)
        used.append(str(conv_id))
    if not arrays_a:
        raise ValueError(
            "No alignment-training conversations were available for the ordered "
            f"model pair {model_a} -> {model_b}. Use --alignment-mode "
            "within_conversation only for explicitly exploratory analysis."
        )
    return np.concatenate(arrays_a), np.concatenate(arrays_b), used


def run_track1_5(
    replay_dir: str,
    turn_variables_path: str,
    out_dir: str,
    conv_ids: list[str] | None = None,
    snapshot: str = "full_response",
    n_layers: int = 4,
    max_turns: int = 10,
    alignment_mode: str = "leave_one_conversation_out",
    alignment_rank: int = 32,
    permutations: int = 2000,
    seed: int = 0,
    allow_unvalidated_models: bool = False,
    plot_variables: bool = True,
) -> Path:
    """Run Track 1.5 and return the output root."""
    turns_path = Path(turn_variables_path)
    turns = pd.read_csv(turns_path)
    required = {"conv_id", "turn", "agent_turn", "speaker", "model"}
    missing = required.difference(turns.columns)
    if missing:
        raise ValueError(f"Turn-variable table is missing columns: {sorted(missing)}")
    activation_index, layers_by_model, replay_info = load_activation_index(
        replay_dir, snapshot
    )
    eligible = []
    for conv_id, frame in turns.groupby("conv_id", sort=True):
        try:
            model_a, model_b = _conversation_models(frame)
            _model_turns(frame, "a", max_turns)
            _model_turns(frame, "b", max_turns)
        except ValueError:
            continue
        if model_a == model_b:
            continue
        if model_a not in layers_by_model or model_b not in layers_by_model:
            continue
        eligible.append(str(conv_id))
    selected = conv_ids or eligible
    unknown = set(selected).difference(eligible)
    if unknown:
        raise ValueError(
            "Requested conversations are not eligible mixed-play conversations "
            f"with {max_turns} turns per model and replay activations: {sorted(unknown)}"
        )
    if not selected:
        raise ValueError("No eligible mixed-play conversations were found.")

    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    registry = {variable.name: variable for variable in VARIABLES}
    variables = [
        name for name in registry
        if name in turns.columns and name not in {"stance_score"}
    ]
    rng = np.random.default_rng(seed)
    run_records = []

    for conv_id in selected:
        frame = turns[turns["conv_id"].eq(conv_id)].copy()
        model_a, model_b = _conversation_models(frame)
        validation = replay_info["validations"]
        invalid = {
            model: validation.get(model, "not_evaluated")
            for model in (model_a, model_b)
            if validation.get(model) != "passed"
        }
        if invalid and not allow_unvalidated_models:
            raise ValueError(
                f"Replay validation is not passed for {invalid}. Re-run with "
                "--allow-unvalidated-models to produce explicitly exploratory RSMs."
            )
        rows_a = _model_turns(frame, "a", max_turns)
        rows_b = _model_turns(frame, "b", max_turns)
        turns_a = rows_a["agent_turn"].astype(int).tolist()
        turns_b = rows_b["agent_turn"].astype(int).tolist()
        layers_a = evenly_spaced(sorted(layers_by_model[model_a]), n_layers)
        layers_b = evenly_spaced(sorted(layers_by_model[model_b]), n_layers)
        conv_root = output_root / _safe_name(conv_id)
        activation_root = conv_root / "activation_rsms"
        variable_root = conv_root / "variable_rsms"
        plot_root = conv_root / "figures"
        activation_root.mkdir(parents=True, exist_ok=True)
        variable_root.mkdir(parents=True, exist_ok=True)
        plot_root.mkdir(parents=True, exist_ok=True)

        activation_rsms = {}
        diagnostics = []
        training_conversations = set()
        for depth_index, (layer_a, layer_b) in enumerate(
            zip(layers_a, layers_b), start=1
        ):
            train_a, train_b, train_ids = _training_pairs(
                turns, activation_index, conv_id, model_a, model_b,
                layer_a, layer_b, max_turns, alignment_mode,
            )
            alignment = fit_alignment(train_a, train_b, max_rank=alignment_rank)
            target_a = _activation_rows(
                activation_index, rows_a, model_a, layer_a
            )
            target_b = _activation_rows(
                activation_index, rows_b, model_b, layer_b
            )
            projected_a = alignment.transform_a(target_a)
            projected_b = alignment.transform_b(target_b)
            matrix = cosine_cross_similarity(projected_a, projected_b)
            label = f"depth_{depth_index:02d}__a_{layer_a}__b_{layer_b}"
            activation_rsms[label] = matrix
            _save_matrix(
                matrix, activation_root / f"{label}.csv", turns_a, turns_b
            )
            paired_cosine = np.diag(matrix)
            diagnostics.append({
                "depth_index": depth_index,
                "layer_a": layer_a,
                "layer_b": layer_b,
                "alignment_rank": alignment.rank,
                "n_train_pairs": alignment.n_train_pairs,
                "n_training_conversations": len(train_ids),
                "heldout_paired_turn_cosine_mean": float(
                    np.nanmean(paired_cosine)
                ),
                "heldout_paired_turn_cosine_median": float(
                    np.nanmedian(paired_cosine)
                ),
            })
            training_conversations.update(train_ids)
        mean_activation = np.nanmean(
            np.stack(list(activation_rsms.values())), axis=0
        )
        activation_rsms["mean_across_layers"] = mean_activation
        _save_matrix(
            mean_activation, activation_root / "mean_across_layers.csv",
            turns_a, turns_b,
        )
        np.savez_compressed(
            activation_root / "activation_rsms.npz", **activation_rsms
        )
        pd.DataFrame(diagnostics).to_csv(
            conv_root / "alignment_diagnostics.csv", index=False
        )

        figure, axes = plt.subplots(
            1, n_layers, figsize=(4.2 * n_layers, 4.2), squeeze=False
        )
        finite = np.concatenate([
            matrix[np.isfinite(matrix)] for key, matrix in activation_rsms.items()
            if key != "mean_across_layers"
        ])
        vmin, vmax = np.quantile(finite, [0.02, 0.98])
        for axis, (label, matrix) in zip(
            axes[0],
            [(key, value) for key, value in activation_rsms.items()
             if key != "mean_across_layers"],
        ):
            image = axis.imshow(
                matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
            )
            axis.set_title(label.replace("__", "\n"))
            axis.set_xlabel("Model B agent turn")
            axis.set_ylabel("Model A agent turn")
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Cosine similarity")
        figure.suptitle(
            f"{conv_id}\nMean-centred, aligned activation RSMs", y=1.04
        )
        figure.savefig(
            plot_root / "activation_rsms_by_layer.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(figure)

        variable_metadata = []
        variable_matrices = {}
        ordered_variables = ["stance_score", *variables]
        for name in ordered_variables:
            variable = registry[name]
            if variable.task == "continuous":
                values_a = pd.to_numeric(rows_a[name], errors="coerce").to_numpy()
                values_b = pd.to_numeric(rows_b[name], errors="coerce").to_numpy()
                matrix, metadata = continuous_rsm(values_a, values_b)
            else:
                values_a = rows_a[name].to_numpy()
                values_b = rows_b[name].to_numpy()
                matrix, metadata = categorical_rsm(values_a, values_b)
            safe_name = _safe_name(name)
            variable_matrices[name] = matrix
            _save_matrix(matrix, variable_root / f"{safe_name}.csv", turns_a, turns_b)
            variable_metadata.append({
                "variable": name,
                "source": variable.source,
                "task": variable.task,
                "timing": variable.timing,
                "n_model_a": int(pd.Series(values_a).notna().sum()),
                "n_model_b": int(pd.Series(values_b).notna().sum()),
                **metadata,
            })
            if plot_variables:
                _plot_matrix(
                    matrix,
                    plot_root / f"variable__{safe_name}.png",
                    f"{name} RSM",
                    turns_a,
                    turns_b,
                    vmin=0.0,
                    vmax=1.0,
                )
        np.savez_compressed(
            variable_root / "variable_rsms.npz",
            **{_safe_name(name): matrix for name, matrix in variable_matrices.items()},
        )
        pd.DataFrame(variable_metadata).to_csv(
            conv_root / "variable_rsm_metadata.csv", index=False
        )

        rsa_rows = []
        for activation_name, activation_matrix in activation_rsms.items():
            for variable_name, variable_matrix in variable_matrices.items():
                rho, p_value, n_cells = rsa_correlation(
                    activation_matrix, variable_matrix, permutations, rng
                )
                rsa_rows.append({
                    "activation_rsm": activation_name,
                    "variable": variable_name,
                    "spearman_rho": rho,
                    "permutation_p": p_value,
                    "n_cells": n_cells,
                })
        rsa = pd.DataFrame(rsa_rows)
        rsa["fdr_q"] = rsa.groupby("activation_rsm", group_keys=False)[
            "permutation_p"
        ].apply(benjamini_hochberg)
        rsa.to_csv(conv_root / "rsa_variable_explanations.csv", index=False)

        conversation_manifest = {
            "schema_version": 1,
            "track": "1.5",
            "conv_id": conv_id,
            "model_a": model_a,
            "model_b": model_b,
            "snapshot": snapshot,
            "max_turns_per_model": max_turns,
            "n_layers": n_layers,
            "layers_a": layers_a,
            "layers_b": layers_b,
            "mean_centered": True,
            "alignment": {
                "mode": alignment_mode,
                "method": "separate PCA to common rank, then orthogonal Procrustes",
                "max_rank": alignment_rank,
                "training_conversations": sorted(training_conversations),
                "target_in_alignment_fit": alignment_mode == "within_conversation",
            },
            "activation_similarity": "cosine in aligned common PCA space",
            "continuous_variable_similarity": (
                "RBF similarity of pooled-within-conversation mean-centred, "
                "standardized scalar values"
            ),
            "categorical_variable_similarity": "exact match",
            "permutations": permutations,
            "seed": seed,
            "replay_validation": {
                model_a: validation.get(model_a, "not_evaluated"),
                model_b: validation.get(model_b, "not_evaluated"),
            },
            "exploratory_unvalidated_models": bool(invalid),
        }
        with (conv_root / "manifest.json").open("w") as handle:
            json.dump(conversation_manifest, handle, indent=2, allow_nan=True)
        run_records.append(conversation_manifest)

    run_manifest = {
        "schema_version": 1,
        "track": "1.5",
        "replay_manifest": str(replay_info["manifest_path"].resolve()),
        "replay_manifest_sha256": _sha256(replay_info["manifest_path"]),
        "turn_variables": str(turns_path.resolve()),
        "turn_variables_sha256": _sha256(turns_path),
        "conversations": run_records,
    }
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(run_manifest, handle, indent=2, allow_nan=True)
    return output_root

