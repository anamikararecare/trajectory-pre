"""Read-only data assembly helpers for the Track 1 Streamlit dashboard."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.debate_prompts import load_topics
from src.track1_probing.replay import reconstruct_history
from src.track1_probing.variables import DYNAMIC_PERSONA_FIELDS, aggregate_annotations

SNAPSHOT_ORDER = [
    "pre_generation",
    "early_response",
    "full_response",
    "final_window",
    "final_token",
]
ACTIVATION_STATUSES = [
    "original",
    "replayed_validated",
    "replayed_warning",
    "unavailable",
]
_KEY_PATTERN = re.compile(
    r"^turn_(?P<turn>\d+)__speaker_(?P<speaker>[^_]+)"
    r"__model_(?P<model>.+?)__layer_(?P<layer>-?\d+)"
    r"__snapshot_(?P<snapshot>.+?)__window_(?P<window>\d+)$"
)


def read_json(path: str | os.PathLike) -> dict:
    with open(path) as handle:
        return json.load(handle)


def discover_replay_runs(data_dir: str) -> list[str]:
    """Prefer complete corpus replays over validation-only gate manifests."""
    pattern = os.path.join(data_dir, "replayed_activations", "*", "manifest.json")
    ranked = []
    for path in glob.glob(pattern):
        directory = str(Path(path).parent)
        try:
            manifest = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        array_count = len(glob.glob(os.path.join(directory, "*.npz")))
        ranked.append((
            not bool(manifest.get("validation_only")),
            array_count > 0,
            str(manifest.get("created_at", "")),
            directory,
        ))
    return [row[-1] for row in sorted(ranked, reverse=True)]


def discover_result_runs(results_root: str) -> list[str]:
    pattern = os.path.join(results_root, "**", "snapshot_probe_scores.csv")
    return sorted((str(Path(path).parent) for path in glob.glob(pattern, recursive=True)), reverse=True)


def parse_activation_key(key: str) -> dict | None:
    match = _KEY_PATTERN.match(key)
    if not match:
        return None
    record = match.groupdict()
    for column in ("turn", "layer", "window"):
        record[column] = int(record[column])
    return record


def load_replay_vectors(
    replay_dir: str,
    model: str,
    layer: int,
    snapshots: list[str] | tuple[str, ...] = tuple(SNAPSHOT_ORDER),
) -> pd.DataFrame:
    """Load only one model/layer slice for interactive activation plots."""
    wanted = set(snapshots)
    rows = []
    for path in sorted(glob.glob(os.path.join(replay_dir, "*.npz"))):
        conv_id = Path(path).stem
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                record = parse_activation_key(key)
                if (
                    record is None
                    or record["model"] != model
                    or record["layer"] != layer
                    or record["snapshot"] not in wanted
                ):
                    continue
                vector = np.asarray(arrays[key], dtype=np.float32).reshape(-1)
                rows.append({
                    "conv_id": conv_id,
                    **record,
                    "activation": vector,
                    "activation_norm": float(np.linalg.norm(vector)),
                })
    return pd.DataFrame(rows)


def validation_status_by_model(manifest: dict) -> dict[str, str]:
    return {
        model: details.get("validation", {}).get("status", "not_evaluated")
        for model, details in manifest.get("models", {}).items()
    }


def activation_status(
    snapshot: str,
    has_original: bool,
    has_replay: bool,
    validation_status: str | None,
) -> str:
    if snapshot == "full_response" and has_original:
        return "original"
    if has_replay and validation_status == "passed":
        return "replayed_validated"
    if has_replay:
        return "replayed_warning"
    return "unavailable"


def replay_array_index(replay_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(replay_dir, "*.npz"))):
        conv_id = Path(path).stem
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                record = parse_activation_key(key)
                if record:
                    rows.append({"conv_id": conv_id, "key": key, **record})
    columns = ["conv_id", "key", "turn", "speaker", "model", "layer", "snapshot", "window"]
    return pd.DataFrame(rows, columns=columns)


def original_array_index(data_dir: str, transcripts: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "activations", "*.npz"))):
        conv_id = Path(path).stem
        transcript = transcripts.get(conv_id)
        if transcript is None:
            continue
        turn_lookup = {int(turn["turn"]): turn for turn in transcript["turns"]}
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                try:
                    layer_text, turn_text = key.split("__", 1)
                    layer, turn = int(layer_text), int(turn_text)
                except ValueError:
                    continue
                turn_record = turn_lookup.get(turn)
                if turn_record is None:
                    continue
                rows.append({
                    "conv_id": conv_id, "turn": turn,
                    "speaker": turn_record["speaker"],
                    "model": turn_record.get("model"),
                    "layer": layer, "snapshot": "full_response",
                    "window": np.nan, "key": key,
                })
    return pd.DataFrame(rows)


def load_transcripts(data_dir: str) -> tuple[dict[str, dict], pd.DataFrame]:
    transcripts, rows = {}, []
    for path in sorted(glob.glob(os.path.join(data_dir, "transcripts", "*.json"))):
        transcript = read_json(path)
        conv_id = transcript["conv_id"]
        transcripts[conv_id] = transcript
        for turn in transcript["turns"]:
            speaker = turn["speaker"]
            partner = "b" if speaker == "a" else "a"
            rows.append({
                "conv_id": conv_id, "turn": int(turn["turn"]),
                "agent_turn": turn.get("agent_turn"), "speaker": speaker,
                "model": turn.get("model", transcript.get(f"agent_{speaker}_model")),
                "partner_model": transcript.get(f"agent_{partner}_model"),
                "role": turn.get("role"), "topic_id": transcript.get("topic_id"),
                "condition": transcript.get("condition"), "text": turn.get("text", ""),
                "stance_score": turn.get("stance_score"),
                "stance_confidence": turn.get("stance_confidence"),
            })
    return transcripts, pd.DataFrame(rows)


def validation_rows(manifest: dict) -> pd.DataFrame:
    rows = []
    for model, details in manifest.get("models", {}).items():
        for row in details.get("validation_rows", []):
            rows.append({"manifest_model": model, **row})
    return pd.DataFrame(rows)


def turn_metadata(manifest: dict) -> pd.DataFrame:
    return pd.DataFrame(manifest.get("turn_metadata", []))


def token_alignment_warnings(metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return pd.DataFrame(columns=["conv_id", "turn", "speaker", "warning"])
    rows = []
    for row in metadata.to_dict("records"):
        warnings = []
        start, end = row.get("response_start_index"), row.get("response_end_index")
        count = row.get("response_token_count")
        if None not in (start, end, count) and end - start + 1 != count:
            warnings.append("response boundary length does not match token count")
        if row.get("early_response_window", 0) > (count or 0):
            warnings.append("early window exceeds response token count")
        if row.get("final_window", 0) > (count or 0):
            warnings.append("final window exceeds response token count")
        if row.get("response_special_tokens_included"):
            warnings.append("primary response pool includes special tokens")
        if row.get("eos_included"):
            warnings.append("primary response pool includes EOS")
        for warning in warnings:
            rows.append({
                "conv_id": row.get("conv_id"), "turn": row.get("turn"),
                "speaker": row.get("speaker"), "model": row.get("model"),
                "warning": warning,
            })
    return pd.DataFrame(rows)


def replay_overview(
    manifest: dict,
    replay_index: pd.DataFrame,
    warnings: pd.DataFrame,
) -> dict:
    eligibility = pd.DataFrame(manifest.get("eligibility", []))
    metadata = turn_metadata(manifest)
    if not eligibility.empty and "eligible" in eligibility:
        eligible = eligibility[eligibility["eligible"].fillna(False).astype(bool)]
    else:
        eligible = eligibility.iloc[0:0]
    completed_keys = set()
    if not metadata.empty:
        completed_keys = set(zip(metadata["conv_id"], metadata["turn"], metadata["speaker"], metadata["model"]))
    eligible_keys = set()
    if not eligible.empty:
        eligible_keys = set(zip(eligible["conv_id"], eligible["turn"], eligible["speaker"], eligible["model"]))
    failed_eligibility = (
        eligibility[eligibility["eligible"].fillna(False).astype(bool).eq(False)]
        if not eligibility.empty and "eligible" in eligibility else eligibility.iloc[0:0]
    )
    snapshot_counts = (
        replay_index.groupby("snapshot").size().reindex(SNAPSHOT_ORDER, fill_value=0).to_dict()
        if not replay_index.empty else {snapshot: 0 for snapshot in SNAPSHOT_ORDER}
    )
    return {
        "replay_id": manifest.get("replay_id"),
        "eligible_turns": len(eligible_keys),
        "completed_turns": len(completed_keys),
        "failed_or_missing_turns": len(failed_eligibility) + len(eligible_keys - completed_keys),
        "eligible_models": sorted(eligible["model"].dropna().unique()) if not eligible.empty else [],
        "eligible_speakers": sorted(eligible["speaker"].dropna().unique()) if not eligible.empty else [],
        "snapshot_counts": snapshot_counts,
        "alignment_warning_count": len(warnings),
    }


def build_activation_coverage(
    turns: pd.DataFrame,
    original_index: pd.DataFrame,
    replay_index: pd.DataFrame,
    manifest: dict,
) -> pd.DataFrame:
    statuses = validation_status_by_model(manifest)
    rows = []
    original_keys = set()
    if not original_index.empty:
        original_keys = set(zip(
            original_index["conv_id"], original_index["turn"],
            original_index["speaker"], original_index["layer"],
        ))
    replay_lookup = {}
    if not replay_index.empty:
        for row in replay_index.to_dict("records"):
            replay_lookup[(
                row["conv_id"], row["turn"], row["speaker"],
                row["model"], row["layer"], row["snapshot"],
            )] = row["window"]
    layers_by_model = {}
    for model, details in manifest.get("models", {}).items():
        layers_by_model[model] = details.get("layers", [])
    for turn in turns.to_dict("records"):
        model = turn["model"]
        layers = layers_by_model.get(model, [])
        for layer in layers:
            for snapshot in SNAPSHOT_ORDER:
                replay_key = (
                    turn["conv_id"], turn["turn"], turn["speaker"],
                    model, layer, snapshot,
                )
                has_replay = replay_key in replay_lookup
                has_original = (
                    turn["conv_id"], turn["turn"], turn["speaker"], layer
                ) in original_keys
                rows.append({
                    "conv_id": turn["conv_id"], "turn": turn["turn"],
                    "speaker": turn["speaker"], "model": model,
                    "layer": layer, "snapshot": snapshot,
                    "window": replay_lookup.get(replay_key),
                    "has_original": has_original,
                    "has_replay": has_replay,
                    "validation_status": statuses.get(model),
                    "activation_status": activation_status(
                        snapshot, has_original, has_replay, statuses.get(model)
                    ),
                })
    return pd.DataFrame(rows)


def load_optional_csv(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def build_turn_table(
    data_dir: str,
    replay_dir: str,
    results_dir: str | None = None,
    geometry_path: str | None = None,
    annotations_path: str | None = None,
) -> dict:
    manifest = read_json(os.path.join(replay_dir, "manifest.json"))
    transcripts, turns = load_transcripts(data_dir)
    replay_index = replay_array_index(replay_dir)
    original_index = original_array_index(data_dir, transcripts)
    metadata = turn_metadata(manifest)
    validations = validation_rows(manifest)
    warnings = token_alignment_warnings(metadata)
    coverage = build_activation_coverage(turns, original_index, replay_index, manifest)

    if not metadata.empty:
        metadata_columns = [
            column for column in (
                "conv_id", "turn", "speaker", "prompt_token_count",
                "response_token_count", "response_start_index", "response_end_index",
                "early_response_window", "early_response_text", "final_window",
                "chat_template_sha256", "legacy_validation_eos_included",
                "response_special_tokens_included", "eos_included",
            ) if column in metadata
        ]
        turns = turns.merge(
            metadata[metadata_columns].drop_duplicates(["conv_id", "turn", "speaker"]),
            on=["conv_id", "turn", "speaker"], how="left", validate="one_to_one",
        )
    if not validations.empty:
        similarity = validations.groupby(
            ["conv_id", "turn", "speaker"], as_index=False
        ).agg(
            replay_cosine_similarity=("cosine_similarity", "median"),
            replay_relative_error=("relative_error", "median"),
            replay_norm_ratio=("norm_ratio", "median"),
        )
        turns = turns.merge(
            similarity, on=["conv_id", "turn", "speaker"],
            how="left", validate="one_to_one",
        )
    geometry = load_optional_csv(geometry_path)
    if not geometry.empty:
        keep = [
            column for column in (
                "conv_id", "turn", "speaker", "sp_pc1", "sp_pc2",
                "semantic_velocity", "semantic_acceleration", "basin_leaning",
                "partnerward_basin_velocity", "off_axis_distance",
            ) if column in geometry
        ]
        turns = turns.merge(
            geometry[keep], on=["conv_id", "turn", "speaker"],
            how="left", validate="one_to_one",
        )
    annotations = load_optional_csv(annotations_path)
    if not annotations.empty:
        if annotations.duplicated(["conv_id", "turn"]).any():
            if "annotator_id" not in annotations:
                raise ValueError("Duplicate dashboard annotations require annotator_id.")
            annotations = aggregate_annotations(annotations)
        turns = turns.merge(
            annotations, on=["conv_id", "turn"], how="left", validate="one_to_one"
        )

    persona_columns = [column for column in DYNAMIC_PERSONA_FIELDS if column in turns]
    turns = turns.sort_values(["conv_id", "speaker", "turn"]).copy()
    for column in persona_columns:
        state = turns.groupby(["conv_id", "speaker"])[column].transform(
            lambda values: pd.to_numeric(values, errors="coerce").rolling(3, min_periods=1).mean()
        )
        turns[f"{column}_trailing3"] = state
        baseline = turns[turns["condition"].eq("self_play")].groupby("model")[f"{column}_trailing3"].mean()
        turns[f"{column}_self_play_baseline"] = turns["model"].map(baseline)
        turns[f"{column}_deviation_from_self_play"] = (
            state - turns[f"{column}_self_play_baseline"]
        )
        turns[f"{column}_movement"] = turns.groupby(
            ["conv_id", "speaker"]
        )[f"{column}_trailing3"].diff()

    results = {}
    if results_dir:
        preliminary_manifest_path = os.path.join(
            results_dir, "preliminary_manifest.json"
        )
        if os.path.exists(preliminary_manifest_path):
            results["preliminary_manifest"] = read_json(preliminary_manifest_path)
        for filename, key in (
            ("snapshot_probe_scores.csv", "scores"),
            ("oof_predictions.csv", "oof"),
            ("fold_scores.csv", "folds"),
            ("original_replay_sensitivity.csv", "sensitivity"),
            ("activation_norm_summary.csv", "activation_norms"),
            ("activation_transition_summary.csv", "activation_transitions"),
            ("experiment_1i_partner_transfer.csv", "transfer"),
            ("experiment_1a_measurement_audit.csv", "audit"),
            ("skipped_targets.csv", "skipped"),
            ("artifact_manifest.csv", "artifact_manifest"),
            ("figures/figure_index.csv", "figures"),
        ):
            results[key] = load_optional_csv(os.path.join(results_dir, filename))
    return {
        "manifest": manifest, "transcripts": transcripts, "turns": turns,
        "replay_index": replay_index, "original_index": original_index,
        "coverage": coverage, "metadata": metadata, "validations": validations,
        "warnings": warnings, **results,
    }


def reconstructed_prompt(
    transcript: dict,
    turn_index: int,
    topics_path: str = "configs/topics.yaml",
) -> list[dict]:
    topics = {topic.id: topic for topic in load_topics(topics_path)}
    return [
        {"role": message.role, "content": message.content}
        for message in reconstruct_history(transcript, turn_index, topics)
    ]
