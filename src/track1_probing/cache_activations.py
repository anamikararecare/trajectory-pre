"""Load immutable Track 1 turns with original and validated replay arrays."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from src.track1_probing.variables import (
    add_lagged_behavioral_state,
    add_persona_baselines,
    derive_stance_variables,
    derive_annotation_variables,
    merge_annotations,
)


_REPLAY_KEY = re.compile(
    r"^turn_(?P<turn>\d+)__speaker_(?P<speaker>[^_]+)__model_(?P<model>.+?)__layer_(?P<layer>-?\d+)__snapshot_(?P<snapshot>.+?)__window_(?P<window>\d+)$"
)


GEOMETRY_COLUMNS = (
    "semantic_velocity", "semantic_acceleration", "basin_leaning",
    "partnerward_basin_velocity", "off_axis_distance",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_replay_manifest(data_dir: str, manifest: dict) -> None:
    if manifest.get("validation_only"):
        raise ValueError("A validation-only replay cannot be used for analysis.")
    records = manifest.get("transcripts", [])
    recorded = {item["path"]: item["sha256"] for item in records}
    if len(recorded) != len(records):
        raise ValueError("Replay manifest contains duplicate transcript paths.")
    current_paths = sorted(glob.glob(os.path.join(data_dir, "transcripts", "*.json")))
    current = {os.path.relpath(path, data_dir): _sha256(path) for path in current_paths}
    if recorded != current:
        raise ValueError("Replay manifest transcript hashes do not match the frozen corpus.")


def load_dataset(
    data_dir: str,
    replay_dir: str | None = None,
    annotations: str | None = None,
    geometry_path: str | None = None,
    allow_unvalidated_replay: bool = False,
) -> pd.DataFrame:
    """Load turns and optionally merge gate-approved replay snapshots."""
    validation_by_model = {}
    replay_metadata = {}
    if replay_dir:
        with open(os.path.join(replay_dir, "manifest.json")) as handle:
            manifest = json.load(handle)
        _validate_replay_manifest(data_dir, manifest)
        validation_by_model = {
            model: details.get("validation", {}).get("status", "not_evaluated")
            for model, details in manifest.get("models", {}).items()
        }
        replay_metadata = {
            (item["conv_id"], item["turn"], item["speaker"]): item
            for item in manifest.get("turn_metadata", [])
        }
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "transcripts", "*.json"))):
        with open(path) as handle:
            transcript = json.load(handle)
        conv_id = transcript["conv_id"]
        original_path = os.path.join(data_dir, "activations", f"{conv_id}.npz")
        original = np.load(original_path) if os.path.exists(original_path) else None
        layers = sorted({int(key.split("__")[0]) for key in original.files}) if original is not None else []
        replay_path = os.path.join(replay_dir, f"{conv_id}.npz") if replay_dir else None
        replay = np.load(replay_path) if replay_path and os.path.exists(replay_path) else None
        replay_lookup = {}
        if replay is not None:
            for key in replay.files:
                match = _REPLAY_KEY.match(key)
                if not match:
                    raise ValueError(f"Malformed replay activation key: {key}")
                identity = (int(match["turn"]), match["speaker"], match["model"], int(match["layer"]), match["snapshot"])
                if identity in replay_lookup:
                    raise ValueError(f"Duplicate replay activation identity: {identity}")
                replay_lookup[identity] = key
        for turn in transcript["turns"]:
            model = turn.get("model", transcript.get(f"agent_{turn['speaker']}_model"))
            row = {
                "conv_id": conv_id, "topic_id": transcript["topic_id"],
                "condition": transcript.get("condition", "unknown"),
                "speaker": turn["speaker"], "model": model, "role": turn["role"],
                "turn": turn["turn"], "agent_turn": turn.get("agent_turn"),
                "transcript_context_text": "\n".join(
                    prior.get("text", "") for prior in transcript["turns"][: turn["turn"]]
                ),
                "text": turn.get("text", ""), "stance_score": turn.get("stance_score"),
                "stance_confidence": turn.get("stance_confidence"),
                "replay_validation_status": validation_by_model.get(model),
                "early_response_text": replay_metadata.get(
                    (conv_id, turn["turn"], turn["speaker"]), {}
                ).get("early_response_text"),
            }
            for layer in layers:
                key = f"{layer}__{turn['turn']}"
                row[f"layer_{layer}"] = original[key] if key in original.files else None
            status = validation_by_model.get(model)
            if replay is not None and (status == "passed" or allow_unvalidated_replay):
                for (recorded_turn, speaker, replay_model, layer, snapshot), key in replay_lookup.items():
                    if (recorded_turn == turn["turn"] and speaker == turn["speaker"] and replay_model == model):
                        row[f"layer_{layer}__{snapshot}"] = replay[key]
            rows.append(row)
        if original is not None:
            original.close()
        if replay is not None:
            replay.close()
    frame = derive_stance_variables(pd.DataFrame(rows))
    if geometry_path:
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
        geometry = pd.read_csv(geometry_path)
        required = {"conv_id", "turn", "speaker", *GEOMETRY_COLUMNS}
        missing = required.difference(geometry.columns)
        if missing:
            raise ValueError(f"Geometry file is missing required columns: {sorted(missing)}")
        geometry_columns = ["conv_id", "turn", "speaker", *GEOMETRY_COLUMNS]
        frame = frame.merge(
            geometry[geometry_columns], on=["conv_id", "turn", "speaker"],
            how="left", validate="one_to_one",
        )
    frame = merge_annotations(frame, annotations)
    frame = derive_annotation_variables(frame)
    frame = add_persona_baselines(frame)
    return add_lagged_behavioral_state(frame)


def get_layer_columns(df: pd.DataFrame, snapshot: str | None = None) -> list[str]:
    if snapshot is None:
        columns = [column for column in df if column.startswith("layer_") and "__" not in column]
    else:
        columns = [column for column in df if column.startswith("layer_") and column.endswith(f"__{snapshot}")]
    return sorted(columns, key=lambda column: int(column.split("_", 1)[1].split("__", 1)[0]))


def get_snapshots(df: pd.DataFrame) -> list[str]:
    found = {column.split("__", 1)[1] for column in df if column.startswith("layer_") and "__" in column}
    order = ["pre_generation", "early_response", "full_response", "final_window", "final_token"]
    return [snapshot for snapshot in order if snapshot in found]
