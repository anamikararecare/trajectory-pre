"""Deterministic activation replay for immutable Track 1 transcripts."""

from __future__ import annotations

import glob
import gc
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.common.debate_prompts import build_opening_message, build_system_prompt, load_topics
from src.common.llm_client import ChatMessage, build_client, load_model_registry

PRIMARY_SNAPSHOTS = ("pre_generation", "early_response", "full_response", "final_window")
ALL_SNAPSHOTS = PRIMARY_SNAPSHOTS + ("final_token",)
MEDIAN_COSINE_TOLERANCE = 0.999
P05_COSINE_TOLERANCE = 0.995


def activation_key(
    turn: int, speaker: str, model: str, layer: int, snapshot: str, window: int
) -> str:
    if snapshot not in ALL_SNAPSHOTS:
        raise ValueError(f"Unknown snapshot: {snapshot}")
    return (
        f"turn_{turn}__speaker_{speaker}__model_{model}__layer_{layer}"
        f"__snapshot_{snapshot}__window_{window}"
    )


def snapshot_window(snapshot: str, metadata: dict) -> int:
    """Return the token-window size recorded for a replay snapshot."""
    if snapshot not in ALL_SNAPSHOTS:
        raise ValueError(f"Unknown snapshot: {snapshot}")
    if snapshot in ("pre_generation", "final_token"):
        return 1
    if snapshot == "final_window":
        return metadata["final_window"]
    return metadata[f"{snapshot}_window"]


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def reconstruct_history(transcript: dict, turn_index: int, topics_by_id: dict) -> list[ChatMessage]:
    """Reconstruct the speaking model's perspective immediately before a turn."""
    turn = transcript["turns"][turn_index]
    speaker = turn["speaker"]
    topic = topics_by_id[transcript["topic_id"]]
    paper = bool(transcript.get("paper_compatible", False))
    history = [
        ChatMessage(
            "system",
            build_system_prompt(topic, transcript[f"agent_{speaker}_role"], paper),
        ),
        ChatMessage("user", build_opening_message(topic, paper)),
    ]
    for prior in transcript["turns"][:turn_index]:
        history.append(
            ChatMessage(
                "assistant" if prior["speaker"] == speaker else "user",
                prior["text"],
            )
        )
    return history


def comparison_metrics(original: np.ndarray, replayed: np.ndarray) -> dict:
    a, b = original.astype(np.float64), replayed.astype(np.float64)
    a_norm, b_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    denom = a_norm * b_norm
    return {
        "cosine_similarity": float(np.dot(a, b) / denom) if denom else float("nan"),
        "relative_error": float(np.linalg.norm(b - a) / a_norm) if a_norm else float("nan"),
        "norm_ratio": float(b_norm / a_norm) if a_norm else float("nan"),
        "mean_bias": float(np.mean(b - a)),
    }


def summarize_validation(rows: list[dict]) -> dict:
    if not rows:
        return {
            "status": "not_evaluated",
            "n": 0,
            "reason": "no original full-response activations available for this model",
        }
    cosine = np.asarray([row["cosine_similarity"] for row in rows], dtype=float)
    norm_ratio = np.asarray([row["norm_ratio"] for row in rows], dtype=float)
    median = float(np.nanmedian(cosine))
    p05 = float(np.nanpercentile(cosine, 5))
    median_norm = float(np.nanmedian(norm_ratio))
    layer_p05 = {}
    for layer in sorted({row.get("layer") for row in rows if row.get("layer") is not None}):
        values = [row["cosine_similarity"] for row in rows if row.get("layer") == layer]
        layer_p05[str(layer)] = float(np.nanpercentile(values, 5))
    min_layer_p05 = min(layer_p05.values(), default=p05)
    length_correlation = float("nan")
    if len(rows) >= 10 and all("response_token_count" in row for row in rows):
        lengths = pd.Series([row["response_token_count"] for row in rows]).rank().to_numpy()
        errors = pd.Series([row["relative_error"] for row in rows]).rank().to_numpy()
        if np.std(lengths) and np.std(errors):
            length_correlation = float(np.corrcoef(lengths, errors)[0, 1])
    no_length_trend = not np.isfinite(length_correlation) or abs(length_correlation) <= 0.2
    passed = (
        median >= MEDIAN_COSINE_TOLERANCE
        and p05 >= P05_COSINE_TOLERANCE
        and min_layer_p05 >= P05_COSINE_TOLERANCE
        and 0.98 <= median_norm <= 1.02
        and no_length_trend
    )
    return {
        "status": "passed" if passed else "failed",
        "n": len(rows),
        "median_cosine_similarity": median,
        "p05_cosine_similarity": p05,
        "layer_p05_cosine_similarity": layer_p05,
        "length_relative_error_spearman": length_correlation,
        "median_relative_error": float(np.nanmedian([r["relative_error"] for r in rows])),
        "median_norm_ratio": median_norm,
        "max_abs_layerwise_bias": float(np.nanmax(np.abs([r["mean_bias"] for r in rows]))),
        "tolerances": {
            "median_cosine_similarity": MEDIAN_COSINE_TOLERANCE,
            "p05_cosine_similarity": P05_COSINE_TOLERANCE,
            "median_norm_ratio": [0.98, 1.02],
            "max_abs_length_error_correlation": 0.2,
        },
        "comparability_warning": None if passed else "Excluded from primary analyses.",
    }


def replay_eligibility(transcript: dict, turn: dict, registry: dict) -> tuple[bool, str]:
    current = registry.get(turn["model"])
    if not current:
        return False, "model missing from registry"
    if current.get("backend") != "hf_local" or not current.get("hookable", False):
        return False, "model is not locally hookable"
    saved = transcript.get("model_specs", {}).get(f"agent_{turn['speaker']}", {})
    if saved.get("hf_id") and saved["hf_id"] != current.get("hf_id"):
        return False, "saved and current model identifiers differ"
    saved_revision = saved.get("resolved_revision") or saved.get("revision")
    if not saved_revision:
        return False, "exact saved model revision unavailable"
    current_revision = current.get("revision")
    if saved_revision and current_revision and saved_revision != current_revision:
        return False, "saved and current model revisions differ"
    return True, "eligible"


def _select_validation(tasks: list[dict], limit: int) -> list[dict]:
    candidates = [task for task in tasks if task["original_keys"]]
    candidates.sort(
        key=lambda task: (
            len(task["turn"]["text"]),
            task["transcript"]["topic_id"],
            task["transcript"]["conv_id"],
            task["turn"]["turn"],
        )
    )
    if len(candidates) <= limit:
        return candidates
    return [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, limit)]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def replay_corpus(
    data_dir: str,
    out_root: str | None = None,
    replay_id: str | None = None,
    registry_path: str = "configs/models.yaml",
    topics_path: str = "configs/topics.yaml",
    model_keys: Iterable[str] | None = None,
    load_in_4bit: bool = False,
    validation_turns: int = 24,
    allow_failed_validation: bool = False,
    validation_only: bool = False,
) -> Path:
    """Validate first, then replay exact recorded text for eligible local models."""
    data_path = Path(data_dir)
    replay_id = replay_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(out_root or data_path / "replayed_activations") / replay_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Replay directory already contains files: {root}")
    root.mkdir(parents=True, exist_ok=True)

    registry = load_model_registry(registry_path)
    selected = set(model_keys or registry)
    unknown_models = selected.difference(registry)
    if unknown_models:
        raise ValueError(f"Unknown model keys: {sorted(unknown_models)}")
    topics = {topic.id: topic for topic in load_topics(topics_path)}
    tasks_by_model, eligibility, transcript_records = defaultdict(list), [], []

    for path in sorted(glob.glob(str(data_path / "transcripts" / "*.json"))):
        with open(path) as handle:
            transcript = json.load(handle)
        transcript_records.append(
            {"conv_id": transcript["conv_id"], "path": os.path.relpath(path, data_path), "sha256": _sha256(path)}
        )
        original_path = data_path / "activations" / f"{transcript['conv_id']}.npz"
        original_files = set()
        if original_path.exists():
            with np.load(original_path) as original:
                original_files = set(original.files)
        for index, turn in enumerate(transcript["turns"]):
            ok, reason = replay_eligibility(transcript, turn, registry)
            if turn["model"] not in selected:
                ok, reason = False, "model not selected"
            eligibility.append(
                {key: value for key, value in {
                    "conv_id": transcript["conv_id"], "turn": turn["turn"],
                    "speaker": turn["speaker"], "model": turn["model"],
                    "eligible": ok, "reason": reason,
                }.items()}
            )
            if ok:
                layers = registry[turn["model"]].get("default_layers", [])
                tasks_by_model[turn["model"]].append(
                    {
                        "transcript": transcript, "turn_index": index, "turn": turn,
                        "layers": layers, "original_path": str(original_path),
                        "original_keys": [f"{layer}__{turn['turn']}" for layer in layers if f"{layer}__{turn['turn']}" in original_files],
                    }
                )

    manifest = {
        "schema_version": 1, "replay_id": replay_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_data_dir": str(data_path.resolve()), "frozen_corpus": True,
        "sampling_disabled": True, "validation_only": validation_only,
        "original_pooling": "response tokens including inferred terminal EOS",
        "primary_pooling": "recorded response tokens excluding EOS and special tokens",
        "response_source": "exact recorded transcript text",
        "primary_snapshots": list(PRIMARY_SNAPSHOTS),
        "sensitivity_snapshots": ["final_token"], "code_revision": _code_revision(),
        "precision": "bfloat16 on CUDA; float32 on CPU",
        "quantization": "4bit" if load_in_4bit else "none",
        "transcripts": transcript_records, "eligibility": eligibility,
        "models": {}, "turn_metadata": [],
    }
    for model_key in sorted(selected):
        spec = registry[model_key]
        manifest["models"][model_key] = {
            "hf_id": spec.get("hf_id"),
            "layers": spec.get("default_layers", []),
            "validation": {
                "status": "not_evaluated", "n": 0,
                "reason": "no eligible recorded turns",
            },
        }
    arrays_by_conv: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    client = None
    for model_key, tasks in tasks_by_model.items():
        if client is not None:
            del client
            client = None
            gc.collect()
            _empty_cuda_cache()
        saved_revisions = {
            (task["transcript"].get("model_specs", {}).get(
                f"agent_{task['turn']['speaker']}", {}
            ).get("resolved_revision"))
            for task in tasks
        } - {None}
        if len(saved_revisions) != 1:
            manifest["models"][model_key] = {
                "validation": {"status": "not_evaluated", "reason": "non-unique saved revision"}
            }
            continue
        pinned_registry = {key: dict(value) for key, value in registry.items()}
        pinned_revision = next(iter(saved_revisions))
        pinned_registry[model_key]["revision"] = pinned_revision
        pinned_registry[model_key]["tokenizer_revision"] = pinned_revision
        client = build_client(model_key, pinned_registry, load_in_4bit=load_in_4bit)
        validation_rows, cached = [], {}
        for task in _select_validation(tasks, validation_turns):
            turn = task["turn"]
            snapshots, metadata = client.teacher_force_snapshots(
                reconstruct_history(task["transcript"], task["turn_index"], topics),
                turn["text"], task["layers"],
                task["transcript"].get("generation", {}).get("max_new_tokens"),
            )
            task_id = (task["transcript"]["conv_id"], turn["turn"])
            cached[task_id] = (snapshots, metadata)
            with np.load(task["original_path"]) as original:
                for layer in task["layers"]:
                    old_key = f"{layer}__{turn['turn']}"
                    if old_key in original.files:
                        validation_rows.append({
                            "conv_id": task_id[0], "turn": turn["turn"],
                            "speaker": turn["speaker"], "model": model_key, "layer": layer,
                            "topic_id": task["transcript"]["topic_id"],
                            "condition": task["transcript"].get("condition"),
                            "role": turn.get("role"),
                            "response_token_count": metadata["response_token_count"],
                            **comparison_metrics(original[old_key], snapshots[(layer, "legacy_full_response")]),
                        })
        validation = summarize_validation(validation_rows)
        spec = registry[model_key]
        manifest["models"][model_key] = {
            "hf_id": spec.get("hf_id"), "model_revision": pinned_revision,
            "tokenizer_revision": pinned_revision,
            "layers": spec.get("default_layers", []), "validation": validation,
            "validation_rows": validation_rows,
        }
        if validation_only or (validation["status"] != "passed" and not allow_failed_validation):
            continue
        for task in tasks:
            turn = task["turn"]
            task_id = (task["transcript"]["conv_id"], turn["turn"])
            if task_id in cached:
                snapshots, metadata = cached[task_id]
            else:
                snapshots, metadata = client.teacher_force_snapshots(
                    reconstruct_history(task["transcript"], task["turn_index"], topics),
                    turn["text"], task["layers"],
                    task["transcript"].get("generation", {}).get("max_new_tokens"),
                )
            for (layer, snapshot), value in snapshots.items():
                if snapshot not in ALL_SNAPSHOTS:
                    continue
                window = snapshot_window(snapshot, metadata)
                arrays_by_conv[task_id[0]][activation_key(
                    turn["turn"], turn["speaker"], model_key, layer, snapshot, window
                )] = value
            manifest["turn_metadata"].append({
                "conv_id": task_id[0], "turn": turn["turn"], "speaker": turn["speaker"],
                "model": model_key, "validation_sample": task_id in cached, **metadata,
            })

    if client is not None:
        del client
        client = None
        gc.collect()
        _empty_cuda_cache()

    for conv_id, arrays in arrays_by_conv.items():
        np.savez_compressed(root / f"{conv_id}.npz", **arrays)
    with open(root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return root
